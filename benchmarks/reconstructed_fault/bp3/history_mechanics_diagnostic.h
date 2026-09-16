// Disposable benchmark diagnostics; no retained particle/fault data are mutated.
#include "history_common_tests.h"
namespace aspect
{
  namespace BP3Benchmark
  {
    template <int dim>
    void export_checkpoint_bulk(const SimulatorAccess<dim> &sim)
    {
      if (!std::getenv("ASPECT_BP3_EXPORT_CHECKPOINT_BULK")) return;
      std::ofstream out(sim.get_output_directory()+"bulk_owned_rank"+
                        std::to_string(Utilities::MPI::this_mpi_process(sim.get_mpi_communicator()))+".csv");
      out.exceptions(std::ios::failbit|std::ios::badbit);
      out<<std::setprecision(17)<<"cell,local,value\n";
      std::vector<types::global_dof_index> indices(sim.get_fe().n_dofs_per_cell());
      // Checkpoints may repartition/renumber unchanged cells. CellId plus the
      // local FE index preserves the exact polynomial independently of ranks.
      for (const auto &cell:sim.get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(indices);
            for (unsigned int i=0;i<indices.size();++i)
              if (sim.introspection().is_stokes_component(sim.get_fe().system_to_component_index(i).first))
                out<<cell->id().to_string()<<','<<i<<','<<sim.get_solution()[indices[i]]<<'\n';
          }
      out.close();
      if (sim.get_pcout().is_active())
        {
          std::ofstream meta(sim.get_output_directory()+"checkpoint_bulk_metadata.csv");
          meta<<std::setprecision(17)<<"step,time,dt,cells\n"<<sim.get_timestep_number()<<','<<sim.get_time()<<','
              <<sim.get_timestep()<<','<<sim.get_triangulation().n_global_active_cells()<<'\n';
        }
      // A rank-local throw reaches MPI_Abort; all rank files must have closed
      // before any process requests this intentional extraction stop.
      MPI_Barrier(sim.get_mpi_communicator());
      sim.get_pcout()<<"Checkpoint bulk export complete; no timestep entered."<<std::endl;
      throw std::runtime_error("Intentional checkpoint extraction stop.");
    }

    template <int dim>
    void compare_frozen_surface(const SimulatorAccess<dim> &sim, AffineConstraints<double> &pending)
    {
      const char *directory=std::getenv("ASPECT_BP3_FROZEN_SURFACE");
      if (!directory || !history_load_ready) return;
      history_load_ready=false;
      AssertThrow(sim.get_timestep_number()==12,ExcMessage("Frozen surface audit requires step 12."));
      const auto &intro=sim.introspection();
      LinearAlgebra::BlockVector owned(intro.index_sets.system_partitioning,sim.get_mpi_communicator());
      owned=sim.get_solution();
      std::map<std::string,std::vector<types::global_dof_index>> cells;
      for (const auto &cell:sim.get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {auto &indices=cells[cell->id().to_string()]; indices.resize(sim.get_fe().n_dofs_per_cell());cell->get_dof_indices(indices);}
      std::set<types::global_dof_index> found;
      std::string line;
      for (unsigned int rank=0;rank<4;++rank)
        {
          std::ifstream in(std::string(directory)+"/bulk_owned_rank"+std::to_string(rank)+".csv");
          std::getline(in,line);
          while (std::getline(in,line))
            {
              std::replace(line.begin(),line.end(),',',' ');
              std::istringstream row(line); std::string id;unsigned int local;double value;
              AssertThrow(static_cast<bool>(row>>id>>local>>value),ExcMessage("Invalid checkpoint cell coefficient."));
              const auto cell=cells.find(id);if (cell==cells.end()) continue;
              AssertThrow(local<cell->second.size(),ExcMessage("Incompatible local FE index."));
              const auto i=cell->second[local];
              if (sim.get_dof_handler().locally_owned_dofs().is_element(i))
                {
                  if (found.count(i)) AssertThrow(owned[i]==value,ExcMessage("Saved shared Stokes coefficient disagrees."));
                  owned[i]=value;found.insert(i);
                }
            }
        }
      AssertThrow(found.size()==intro.index_sets.stokes_partitioning[0].n_elements()+intro.index_sets.stokes_partitioning[1].n_elements(),
                  ExcMessage("Incomplete saved Stokes polynomial extraction."));
      AffineConstraints<double> physical(pending); physical.close(); physical.distribute(owned);
      LinearAlgebra::BlockVector frozen(sim.get_current_linearization_point()); frozen=owned;
      const auto &manager=sim.get_reconstructed_fault_manager();
      ReconstructedFaultVector V(1,manager.get_timestep_committed_slip_rate(0));
      std::ifstream rates(std::string(directory)+"/fault_12.csv");
      std::getline(rates,line); unsigned int count=0;
      while (std::getline(rates,line))
        {
          std::replace(line.begin(),line.end(),',',' ');
          std::istringstream row(line); double xd,x,y,time,dt,v;
          AssertThrow(static_cast<bool>(row>>xd>>x>>y>>time>>dt>>v),ExcMessage("Invalid saved fault row."));
          AssertThrow(count<V[0].size() && Point<dim>(x,y).distance(manager.get_fault(0).vertex(count))<1e-8
                      && std::abs(time-sim.get_time())<1e-6 && std::abs(dt-sim.get_timestep())<1e-6,
                      ExcMessage("Frozen comparison does not match saved geometry/time."));
          V[0][count++]=v;
        }
      AssertThrow(count==V[0].size(),ExcMessage("Incomplete saved fault rates."));
      sim.get_reconstructed_fault_surface_system().linearize_surface_system(frozen,V);
      common_history_tests(sim,frozen);
      // One-sided perturbations preserve feasibility at contact. Check the
      // unreplaced residual and K=-dR/dV, not the restricted convergence rows.
      auto &surface=sim.get_reconstructed_fault_surface_system();
      std::ofstream fd;
      if (sim.get_pcout().is_active())
        {fd.open(sim.get_output_directory()+"history_tangent_fd.csv");fd<<std::setprecision(17)<<"mode,h,node,finite_difference,minus_K\n";}
      for (unsigned int mode=0;mode<2;++mode)
        {
          if (mode) setenv("ASPECT_FAULT_HISTORY_FE","1",1);
          const auto base=surface.linearize_surface_system(frozen,V);
          ReconstructedFaultVector direction=V,action;
          std::fill(direction[0].begin(),direction[0].end(),0.);direction[0][756]=1.;
          surface.apply_surface_jacobian(direction,action);
          for (double h:{1e-17,1e-18,1e-19})
            {
              auto trial=V;trial[0][756]+=h;
              const auto response=surface.evaluate_surface_residual(frozen,trial);
              if (fd) for (unsigned int node:{755u,756u,757u})
                fd<<mode<<','<<h<<','<<node<<','<<(response.values[0][node]-base.values[0][node])/h<<','<<-action[0][node]<<'\n';
            }
        }
      unsetenv("ASPECT_FAULT_HISTORY_FE");
      if (fd) fd.close();
      MPI_Barrier(sim.get_mpi_communicator());
      sim.get_pcout()<<"Frozen surface comparison complete at saved step-12 bulk/V with retained step-11 history."<<std::endl;
      throw std::runtime_error("Intentional frozen surface comparison stop before Newton.");
    }
  }
}
