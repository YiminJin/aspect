// Benchmark-only frozen-history diagnostic; never publishes a timestep.
#include <cstring>

namespace aspect
{
  namespace BP3Benchmark
  {
    template <int dim>
    struct JunctionSnapshot
    {
      std::unique_ptr<SimulatorAccess<dim>> simulator;
      std::unique_ptr<LinearAlgebra::BlockVector> bulk;
      std::vector<std::vector<double>> properties;
      std::vector<Point<dim>> vertices;
      std::vector<double> committed_V;
      std::map<types::particle_index,std::pair<Point<dim>,std::vector<double>>> particles;
    };

    template <int dim>
    JunctionSnapshot<dim> &junction_snapshot()
    {
      static JunctionSnapshot<dim> state;
      return state;
    }

    template <int dim>
    void capture_junction_history(const SimulatorAccess<dim> &sim, AffineConstraints<double> &)
    {
      const bool theta_audit=std::getenv("ASPECT_BP3_THETA_EXACT_DIAGNOSTIC");
      const bool work_audit=std::getenv("ASPECT_BP3_WORK_MEASURE");
      const bool within_step=std::getenv("ASPECT_BP3_WITHIN_STEP_DIAGNOSTIC");
      if ((!within_step && !theta_audit && !work_audit && !std::getenv("ASPECT_BP3_JUNCTION_DIAGNOSTIC") && !std::getenv("ASPECT_FAULT_HISTORY_FE"))
          || sim.get_timestep_number()!=(within_step ? 10u : work_audit ? 0u : theta_audit ? 13u : 12u))
        return;
      const auto &manager=sim.get_reconstructed_fault_manager();
      if (manager.get_faults().empty() || !manager.slip_rates_are_initialized()) return;
      if (work_audit && manager.get_property_index("background tractions")==numbers::invalid_unsigned_int) return;
      // The solver rebuilds physical constraints after preparing the frozen
      // constitutive state. The last capture is therefore its actual input,
      // after ordinary advection/transfer, not a post-solve visualization.
      auto &state=junction_snapshot<dim>();
      state.simulator=std::make_unique<SimulatorAccess<dim>>(sim);
      state.bulk=std::make_unique<LinearAlgebra::BlockVector>(
        sim.introspection().index_sets.system_partitioning,sim.get_mpi_communicator());
      *state.bulk=sim.get_solution();
      state.properties.clear(); state.vertices.clear(); state.particles.clear();
      const auto &fault=manager.get_fault(0);
      for (unsigned int i=0;i<fault.n_vertices();++i)
        {
          const auto p=fault.get_properties(i);
          state.properties.emplace_back(p.begin(),p.end());
          state.vertices.push_back(fault.vertex(i));
        }
      state.committed_V=manager.get_timestep_committed_slip_rate(0);
      for (const auto &particle:sim.get_phase_field_handler().get_associated_particle_manager().get_particle_handler())
        {
          const auto p=particle.get_properties();
          state.particles.emplace(particle.get_id(),
            std::make_pair(particle.get_location(),std::vector<double>(p.begin(),p.end())));
        }
      if (within_step)
        {
          // Compare identical incoming represented bits in the two disposable
          // processes, not only their post-rollback norms.
          std::uint64_t hash=14695981039346656037ULL;
          const auto add=[&](const double value)
          {
            unsigned char bytes[sizeof(double)];std::memcpy(bytes,&value,sizeof(value));
            for (const auto byte:bytes) { hash^=byte;hash*=1099511628211ULL; }
          };
          for (const auto i:sim.get_dof_handler().locally_owned_dofs()) add((*state.bulk)[i]);
          for (const auto &row:state.properties) for (const auto value:row) add(value);
          for (const auto value:state.committed_V) add(value);
          for (const auto &p:state.particles)
            {add(p.first);for(unsigned int d=0;d<dim;++d)add(p.second.first[d]);for(const auto v:p.second.second)add(v);}
          std::ofstream out(sim.get_output_directory()+"incoming_rank"+
            std::to_string(Utilities::MPI::this_mpi_process(sim.get_mpi_communicator()))+".txt");
          out<<hash<<'\n';
        }
    }

    template <int dim>
    void verify_junction_rollback(const SolverControl &control)
    {
      if ((!std::getenv("ASPECT_BP3_THETA_EXACT_DIAGNOSTIC") && !std::getenv("ASPECT_BP3_JUNCTION_DIAGNOSTIC")
           && !std::getenv("ASPECT_FAULT_HISTORY_FE") && !std::getenv("ASPECT_BP3_WORK_MEASURE")
           && !std::getenv("ASPECT_BP3_WITHIN_STEP_DIAGNOSTIC")) || control.last_check()!=SolverControl::failure)
        return;
      const auto &state=junction_snapshot<dim>();
      AssertThrow(state.simulator && state.bulk,ExcMessage("Junction diagnostic lacks a frozen-state snapshot."));
      const auto &sim=*state.simulator;
      // Compare represented owned entries, not Epetra vector maps: a copied
      // snapshot may carry a different ghost layout from the restored vector.
      double local_bulk_error=0.;
      for (const auto i:sim.get_dof_handler().locally_owned_dofs())
        local_bulk_error=std::max(local_bulk_error,std::abs((*state.bulk)[i]-sim.get_solution()[i]));
      const double bulk_error=Utilities::MPI::max(local_bulk_error,sim.get_mpi_communicator());
      bool equal=bulk_error==0.;
      const auto &manager=sim.get_reconstructed_fault_manager();
      const auto &fault=manager.get_fault(0);
      equal=equal && manager.get_timestep_committed_slip_rate(0)==state.committed_V
            && manager.get_slip_rate(0)==state.committed_V;
      for (unsigned int i=0;i<fault.n_vertices();++i)
        {
          const auto p=fault.get_properties(i);
          equal=equal && p.size()==state.properties[i].size()
                && std::memcmp(p.data(),state.properties[i].data(),p.size()*sizeof(double))==0
                && fault.vertex(i)==state.vertices[i];
        }
      unsigned int count=0;
      for (const auto &particle:sim.get_phase_field_handler().get_associated_particle_manager().get_particle_handler())
        {
          ++count;
          const auto p=particle.get_properties();
          const auto saved=state.particles.find(particle.get_id());
          equal=equal && saved!=state.particles.end();
          if (saved!=state.particles.end())
            equal=equal && particle.get_location()==saved->second.first
                  && p.size()==saved->second.second.size()
                  && std::memcmp(p.data(),saved->second.second.data(),p.size()*sizeof(double))==0;
        }
      equal=equal && count==state.particles.size();
      AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(equal),sim.get_mpi_communicator()),
                  ExcMessage("Noncommitting junction solve changed bulk, V, history, or geometry."));
      if (sim.get_pcout().is_active())
        {
          const auto position=[&](const std::string &name)
          { return manager.get_property_information()[manager.get_property_index(name)].position; };
          const auto theta=position("phase field fault state"), C=position("phase field fault cohesive traction"),
                     I=position("phase field fault previous I h"), bg=position("background tractions");
          std::ofstream out(sim.get_output_directory()+"noncommitting_history.csv");
          out<<"node,x,y,V_committed,Theta_retained,C_retained,Ih_retained,bg_shear,bg_normal\n";
          for (unsigned int i=0;i<state.vertices.size();++i)
            out<<std::setprecision(17)<<i<<','<<state.vertices[i][0]<<','<<state.vertices[i][1]<<','
               <<state.committed_V[i]<<','<<state.properties[i][theta]<<','<<state.properties[i][C]<<','
               <<state.properties[i][I]<<','<<state.properties[i][bg]<<','<<state.properties[i][bg+1]<<'\n';
          sim.get_pcout()<<"BP3 noncommitting rollback verified: complete bulk vector, committed/current V, "
                         <<"all particle and surface properties, particle positions/IDs and fault vertices unchanged."
                         <<std::endl;
        }
    }
  }
}
