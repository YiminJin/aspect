// One frozen comparison followed, only on sign reversal, by one disposable solve.
namespace aspect
{
  namespace BP3Benchmark
  {
    bool theta_exact_ready=false;

    std::vector<std::vector<double>> theta_audit_csv(const std::string &path)
    {
      std::ifstream in(path);
      AssertThrow(in,ExcMessage("Cannot open theta audit input: "+path));
      std::string line;
      std::getline(in,line);
      std::vector<std::vector<double>> rows;
      while (std::getline(in,line))
        {
          std::istringstream row(line); std::string field;
          rows.emplace_back();
          while (std::getline(row,field,',')) rows.back().push_back(std::stod(field));
        }
      return rows;
    }

    template <int dim>
    void exact_theta_comparison(const SimulatorAccess<dim> &sim, AffineConstraints<double> &pending)
    {
      const char *input=std::getenv("ASPECT_BP3_THETA_EXACT_DIAGNOSTIC");
      if (!input || !theta_exact_ready) return;
      theta_exact_ready=false;
      const std::string directory(input);
      AssertThrow(sim.get_timestep_number()==13 && std::getenv("ASPECT_FAULT_NONCOMMITTING_DIAGNOSTIC"),
                  ExcMessage("Theta audit requires disposable mechanics 13."));
      const auto &manager=sim.get_reconstructed_fault_manager();
      const auto &fault=manager.get_fault(0);
      const auto before=theta_audit_csv(directory+"/fault_12.csv");
      const auto after=theta_audit_csv(directory+"/fault_13.csv");
      const auto expected=theta_audit_csv(directory+"/expected_weak.csv");
      AssertThrow(fault.n_vertices()==1236 && after.size()==fault.n_vertices() && before.size()==after.size(),
                  ExcMessage("Theta audit requires the saved 50-m fault."));
      const auto state=manager.get_property_information()[manager.get_property_index("phase field fault state")].position;
      ReconstructedFaultVector V(1,std::vector<double>(after.size()));
      for (unsigned int i=0;i<after.size();++i)
        {
          AssertThrow(Point<dim>(after[i][1],after[i][2]).distance(fault.vertex(i))<1e-9,
                      ExcMessage("Theta audit fault geometry differs."));
          AssertThrow(fault.get_properties(i)[state]==before[i][6],ExcMessage("Mechanics 13 lacks retained Theta12."));
          V[0][i]=after[i][5];
        }
      AssertThrow(std::abs(sim.get_time()-after[0][3])<1e-6 && std::abs(sim.get_timestep()-after[0][4])<1e-6,
                  ExcMessage("Theta audit left the saved mechanics-13 clock."));

      // Import only the exact saved Stokes polynomial. History comes from the
      // ordinary step-12 checkpoint/advection/transfer path, never from step 13.
      const auto &intro=sim.introspection();
      LinearAlgebra::BlockVector owned(intro.index_sets.system_partitioning,sim.get_mpi_communicator());
      owned=sim.get_solution();
      std::map<std::string,std::vector<types::global_dof_index>> cells;
      for (const auto &cell:sim.get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {auto &dofs=cells[cell->id().to_string()];dofs.resize(sim.get_fe().n_dofs_per_cell());cell->get_dof_indices(dofs);}
      std::set<types::global_dof_index> found;
      for (unsigned int rank=0;rank<4;++rank)
        {
          std::ifstream in(directory+"/bulk_owned_rank"+std::to_string(rank)+".csv");
          AssertThrow(in,ExcMessage("Missing saved Stokes polynomial."));
          std::string line;std::getline(in,line);
          while (std::getline(in,line))
            {
              std::replace(line.begin(),line.end(),',',' ');
              std::istringstream row(line);std::string id;unsigned int local;double value;
              AssertThrow(row>>id>>local>>value,ExcMessage("Invalid saved Stokes coefficient."));
              const auto cell=cells.find(id);if (cell==cells.end())continue;
              const auto i=cell->second.at(local);
              if (sim.get_dof_handler().locally_owned_dofs().is_element(i))
                {
                  if (found.count(i))AssertThrow(owned[i]==value,ExcMessage("Inconsistent shared Stokes value."));
                  owned[i]=value;found.insert(i);
                }
            }
        }
      AssertThrow(found.size()==intro.index_sets.stokes_partitioning[0].n_elements()+intro.index_sets.stokes_partitioning[1].n_elements(),
                  ExcMessage("Incomplete saved Stokes field."));
      AffineConstraints<double> physical(pending);physical.close();physical.distribute(owned);
      LinearAlgebra::BlockVector frozen(sim.get_current_linearization_point());frozen=owned;
      auto &surface=sim.get_reconstructed_fault_surface_system();
      unsetenv("ASPECT_FAULT_THETA_UPDATE_DIAGNOSTIC");
      setenv("ASPECT_FAULT_THETA_QP_EXPORT",(sim.get_output_directory()+"frozen_A").c_str(),1);
      const auto A=surface.linearize_surface_system(frozen,V);
      double error=0.;
      for (unsigned int i=0;i<after.size();++i)
        {
          const double scale=std::max({1.,std::abs(expected[i][2]),std::abs(expected[i][4])});
          error=std::max(error,std::abs(A.friction_traction[0][i]-expected[i][4])/scale);
          error=std::max(error,std::abs(A.values[0][i]-expected[i][6])/scale);
        }
      sim.get_pcout()<<std::setprecision(17)<<"Theta exact A reproduction: physical-term relative error="<<error<<std::endl;
      AssertThrow(error<1e-10,ExcMessage("Original saved mechanical friction/residual was not reproduced."));
      setenv("ASPECT_FAULT_THETA_UPDATE_DIAGNOSTIC",(directory+"/frozen_update.txt").c_str(),1);
      setenv("ASPECT_FAULT_THETA_QP_EXPORT",(sim.get_output_directory()+"frozen_B").c_str(),1);
      const auto B=surface.linearize_surface_system(frozen,V);
      const unsigned int contact=796;
      const bool reverse=A.values[0][contact]<0. && B.values[0][contact]>0.;
      if (sim.get_pcout().is_active())
        {
          std::ofstream out(sim.get_output_directory()+"exact_weak.csv");
          out<<std::setprecision(17)<<"node,xd,weight,A_friction,B_friction,A_R,B_R,A_q,B_q,A_C,B_C,A_sigma,B_sigma\n";
          for (unsigned int i=0;i<after.size();++i)
            out<<i<<','<<after[i][0]<<','<<expected[i][1]<<','<<A.friction_traction[0][i]<<','<<B.friction_traction[0][i]
               <<','<<A.values[0][i]<<','<<B.values[0][i]<<','<<A.shear_traction[0][i]<<','<<B.shear_traction[0][i]
               <<','<<A.cohesive_traction[0][i]<<','<<B.cohesive_traction[0][i]<<','<<A.normal_traction[0][i]<<','<<B.normal_traction[0][i]<<'\n';
        }
      sim.get_pcout()<<"Theta exact conditional gate: A_R="<<A.values[0][contact]<<", B_R="<<B.values[0][contact]
                     <<", sign reversed="<<reverse<<std::endl;
      if (!reverse)
        {MPI_Barrier(sim.get_mpi_communicator());throw std::runtime_error("Theta exact comparison complete: no sign reversal, no solve.");}

      // Verify that the alternate state is frozen, not updated with trial V.
      // Use an adjacent non-contact direction; pressure is in physical units.
      ReconstructedFaultVector direction(1,std::vector<double>(after.size(),0.)), action;
      direction[0][797]=1.;surface.apply_surface_jacobian(direction,action);
      std::ofstream fd;
      if (sim.get_pcout().is_active())
        {fd.open(sim.get_output_directory()+"derivative_checks.csv");fd<<"kind,h,relative_error\n";}
      for (const double h:{1e-16,1e-17})
        {
          auto trial=V;trial[0][797]+=h;
          const auto r=surface.evaluate_surface_residual(frozen,trial);
          double e=0.,scale=0.;
          for (unsigned int i=795;i<=798;++i)
            {e=std::max(e,std::abs((r.values[0][i]-B.values[0][i])/h+action[0][i]));scale=std::max(scale,std::abs(action[0][i]));}
          if (fd)fd<<std::setprecision(17)<<"K,"<<h<<','<<e/scale<<'\n';
          AssertThrow(e/scale<1e-4,ExcMessage("Frozen-theta K derivative check failed."));
        }
      LinearAlgebra::BlockVector perturbation(owned);perturbation=0.;perturbation.block(intro.block_indices.pressure)=1.;
      AffineConstraints<double> homogeneous(pending);
      for (const auto &line:pending.get_lines())homogeneous.set_inhomogeneity(line.index,0.);
      homogeneous.close();homogeneous.distribute(perturbation);
      LinearAlgebra::BlockVector ghost(sim.get_current_linearization_point());ghost=perturbation;
      ReconstructedFaultVector G;surface.apply_G(ghost,G);
      LinearAlgebra::BlockVector trial(frozen);trial.add(100.,ghost);
      const auto r=surface.evaluate_surface_residual(trial,V);
      double e=0.,scale=0.;
      for (unsigned int i=795;i<=798;++i)
        {e=std::max(e,std::abs((r.values[0][i]-B.values[0][i])/100.-G[0][i]));scale=std::max(scale,std::abs(G[0][i]));}
      if (fd)fd<<std::setprecision(17)<<"G_pressure,100,"<<e/scale<<'\n';
      AssertThrow(e/scale<1e-7,ExcMessage("Frozen-theta pressure G check failed."));
      if (fd)fd.close();
      setenv("ASPECT_FAULT_THETA_QP_EXPORT",(sim.get_output_directory()+"solve").c_str(),1);
      sim.get_pcout()<<"Theta exact gate and derivatives passed; allowing one noncommitting mechanics-13 solve."<<std::endl;
    }
  }
}
