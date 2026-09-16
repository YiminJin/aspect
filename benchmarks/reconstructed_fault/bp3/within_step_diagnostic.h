// Candidate-state diagnostics: immutable incoming state during mechanics;
// the explicitly selected subdivision replay uses ordinary terminal commit.
#include "../../../tests/phase_field_fault_test_access.h"
namespace aspect
{
  namespace BP3Benchmark
  {
    bool within_step_ready=false;
    std::vector<double> coupled_old_theta;
    unsigned int coupled_pending_step=numbers::invalid_unsigned_int;

    template <int dim>
    void within_step_check_impl(const SimulatorAccess<dim> &sim, AffineConstraints<double> &pending)
    {
      const char *path=std::getenv("ASPECT_BP3_WITHIN_STEP_DIAGNOSTIC");
      const char *substeps=std::getenv("ASPECT_BP3_COUPLED_STATE_REPLAY");
      if (substeps) path=std::getenv("ASPECT_BP3_COUPLED_STATE_INPUT");
      if (!path || !within_step_ready) return;
      within_step_ready=false;
      const auto saved=theta_audit_csv(std::string(path)+"/fault_9.csv");
      const auto target=theta_audit_csv(std::string(path)+"/fault_10.csv");
      auto &manager=sim.get_reconstructed_fault_manager();
      const auto &fault=manager.get_fault(0);
      const auto state=manager.get_property_information()[manager.get_property_index("phase field fault state")].position;
      const unsigned int count=substeps ? std::stoi(substeps) : 1;
      const unsigned int k=sim.get_timestep_number();
      AssertThrow(fault.n_vertices()==1236 && saved.size()==1236 && k>=10 && k<10+count,
                  ExcMessage("Within-step diagnostic has the wrong incoming geometry/clock."));
      const double dt=target[0][4]/count;
      AssertThrow(std::abs(sim.get_time()-(saved[0][3]+(k-9)*dt))<1e-6 && std::abs(sim.get_timestep()-dt)<1e-6,
                  ExcMessage("Within-step diagnostic changed the selected real timestep."));
      if (k==10)
        for (unsigned int i=0;i<saved.size();++i)
          AssertThrow(fault.get_properties(i)[state]==saved[i][6]
                      && manager.get_timestep_committed_slip_rate(0)[i]==saved[i][5],
                      ExcMessage("Within-step diagnostic does not contain retained step-9 Theta/V."));
      if (substeps)
        {
          AssertThrow(coupled_pending_step==numbers::invalid_unsigned_int,
                      ExcMessage("Previous coupled-state substep was not audited exactly once."));
          coupled_pending_step=k;coupled_old_theta.resize(fault.n_vertices());
          for(unsigned int i=0;i<fault.n_vertices();++i) coupled_old_theta[i]=fault.get_properties(i)[state];
        }

      // This fresh-start-only replay did not serialize its transient I_h cache.
      // Reconstruct the SAME frozen input, not a new rounded normal integral.
      // Histories and production restart/cache policy remain untouched.
      auto &model=const_cast<MaterialModel::PhaseFieldFault<dim>&>(
        Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(sim.get_material_model()));
      using Access=MaterialModel::internal::PhaseFieldFaultTestAccess<dim>;
      auto integrals=Access::current_normalization_integrals(model);
      double maximum_difference=0.;
      for (unsigned int i=0;i<saved.size();++i)
        {
          maximum_difference=std::max(maximum_difference,std::abs(integrals[0][i]/saved[i][8]-1.));
          integrals[0][i]=saved[i][8];
        }
      sim.get_pcout()<<std::setprecision(17)<<"Within-step frozen I_h reconstruction: relative difference="<<maximum_difference<<std::endl;
      AssertThrow(maximum_difference<1e-12,ExcMessage("Restart reconstructed a materially different I_h; do not replace it silently."));
      Access::restore_diagnostic_normalization(model,integrals);
      if (substeps && k>10) return; // Derivative checks are independent of history publication.

      // Freeze bulk and history while differentiating the complete nodal map.
      // Independent adjacent columns expose a falsely symmetrized state term.
      const auto &intro=sim.introspection();
      auto &surface=sim.get_reconstructed_fault_surface_system();
      LinearAlgebra::BlockVector owned(intro.index_sets.system_partitioning,sim.get_mpi_communicator());
      owned=sim.get_current_linearization_point();
      AffineConstraints<double> physical(pending);physical.close();physical.distribute(owned);
      LinearAlgebra::BlockVector base(sim.get_current_linearization_point());base=owned;
      ReconstructedFaultVector V(1,manager.get_timestep_committed_slip_rate(0));
      surface.linearize_surface_system(base,V);
      std::ofstream out;
      if (sim.get_pcout().is_active())
        {out.open(sim.get_output_directory()+"derivatives.csv");out<<std::setprecision(17)<<"direction,epsilon,error\n";}
      const auto &friction=MaterialModel::internal::PhaseFieldFaultTestAccess<dim>::fault_friction(model);
      for (const double v:{1e-16,1e-12,1e-9,1e-8})
        {
          const double h=1e-3*v,old=saved[796][6],dt=sim.get_timestep();
          const double exact=friction.update_state_derivative_wrt_slip_rate(v,old,dt);
          const double fd=(friction.update_state(v+h,old,dt)-friction.update_state(v-h,old,dt))/(2*h);
          const double error=std::abs(fd/exact-1.);
          if(out)out<<"aging,"<<v<<','<<error<<'\n';
          AssertThrow(error<2e-6,ExcMessage("Exact aging derivative failed its independent finite difference."));
        }
      for (const std::vector<double> mixture:{std::vector<double>{1.,0.},std::vector<double>{.4,.6},std::vector<double>{0.,1.}})
        {
          const double old=saved[796][6],h=1e-4*old,v=V[0][796];
          const double fd=(friction.friction_coefficient(mixture,v,old+h)-friction.friction_coefficient(mixture,v,old-h))/(2*h);
          const double exact=friction.friction_coefficient_derivative_wrt_state(mixture,v,old);
          const double error=std::abs(fd/exact-1.);
          if(out)out<<"mu_state,"<<mixture[1]<<','<<error<<'\n';
          AssertThrow(error<1e-7,ExcMessage("Friction state derivative failed its independent finite difference."));
        }
      for (const unsigned int node:{796u,797u,1100u})
        {
          ReconstructedFaultVector dV(1,std::vector<double>(V[0].size())),action;
          dV[0][node]=V[0][node];surface.apply_surface_jacobian(dV,action);
          for (const double epsilon:{1e-3,1e-4})
            {
              auto vp=V,vm=V;vp[0][node]+=epsilon*dV[0][node];vm[0][node]-=epsilon*dV[0][node];
              const auto rp=surface.evaluate_surface_residual(base,vp),rm=surface.evaluate_surface_residual(base,vm);
              double error=0.,scale=0.;
              for(unsigned int i=0;i<V[0].size();++i)
                {error=std::max(error,std::abs((rp.values[0][i]-rm.values[0][i])/(2*epsilon)+action[0][i]));scale=std::max(scale,std::abs(action[0][i]));}
              if(out)out<<node<<','<<epsilon<<','<<error/scale<<'\n';
              AssertThrow(error/scale<2e-6,ExcMessage("Candidate nodal-state Jacobian failed its central directional check."));
            }
        }
      LinearAlgebra::BlockVector direction(owned);direction=0.;direction.block(intro.block_indices.pressure)=1.;
      AffineConstraints<double> homogeneous(pending);
      for (const auto &line:pending.get_lines()) homogeneous.set_inhomogeneity(line.index,0.);
      homogeneous.close();homogeneous.distribute(direction);
      LinearAlgebra::BlockVector ghost(base);ghost=direction;
      ReconstructedFaultVector G;surface.apply_G(ghost,G);
      auto xp=base,xm=base;xp.add(100.,ghost);xm.add(-100.,ghost);
      const auto rp=surface.evaluate_surface_residual(xp,V),rm=surface.evaluate_surface_residual(xm,V);
      double error=0.,scale=0.;
      for(unsigned int i=0;i<V[0].size();++i)
        {error=std::max(error,std::abs((rp.values[0][i]-rm.values[0][i])/200.-G[0][i]));scale=std::max(scale,std::abs(G[0][i]));}
      if(out)out<<"pressure,100,"<<error/scale<<'\n';
      AssertThrow(error/scale<1e-7,ExcMessage("Candidate-state G pressure derivative failed."));
      sim.get_pcout()<<"Within-step input/derivative checks passed; same retained history, no state publication."<<std::endl;
    }

    template <int dim>
    void verify_coupled_state_commit(const SimulatorAccess<dim> &sim)
    {
      if (!std::getenv("ASPECT_BP3_COUPLED_STATE_REPLAY")) return;
      AssertThrow(coupled_pending_step==sim.get_timestep_number(),
                  ExcMessage("Candidate-state commit audit is missing its preceding history."));
      const auto &manager=sim.get_reconstructed_fault_manager();
      const auto &fault=manager.get_fault(0);
      const auto state=manager.get_property_information()[manager.get_property_index("phase field fault state")].position;
      const auto &V=manager.get_timestep_committed_slip_rate(0);
      const auto &model=Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(sim.get_material_model());
      const auto &friction=MaterialModel::internal::PhaseFieldFaultTestAccess<dim>::fault_friction(model);
      double error=0.;
      for (unsigned int i=0;i<V.size();++i)
        {
          const double candidate=friction.update_state(V[i],coupled_old_theta[i],sim.get_timestep());
          error=std::max(error,std::abs(fault.get_properties(i)[state]/candidate-1.));
        }
      // The last converged assembly precedes publication. Compare its actual
      // QP state with the interpolation of the now committed nodal candidates.
      const auto rank=std::to_string(Utilities::MPI::this_mpi_process(sim.get_mpi_communicator()));
      const auto file=sim.get_output_directory()+"state_qp_rank"+rank+".csv";
      std::ifstream in(file);AssertThrow(in,ExcMessage("Missing converged candidate-state samples."));
      std::string line;std::getline(in,line);unsigned int samples=0;
      while (std::getline(in,line))
        {
          std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);
          std::string cell;unsigned int qp,j;double xi,w,v,theta;
          AssertThrow(row>>cell>>qp>>j>>xi>>w>>v>>theta,ExcMessage("Invalid candidate-state sample."));
          const double actual=(1-xi)*fault.get_properties(j)[state]+xi*fault.get_properties(j+1)[state];
          error=std::max(error,std::abs(actual/theta-1.));++samples;
        }
      error=Utilities::MPI::max(error,sim.get_mpi_communicator());
      samples=Utilities::MPI::sum(samples,sim.get_mpi_communicator());
      AssertThrow(error<1e-12 && samples>0,ExcMessage("Committed Theta differs from the state used by coupled mechanics."));
      std::filesystem::copy_file(file,sim.get_output_directory()+"state_qp_step"+
        std::to_string(sim.get_timestep_number())+"_rank"+rank+".csv");
      if (sim.get_pcout().is_active())
        {
          std::ofstream out(sim.get_output_directory()+"candidate_commit_"+std::to_string(sim.get_timestep_number())+".csv");
          out<<std::setprecision(17)<<"step,time,dt,max_relative_error,qp_samples\n"
             <<sim.get_timestep_number()<<','<<sim.get_time()<<','<<sim.get_timestep()<<','<<error<<','<<samples<<'\n';
        }
      sim.get_pcout()<<"Coupled-state commit verified: step="<<sim.get_timestep_number()
        <<", candidate/interpolation error="<<error<<", samples="<<samples<<"; one update from immutable old Theta."<<std::endl;
      coupled_pending_step=numbers::invalid_unsigned_int;
    }

    template <int dim>
    void within_step_check(const SimulatorAccess<dim> &sim, AffineConstraints<double> &pending)
    {
      try { within_step_check_impl(sim,pending); }
      catch (const std::exception &e)
        {
          std::cerr<<"Within-step preflight rank "<<Utilities::MPI::this_mpi_process(sim.get_mpi_communicator())
                   <<": "<<e.what()<<std::endl;
          throw;
        }
    }
  }
}
