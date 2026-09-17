// Archived K5 probe; not included by the maintained BP3 plugin.
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
      const unsigned int first=within_step_target();
      const auto saved=theta_audit_csv(std::string(path)+"/fault_"+std::to_string(first-1)+".csv");
      const auto target=theta_audit_csv(std::string(path)+"/fault_"+std::to_string(first)+".csv");
      auto &manager=sim.get_reconstructed_fault_manager();
      const auto &fault=manager.get_fault(0);
      const auto state=manager.get_property_information()[manager.get_property_index("phase field fault state")].position;
      const unsigned int count=substeps ? std::stoi(substeps) : 1;
      const unsigned int k=sim.get_timestep_number();
      AssertThrow(fault.n_vertices()==1236 && saved.size()==1236 && k>=first && k<first+count,
                  ExcMessage("Within-step diagnostic has the wrong incoming geometry/clock."));
      const double dt=target[0][4]/count;
      AssertThrow(std::abs(sim.get_time()-(saved[0][3]+(k-first+1)*dt))<1e-6 && std::abs(sim.get_timestep()-dt)<1e-6,
                  ExcMessage("Within-step diagnostic changed the selected real timestep."));
      if (k==first)
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
      if ((std::getenv("ASPECT_BP3_NOTCH_BOUNDARY_PROBE") || std::getenv("ASPECT_FAULT_FREE_TRACE_DIAGNOSTIC"))
          && sim.get_pcout().is_active())
        {
          std::ofstream out(sim.get_output_directory()+"notch_input_properties.csv");
          out<<std::setprecision(17)<<"node,xd";
          for (const auto &property:manager.get_property_information())
            for (unsigned int c=0;c<property.n_components;++c)
              out<<','<<property.name<<'_'<<c;
          out<<'\n';
          for (unsigned int i=0;i<fault.n_vertices();++i)
            {
              out<<i<<','<<BP3::down_dip(fault.vertex(i)[0],fault.vertex(i)[1]);
              for (const auto &property:manager.get_property_information())
                for (unsigned int c=0;c<property.n_components;++c)
                  out<<','<<fault.get_properties(i)[property.position+c];
              out<<'\n';
            }
        }
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
      // The early experiment holds Theta fixed and tests no state-update
      // derivative. Its short dt also makes this legacy tiny-rate finite
      // difference cancellation-limited; retain it only for the state test.
      for (const double v:(first==2 ? std::vector<double>{} : std::vector<double>{1e-16,1e-12,1e-9,1e-8}))
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
      for (const unsigned int node:{795u,796u,797u,1100u})
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
      if (std::getenv("ASPECT_FAULT_FREE_TRACE_DIAGNOSTIC"))
        {
          // The new free column must use the same source in residual and B.
          // Its deep-side derivative is exactly zero, not a small penalty.
          auto &bulk=sim.get_reconstructed_fault_stokes_coupling();
          bulk.linearize_B(base);
          auto dv=V;for(auto &fault_values:dv)std::fill(fault_values.begin(),fault_values.end(),0.);
          dv[0][795]=BP3::Vp;
          LinearAlgebra::BlockVector action(owned),plus(owned),minus(owned);
          bulk.apply_B(dv,action);
          auto vp=V,vm=V;
          vp[0][795]+=.125*BP3::Vp;vm[0][795]-=.125*BP3::Vp;
          bulk.evaluate_slip_dependent_bulk_residual(base,vp,plus);
          bulk.evaluate_slip_dependent_bulk_residual(base,vm,minus);
          plus.add(-1.,minus);plus*=4.;plus+=action;
          const double B_error=plus.l2_norm()/action.l2_norm();
          AssertThrow(B_error<2e-10,ExcMessage("Independent trace B is inconsistent with the bulk residual."));

          // Use a constrained, nontrivial velocity direction to check full G,
          // then independently integrate the shear virtual work on bulk QPs.
          direction=action;direction.block(intro.block_indices.pressure)=0.;
          direction*=BP3::Vp/direction.block(intro.block_indices.velocities).linfty_norm();
          homogeneous.distribute(direction);ghost=direction;
          surface.apply_G(ghost,G);
          xp=base;xm=base;xp.add(.125,ghost);xm.add(-.125,ghost);
          const auto gp=surface.evaluate_surface_residual(xp,V),gm=surface.evaluate_surface_residual(xm,V);
          double g_error=0.,g_scale=0.;
          for(unsigned int i=0;i<V[0].size();++i)
            {g_error=std::max(g_error,std::abs(4.*(gp.values[0][i]-gm.values[0][i])-G[0][i]));g_scale=std::max(g_scale,std::abs(G[0][i]));}
          AssertThrow(g_error/g_scale<2e-7,ExcMessage("Independent trace G velocity derivative failed."));
          const auto &quad=intro.quadratures.velocities;
          FEValues<dim> fe(sim.get_mapping(),sim.get_fe(),quad,
            update_values|update_gradients|update_quadrature_points|update_JxW_values);
          std::vector<double> phase(quad.size()),temp(quad.size());
          std::vector<SymmetricTensor<2,dim>> strain(quad.size());
          std::vector<std::vector<double>> chemical(intro.chemical_composition_field_indices().size(),std::vector<double>(quad.size()));
          const bool early=first==2;
          std::vector<std::vector<double>> all_fields(early ? intro.n_compositional_fields : 0,std::vector<double>(quad.size()));
          std::array<unsigned int,3> stress_fields{};
          std::ofstream history;
          if (early)
            {
              for (const auto &m:sim.get_parameters().mapped_particle_properties)
                if (m.second.first=="maxwell stress") stress_fields[m.second.second]=m.first;
              history.open(sim.get_output_directory()+"early_working_history_rank"+
                std::to_string(Utilities::MPI::this_mpi_process(sim.get_mpi_communicator()))+".csv");
              history<<std::setprecision(17)<<"cell,qp,x,y,segment,xi,weight,phi,chi,kappa,old_xx,old_yy,old_xy,old_shear,old_minus_normal\n";
            }
          double work=0.;unsigned int deep_samples=0,free_samples=0;
          for(const auto &cell:sim.get_dof_handler().active_cell_iterators())if(cell->is_locally_owned())
            {
              fe.reinit(cell);
              const auto &association=manager.get_stokes_qp_fault_associations(cell->id(),quad,fe.get_quadrature_points());
              fe[FEValuesExtractors::Scalar(intro.variable("phase_field").first_component_index)].get_function_values(base,phase);
              fe[intro.extractors.temperature].get_function_values(base,temp);
              fe[intro.extractors.velocities].get_function_symmetric_gradients(ghost,strain);
              for(unsigned int c=0;c<chemical.size();++c)
                fe[intro.extractors.compositional_fields[intro.chemical_composition_field_indices()[c]]].get_function_values(base,chemical[c]);
              for(unsigned int c=0;c<all_fields.size();++c)
                fe[intro.extractors.compositional_fields[c]].get_function_values(base,all_fields[c]);
              for(unsigned int q=0;q<quad.size();++q)if(association[q].active)
                {
                  const auto &a=association[q];
                  if(a.segment_index==794)
                    {++deep_samples;AssertThrow(a.shape_0==1. && a.shape_1==0.,ExcMessage("Deep trace retained a free-node weight."));}
                  const auto point=fe.quadrature_point(q);
                  const bool export_history=early && BP3::down_dip(point[0],point[1])>=37000.
                    && BP3::down_dip(point[0],point[1])<=43000.;
                  if(a.segment_index!=795 && !export_history)continue;
                  typename MaterialModel::PhaseFieldFault<dim>::ReconstructedFaultBulkPointInputs input;
                  input.fault_index=a.fault_index;input.segment_index=a.segment_index;input.xi=a.xi;
                  input.phase_field=phase[q];input.previous_phase_field=phase[q];input.temperature=temp[q];
                  std::vector<double> composition;
                  for(const auto &c:chemical)composition.push_back(c[q]);
                  input.bulk_material_fractions=MaterialModel::MaterialUtilities::compute_composition_fractions(composition);
                  const auto response=model.evaluate_reconstructed_fault_bulk_point(input);
                  if(export_history)
                    {
                      // This is the retained constrained FE history actually
                      // supplied to mechanics, before any new stress commit.
                      std::vector<double> all(all_fields.size());
                      for(unsigned int c=0;c<all.size();++c)all[c]=all_fields[c][q];
                      SymmetricTensor<2,dim> old;
                      for(unsigned int c=0;c<3;++c)old[SymmetricTensor<2,dim>::unrolled_to_component_indices(c)]=all[stress_fields[c]];
                      const auto frozen=model.evaluate_frozen_maxwell_stress(temp[q],all,old);
                      history<<cell->id()<<','<<q<<','<<point[0]<<','<<point[1]<<','<<a.segment_index<<','<<a.xi
                        <<','<<fe.JxW(q)*response.localization_factor<<','<<phase[q]<<','<<response.localization_factor<<','<<response.kappa
                        <<','<<frozen[0][0]<<','<<frozen[1][1]<<','<<frozen[0][1]
                        <<','<<frozen*symmetrize(outer_product(a.tangent,a.normal))
                        <<','<<-frozen*symmetrize(outer_product(a.normal,a.normal))<<'\n';
                    }
                  if(a.segment_index!=795)continue;
                  ++free_samples;
                  AssertThrow(a.shape_1==a.xi,ExcMessage("Free-side Q1 geometry changed."));
                  work+=fe.JxW(q)*2.*response.kappa*response.localization_factor*BP3::Vp*a.shape_0
                    *(symmetrize(outer_product(a.tangent,a.normal))*strain[q]);
                }
            }
          work=Utilities::MPI::sum(work,sim.get_mpi_communicator());
          deep_samples=Utilities::MPI::sum(deep_samples,sim.get_mpi_communicator());
          free_samples=Utilities::MPI::sum(free_samples,sim.get_mpi_communicator());
          const double assembled=action*direction;
          const double work_error=std::abs(work-assembled)/std::max(std::abs(work),std::abs(assembled));
          AssertThrow(deep_samples>0 && free_samples>0 && work_error<2e-11,
                      ExcMessage("Independent trace violates source/surface virtual work."));
          if(out)out<<"trace_B,.125,"<<B_error<<"\ntrace_G_velocity,.125,"<<g_error/g_scale
            <<"\ntrace_virtual_work,1,"<<work_error<<'\n';
          sim.get_pcout()<<"Independent trace checks passed: B="<<B_error<<", G="<<g_error/g_scale
            <<", virtual work="<<work_error<<", deep/free QPs="<<deep_samples<<'/'<<free_samples<<std::endl;
        }
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
