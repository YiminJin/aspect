/* Frozen BP3 response discrimination. No state, history, or operator is changed. */
#include <aspect/simulator_signals.h>
#include <aspect/simulator/assemblers/reconstructed_fault_stokes.h>
#include <aspect/reconstructed_fault/surface_system.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/material_model/utilities.h>
#include <aspect/plugins.h>
#include <aspect/time_stepping/interface.h>
#include <deal.II/lac/solver_gmres.h>
#include <fstream>
#include <iomanip>
#include "reconstructed_fault_frozen_profile.h"
#include "reconstructed_fault_velocity_export.h"
#include "bp3_length_scale_checks.h"

namespace aspect
{
  namespace TimeStepping
  {
    // The server and local builds differ. Preserve both production controllers;
    // accept only negligible clock differences, never a material safety cutback.
    template <int dim>
    class ReplayClock : public TimeStepping::Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        double execute() override
        {
          if (times.empty())
            {
              std::ifstream in(std::getenv("ASPECT_BP3_TIMESTEP_SEQUENCE"));
              AssertThrow(in,ExcMessage("Missing mechanical-probe clock."));
              std::string line;std::getline(in,line);
              while (std::getline(in,line))
                {
                  std::replace(line.begin(),line.end(),',',' ');
                  std::istringstream row(line);unsigned int step;double t,dt;
                  AssertThrow(row>>step>>t>>dt && step==times.size(),ExcMessage("Invalid mechanical-probe clock."));
                  times.push_back(t);steps.push_back(dt);
                }
            }
          const auto k=this->get_timestep_number();
          AssertThrow(k<times.size() && std::abs(this->get_time()-times[k])<=1e-8*std::max(1.,times[k]),
                      ExcMessage("Mechanical probe left the saved clock."));
          return k+1<steps.size()?steps[k+1]:std::numeric_limits<double>::max();
        }
        std::pair<TimeStepping::Reaction,double> determine_reaction(const TimeStepping::TimeStepInfo &info) override
        {
          const auto k=this->get_timestep_number();
          if (k+1<steps.size())
            {
              this->get_pcout()<<std::setprecision(17)<<"Mechanical probe clock: next="<<k+1<<" requested="<<steps[k+1]<<" selected="<<info.next_time_step_size<<std::endl;
              AssertThrow(std::abs(info.next_time_step_size-steps[k+1])<=1e-8*steps[k+1],
                          ExcMessage("Production controller requires a material cutback; stop the diagnostic."));
            }
          return {TimeStepping::Reaction::advance,std::numeric_limits<double>::max()};
        }
      private:
        std::vector<double> times,steps;
    };

    ASPECT_REGISTER_TIME_STEPPING_MODEL(ReplayClock,"mechanical probe clock","Saved-clock cap with cross-build roundoff checks; production guards remain active.")
  }

  namespace MechanicalModes
  {
    using Vector = LinearAlgebra::BlockVector;
    using FaultVector = ReconstructedFaultVector;
    struct Action
    {
      std::function<void(Vector &,const Vector &)> action;
      void vmult(Vector &dst,const Vector &src) const { action(dst,src); }
    };

    template <int dim>
    void probe(const SimulatorAccess<dim> &sim,
               const typename SimulatorSignals<dim>::FaultBulkAction &,
               const typename SimulatorSignals<dim>::FaultBulkAction &preconditioner,
               const typename SimulatorSignals<dim>::FaultPressureAction &,
               const Vector &,const Vector &,double,unsigned int budget,double)
    {
      const unsigned int target_step=std::getenv("ASPECT_MECHANICAL_PROBE_STEP")?std::atoi(std::getenv("ASPECT_MECHANICAL_PROBE_STEP")):11;
      const unsigned int target_iteration=std::getenv("ASPECT_MECHANICAL_PROBE_NEWTON")?std::atoi(std::getenv("ASPECT_MECHANICAL_PROBE_NEWTON")):6;
      if (sim.get_timestep_number()!=target_step || sim.get_nonlinear_iteration()!=target_iteration) return;
      FrozenMechanicalProfile::snapshot_or_verify(sim);
      if (std::getenv("ASPECT_BP3_LENGTH_QUALIFICATION")) return;
      auto &surface=sim.get_reconstructed_fault_surface_system();
      auto &manager=sim.get_reconstructed_fault_manager();
      const auto &base_residual=surface.get_linearization_residual();
      // The saved and reproduced step 11 both converged at iteration 6.
      // Select that uncommitted linearization, not an extra absolute criterion.
      const double base_rms=surface.surface_residual_rms(base_residual,manager.prescribed_slip_rate_mask());
      const auto &intro=sim.introspection();
      const auto comm=sim.get_mpi_communicator();
      const unsigned int rank=Utilities::MPI::this_mpi_process(comm);
      const auto &faults=manager.get_faults();
      AssertThrow(dim==2 && faults.size()==1,ExcMessage("This diagnostic requires the single 2D BP3 fault."));
      const auto &material=Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(sim.get_material_model());
      AssertThrow(!material.uses_adiabatic_friction_pressure(),ExcMessage("True normal feedback is required."));
      FaultVector V(1,manager.get_slip_rate(0));
      const unsigned int n=V[0].size();
      Vector saved(intro.index_sets.system_partitioning,comm),base(saved);
      saved=sim.get_solution();base=sim.get_current_linearization_point();
      AffineConstraints<double> homogeneous(sim.get_current_constraints());
      for (const auto &line:homogeneous.get_lines()) homogeneous.set_inhomogeneity(line.index,0.);
      IndexSet stokes_indices(sim.get_dof_handler().n_dofs());
      stokes_indices.add_range(0,intro.index_sets.stokes_partitioning[0].size()+intro.index_sets.stokes_partitioning[1].size());
      const auto stokes_constraints=homogeneous.get_view(stokes_indices);
      const auto physical=[&](const Vector &solver)
      {
        Vector owned(intro.index_sets.system_partitioning,comm);
        owned.block(0)=solver.block(0);owned.block(1)=solver.block(1);
        owned.block(1)*=sim.get_pressure_scaling();
        homogeneous.distribute(owned);owned.compress(VectorOperation::insert);
        Vector ghosted(intro.index_sets.system_partitioning,intro.index_sets.system_relevant_partitioning,comm);
        ghosted=owned;return ghosted;
      };
      // Keep A and homogeneous constraints exactly as in the assembled solve.
      // Pressure remains a solved unknown, including in the frozen-normal case.
      Action A{[&](Vector &dst,const Vector &src)
      {
        Vector z(src);stokes_constraints.set_zero(z);z.compress(VectorOperation::insert);
        const auto &matrix=sim.get_system_matrix();
        matrix.block(0,0).vmult(dst.block(0),z.block(0));
        matrix.block(0,1).vmult_add(dst.block(0),z.block(1));
        matrix.block(1,0).vmult(dst.block(1),z.block(0));
        matrix.block(1,1).vmult_add(dst.block(1),z.block(1));
        stokes_constraints.set_zero(dst);dst.compress(VectorOperation::insert);
      }};
      Action prec{preconditioner};
      std::ofstream summary,nodes;
      if (!rank)
        {
          summary.open(sim.get_output_directory()+"mechanical_modes.csv");
          summary<<std::setprecision(17)<<"mode,step,newton,time,dt,base_surface_RMS_Pa,iterations,fresh_relative,mass_norm,mechanical_shear,pressure_feedback,deviatoric_normal_feedback,instantaneous_friction,damping,full_restoring,frozen_normal_restoring,work_pair_relative,action_relative,fd_relative,fd_frozen_relative\n";
          nodes.open(sim.get_output_directory()+"mechanical_mode_nodes.csv");
          nodes<<std::setprecision(17)<<"mode,node,xd,V,deltaV,mass_row,base_R,delta_q,delta_p,minus_delta_tau_N,minus_mu_delta_p,mu_delta_tau_N,minus_sigma_muV_deltaV,minus_damping,delta_R,delta_R_frozen_normal,K_deltaV,G_shear_delta_x,mass_diagonal,mass_upper\n";
        }
      std::ofstream raw(sim.get_output_directory()+"mechanical_mode_qp_rank"+std::to_string(rank)+".csv");
      raw<<std::setprecision(17)<<"mode,cell,qp,segment,xi,x,y,weight,V,sigma,mu,deltaV,delta_q,delta_p,minus_delta_tau_N,minus_sigma_muV_deltaV,delta_R,delta_R_frozen_normal,JxW,kappa,chi,d_us_dn,d_un_ds,u_s,u_n\n";
      const bool decompose=std::getenv("ASPECT_MECHANICAL_DECOMPOSITION");
      std::ofstream velocities,parts;
      if (decompose)
        {
          velocities.open(sim.get_output_directory()+"mechanical_velocity_cells_rank"+std::to_string(rank)+".csv");
          velocities<<std::setprecision(17)<<"mode,cell,x,y,h";
          for (unsigned int q=0;q<9;++q) velocities<<",ux"<<q<<",uy"<<q;
          velocities<<'\n';
          if (!rank)
            {parts.open(sim.get_output_directory()+"mechanical_shear_parts.csv");parts<<std::setprecision(17)<<"mode,mass,signed_d_us_dn,signed_d_un_ds,bulk_relaxation,closure_relative\n";}
        }

      for (unsigned int mode=0;mode<3;++mode)
        {
          const bool bp5=std::getenv("ASPECT_BP5_SHORT_TEST");
          if (bp5 && mode==1) continue;
          const bool length_study=std::getenv("ASPECT_BP3_LENGTH_STUDY");
          if (mode==0 && std::getenv("ASPECT_MECHANICAL_WIDTH_PROBE") && !length_study) continue;
          const std::string name=mode==0?(bp5?"target_3125m":length_study?"target_1500m":"broad_3000m"):mode==1?"six_nodes_600m":"alternating_200m";
          FaultVector dv(1,std::vector<double>(n));
          for (unsigned int i=0;i<n;++i)
            {
              const double xd=(100000.-faults[0].vertex(i)[1])/std::sin(numbers::PI/3.);
              // Four long wavelengths under the same taper on both meshes;
              // the material transition itself remains at 15--18 km.
              const double z=(xd-16500.)/(bp5?6250.:1500.);
              if (std::abs(z)<1.)
                {
                  const double taper=std::pow(std::cos(numbers::PI*z/2.),2);
                  const double wave=mode==0?(bp5?3125.:1500.):(mode==1?600.:200.);
                  dv[0][i]=1e-12*taper*(mode==0 && !length_study?1.:std::cos(2*numbers::PI*(xd-16500.)/wave));
                }
            }
          // For a prescribed small velocity variation, bulk relaxation is
          // A dx=B dV. Surface response is G dx-K dV, not a B-transpose model.
          Vector full_B(intro.index_sets.system_partitioning,comm);
          sim.get_reconstructed_fault_stokes_coupling().apply_B(dv,full_B);
          Vector rhs(intro.index_sets.stokes_partitioning,comm),dx(rhs),residual(rhs);
          rhs.block(0)=full_B.block(0);rhs.block(1)=full_B.block(1);
          stokes_constraints.set_zero(rhs);rhs.compress(VectorOperation::insert);
          const double rhs_norm=rhs.l2_norm();
          const double tolerance=1e-10*rhs_norm;
          unsigned int iterations=0;double fresh;
          do
            {
              SolverControl control(budget-iterations,tolerance);
              SolverFGMRES<Vector> solver(control,typename SolverFGMRES<Vector>::AdditionalData(sim.get_parameters().stokes_gmres_restart_length));
              bool failed=false;
              try {solver.solve(A,dx,rhs,prec);} catch (const SolverControl::NoConvergence &) {failed=true;}
              iterations+=std::max(1u,control.last_step());
              A.vmult(residual,dx);residual-=rhs;fresh=residual.l2_norm();
              AssertThrow(fresh<=tolerance || (!failed && iterations<budget),ExcMessage("Mechanical mode failed its fresh A residual check."));
            } while (fresh>tolerance);
          const Vector direction=physical(dx);
          if (decompose) export_velocity_cells(sim,direction,name,velocities);
          FaultVector G,K;
          surface.apply_G(direction,G);surface.apply_surface_jacobian(dv,K);

          // Reproduce the production owned-bulk-QP work measure. Old FE history,
          // incoming Q1 state, geometry and material remain frozen throughout.
          const auto &quadrature=intro.quadratures.velocities;
          FEValues<dim> fe(sim.get_mapping(),sim.get_fe(),quadrature,update_values|update_gradients|update_quadrature_points|update_JxW_values);
          const unsigned int nq=quadrature.size();
          std::vector<double> phi(nq),temperature(nq),pressure(nq),dp(nq);
          std::vector<SymmetricTensor<2,dim>> strain(nq),deps(nq);
          std::vector<Tensor<2,dim>> gradient(nq);
          std::vector<Tensor<1,dim>> velocity(nq);
          double normal_gradient_work=0.,tangent_gradient_work=0.;
          std::vector<std::vector<double>> composition(intro.n_compositional_fields,std::vector<double>(nq));
          std::array<unsigned int,SymmetricTensor<2,dim>::n_independent_components> stress_fields;
          stress_fields.fill(numbers::invalid_unsigned_int);
          for (const auto &entry:sim.get_parameters().mapped_particle_properties)
            if (entry.second.first=="maxwell stress") stress_fields[entry.second.second]=entry.first;
          // Columns: mass row, reproduced base, dq, dp, -dtau:N, -mu dp,
          // +mu dtau:N, -sigma mu_V dV, -eta dV, total, frozen, shear-G,
          // centered full/frozen finite differences, mass*dV.
          std::vector<std::vector<double>> loads(15,std::vector<double>(n));
          for (const auto &cell:sim.get_dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                fe.reinit(cell);
                const auto &associations=manager.get_stokes_qp_fault_associations(cell->id(),quadrature,fe.get_quadrature_points());
                const auto &working=sim.get_current_linearization_point();
                fe[FEValuesExtractors::Scalar(intro.variable("phase_field").first_component_index)].get_function_values(working,phi);
                fe[intro.extractors.temperature].get_function_values(working,temperature);
                fe[intro.extractors.pressure].get_function_values(working,pressure);
                fe[intro.extractors.velocities].get_function_symmetric_gradients(working,strain);
                fe[intro.extractors.pressure].get_function_values(direction,dp);
                fe[intro.extractors.velocities].get_function_symmetric_gradients(direction,deps);
                fe[intro.extractors.velocities].get_function_gradients(direction,gradient);
                fe[intro.extractors.velocities].get_function_values(direction,velocity);
                for (unsigned int c=0;c<composition.size();++c)
                  fe[intro.extractors.compositional_fields[c]].get_function_values(working,composition[c]);
                for (unsigned int q=0;q<nq;++q)
                  if (associations[q].active)
                    {
                      const auto &a=associations[q];const unsigned int j=a.segment_index;
                      typename MaterialModel::PhaseFieldFault<dim>::ReconstructedFaultPointInputs in;
                      in.fault_index=0;in.segment_index=j;in.xi=a.xi;in.position=fe.quadrature_point(q);
                      in.phase_field=in.previous_phase_field=phi[q];in.temperature=temperature[q];
                      in.dynamic_pressure=pressure[q];in.strain_rate=strain[q];
                      in.slip_tensor=symmetrize(outer_product(a.tangent,a.normal));
                      in.normal_tensor=symmetrize(outer_product(a.normal,a.normal));
                      for (unsigned int c=0;c<stress_fields.size();++c)
                        in.old_maxwell_stress[SymmetricTensor<2,dim>::unrolled_to_component_indices(c)]=composition[stress_fields[c]][q];
                      std::vector<double> chemical;
                      for (const auto c:intro.chemical_composition_field_indices()) chemical.push_back(composition[c][q]);
                      in.bulk_material_fractions=MaterialModel::MaterialUtilities::compute_composition_fractions(chemical);
                      const double N[2]={a.shape_0,a.shape_1};
                      in.slip_rate=N[0]*V[0][j]+N[1]*V[0][j+1];
                      const double v=N[0]*dv[0][j]+N[1]*dv[0][j+1];
                      const auto r=material.evaluate_reconstructed_fault_point(in);
                      if (std::getenv("ASPECT_BP5_SHORT_TEST"))
                        {
                          const auto fractions=MaterialModel::internal::PhaseFieldFaultTestAccess<dim>::
                            surface_material_state_at_projection(material,0,j,a.xi).first;
                          const auto state=manager.get_property_information()[manager.get_property_index("phase field fault state")].position;
                          const double theta=N[0]*faults[0].get_properties(j)[state]+N[1]*faults[0].get_properties(j+1)[state];
                          const double direct_effect=.004*fractions[0]+.04*fractions[1];
                          const double expected=direct_effect*std::asinh(in.slip_rate/(2e-6)*
                            std::exp((.6+.03*std::log(theta*1e-6/.1))/direct_effect));
                          AssertThrow(std::abs(r.friction_coefficient/expected-1)<1e-12,
                                      ExcMessage("BP5 production-QP friction parameter/state mismatch."));
                        }
                      const double w=fe.JxW(q)*r.localization_factor;
                      if (w==0.) continue;
                      const double shear_G=2*r.kappa*(in.slip_tensor*deps[q]);
                      // 2 kappa S:eps = kappa (d_n u_s + d_s u_n).
                      // Keep both signed contractions: either may oppose the
                      // input mode. Their sum is the existing shear work.
                      const double d_us_dn=a.tangent*(gradient[q]*a.normal);
                      const double d_un_ds=a.normal*(gradient[q]*a.tangent);
                      normal_gradient_work+=w*v*r.kappa*d_us_dn;
                      tangent_gradient_work+=w*v*r.kappa*d_un_ds;
                      const double direct=2*r.kappa*r.localization_factor*(in.slip_tensor*in.slip_tensor);
                      const double dq=shear_G-direct*v;
                      const double minus_dtn=-2*r.kappa*(in.normal_tensor*deps[q]);
                      const double damping=r.damping_traction/in.slip_rate;
                      const double friction=-(r.minus_derivative_wrt_slip_rate-direct-damping)*v;
                      const double frozen=dq+friction-damping*v;
                      const double full=frozen-r.friction_coefficient*(dp[q]+minus_dtn);
                      // The matched diagnostic changes only normal feedback:
                      // mu(V)*sigma_base retains exactly the baseline resistance.
                      const double eps=0.01;
                      auto plus=in,minus=in;
                      plus.slip_rate+=eps*v;minus.slip_rate-=eps*v;
                      plus.strain_rate+=eps*deps[q];minus.strain_rate-=eps*deps[q];
                      plus.dynamic_pressure+=eps*dp[q];minus.dynamic_pressure-=eps*dp[q];
                      const auto rp=material.evaluate_reconstructed_fault_point(plus),rm=material.evaluate_reconstructed_fault_point(minus);
                      const double fd=(rp.residual_density-rm.residual_density)/(2*eps);
                      const double fdf=(rp.shear_traction-rm.shear_traction-(rp.friction_coefficient-rm.friction_coefficient)*r.normal_traction-rp.damping_traction+rm.damping_traction)/(2*eps);
                      const double terms[15]={1.,r.residual_density,dq,dp[q],minus_dtn,-r.friction_coefficient*dp[q],-r.friction_coefficient*minus_dtn,friction,-damping*v,full,frozen,shear_G,fd,fdf,v};
                      for (unsigned int c=0;c<15;++c)
                        for (unsigned int b=0;b<2;++b) loads[c][j+b]+=w*N[b]*terms[c];
                      const double xd=(100000.-in.position[1])/std::sin(numbers::PI/3.);
                      if (xd>12000. && xd<20000.)
                        raw<<name<<','<<cell->id()<<','<<q<<','<<j<<','<<a.xi<<','<<in.position[0]<<','<<in.position[1]<<','<<w<<','<<in.slip_rate<<','<<r.normal_traction<<','<<r.friction_coefficient<<','<<v<<','<<dq<<','<<dp[q]<<','<<minus_dtn<<','<<friction<<','<<full<<','<<frozen<<','<<fe.JxW(q)<<','<<r.kappa<<','<<r.localization_factor<<','<<d_us_dn<<','<<d_un_ds<<','<<velocity[q]*a.tangent<<','<<velocity[q]*a.normal<<'\n';
                    }
              }
          for (auto &load:loads)
            {const auto local=load;Utilities::MPI::sum(local,comm,load);}
          double mass=0.,work=0.,action_error=0.,fd_error=0.,fdf_error=0.,action_scale=0.,frozen_scale=0.;
          std::array<double,7> projected{};
          for (unsigned int i=0;i<n;++i)
            {
              mass+=dv[0][i]*loads[14][i];work+=dv[0][i]*loads[11][i];
              constexpr unsigned int columns[7]={2,5,6,7,8,9,10};
              for (unsigned int k=0;k<7;++k) projected[k]-=dv[0][i]*loads[columns[k]][i];
              action_error+=std::pow(loads[9][i]-(G[0][i]-K[0][i]),2);
              fd_error+=std::pow(loads[9][i]-loads[12][i],2);
              fdf_error+=std::pow(loads[10][i]-loads[13][i],2);
              action_scale+=loads[9][i]*loads[9][i];frozen_scale+=loads[10][i]*loads[10][i];
              AssertThrow(std::abs(loads[1][i]-base_residual.values[0][i])<=1e-5*loads[0][i],ExcMessage("Probe does not reproduce the production weak residual."));
              if (!rank)
                {
                  nodes<<name<<','<<i<<','<<(100000.-faults[0].vertex(i)[1])/std::sin(numbers::PI/3.)<<','<<V[0][i]<<','<<dv[0][i];
                  for (unsigned int k=0;k<11;++k) nodes<<','<<loads[k][i];
                  nodes<<','<<K[0][i]<<','<<loads[11][i]<<','<<base_residual.mass_diagonal[0][i]<<','
                       <<(i+1<n?base_residual.mass_off_diagonal[0][i]:0.)<<'\n';
                }
            }
          const double bulk_work=dx.block(0)*rhs.block(0);
          normal_gradient_work=Utilities::MPI::sum(normal_gradient_work,comm);
          tangent_gradient_work=Utilities::MPI::sum(tangent_gradient_work,comm);
          const double split_error=std::abs(normal_gradient_work+tangent_gradient_work-work)/std::abs(work);
          AssertThrow(split_error<1e-12,ExcMessage("Signed velocity-gradient contributions do not reproduce shear work."));
          if (parts) parts<<name<<','<<mass<<','<<normal_gradient_work/mass<<','<<tangent_gradient_work/mass<<','<<work/mass<<','<<split_error<<'\n';
          const double work_error=std::abs(work-bulk_work)/std::max(std::abs(work),std::abs(bulk_work));
          action_error=std::sqrt(action_error/action_scale);fd_error=std::sqrt(fd_error/action_scale);fdf_error=std::sqrt(fdf_error/frozen_scale);
          AssertThrow(work_error<1e-8 && action_error<1e-8 && fd_error<1e-5 && fdf_error<1e-5,ExcMessage("Mechanical mode derivative/work verification failed."));
          if (!rank)
            {
              summary<<name<<','<<sim.get_timestep_number()<<','<<sim.get_nonlinear_iteration()<<','<<sim.get_time()<<','<<sim.get_timestep()<<','<<base_rms<<','<<iterations<<','<<fresh/rhs_norm<<','<<mass;
              for (const double p:projected) summary<<','<<p/mass;
              summary<<','<<work_error<<','<<action_error<<','<<fd_error<<','<<fdf_error<<'\n';summary.flush();
            }
          sim.get_pcout()<<"Mechanical probe "<<name<<": iterations="<<iterations<<", fresh relative="<<fresh/rhs_norm<<", restoring="<<projected[5]/mass<<", frozen-normal="<<projected[6]/mass<<std::endl;
        }
      Vector check(saved);check=sim.get_solution();check-=saved;
      AssertThrow(check.l2_norm()==0. && manager.get_slip_rate(0)==V[0],ExcMessage("Mechanical probe changed physical state."));
      check=sim.get_current_linearization_point();check-=base;
      AssertThrow(check.l2_norm()==0.,ExcMessage("Mechanical probe changed the working history/state."));
      raw.close();nodes.close();summary.close();velocities.close();parts.close();
      sim.get_pcout()<<"MECHANICAL MODES VERIFIED; intentional stop before trial/history publication."<<std::endl;
      MPI_Barrier(comm);
      AssertThrow(false,ExcMessage("Intentional noncommitting mechanical-mode stop."));
    }

    template <int dim> void connect(SimulatorSignals<dim> &signals)
    {
      signals.post_advection_solver.connect(&check_bp3_length_scale<dim>);
      signals.post_reconstructed_fault_linear_solver.connect(&probe<dim>);
      signals.post_constraints_creation.connect(&FrozenMechanicalProfile::constraints<dim>);
      signals.post_advection_solver.connect(&FrozenMechanicalProfile::retain_normalization<dim>);
    }
  }
  ASPECT_REGISTER_SIGNALS_CONNECTOR(MechanicalModes::connect<2>,MechanicalModes::connect<3>)
}
