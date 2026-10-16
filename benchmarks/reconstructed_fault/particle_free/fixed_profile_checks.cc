/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/simulator.h>
#include <aspect/postprocess/interface.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/phase_field.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/reconstructed_fault/surface_system.h>
#include <aspect/simulator/assemblers/reconstructed_fault_stokes.h>
#include "../../../tests/phase_field_fault_test_access.h"

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <fstream>
#include <iomanip>

namespace aspect
{
  namespace Postprocess
  {
    // Existing test friendship. This observer freezes the accepted initial
    // localization and drives only native mechanical steps; no phase solve,
    // advection or population update is hidden between those steps.
    template <int dim>
    class MomentCycle : public Interface<dim>, public SimulatorAccess<dim>
    {
        using Model = MaterialModel::PhaseFieldFault<dim>;
        using Access = MaterialModel::internal::PhaseFieldFaultTestAccess<dim>;
        using BV = LinearAlgebra::BlockVector;

        void source_checks(Model &model, Simulator<dim> &sim)
        {
          const auto saved_step=sim.timestep_number;
          const auto saved_dt=sim.time_step;
          auto &rates=const_cast<std::vector<double> &>(this->get_reconstructed_fault_manager().get_slip_rate(0));
          const auto saved_rates=rates;
          const BV anchor(model.accepted_fe_strain_state());
          const auto q_anchor=property("phase field fault cohesive traction");
          const auto tensor_anchor=model.uses_dg_trial_stress_history() ? Access::dg_history_values(model) : std::vector<double>();
          BV prescribed(sim.solution);
          struct Field : Function<dim>
          {
            explicit Field(const unsigned int n) : Function<dim>(n) {}
            double value(const Point<dim> &p,const unsigned int c) const override
            {
              return c==0 ? 1.e-5*p[0]*p[0]*p[1] : (c==1 ? .5e-5*p[0]*p[1]*p[1] : 0.);
            }
          } field(this->get_fe().n_components());
          VectorTools::interpolate(this->get_mapping(),this->get_dof_handler(),field,prescribed,
                                   this->introspection().component_masks.velocities);
          for (unsigned int j=0; j<rates.size(); ++j)
            rates[j]=1.e-5*(1+.3*std::sin(.37*j));
          const auto I=Access::current_normalization_integrals(model).at(0);
          const auto quad=model.mechanical_quadrature();
          FEValues<dim> fe(this->get_mapping(),this->get_fe(),quad,update_values|update_quadrature_points|update_JxW_values);
          const unsigned int pc=this->introspection().variable("phase_field").first_component_index;
          std::vector<double> phi(quad.size());
          std::ofstream out(this->get_output_directory()+"source-checks-"+std::to_string(saved_step)+".csv");
          out << std::setprecision(17) << "dt_over_lambda,max_stress_error_Pa,scale_Pa,integrated_stress_error_Pa_m2\n";
          sim.timestep_number=saved_step+1;
          for (const double ratio :
          {
            1.e-12,1.e-6,.1,1.,4.
          })
          {
            sim.time_step=100*ratio;
            const auto c=Access::maxwell_coefficients(1.e8,1.e6,sim.time_step);
            const auto first=Access::prepared_fe_state(model,prescribed);
            const auto second=Access::prepared_fe_state(model,prescribed);
            const double valid_rate=rates[0];
            rates[0]=std::numeric_limits<double>::quiet_NaN();
            bool rejected=false;
            try
              {
                Access::prepared_fe_state(model,prescribed);
              }
            catch (const std::exception &)
              {
                rejected=true;
              }
            rates[0]=valid_rate;
            AssertThrow(rejected,ExcMessage("Invalid surface candidate was not rejected."));
            double error=0.,scale=1.,integrated_error=0.;
            SymmetricTensor<2,dim> S;
            S[0][1]=.5;
            for (const auto &cell : this->get_dof_handler().active_cell_iterators())
              if (cell->is_locally_owned())
                {
                  fe.reinit(cell);
                  fe[FEValuesExtractors::Scalar(pc)].get_function_values(prescribed,phi);
                  for (unsigned int q=0; q<quad.size(); ++q)
                    {
                      const auto x=fe.quadrature_point(q);
                      // Analytic gradient of the prescribed Q2 field, embedded
                      // explicitly in 3D before taking its deviator.
                      SymmetricTensor<2,3> strain;
                      strain[0][0]=2.e-5*x[0]*x[1];
                      strain[1][1]=1.e-5*x[0]*x[1];
                      strain[0][1]=.5e-5*x[0]*x[0]+.25e-5*x[1]*x[1];
                      strain=deviator(strain);
                      SymmetricTensor<2,dim> eps;
                      for (unsigned int i=0; i<2; ++i)
                        for (unsigned int j=i; j<2; ++j)
                          eps[i][j]=strain[i][j];
                      const unsigned int segment=std::min<unsigned int>(I.size()-2,x[0]*(I.size()-1));
                      const double xi=x[0]*(I.size()-1)-segment;
                      const double normalization=(1-xi)*I[segment]+xi*I[segment+1];
                      const double V=(1-xi)*rates[segment]+xi*rates[segment+1];
                      const double chi=(1/this->get_phase_field_handler().energetic_degradation({.5,.5},phi[q])-1)/normalization;
                      const auto incoming=model.retained_history_stress(cell->id(),x, {});
                      const auto expected=c.first*incoming+2*c.second*(eps-chi*V*S);
                      const auto actual=Access::prepared_fe_stress(model,first,cell->id(),x);
                      const auto repeat=Access::prepared_fe_stress(model,second,cell->id(),x);
                      AssertThrow(actual==repeat,ExcMessage("Repeated native candidates differ."));
                      const auto tensor_difference=actual-expected;
                      const double full_error=std::sqrt(tensor_difference*tensor_difference+trace(tensor_difference)*trace(tensor_difference));
                      error=std::max(error,full_error);
                      scale=std::max(scale,c.first*incoming.norm()+2*c.second*(eps.norm()+std::abs(chi*V)*S.norm()));
                      integrated_error+=fe.JxW(q)*full_error;
                    }
                }
            AssertThrow(error<=1.e-12*scale,ExcMessage("Native history loses the analytic Q2 strain source."));
            out << ratio << ',' << error << ',' << scale << ',' << integrated_error << '\n';
          }
          AssertThrow(model.accepted_fe_strain_state()==anchor
                      && property("phase field fault cohesive traction")==q_anchor
                      && (!model.uses_dg_trial_stress_history() || Access::dg_history_values(model)==tensor_anchor),
                      ExcMessage("Source/candidate checks mutated accepted histories."));
          sim.timestep_number=saved_step;
          sim.time_step=saved_dt;
          rates=saved_rates;
        }

        void representation_checks()
        {
          const FE_DGQ<dim> basis(2), coarse(1);
          auto cell=this->get_triangulation().begin_active();
          while (!cell->is_locally_owned()) ++cell;
          const unsigned int n=basis.n_dofs_per_cell();
          std::ofstream out(this->get_output_directory()+"representation.csv");
          out << std::setprecision(17) << "gauss_points,mass_rank,min_singular,max_singular,DGQ1_gradient_L2\n";
          for (const unsigned int order :
          {
            2u,3u
          })
          {
            const QGauss<dim> quad(order);
            FEValues<dim> fe(this->get_mapping(),basis,quad,update_values|update_JxW_values);
            fe.reinit(cell);
            LAPACKFullMatrix<double> mass(n,n);
            for (unsigned int q=0; q<quad.size(); ++q)
              for (unsigned int i=0; i<n; ++i)
                for (unsigned int j=0; j<n; ++j)
                  mass(i,j)+=fe.JxW(q)*fe.shape_value(i,q)*fe.shape_value(j,q);
            mass.compute_svd();
            unsigned int rank=0;
            for (unsigned int i=0; i<n; ++i)
              rank+=mass.singular_value(i)>1.e-12*mass.singular_value(0);
            AssertThrow(rank==(order==2 ? 4u : 9u),ExcMessage("Unexpected DGQ2 mass rank on the actual mapping."));
            out << order << ',' << rank << ',' << mass.singular_value(n-1) << ',' << mass.singular_value(0) << ",0\n";
          }
          // Project the analytic Q2 source epsilon_xy, whose y^2 term a
          // bilinear history space cannot retain, using actual physical cells.
          const QGauss<dim> quad(4);
          FEValues<dim> fe(this->get_mapping(),coarse,quad,update_values|update_quadrature_points|update_JxW_values);
          const unsigned int nc=coarse.n_dofs_per_cell();
          double error=0.;
          for (const auto &c : this->get_triangulation().active_cell_iterators())
            if (c->is_locally_owned())
              {
                fe.reinit(c);
                FullMatrix<double> mass(nc,nc);
                Vector<double> rhs(nc), coefficients(nc);
                for (unsigned int q=0; q<quad.size(); ++q)
                  {
                    const auto x=fe.quadrature_point(q);
                    const double source=.5e-5*x[0]*x[0]+.25e-5*x[1]*x[1];
                    for (unsigned int i=0; i<nc; ++i)
                      {
                        rhs[i]+=fe.JxW(q)*source*fe.shape_value(i,q);
                        for (unsigned int j=0; j<nc; ++j)
                          mass(i,j)+=fe.JxW(q)*fe.shape_value(i,q)*fe.shape_value(j,q);
                      }
                  }
                mass.gauss_jordan();
                mass.vmult(coefficients,rhs);
                for (unsigned int q=0; q<quad.size(); ++q)
                  {
                    const auto x=fe.quadrature_point(q);
                    double actual=0.;
                    for (unsigned int i=0; i<nc; ++i) actual+=coefficients[i]*fe.shape_value(i,q);
                    error+=fe.JxW(q)*std::pow(actual-.5e-5*x[0]*x[0]-.25e-5*x[1]*x[1],2);
                  }
              }
          AssertThrow(std::sqrt(error)>1.e-10,ExcMessage("DGQ1 negative control did not expose lost source modes."));
          out << "4,0,0,0," << std::sqrt(error) << '\n';
        }

        void operator_checks(Model &model, Simulator<dim> &sim, const unsigned int step,
                             const BV &start, const BV &mechanical_trial,
                             const ReconstructedFaultVector &rates)
        {
          const auto &intro=this->introspection();
          const auto vb=intro.block_indices.velocities;
          const auto pb=intro.block_indices.pressure;
          auto &surface=*sim.reconstructed_fault_surface_system;
          const BV saved_linearization(sim.current_linearization_point);
          const auto tensor_anchor=model.uses_dg_trial_stress_history() ? Access::dg_history_values(model) : std::vector<double>();
          // A zero-velocity/pressure probe exposes the nontrivial native frozen
          // history load; near-zero equilibrium residuals alone are insufficient.
          BV probe(mechanical_trial);
          BV owned(intro.index_sets.system_partitioning,this->get_mpi_communicator());
          owned=probe;
          owned.block(vb)=0.;
          owned.block(pb)=0.;
          probe=owned;
          Access::clear_native_stress_samples(model);
          sim.endpoint_mechanical_defects(probe,rates);
          const BV clean_rhs(sim.system_rhs);
          const auto clean_surface=surface.evaluate_surface_residual(probe,rates);
          std::ofstream loads(this->get_output_directory()+"loads-"+std::to_string(step)+".csv");
          loads << std::setprecision(17) << "index,bulk_rhs\n";
          for (const auto i : clean_rhs.block(vb).locally_owned_elements())
            loads << i << ',' << clean_rhs.block(vb)[i] << '\n';
          std::ofstream shear(this->get_output_directory()+"shear-load-"+std::to_string(step)+".csv");
          shear << std::setprecision(17) << "node,shear,cohesive,friction,residual\n";
          for (unsigned int j=0; j<rates[0].size(); ++j)
            shear << j << ',' << clean_surface.shear_traction[0][j] << ',' << clean_surface.cohesive_traction[0][j]
                  << ',' << clean_surface.friction_traction[0][j] << ',' << clean_surface.values[0][j] << '\n';

          auto &particles=sim.phase_field_handler->get_associated_particle_manager().get_particle_handler();
          const auto &properties=sim.phase_field_handler->get_associated_particle_manager().get_property_manager().get_data_info();
          const auto sp=properties.get_position_by_field_name("maxwell stress");
          std::map<types::particle_index,std::vector<double>> saved_particles;
          for (auto &particle : particles)
            {
              saved_particles[particle.get_id()]= {particle.get_properties().begin(),particle.get_properties().end()};
              for (unsigned int c=0; c<SymmetricTensor<2,dim>::n_independent_components; ++c)
                particle.get_properties()[sp+c]=1.e7*(c+1)+particle.get_id();
            }
          BV poisoned_probe(probe), poisoned_start(start);
          for (auto *state :
               {
                 &poisoned_probe,&poisoned_start
               })
            {
              owned=*state;
              for (const auto &mapping : sim.parameters.mapped_particle_properties)
                if (mapping.second.first=="maxwell stress")
                  owned.block(intro.block_indices.compositional_fields[mapping.first])=2.e7*(mapping.second.second+1);
              *state=owned;
            }
          Access::clear_native_stress_samples(model);
          sim.endpoint_mechanical_defects(poisoned_probe,rates);
          BV rhs_error(sim.system_rhs);
          rhs_error-=clean_rhs;
          double load_error=std::max(rhs_error.block(vb).linfty_norm(),rhs_error.block(pb).linfty_norm());
          const auto dirty_surface=surface.evaluate_surface_residual(poisoned_probe,rates);
          for (unsigned int j=0; j<rates[0].size(); ++j)
            {
              load_error=std::max(load_error,std::abs(dirty_surface.shear_traction[0][j]-clean_surface.shear_traction[0][j]));
              load_error=std::max(load_error,std::abs(dirty_surface.values[0][j]-clean_surface.values[0][j]));
            }
          sim.solution=poisoned_start;
          sim.current_linearization_point=poisoned_start;
          ReconstructedFaultVector poisoned_rates;
          sim.solve_reconstructed_fault_stokes(false,&poisoned_rates);
          BV solution_error(sim.solution);
          solution_error-=mechanical_trial;
          double solve_error=std::max(solution_error.block(vb).linfty_norm(),solution_error.block(pb).linfty_norm());
          for (unsigned int j=0; j<rates[0].size(); ++j)
            solve_error=std::max(solve_error,std::abs(poisoned_rates[0][j]-rates[0][j]));
          for (auto &particle : particles)
            std::copy(saved_particles.at(particle.get_id()).begin(),saved_particles.at(particle.get_id()).end(),particle.get_properties().begin());
          sim.solution=start;
          sim.current_linearization_point=saved_linearization;
          Access::clear_native_stress_samples(model);
          AssertThrow(load_error==0. && solve_error==0.
                      && (!model.uses_dg_trial_stress_history() || Access::dg_history_values(model)==tensor_anchor),
                      ExcMessage("Particle or mapped stress fed native loads/solve or changed DG history."));

          // Actual B/G work pair with independent surface coefficients and a Q2
          // bubble velocity (zero at the constrained boundary). Also integrate
          // the bilinear form independently on the same physical measure/rule.
          BV bubble(start);
          bubble=0.;
          struct Bubble : Function<dim>
          {
            explicit Bubble(const unsigned int n) : Function<dim>(n) {}
            double value(const Point<dim> &x,const unsigned int c) const override
            {
              return c==0 ? x[0]*(1-x[0])*x[1]*(1-x[1])*(x[1]-.5) : 0.;
            }
          } field(this->get_fe().n_components());
          VectorTools::interpolate(this->get_mapping(),this->get_dof_handler(),field,bubble,intro.component_masks.velocities);
          ReconstructedFaultVector direction=rates, g;
          for (unsigned int j=0; j<direction[0].size(); ++j)
            direction[0][j]=.2+.1*std::sin(.43*j);
          sim.reconstructed_fault_stokes_coupling->linearize_B(mechanical_trial);
          surface.linearize_surface_system(mechanical_trial,rates);
          BV b(bubble);
          sim.reconstructed_fault_stokes_coupling->apply_B(direction,b);
          surface.apply_G(bubble,g);
          const double left=bubble.block(vb)*b.block(vb);
          double right=0., independent=0.;
          for (unsigned int j=0; j<direction[0].size(); ++j)
            right+=direction[0][j]*g[0][j];
          const auto I=Access::current_normalization_integrals(model).at(0);
          const auto quad=model.mechanical_quadrature();
          FEValues<dim> fe(this->get_mapping(),this->get_fe(),quad,update_values|update_gradients|update_quadrature_points|update_JxW_values);
          std::vector<double> phi(quad.size());
          std::vector<SymmetricTensor<2,dim>> bubble_strain(quad.size());
          const auto c=Access::maxwell_coefficients(1.e8,1.e6,sim.time_step);
          for (const auto &cell : this->get_dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                fe.reinit(cell);
                fe[FEValuesExtractors::Scalar(intro.variable("phase_field").first_component_index)].get_function_values(start,phi);
                fe[intro.extractors.velocities].get_function_symmetric_gradients(bubble,bubble_strain);
                for (unsigned int q=0; q<quad.size(); ++q)
                  {
                    const auto x=fe.quadrature_point(q);
                    const unsigned int j=std::min<unsigned int>(I.size()-2,x[0]*(I.size()-1));
                    const double xi=x[0]*(I.size()-1)-j;
                    const double chi=(1/this->get_phase_field_handler().energetic_degradation({.5,.5},phi[q])-1)/((1-xi)*I[j]+xi*I[j+1]);
                    const double v=(1-xi)*direction[0][j]+xi*direction[0][j+1];
                    const double shear=bubble_strain[q][0][1];
                    independent+=fe.JxW(q)*2*c.second*chi*v*shear;
                  }
              }
          const double scale=std::max({1.,std::abs(left),std::abs(right),std::abs(independent)});
          AssertThrow(std::abs(independent)>1. && std::abs(left-right)<=1.e-10*scale && std::abs(left-independent)<=1.e-10*scale,
                      ExcMessage("Actual B/G maps are not the expected work adjoints."));
          std::ofstream checks(this->get_output_directory()+"operator-checks.csv",step==1 ? std::ios::out : std::ios::app);
          if (step==1) checks << "step,poison_load_error,poison_solve_error,B_work,G_work,independent_work,adjoint_error\n";
          checks << std::setprecision(17) << step << ',' << load_error << ',' << solve_error << ','
                 << left << ',' << right << ',' << independent << ',' << std::abs(left-right)/scale << '\n';
        }

        void reject_complete_step(Model &model, Simulator<dim> &sim)
        {
          auto &manager=*sim.reconstructed_fault_manager;
          auto &pm=sim.phase_field_handler->get_associated_particle_manager();
          auto &particles=pm.get_particle_handler();
          const auto hp=pm.get_property_manager().get_data_info().get_position_by_field_name("crack_driving_force");
          const BV solution(sim.solution), linearization(sim.current_linearization_point);
          const BV accepted(model.accepted_fe_strain_state());
          const auto q=property("phase field fault cohesive traction");
          const auto previous_I=property("phase field fault previous I h");
          const auto rates=manager.get_timestep_committed_slip_rate(0);
          const auto tensor=model.uses_dg_trial_stress_history() ? Access::dg_history_values(model) : std::vector<double>();
          std::map<types::particle_index,std::vector<double>> properties;
          for (const auto &p : particles)
            properties[p.get_id()]= {p.get_properties().begin(),p.get_properties().end()};
          auto particle=particles.begin();
          const double valid_H=particle->get_properties()[hp];
          particle->get_properties()[hp]=std::numeric_limits<double>::quiet_NaN();
          std::string failure;
          try
            {
              sim.solve_reconstructed_fault_stokes();
            }
          catch (const std::exception &e)
            {
              failure=e.what();
            }
          particle->get_properties()[hp]=valid_H;
          AssertThrow(failure.find("Stored crack-driving history")!=std::string::npos,
                      ExcMessage("Complete-step failure did not reach the intended prepublication validation."));
          AssertThrow(sim.solution==solution && sim.current_linearization_point==linearization
                      && model.accepted_fe_strain_state()==accepted
                      && property("phase field fault cohesive traction")==q
                      && property("phase field fault previous I h")==previous_I
                      && manager.get_timestep_committed_slip_rate(0)==rates
                      && (!model.uses_dg_trial_stress_history() || Access::dg_history_values(model)==tensor),
                      ExcMessage("Rejected mechanical step changed accepted native histories."));
          for (const auto &p : particles)
            AssertThrow(std::equal(properties.at(p.get_id()).begin(),properties.at(p.get_id()).end(),p.get_properties().begin()),
                        ExcMessage("Rejected mechanical step changed particle samples/history."));
          std::ofstream out(this->get_output_directory()+"rejection.txt");
          out << "Native mechanical convergence followed by invalid-H prepublication rejection; all physical inputs restored exactly.\n";
        }

        std::vector<double> property(const std::string &name) const
        {
          const auto &manager = this->get_reconstructed_fault_manager();
          const auto position = manager.get_property_information()[manager.get_property_index(name)].position;
          std::vector<double> values(manager.get_fault(0).n_vertices());
          for (unsigned int i=0; i<values.size(); ++i)
            values[i] = manager.get_fault(0).get_properties(i)[position];
          return values;
        }

        void write_state(const unsigned int step, const Model &model,
                         const BV &state, const std::vector<double> &slip)
        {
          const auto &intro = this->introspection();
          const auto quad = model.mechanical_quadrature();
          FEValues<dim> fe(this->get_mapping(), this->get_fe(), quad,
                           update_values|update_gradients|update_quadrature_points|update_JxW_values);
          std::vector<Tensor<1,dim>> u(quad.size());
          std::vector<SymmetricTensor<2,dim>> strain(quad.size());
          std::vector<double> phi(quad.size()), pressure(quad.size());
          const auto pc = intro.variable("phase_field").first_component_index;
          if (step==0)
            {
              const auto phase_block=intro.variable("phase_field").block_index;
              std::ofstream seed(this->get_output_directory()+"accepted-phase.csv");
              seed << std::setprecision(17) << "index,phi\n";
              for (const auto i : state.block(phase_block).locally_owned_elements())
                seed << i << ',' << state.block(phase_block)[i] << '\n';
            }
          std::ofstream bulk(this->get_output_directory()+"bulk-"+std::to_string(step)+".csv");
          bulk << std::setprecision(17) << "x,y,weight,phi,ux,uy,p,tau_xx,tau_yy,tau_xy,tau_zz\n";
          double storage = 0.;
          const double G = 1.e6;
          for (const auto &cell : this->get_dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                fe.reinit(cell);
                fe[intro.extractors.velocities].get_function_values(state,u);
                fe[intro.extractors.velocities].get_function_symmetric_gradients(state,strain);
                fe[intro.extractors.pressure].get_function_values(state,pressure);
                fe[FEValuesExtractors::Scalar(pc)].get_function_values(state,phi);
                for (unsigned int q=0; q<quad.size(); ++q)
                  {
                    const auto x = fe.quadrature_point(q);
                    const auto tau = model.retained_history_stress(cell->id(),x, {});
                    storage += fe.JxW(q)*(tau*tau+trace(tau)*trace(tau))/(4*G);
                    bulk << x[0] << ',' << x[1] << ',' << fe.JxW(q) << ',' << phi[q]
                         << ',' << u[q][0] << ',' << u[q][1] << ',' << pressure[q]
                         << ',' << tau[0][0] << ',' << tau[1][1] << ',' << tau[0][1]
                         << ',' << -trace(tau) << '\n';
                  }
              }
          const auto I = Access::current_normalization_integrals(model).at(0);
          const auto q = property("phase field fault cohesive traction");
          const auto &V = this->get_reconstructed_fault_manager().get_slip_rate(0);
          std::ofstream surface(this->get_output_directory()+"surface-"+std::to_string(step)+".csv");
          surface << std::setprecision(17) << "x,I,q,Z,s_elas,V,slip\n";
          for (unsigned int j=0; j<I.size(); ++j)
            surface << double(j)/(I.size()-1) << ',' << I[j] << ',' << q[j]
                    << ',' << I[j]*q[j] << ',' << I[j]*q[j]/G << ',' << V[j] << ',' << slip[j] << '\n';
          std::ofstream energy(this->get_output_directory()+"storage.csv", step ? std::ios::app : std::ios::out);
          if (step==0) energy << "step,bulk_J_m,cohesive_J_m\n";
          energy << std::setprecision(17) << step << ',' << storage << ',' << Access::cohesive_storage(model) << '\n';
        }

      public:
        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          AssertThrow(dim==2 && this->get_timestep_number()==0
                      && Utilities::MPI::n_mpi_processes(this->get_mpi_communicator())==1,
                      ExcMessage("PF2 is a fresh serial frozen-localization diagnostic."));
          auto &sim = const_cast<Simulator<dim> &>(this->get_simulator());
          auto &model = Plugins::get_plugin_as_type<Model>(*sim.material_model);
          auto &manager = *sim.reconstructed_fault_manager;
          model.prepare_reconstructed_fault_mechanical_solve();
          const unsigned int pb = this->introspection().variable("phase_field").block_index;
          const BV initial(sim.solution);
          const auto initial_I = Access::current_normalization_integrals(model);
          const auto old_time = sim.time;
          const auto old_dt = sim.time_step;
          std::vector<double> slip(manager.get_fault(0).n_vertices(),0.);
          write_state(0,model,sim.solution,slip);
          representation_checks();
          source_checks(model,sim);
          if (model.uses_dg_trial_stress_history())
            {
              auto particle=this->get_phase_field_handler().get_associated_particle_manager().get_particle_handler().begin();
              const auto position=particle->get_location();
              auto moved=position;
              moved[0]+=1.e-8;
              particle->set_location(moved);
              bool motion_rejected=false;
              try
                {
                  Access::validate_native_history(model);
                }
              catch (const std::exception &)
                {
                  motion_rejected=true;
                }
              particle->set_location(position);
              AssertThrow(motion_rejected,ExcMessage("DG history admitted material transport."));
              BV altered(this->introspection().index_sets.system_partitioning,this->get_mpi_communicator());
              altered=sim.solution;
              // A tiny admissible phase change must still be rejected: PF2 has
              // fixed localization, rather than a permissive phase approximation.
              for (const auto i : altered.block(pb).locally_owned_elements())
                altered.block(pb)[i]*=.999;
              sim.solution.block(pb)=altered.block(pb);
              bool phase_rejected=false;
              try
                {
                  Access::validate_native_history(model);
                }
              catch (const std::exception &)
                {
                  phase_rejected=true;
                }
              sim.solution=initial;
              AssertThrow(phase_rejected,ExcMessage("DG history admitted changing localization."));
            }
          std::ofstream metrics(this->get_output_directory()+"steps.csv");
          metrics << std::setprecision(17) << "step,time,dt,max_V,max_q,max_s_elas,Z_error,phase_change,I_change,repeat_error\n";
          for (unsigned int step=1; step<=3; ++step)
            {
              sim.timestep_number = step;
              sim.time_step = .125;
              sim.time = step*sim.time_step;
              sim.old_old_solution = sim.old_solution;
              sim.old_solution = sim.solution;
              sim.current_linearization_point = sim.solution;
              const auto I = Access::current_normalization_integrals(model).at(0);
              auto Z = property("phase field fault cohesive traction");
              for (unsigned int j=0; j<Z.size(); ++j) Z[j] *= I[j];
              const BV anchor(model.accepted_fe_strain_state());
              const auto q_anchor = property("phase field fault cohesive traction");
              const BV start(sim.solution);
              ReconstructedFaultVector first, second;
              sim.solve_reconstructed_fault_stokes(false,&first);
              const BV trial(sim.solution);
              sim.solution = start;
              sim.current_linearization_point = start;
              sim.solve_reconstructed_fault_stokes(false,&second);
              BV difference(sim.solution);
              difference -= trial;
              double repeat_error = difference.linfty_norm();
              for (unsigned int j=0; j<first[0].size(); ++j)
                repeat_error = std::max(repeat_error,std::abs(first[0][j]-second[0][j]));
              AssertThrow(repeat_error==0. && model.accepted_fe_strain_state()==anchor
                          && property("phase field fault cohesive traction")==q_anchor,
                          ExcMessage("Repeated mechanical trials changed accepted history or output."));
              operator_checks(model,sim,step,start,trial,first);
              sim.solution = start;
              sim.current_linearization_point = start;
              if (step==1)
                reject_complete_step(model,sim);
              sim.solve_reconstructed_fault_stokes();
              const auto &rates = manager.get_slip_rate(0);
              const auto q = property("phase field fault cohesive traction");
              const auto I_new = Access::current_normalization_integrals(model).at(0);
              const auto c = Access::maxwell_coefficients(1.e8,1.e6,sim.time_step);
              double error = 0., max_s = 0.;
              for (unsigned int j=0; j<Z.size(); ++j)
                {
                  error = std::max(error,std::abs(I_new[j]*q[j]-c.first*Z[j]-c.second*rates[j]));
                  max_s = std::max(max_s,I_new[j]*q[j]/1.e6);
                  slip[j] += sim.time_step*rates[j];
                }
              AssertThrow(error<=1.e-10 && sim.solution.block(pb)==initial.block(pb)
                          && Access::current_normalization_integrals(model)==initial_I,
                          ExcMessage("Frozen cohesive path changed localization or violated Z recurrence."));
              if (model.uses_dg_trial_stress_history())
                AssertThrow(model.accepted_fe_strain_state().block(this->introspection().block_indices.velocities).linfty_norm()==0.,
                            ExcMessage("DG history evolved the unused FE displacement anchor."));
              bool duplicate_rejected = false;
              const BV published(model.accepted_fe_strain_state());
              const auto published_tensor=model.uses_dg_trial_stress_history() ? Access::dg_history_values(model) : std::vector<double>();
              try
                {
                  model.commit_reconstructed_fault_mechanical_history(sim.solution);
                }
              catch (const std::exception &)
                {
                  duplicate_rejected = true;
                }
              AssertThrow(duplicate_rejected && model.accepted_fe_strain_state()==published
                          && property("phase field fault cohesive traction")==q
                          && (!model.uses_dg_trial_stress_history() || Access::dg_history_values(model)==published_tensor),
                          ExcMessage("Duplicate publication was accepted or mutated history."));
              const double max_q = *std::max_element(q.begin(),q.end());
              AssertThrow(max_s>0. && max_q>0.,ExcMessage("Fixture did not develop physical cohesive storage."));
              metrics << step << ',' << sim.time << ',' << sim.time_step << ','
                      << *std::max_element(rates.begin(),rates.end()) << ',' << max_q << ','
                      << max_s << ',' << error << ",0,0," << repeat_error << '\n';
              write_state(step,model,sim.solution,slip);
            }
          source_checks(model,sim);
          // The enclosing simulator executes only initialization. Identify the
          // observer's mechanical path explicitly instead of advancing its clock.
          sim.time = old_time;
          sim.time_step = old_dt;
          sim.timestep_number = 0;
          this->get_pcout() << "   PF2 frozen-localization native mechanical path passed; three physical history commits." << std::endl;
          return {"PF2 fixed profile:","passed"};
        }
    };
    ASPECT_REGISTER_POSTPROCESSOR(MomentCycle,"PF2 fixed profile checks",
                                  "Bounded native mechanical path with frozen initial localization; no outer phase evolution or transport.")
  }
}
