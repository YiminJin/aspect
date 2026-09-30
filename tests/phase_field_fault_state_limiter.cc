// Exercise the real timestep plugin against committed nodal state. Synthetic
// probe histories are restored before leaving the accepted-state observer.
#include "phase_field_fault_stage_i.cc"
#include <aspect/postprocess/interface.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/time_stepping/reconstructed_fault.h>
#include <aspect/plugins.h>

namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    class VerifyFaultStateLimiter : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          const auto &model = Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(
            this->get_material_model());
          const auto &law = model.get_fault_friction();
          TimeStepping::ReconstructedFault<dim> limiter;
          limiter.initialize_simulator(this->get_simulator());
          limiter.initialize();
          ParameterHandler prm;
          TimeStepping::ReconstructedFault<dim>::declare_parameters(prm);
          limiter.parse_parameters(prm);
          const auto set_limit = [&](const std::string &value)
          {
            prm.enter_subsection("Time stepping");
            prm.enter_subsection("Reconstructed fault time step");
            prm.set("Maximum logarithmic state change", value);
            prm.leave_subsection();
            prm.leave_subsection();
            limiter.parse_parameters(prm);
          };
          const auto old_proposal = [&]()
          { return model.compute_reconstructed_fault_time_step(this->get_parameters().CFL_number); };
          AssertThrow(limiter.execute() == old_proposal(), ExcMessage("Default limiter changed the old proposal."));

          if (!law.has_state_variable())
            {
              set_limit("0.1");
              AssertThrow(limiter.execute() == old_proposal(), ExcMessage("State limiter affected stateless friction."));
              this->get_pcout() << "State limiter: stateless law bypass PASS" << std::endl;
              return {};
            }

          auto &manager = this->get_reconstructed_fault_manager();
          const auto position = manager.get_property_information()[manager.get_property_index(
            "phase field fault state")].position;
          std::vector<std::vector<double>> saved_velocity, saved_theta, velocity;
          const double Dc = law.get_characteristic_slip_distance();
          for (unsigned int f=0; f<manager.get_faults().size(); ++f)
            {
              auto &fault = manager.get_fault(f);
              saved_velocity.push_back(manager.get_timestep_committed_slip_rate(f));
              saved_theta.emplace_back();
              velocity.emplace_back(fault.n_vertices(),1e-6);
              velocity.back().back() = 2e-6;
              for (unsigned int i=0; i<fault.n_vertices(); ++i)
                saved_theta.back().push_back(fault.get_properties(i)[position]);
            }
          const auto commit_velocity = [&](const std::vector<std::vector<double>> &values)
          {
            manager.begin_slip_rate_nonlinear_solve();
            manager.begin_slip_rate_trial();
            manager.set_slip_rate_trial_values(values);
            manager.accept_slip_rate_trial();
            manager.validate_slip_rate_nonlinear_commit();
            manager.commit_slip_rate_nonlinear_solve();
          };
          commit_velocity(velocity);
          const double ceiling = std::min(old_proposal(),this->get_parameters().maximum_time_step);
          const auto steady_state = [&]()
          {
            for (unsigned int f=0; f<velocity.size(); ++f)
              for (unsigned int i=0; i<velocity[f].size(); ++i)
                manager.get_fault(f).get_properties(i)[position] = Dc/velocity[f][i];
          };
          set_limit("0.1");
          steady_state();
          AssertThrow(limiter.execute()==ceiling, ExcMessage("Equilibrium state limited the timestep."));

          // Put the restrictive state at the final vertex, not the first one.
          auto &fault = manager.get_fault(velocity.size()-1);
          const auto last = fault.n_vertices()-1;
          const double V = velocity.back().back(), steady = Dc/V;
          for (const double theta : {0.01, 10.*steady, 1e-30})
            {
              steady_state();
              fault.get_properties(last)[position]=theta;
              const double dt=limiter.execute();
              const double direction=theta<steady ? 1. : -1.;
              // Independent inverse of the constant-rate ODE solution, using
              // log1p/expm1 to resolve very small initial state and timestep.
              const double expected=-steady*std::log1p(theta*std::expm1(direction*.1)/(theta-steady));
              AssertThrow(std::abs(dt/expected-1.)<2e-12,
                          ExcMessage("State limiter differs from the unweighted analytic bound."));
              AssertThrow(std::abs(std::log(law.update_state(V,theta,dt)/theta))<=.1,
                          ExcMessage("State limiter returned an unsafe bracket end."));
              AssertThrow(fault.get_properties(last)[position]==theta,
                          ExcMessage("State prediction changed committed Theta."));
              AssertThrow(limiter.execute()==dt, ExcMessage("State prediction was not repeatable."));
              manager.begin_slip_rate_nonlinear_solve();
              manager.begin_slip_rate_trial();
              auto trial=velocity;
              for (auto &rates:trial) for (double &rate:rates) rate*=100.;
              manager.set_slip_rate_trial_values(trial);
              AssertThrow(limiter.execute()==dt, ExcMessage("State predictor read trial instead of committed velocity."));
              manager.rollback_slip_rate_trial();
              manager.rollback_slip_rate_nonlinear_solve();
              AssertThrow(Utilities::MPI::min(dt,this->get_mpi_communicator())
                          ==Utilities::MPI::max(dt,this->get_mpi_communicator()),
                          ExcMessage("Replicated state predictions differ across ranks."));
            }
          // A finite large bound must not be spuriously activated merely
          // because forming predicted/old overflows double precision.
          steady_state();
          fault.get_properties(last)[position]=1e-310;
          set_limit("1000");
          AssertThrow(limiter.execute()==ceiling, ExcMessage("Overflow of a state ratio caused spurious limiting."));
          set_limit(Utilities::to_string(std::numeric_limits<double>::max()));
          AssertThrow(limiter.execute()==old_proposal(), ExcMessage("Maximum finite double did not disable limiting."));
          commit_velocity(saved_velocity);
          for (unsigned int f=0; f<velocity.size(); ++f)
            {
              for (unsigned int i=0; i<velocity[f].size(); ++i)
                manager.get_fault(f).get_properties(i)[position]=saved_theta[f][i];
            }
          this->get_pcout() << "State limiter: maximum finite double, equilibrium, increasing/decreasing state, tiny timestep, "
                           << "all vertices, committed velocity, nonmutation and MPI agreement PASS" << std::endl;
          return {};
        }
    };
    ASPECT_REGISTER_POSTPROCESSOR(VerifyFaultStateLimiter,"verify fault state limiter",
                                  "Test the reconstructed-fault timestep state predictor without committing its probes.")
  }
}
