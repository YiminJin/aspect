// R1 observation of the unchanged production parser and limiter. No repairs.
#include "../../../tests/phase_field_fault_state_limiter.cc"
#include <aspect/time_stepping/reconstructed_fault.h>
#include <iomanip>

namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    class R1LimiterProbe : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          using Limiter = TimeStepping::ReconstructedFault<dim>;
          const auto &material = Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(
            this->get_material_model());
          auto &manager = this->get_reconstructed_fault_manager();
          auto &fault = manager.get_fault(0);
          const auto position = manager.get_property_information()[manager.get_property_index(
            "phase field fault state")].position;
          const double saved_theta = fault.get_properties(0)[position];
          const double law_dt = material.compute_reconstructed_fault_time_step(this->get_parameters().CFL_number);
          std::ostringstream exact;
          exact << std::setprecision(17) << std::numeric_limits<double>::max();
          for (const auto &value : std::vector<std::string>{"default", "infinity", "0", "0.1", exact.str()})
            {
              ParameterHandler prm;
              Limiter limiter;
              limiter.initialize_simulator(this->get_simulator());
              limiter.initialize();
              try
                {
                  Limiter::declare_parameters(prm);
                  prm.enter_subsection("Time stepping");
                  prm.enter_subsection("Reconstructed fault time step");
                  if (value != "default") prm.set("Maximum logarithmic state change", value);
                  const std::string spelling = prm.get("Maximum logarithmic state change");
                  const double parsed = prm.get_double("Maximum logarithmic state change");
                  prm.leave_subsection(); prm.leave_subsection();
                  limiter.parse_parameters(prm);
                  this->get_pcout() << std::setprecision(17) << "R1 parameter: input=" << value
                    << " text=" << spelling << " parsed=" << parsed
                    << " exact_max=" << (parsed == std::numeric_limits<double>::max()) << std::endl;
                  // A zero bound can search down to machine precision; observe
                  // parsing here rather than spending an unbounded interval on it.
                  if (parsed != 0.)
                    {
                      const double dt = limiter.execute();
                      this->get_pcout() << "R1 proposal: input=" << value << " dt=" << dt
                        << " law_dt=" << law_dt << " identical=" << (dt==law_dt) << std::endl;
                      // Deliberately invalid synthetic history detects whether
                      // the disabled path actually avoids the state predictor.
                      fault.get_properties(0)[position] = std::numeric_limits<double>::quiet_NaN();
                      bool bypass = false;
                      try { bypass = limiter.execute()==law_dt; }
                      catch (const std::exception &) {}
                      fault.get_properties(0)[position] = saved_theta;
                      this->get_pcout() << "R1 state bypass: input=" << value << " bypass=" << bypass << std::endl;
                    }
                }
              catch (const std::exception &error)
                {
                  fault.get_properties(0)[position] = saved_theta;
                  this->get_pcout() << "R1 parameter rejected: input=" << value << "\n" << error.what() << std::endl;
                }
            }
          AssertThrow(fault.get_properties(0)[position] == saved_theta, ExcInternalError());
          this->get_pcout() << "R1 parameter probe restored Theta" << std::endl;
          return {};
        }
    };
    ASPECT_REGISTER_POSTPROCESSOR(R1LimiterProbe,"R1 limiter probe",
                                  "Observe parser and disabled-path behavior; restore every synthetic state probe.")
  }
}
