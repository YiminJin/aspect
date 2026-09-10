#include "phase_field_fault_stage_i.cc"
#include <cstdlib>

namespace aspect
{
  template <int dim>
  void connect_residual_consistency(SimulatorSignals<dim> &signals)
  {
    signals.start_timestep.connect([](const SimulatorAccess<dim> &)
    {
      AssertThrow(setenv("ASPECT_K1_FLOOR_AUDIT", "1", 1) == 0,
                  ExcMessage("Could not enable the residual consistency regression."));
    });
  }
  ASPECT_REGISTER_SIGNALS_CONNECTOR(connect_residual_consistency<2>,
                                    connect_residual_consistency<3>)
}
