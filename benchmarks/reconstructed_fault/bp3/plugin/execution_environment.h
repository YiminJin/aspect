#ifndef ASPECT_BENCHMARK_BP3_EXECUTION_ENVIRONMENT_H
#define ASPECT_BENCHMARK_BP3_EXECUTION_ENVIRONMENT_H

#include <cstdlib>
#include <initializer_list>

namespace BP3
{
  // Live core experiments: particle/rk_2.cc freezes transport;
  // material_model/phase_field_fault.cc accepts legacy completion inputs;
  // simulator/reconstructed_fault_interface_preconditioner.h changes the
  // preconditioner. Presence-based switches must be unset, even if set to "0".
  inline const char *unexpected_execution_switch()
  {
    for (const char *name : {
           "ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION",
           "ASPECT_IH_BOTTOM_COMPLETION_DIAGNOSTIC",
           "ASPECT_BP3_UNIFORM_SLIDING",
           "ASPECT_FAULT_INTERFACE_MODES"})
      if (std::getenv(name)) return name;
    return nullptr;
  }
}

#endif
