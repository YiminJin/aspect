#ifndef ASPECT_BENCHMARK_BP3_EXECUTION_ENVIRONMENT_H
#define ASPECT_BENCHMARK_BP3_EXECUTION_ENVIRONMENT_H

#include <cstdlib>
#include <initializer_list>

namespace BP3
{
  // Presence-based selectors are active even when set to "0". The maintained
  // environment clears them; reject an accidental direct launch as well.
  inline const char *unexpected_execution_switch()
  {
    for (const char *name : {
           "ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION",
           "ASPECT_IH_BOTTOM_COMPLETION_DIAGNOSTIC",
           "ASPECT_BP3_UNIFORM_SLIDING",
           "ASPECT_BP3_TOP_SOURCE_EXPERIMENT",
           "ASPECT_FAULT_INTERFACE_MODES",
           "ASPECT_BP5_SHORT_TEST",
           "ASPECT_DISTURBANCE_EPS",
           "ASPECT_DISTURBANCE_CONTROL",
           "ASPECT_DISTURBANCE_DT"})
      if (std::getenv(name)) return name;
    return nullptr;
  }
}

#endif
