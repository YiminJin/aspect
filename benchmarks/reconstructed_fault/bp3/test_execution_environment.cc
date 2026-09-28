#include "plugin/execution_environment.h"
#include <cstring>
#include <iostream>

int main()
{
  for (const char *name : {"ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION",
       "ASPECT_IH_BOTTOM_COMPLETION_DIAGNOSTIC", "ASPECT_BP3_UNIFORM_SLIDING",
       "ASPECT_BP3_TOP_SOURCE_EXPERIMENT", "ASPECT_FAULT_INTERFACE_MODES",
       "ASPECT_BP5_SHORT_TEST", "ASPECT_DISTURBANCE_EPS",
       "ASPECT_DISTURBANCE_CONTROL", "ASPECT_DISTURBANCE_DT"})
    unsetenv(name);
  if (BP3::unexpected_execution_switch()) return 1;
  for (const char *name : {"ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION",
                         "ASPECT_FAULT_INTERFACE_MODES", "ASPECT_IH_BOTTOM_COMPLETION_DIAGNOSTIC"})
    {
      setenv(name,"0",1);
      const char *found=BP3::unexpected_execution_switch();
      if (!found || std::strcmp(found,name)) return 2;
      unsetenv(name);
    }
  setenv("ASPECT_FAULT_EXPLICIT_B","1",1);
  setenv("ASPECT_FAULT_EXPLICIT_G","1",1);
  setenv("ASPECT_FAULT_VELOCITY_GMG","1",1);
  setenv("ASPECT_FAULT_LINEAR_PERFORMANCE","1",1);
  setenv("ASPECT_BP5_SHORT_TEST","0",1); // retired from this runtime
  if (BP3::unexpected_execution_switch()) return 3;
  std::cout<<"BP3 clean/contaminated environment checks passed.\n";
}
