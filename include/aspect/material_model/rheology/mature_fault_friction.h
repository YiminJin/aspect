/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/

#ifndef _aspect_material_model_rheology_mature_fault_friction_h
#define _aspect_material_model_rheology_mature_fault_friction_h

#include <utility>

namespace aspect
{
  namespace MaterialModel
  {
    namespace Rheology
    {
      /** Regularized positive-speed rate-and-state law. Mechanics uses the
       * incoming age. The exact constant-speed aging solution is applied at
       * publication. All arguments use SI units. */
      struct MatureFaultFriction
      {
        double reference_coefficient, direct_effect, evolution_effect;
        double reference_speed, slip_distance, damping;

        /** Resistance (Pa) and its derivative (Pa s/m) at fixed age. */
        std::pair<double,double> resistance(double speed, double age, double compression) const;

        /** Age in seconds after sliding at constant speed for dt seconds. */
        double advance_age(double speed, double age, double dt) const;
      };
    }

  }
}
#endif
