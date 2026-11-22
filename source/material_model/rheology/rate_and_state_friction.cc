/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/

#include <aspect/material_model/rheology/rate_and_state_friction.h>
#include <aspect/global.h>

namespace aspect
{
  namespace MaterialModel
  {
    namespace Rheology
    {
      RateAndStateFriction::Resistance
      RateAndStateFriction::resistance(const double speed, const double age, const double compression) const
      {
        // Use log(z) to avoid overflow for large reference friction / direct effect.
        AssertThrow(speed > 0. && age > 0. && compression > 0., ExcMessage("Regularized mature friction requires positive speed, age and compression."));
        const double log_z = std::log(speed) - std::log(2.*reference_speed)
                             + (reference_coefficient + evolution_effect * std::log(age*reference_speed/slip_distance))/direct_effect;
        const double small = std::exp(-std::abs(log_z));
        const double asinh_z = log_z > 0 ? log_z + std::log1p(std::hypot(1.,small)) : std::asinh(small);
        const double ratio = (log_z > 0 ? 1. : small)/std::hypot(1.,small);
        return {compression *direct_effect *asinh_z + damping*speed,
                compression *direct_effect *ratio/speed + damping
               };
      }

      double
      RateAndStateFriction::advance_age(const double speed, const double age, const double dt) const
      {
        AssertThrow(speed >= 0. && age > 0. && dt >= 0., ExcMessage("Invalid aging inputs."));
        const double x = speed*dt/slip_distance;
        return age*std::exp(-x) + dt*(x == 0. ? 1. : -std::expm1(-x)/x);
      }
    }
  }
}
