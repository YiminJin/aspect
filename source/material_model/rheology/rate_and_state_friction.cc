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
      FaultTraction
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

      RateStateHistory::RateStateHistory(const RateAndStateFriction &parameters,const double age,
                                         std::vector<double> supplied)
        : law(parameters),initial_age(age),ages(std::move(supplied))
      {
        AssertThrow(initial_age>0. && law.direct_effect>0. && law.reference_speed>0. && law.slip_distance>0.,
                    ExcMessage("RSF requires positive material, time and speed scales."));
      }

      void RateStateHistory::initialize(const unsigned int count)
      {
        if (ages.empty()) ages.assign(count,initial_age);
        AssertThrow(ages.size()==count && std::all_of(ages.begin(),ages.end(),[](double v)
        {
          return v>0.;
        }),
        ExcMessage("Expected one positive RSF age per material identity."));
      }

      FaultTraction RateStateHistory::at_speed(const unsigned int i,const double xi,const double speed,const double pressure) const
      {
        return law.resistance(speed,(1.-xi)*ages[i]+xi*ages[i+1],pressure);
      }

      void RateStateHistory::accept_rate(const unsigned int i,const double rate,const double dt)
      {
        ages[i]=law.advance_age(std::abs(rate),ages[i],dt);
      }

      void RateStateHistory::restore_state(std::vector<double> values)
      {
        AssertThrow(values.size()==ages.size() && std::all_of(values.begin(),values.end(),[](double v)
        {
          return v>0.;
        }),
        ExcMessage("Invalid RSF age history in checkpoint."));
        ages=std::move(values);
      }
    }
  }
}
