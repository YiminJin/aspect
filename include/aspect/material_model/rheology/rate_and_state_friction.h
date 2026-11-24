/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/

#ifndef _aspect_material_model_rheology_rate_and_state_friction_h
#define _aspect_material_model_rheology_rate_and_state_friction_h


#include <aspect/material_model/rheology/fault_resistance.h>

namespace aspect
{
  namespace MaterialModel
  {
    namespace Rheology
    {
      /** Regularized positive-speed rate-and-state law. Mechanics uses the
       * incoming age. The exact constant-speed aging solution is applied at
       * publication. All arguments use SI units. */
      struct RateAndStateFriction
      {
        double reference_coefficient, direct_effect, evolution_effect;
        double reference_speed, slip_distance, damping;

        /** Evaluate the traction and tangent at fixed age and compression. */
        FaultTraction resistance(double speed, double age, double compression) const;

        /** Age in seconds after sliding at constant speed for dt seconds. */
        double advance_age(double speed, double age, double dt) const;
      };

      /** Regularized RSF and its accepted age in seconds, indexed by material
       * identity. The mature response interpolates age before evaluating the law. */
      class RateStateHistory : public FaultResistance
      {
        public:
          RateStateHistory(const RateAndStateFriction &law, double initial_age,
                           std::vector<double> initial_ages = {});
          std::string name() const override
          {
            return "rate and state";
          }
          FaultTraction at_speed(unsigned int i,double xi,double speed,double pressure) const override;
          void initialize(unsigned int count) override;
          void accept_rate(unsigned int i,double rate,double dt) override;
          const std::vector<double> &state() const override
          {
            return ages;
          }
          void restore_state(std::vector<double> values) override;
        private:
          RateAndStateFriction law;
          double initial_age;
          std::vector<double> ages;
      };
    }

  }
}
#endif
