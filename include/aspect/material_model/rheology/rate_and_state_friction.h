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

        /** Eliminate the signed, stage-aged RSF relation on a material strip. */
        SectionRoot solve_strip(double free_rate,
                                double mobility,
                                double age,
                                double h,
                                double compression) const;

        /** Age in seconds after sliding at constant speed for dt seconds. */
        double advance_age(double speed, double age, double dt) const;
      };

      /** Regularized RSF and its accepted age in seconds, indexed by material
       * identity. The mature response interpolates age before evaluating the law. */
      class RateStateHistory : public FaultResistance
      {
        public:
          RateStateHistory(const RateAndStateFriction &law,
                           double                     initial_age,
                           std::vector<double>        initial_ages = {});
          std::string name() const override
          {
            return "rate and state";
          }
          FaultTraction at_speed(unsigned int i,
                                 double       xi,
                                 double       speed,
                                 double       pressure) const override;
          SectionRoot solve_strip(unsigned int i,
                                  double       free_rate,
                                  double       mobility,
                                  double       h,
                                  double       compression) const override;
          double characteristic_slip_distance() const override
          {
            return law.slip_distance;
          }
          std::optional<SlipResolutionParameters> slip_resolution_parameters(unsigned int /*id*/) const override
          {
            return SlipResolutionParameters{law.direct_effect, law.evolution_effect, law.slip_distance};
          }
          double predict_age(double rate, double age, double dt) const override;
          double state_time_limit(double rate, double age, double log_budget) const override;
          void validate_midpoint() const override;
          bool admits_duration(double h, double pressure) const override;
          std::vector<double> parameters() const override;
          void set_initial_ages(std::vector<double> values) override
          {
            ages=std::move(values);
          }
          const std::vector<double> &ages_in_seconds() const override
          {
            return ages;
          }
          void initialize(unsigned int count) override;
          void accept_rate(unsigned int i, double rate, double dt) override;
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
