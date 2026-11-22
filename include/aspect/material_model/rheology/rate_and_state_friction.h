/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/

#ifndef _aspect_material_model_rheology_rate_and_state_friction_h
#define _aspect_material_model_rheology_rate_and_state_friction_h


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

        struct Resistance
        {
          double traction;          //!< Shear resistance (Pa), including damping.
          double speed_derivative;  //!< Fixed-age derivative (Pa s/m).
        };

        /** Evaluate the traction and tangent at fixed age and compression. */
        Resistance resistance(double speed, double age, double compression) const;

        /** Age in seconds after sliding at constant speed for dt seconds. */
        double advance_age(double speed, double age, double dt) const;
      };
    }

  }
}
#endif
