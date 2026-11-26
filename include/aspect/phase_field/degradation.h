/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/
#ifndef _aspect_phase_field_degradation_h
#define _aspect_phase_field_degradation_h

namespace aspect
{
  namespace PhaseField
  {
    /** Numerator and denominator of the excess compliance, evaluated before
     * division. Their bounded ratio is regular even at complete damage. */
    struct CohesiveDegradation
    {
        CohesiveDegradation(double phase, double magnitude);
        double excess_numerator; ///< m phi (1 + phi)
        double intact_factor;    ///< (1 - phi)^2
        /** Scale G A:A to the cohesive energy at a damaged point (Pa). */
        double cohesive_energy(double strain_energy) const;
        /** Mobile fraction of the cohesive-Maxwell response, in [0,1]. */
        double fraction() const;
        double fraction_derivative() const;
        double driving_factor() const;
        double driving_factor_derivative() const;
      private:
        double phase, magnitude;
    };

    /** Curvature-one mature energetic degradation. Return the dimensionless
     * excess compliance 1/g(phi)-1 directly, including its limit at zero.
     * The fully damaged limit is infinite. The caller supplies a physical
     * phase in [0,1] and a positive dimensionless magnitude. */
    double excess_compliance(double phase, double magnitude);
  }
}
#endif
