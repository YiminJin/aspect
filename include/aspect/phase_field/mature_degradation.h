/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/
#ifndef _aspect_phase_field_mature_degradation_h
#define _aspect_phase_field_mature_degradation_h

namespace aspect
{
  namespace PhaseField
  {
    /** Curvature-one mature energetic degradation. Return the dimensionless
     * excess compliance 1/g(phi)-1 directly, including its limit at zero.
     * The fully damaged limit is infinite. The caller supplies a physical
     * phase in [0,1] and a positive dimensionless magnitude. */
    double excess_compliance(double phase, double magnitude);
  }
}
#endif
