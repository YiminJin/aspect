/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/
#include "common.h"
#include <aspect/phase_field/mature_degradation.h>

TEST_CASE("Mature energetic degradation uses the physical phase limits")
{
  using aspect::PhaseField::excess_compliance;
  CHECK(excess_compliance(0.,2.) == 0.);
  CHECK(excess_compliance(1.e-20,2.)/2.e-20 == Approx(1.));
  CHECK(excess_compliance(.5,2.) == Approx(6.));
  CHECK(std::isinf(excess_compliance(1.,2.)));
}
