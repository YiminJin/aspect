/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/
#include "common.h"
#include <aspect/material_model/fixed_fault.h>

TEST_CASE("Mature fault resistance and aging")
{
  const aspect::MaterialModel::Rheology::MatureFaultFriction law = {.6,.025,.015,.001,.008,1.};
  const double rate = .16359216623205447;
  const double age = 10.;
  const double pressure = 5.;
  const auto result = law.resistance(rate,age,pressure);
  REQUIRE(result.first == Approx(3.85-.0325));
  const double h = 1.e-6;
  const double difference = (law.resistance(rate+h,age,pressure).first-law.resistance(rate-h,age,pressure).first)/(2.*h);
  REQUIRE(result.second == Approx(difference).epsilon(1.e-8));
  REQUIRE(law.advance_age(rate,age,.0125) == Approx(7.7554483265394705));
  REQUIRE(law.advance_age(0.,age,.0125) == Approx(age+.0125));
  REQUIRE(law.advance_age(1.e-20,age,.0125) == Approx(age+.0125));
}
