/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/
#include "common.h"
#include <aspect/phase_field/crack_geometry.h>
#include <aspect/phase_field/degradation.h>
#include <deal.II/base/quadrature_lib.h>

TEST_CASE("Mature energetic degradation uses the physical phase limits")
{
  using aspect::PhaseField::excess_compliance;
  CHECK(excess_compliance(0.,2.) == 0.);
  CHECK(excess_compliance(1.e-20,2.)/2.e-20 == Approx(1.));
  CHECK(excess_compliance(.5,2.) == Approx(6.));
  CHECK(std::isinf(excess_compliance(1.,2.)));
}

TEST_CASE("Crack geometry selection, variations and surface normalization")
{
  using aspect::PhaseField::CrackGeometry;
  const std::vector<std::string> names = {"AT1","AT2","CZM"};
  const double phases[] = {0., .25, .75, 1.};
  REQUIRE(CrackGeometry::selection_pattern() == "AT1|AT2|CZM");
  REQUIRE_THROWS(CrackGeometry::select("unknown"));

  for (const auto &name : names)
    {
      CAPTURE(name);
      const auto &law = CrackGeometry::select(name);
      for (const double phase : phases)
        {
          CAPTURE(phase);
          if (name == "AT1")
            {
              CHECK(law.value(phase) == Approx(phase));
              CHECK(law.first_derivative(phase) == Approx(1.));
              CHECK(law.second_derivative(phase) == Approx(0.));
            }
          else if (name == "AT2")
            {
              CHECK(law.value(phase) == Approx(phase*phase));
              CHECK(law.first_derivative(phase) == Approx(2.*phase));
              CHECK(law.second_derivative(phase) == Approx(2.));
            }
          else
            {
              CHECK(law.value(phase) == Approx(1.-(1.-phase)*(1.-phase)));
              CHECK(law.first_derivative(phase) == Approx(2.*(1.-phase)));
              CHECK(law.second_derivative(phase) == Approx(-2.));
            }
        }

      // Integrate the defining c_0 integral independently. Substituting
      // phase = u^2 removes the square-root endpoint singularity.
      const dealii::QGauss<1> rule(24);
      double integral = 0.;
      for (unsigned int q=0; q<rule.size(); ++q)
        {
          const double u = rule.point(q)[0];
          integral += rule.weight(q) * 8.*u*std::sqrt(law.value(u*u));
        }
      CHECK(std::abs(law.normalization()-integral) < 5.e-14);
    }
}

#include <aspect/phase_field/midpoint_fracture.h>

TEST_CASE("Midpoint inequality rows respect the signed stage obstacle")
{
  using aspect::PhaseField::projected_gradient;
  const double lo=-.008, hi=.5*(1.+lo), mass=.03;
  REQUIRE(projected_gradient(lo,2.,mass,lo,hi)==0.);
  REQUIRE(projected_gradient(hi,-2.,mass,lo,hi)==0.);
  REQUIRE(projected_gradient(.2,0.,mass,lo,hi)==0.);
  REQUIRE(projected_gradient(lo,-2.,mass,lo,hi)<0.);
  REQUIRE(projected_gradient(hi,2.,mass,lo,hi)>0.);
  REQUIRE(projected_gradient(.2,.001,mass,lo,hi)==Approx(.001/mass));
}
