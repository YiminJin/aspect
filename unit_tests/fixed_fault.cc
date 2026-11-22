/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/
#include "common.h"
#include <aspect/material_model/rheology/rate_and_state_friction.h>
#include <aspect/reconstructed_fault/stationary_fault.h>

TEST_CASE("Mature fault resistance and aging")
{
  const aspect::MaterialModel::Rheology::RateAndStateFriction law = {.6,.025,.015,.001,.008,1.};
  const double rate = .16359216623205447;
  const double age = 10.;
  const double pressure = 5.;
  const auto result = law.resistance(rate,age,pressure);
  REQUIRE(result.traction == Approx(3.85-.0325));
  const double h = 1.e-6;
  const double difference = (law.resistance(rate+h,age,pressure).traction-law.resistance(rate-h,age,pressure).traction)/(2.*h);
  REQUIRE(result.speed_derivative == Approx(difference).epsilon(1.e-8));
  REQUIRE(law.advance_age(rate,age,.0125) == Approx(7.7554483265394705));
  REQUIRE(law.advance_age(0.,age,.0125) == Approx(age+.0125));
  REQUIRE(law.advance_age(1.e-20,age,.0125) == Approx(age+.0125));
}

TEST_CASE("Stationary fault spans decimal box faces")
{
  using dealii::Point;
  const Point<2> origin(.1,.1), extent(.2,.2);
  std::vector<Point<2>> labels = {{.2,.1},{.2,.2},{.2,.3}};
  aspect::StationaryFault<2> fault;
  fault.initialize(origin,extent,labels,10.,.001);
  CHECK(fault.labels().size() == 3);
  CHECK(fault.project(labels.back()).xi == Approx(1.));
  std::reverse(labels.begin(),labels.end());
  fault.initialize(origin,extent,labels,10.,.001);
  CHECK(fault.project(labels.back()).xi == Approx(1.));
  labels.front()[1] = .299;
  REQUIRE_THROWS(fault.initialize(origin,extent,labels,10.,.001));
}

TEST_CASE("Finite normal association on an inclined stationary trace")
{
  using dealii::Point;
  aspect::StationaryFault<2> fault;
  const double x = .5/std::sqrt(3.);
  fault.initialize(Point<2>(-.5,-.5),Point<2>(1.,1.), {{-x,-.5},{0.,0.},{x,.5}},1.,1.,.2);
  const auto &edge = fault.segment(1);
  CHECK(edge.length+edge.station == Approx(2./std::sqrt(3.)));
  CHECK(edge.normal[0] == Approx(-std::sqrt(3.)/2.));
  CHECK(fault.project(Point<2>(.5*x,.25)).xi == Approx(.5));
  CHECK_FALSE(fault.project(Point<2>(.49,.49)).active);
  CHECK_FALSE(fault.project(Point<2>(-.4,.3)).active);
}

TEST_CASE("Fault strip cuts a cell whose four corners name the same segment")
{
  using dealii::Point;
  aspect::StationaryFault<2> fault;
  fault.initialize(Point<2>(),Point<2>(1.,1.), {{.5,0.},{.5,1.}},1.,1.,.1);
  const auto cut=fault.cut_quadrature(Point<2>(),Point<2>(1.,1.));
  double area=0., admitted=0.;
  for (unsigned int q=0; q<cut.size(); ++q)
    {
      area+=cut.weight(q);
      if (fault.project(cut.point(q)).active) admitted+=cut.weight(q);
    }
  CHECK(area == Approx(1.));
  CHECK(admitted == Approx(.2));
  CHECK(fault.cut_quadrature(Point<2>(.45,.2),Point<2>(.55,.3)).size() == 0);
}
