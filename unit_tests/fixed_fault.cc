/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/
#include "common.h"
#include <aspect/material_model/rheology/rate_and_state_friction.h>
#include <aspect/reconstructed_fault/material_fault_trace.h>

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
  aspect::MaterialFaultTrace<2> fault;
  fault.initialize(origin,extent,labels,.001);
  CHECK(fault.labels().size() == 3);
  CHECK(fault.project(labels.back()).xi == Approx(1.));
  std::reverse(labels.begin(),labels.end());
  fault.initialize(origin,extent,labels,.001);
  CHECK(fault.project(labels.back()).xi == Approx(1.));
  labels.front()[1] = .299;
  REQUIRE_THROWS(fault.initialize(origin,extent,labels,.001));
}

TEST_CASE("Finite normal association on an inclined stationary trace")
{
  using dealii::Point;
  aspect::MaterialFaultTrace<2> fault;
  const double x = .5/std::sqrt(3.);
  fault.initialize(Point<2>(-.5,-.5),Point<2>(1.,1.), {{-x,-.5},{0.,0.},{x,.5}},1.,.2);
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
  aspect::MaterialFaultTrace<2> fault;
  fault.initialize(Point<2>(),Point<2>(1.,1.), {{.5,0.},{.5,1.}},1.,.1);
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

TEST_CASE("Interior material connectivity through rigid rotation")
{
  using dealii::Point;
  aspect::MaterialFaultTrace<2> fault;
  const Point<2> center;
  fault.initialize(Point<2>(-2,-2),Point<2>(4,4), {{-.5,0.},{.5,0.}},1.,.2,true);
  auto labels=fault.labels();
  labels[1].slip=2.;
  const double angle=35.*dealii::numbers::PI/180.;
  for (unsigned int turn=1; turn<=3; ++turn)
    {
      for (auto &label:labels)
        {
          const auto prior=label.position;
          label.position=Point<2>(std::cos(angle)*prior[0]-std::sin(angle)*prior[1],
                                  std::sin(angle)*prior[0]+std::cos(angle)*prior[1]);
        }
      fault.replace_material_labels(labels);
      const auto &segment=fault.segment(0);
      CHECK(segment.length==Approx(1.));
      CHECK(segment.tangent[0]==Approx(std::cos(turn*angle)));
      CHECK(segment.normal[0]==Approx(-std::sin(turn*angle)));
      CHECK(fault.project(center).xi==Approx(.5));
      CHECK(fault.labels()[1].slip==2.);
    }
  labels[1].position=labels[0].position;
  REQUIRE_THROWS(fault.replace_material_labels(labels));
}

TEST_CASE("A material fault retains simple open connectivity")
{
  using dealii::Point;
  aspect::MaterialFaultTrace<2> fault;
  fault.initialize(Point<2>(-2,-2),Point<2>(4,4), {{-1.,0.},{-.5,0.},{.5,0.},{1.,0.}},1.,.2,true);
  auto vertices=fault.labels();
  vertices[1].position=Point<2>(1.,1.);
  vertices[2].position=Point<2>(-1.,1.);
  REQUIRE_THROWS(fault.replace_material_labels(vertices));
}
