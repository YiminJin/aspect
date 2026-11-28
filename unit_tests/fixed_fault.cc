/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/
#include "common.h"
#include <aspect/adiabatic_conditions/interface.h>
#include <aspect/material_model/phase_field_fault/input_compatibility.h>
#include <aspect/material_model/rheology/friction_normal_stress.h>
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

TEST_CASE("Fault RSF timestep inverts the constant-speed aging law")
{
  using namespace aspect::MaterialModel::Rheology;
  const RateAndStateFriction law = {.6,.025,.015,.001,.008,1.};
  const RateStateHistory history(law,10.);
  const double budget = .1;
  CHECK(std::isinf(history.state_time_limit(.001,8.,budget)));
  CHECK(history.characteristic_slip_distance()/.001 == Approx(8.));
  for (const double rate : {0., 1.e-20, .0001, -.01})
    {
      const double dt = history.state_time_limit(rate,10.,budget);
      const double age = law.advance_age(std::abs(rate),10.,dt);
      CHECK(std::log(age/10.) == Approx(rate < 0. ? -budget : budget));
      CHECK(history.predict_age(rate,10.,dt) == age);
    }
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

TEST_CASE("Prescribed curved fault uses material order and local segment normals")
{
  using dealii::Point;
  aspect::MaterialFaultTrace<2> fault;
  const std::vector<Point<2>> vertices = {{0.,.25}, {.625,.25}, {.375,.75}, {1.,.75}};
  fault.initialize(Point<2>(), Point<2>(1.,1.), vertices, .01, .1);
  for (unsigned int j=0; j<vertices.size(); ++j)
    CHECK(fault.labels()[j].position == vertices[j]);

  // The middle segment is perpendicular to the endpoint chord. Moving
  // away from it along its normal puts the probe exactly on that chord.
  const double root_five = std::sqrt(5.);
  const auto &middle = fault.segment(1);
  CHECK(middle.station == Approx(.625));
  CHECK(middle.length == Approx(root_five/4.));
  CHECK(middle.normal[0] == Approx(-2./root_five));
  CHECK(middle.normal[1] == Approx(-1./root_five));
  CHECK(fault.segment(2).station == Approx(.625+root_five/4.));
  const Point<2> on_chord(.5+2./(16.*root_five), .5+1./(16.*root_five));
  const auto foot = fault.project(on_chord);
  REQUIRE(foot.active);
  CHECK(foot.first_vertex == 1);
  CHECK(foot.xi == Approx(.5));
  CHECK((on_chord-vertices[1])*middle.normal == Approx(-1./16.));

  const auto contact_foot = fault.project(Point<2>(0.,.29));
  REQUIRE(contact_foot.active);
  CHECK(contact_foot.first_vertex == 0);
  CHECK(contact_foot.xi == 0.);
  CHECK(fault.project(vertices[1]).first_vertex == 0);
  // Inside overlapping normal supports the closest perpendicular wins.
  const auto overlap_foot = fault.project(Point<2>(.6,.27));
  REQUIRE(overlap_foot.active);
  CHECK(overlap_foot.first_vertex == 1);
  // On the outer side of this bend neither incident segment admits the
  // point. Proximity to the vertex alone does not give it a slip load.
  CHECK_FALSE(fault.project(Point<2>(.65,.24)).active);
}

TEST_CASE("Prescribed mature fault rejects invalid initial topology and coordinates")
{
  using dealii::Point;
  aspect::MaterialFaultTrace<2> fault;
  const Point<2> origin, extent(1.,1.);
  REQUIRE_THROWS(fault.initialize(origin, extent, {{0.,.2}, {.8,.8}, {.2,.8}, {1.,.2}}, .01));
  REQUIRE_THROWS(fault.initialize(origin, extent, {{0.,.2}, {.5,.5}, {.5,.5}, {1.,.8}}, .01));
  REQUIRE_THROWS(fault.initialize(origin, extent, {{0.,.2}, {.5,1.1}, {1.,.8}}, .01));
  REQUIRE_THROWS(fault.initialize(origin, extent, {{0.,.2}, {.5,std::numeric_limits<double>::quiet_NaN()}, {1.,.8}}, .01));
  // The first incident edge runs along the wall at a noncorner endpoint.
  REQUIRE_THROWS(fault.initialize(origin, extent, {{0.,.2}, {0.,.6}, {1.,.8}}, .01));
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

TEST_CASE("Fault normal stress selection preserves the historical route defaults")
{
  using Pressure = aspect::MaterialModel::Rheology::FrictionNormalStress;
  dealii::ParameterHandler prm;
  Pressure::declare_parameters(prm);
  for (auto legacy :
       {
         Pressure::Source::resolved_traction, Pressure::Source::prescribed_constant
       })
    CHECK_FALSE(Pressure::parse(prm,legacy).uses_adiabat());
  prm.enter_subsection("Material model");
  prm.set("Friction normal stress","resolved normal traction");
  prm.leave_subsection();
  aspect::MaterialModel::FaultInputCompatibility::resolve(prm, true, false);
  REQUIRE_THROWS(Pressure::parse(prm,Pressure::Source::prescribed_constant));
  aspect::AdiabaticConditions::declare_parameters<2>(prm);
  prm.enter_subsection("Material model");
  prm.set("Friction normal stress","adiabatic pressure");
  prm.leave_subsection();
  aspect::MaterialModel::FaultInputCompatibility::resolve(prm, true, false);
  for (auto legacy :
       {
         Pressure::Source::resolved_traction, Pressure::Source::prescribed_constant
       })
    CHECK(Pressure::parse(prm,legacy).uses_adiabat());
}
