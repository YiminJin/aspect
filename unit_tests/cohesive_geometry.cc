/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/reconstructed_fault/cohesive_geometry.h>
#include "../contrib/catch/catch.hpp"

namespace CI=aspect::CohesiveInterface;

TEST_CASE("Cohesive complete sampling disk and vertical lifting clearance", "[cohesive_geometry]")
{
  CI::Geometry::Piece p;p.x=3.;p.end=6.;p.cells=8;
  CI::Geometry g({p},.75);
  const CI::Geometry::Bounds domain{dealii::Point<2>(0,-6),dealii::Point<2>(12,6)};
  const auto c=g.clearance(domain,1,.75);
  REQUIRE(std::abs(c.sampling-4.5)<1e-12);
  REQUIRE(std::abs(c.lifting-3.)<1e-12);
  p.x=8.9;p.y=2.2;p.end=10.9;p.a=p.b=1.;
  CI::Geometry turn({p},.75);
  const auto disk=turn.sampling_bounds(1,.75);
  REQUIRE(p.end+.75<12.); // Tip-plus-radius would incorrectly pass.
  REQUIRE(disk.upper[0]>12.);
  REQUIRE(turn.clearance(domain,1,.75).sampling<0.);
  REQUIRE(turn.clearance(domain,1,.75).lifting>0.);
}

TEST_CASE("Cohesive lifting protects polynomial extrema between Gauss points", "[cohesive_geometry]")
{
  CI::Geometry::Piece p;p.x=3.;p.y=5.08;p.end=5.;p.a=.6;p.b=-1.4;
  p.hermite=true;p.c1=-2.;p.cells=1;
  const CI::Geometry::Bounds domain{dealii::Point<2>(0,-6),dealii::Point<2>(12,6)};
  CI::Geometry g({p},.75);
  for(const auto &q:g.surface_points())REQUIRE(q.position[1]+.75<6.);
  REQUIRE(g.sample(3.+.3*2).y+.75>6.);
  const auto c=g.clearance(domain,1,.75);
  REQUIRE(c.sampling>0.);
  REQUIRE(c.lifting<0.);
  REQUIRE(c.lifting_wall==3);
}

TEST_CASE("Cohesive interval envelope contains complete curved support", "[cohesive_geometry]")
{
  CI::Geometry::Piece p;p.x=3.;p.end=6.;p.cells=8;
  CI::Geometry g({p},.75);
  for(const double direction:{.7,-.4,.2})g.append(direction,.3,.75);
  const auto b=g.lifting_bounds();
  for(const auto &piece:g.pieces())
    for(unsigned k=0;k<=2000;++k)
      {
        const double x=piece.x+(piece.end-piece.x)*k/2000.;
        const double y=g.sample(x).y;
        REQUIRE(y-.75>=b.lower[1]);REQUIRE(y+.75<=b.upper[1]);
      }
  // A temporary tail must be protected even while the formed tip is safe.
  const auto c=g.clearance({dealii::Point<2>(0,-6),dealii::Point<2>(6.8,6)},1,.2);
  REQUIRE(c.sampling>0.);REQUIRE(c.lifting<0.);
}
