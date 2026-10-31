/* Copyright (C) 2026 by the authors of ASPECT. GPL v2 or later. */
#include <aspect/reconstructed_fault/cohesive_timestep.h>
#include "../contrib/catch/catch.hpp"

namespace CE=aspect::CohesiveEvolution;
namespace CI=aspect::CohesiveInterface;

TEST_CASE("Admissible tangent bounds assembled force resolution", "[cohesive_timestep]")
{
  CE::Operators op;op.K.reinit(3,3);op.Q.reinit(3,2);
  op.mass={1.,2.};op.weights={1.,1.,1.};
  op.K(1,1)=6.;op.K(1,2)=1.5;op.K(2,1)=-.5;op.K(2,2)=4.;
  op.Q(0,0)=1.;op.Q(1,0)=.4;op.Q(1,1)=.6;op.Q(2,1)=1.;
  CE::Result r;r.free=2;r.coefficients.reinit(3);r.point.resize(3);
  const double a=6.,d=2.,b=.5/std::sqrt(2.);
  r.curvature=.5*(a+d-std::sqrt((a-d)*(a-d)+4*b*b));
  const auto u=CE::linearized_resolution(op,r);
  for(int i=-10;i<=10;++i)for(int j=-10;j<=10;++j)
    {
      const double f0=CE::weak_tolerance*i/10.,f1=2*CE::weak_tolerance*j/10.;
      const double determinant=24.+.75;
      const double x=(4*f0-1.5*f1)/determinant,y=(.5*f0+6*f1)/determinant;
      for(unsigned q=0;q<3;++q)
        REQUIRE(std::abs(op.Q(q,0)*x+op.Q(q,1)*y)<=u.displacement[q]*(1+1e-14));
    }
  r.free=0;REQUIRE(CE::linearized_resolution(op,r).maximum==0.);
}

TEST_CASE("Rate resolution never uses a damage cutoff", "[cohesive_timestep]")
{
  const double old=5.3429853612048589e-8,now=4.6606914226032277e-7;
  REQUIRE(CE::resolved_log_change(old,now,0.,0.,1e-6)==Approx(.330533398331986).epsilon(1e-12));
  REQUIRE(CE::resolved_log_change(old,now,1e-6,1e-6,1e-6)==0.);
  REQUIRE(CE::resolved_log_change(1e-7,.01,1e-10,1e-10,1e-6)>9.);
  REQUIRE(CE::resolved_log_change(-old,-now,0.,0.,1e-6)==Approx(.330533398331986).epsilon(1e-12));
}

TEST_CASE("Continuous defects resolve admission and signed reversal", "[cohesive_timestep]")
{
  CI::Parameters p;p.Gc=3.;p.peak=2000.;p.Dc=6e-5;p.V0=1e-6;
  CI::PointState old;old.theta=10.;
  double previous=0.;
  for(const double dt:{1e-4,5e-5,2.5e-5})
    {
      const double V=.001,inc=V*dt;auto now=old;now.jump=inc;
      now.damage=inc/(2*p.Gc/p.peak);
      now.theta=(old.theta+p.V0*dt/p.Dc)/(1+V*dt/p.Dc);
      const auto e=CE::history_defect(p,old,now,dt,inc,0.,V,0.,0.);
      REQUIRE(e.slip==Approx(.5*inc/p.Dc));
      REQUIRE(e.damage==Approx(.5*now.damage));
      if(previous)REQUIRE(e.slip==Approx(previous/2));
      previous=e.slip;
    }
  const double dt=1e-4,V=-.01,inc=V*dt;auto now=old;now.damage=1.;old.damage=1.;now.jump=inc;
  now.theta=(old.theta+p.V0*dt/p.Dc)/(1+std::abs(V)*dt/p.Dc);
  REQUIRE(CE::resolved_log_change(.01,V,0.,0.,p.V0)==0.);
  REQUIRE(CE::history_defect(p,old,now,dt,inc,.01,V,0.,0.).slip>1e-3);
  REQUIRE(CE::history_defect(p,old,now,dt,inc,V,V,0.,0.).slip<1e-16);
  REQUIRE(CE::history_defect(p,old,now,dt,inc,V,V,.01,0.).resolution>1.);
}

TEST_CASE("Aging defect agrees with independent exponential local error", "[cohesive_timestep]")
{
  CI::Parameters p;p.Dc=6e-5;p.V0=1e-6;
  CI::PointState old;old.damage=1.;old.theta=2.;const double V=.01;
  for(const double dt:{1e-4,5e-5,2.5e-5})
    {
      auto now=old;now.jump=V*dt;now.theta=(old.theta+p.V0*dt/p.Dc)/(1+V*dt/p.Dc);
      const double exact=p.V0/V+(old.theta-p.V0/V)*std::exp(-V*dt/p.Dc);
      const double error=std::abs(std::log(now.theta/exact));
      const auto e=CE::history_defect(p,old,now,dt,V*dt,V,V,0.,0.);
      REQUIRE(e.state/error==Approx(1.).margin(.02));
    }
}
