/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/material_model/phase_field_fault/bulk_cohesive.h>
#include "../contrib/catch/catch.hpp"
#include <cmath>

namespace BC=aspect::MaterialModel::BulkCohesive;

TEST_CASE("Bulk cohesive intact onset and closed limit", "[bulk_cohesive]")
{
  const BC::Parameters p{1e6,1e8,128.,1000.};
  BC::State incoming;BC::Tensor strain;
  const double dt=.01,beta=1./(1.+dt*p.G/p.eta);
  for(const double t0:{-3000.,-2000.,-1500.,0.,500.,1500.,2000.,3000.})
    {
      strain[0][1]=t0/(2*p.G*beta);
      const auto r=BC::update(p,incoming,strain,0.,dt);
      REQUIRE(r.state.a==0.);
      REQUIRE(r.w==0.);
      REQUIRE(r.state.q==std::copysign(std::max(0.,std::abs(t0)-1000.),t0));
      REQUIRE(std::abs(r.driving-128.*std::pow(std::max(0.,std::abs(t0)-1000.),2)/(2e6))<1e-12);
      if(std::abs(t0)==2000.)REQUIRE(r.driving==64.);
      for(const double phi:{1e-20,1e-12,1e-7,.01,.6})
        {
          const auto damaged=BC::update(p,incoming,strain,phi,dt);
          const double h=BC::compliance(phi,p.m).h;
          REQUIRE(std::abs(damaged.state.q-std::copysign(std::max(0.,std::abs(t0)-1000.)/(1+h),t0))<1e-10);
          REQUIRE(std::abs(damaged.state.a-h*damaged.state.q/p.G)<1e-16);
          REQUIRE(std::abs(BC::account(p,incoming,damaged,strain,dt).error)<1e-12);
        }
    }
  REQUIRE_THROWS(BC::update(p,incoming,strain,1.,dt));
}

TEST_CASE("Bulk cohesive history dependent returns and analytic tangents", "[bulk_cohesive]")
{
  const BC::Parameters p{1e6,1e8,128.,1000.};
  for(const double phi:{.005,.1,.6})
    {
      BC::State incoming;incoming.phi=phi;
      const auto h=BC::compliance(phi,p.m);
      incoming.a=h.h*7./p.G;incoming.q=7.;
      incoming.T[0][0]=210.;incoming.T[1][1]=-80.;incoming.T[2][2]=-130.;
      incoming.T[0][1]=p.G*incoming.a+700.;
      for(const double drive:{-8000.,0.,8000.})
        {
          BC::Tensor strain;strain[0][1]=drive/(2*p.G);
          const double dt=.1;
          const auto r=BC::update(p,incoming,strain,phi,dt);
          REQUIRE(std::abs(BC::account(p,incoming,r,strain,dt).error)<1e-11);
          const double yield=r.stress[0][1]-r.state.q;
          REQUIRE(std::abs(yield)<=1000.+1e-10);
          REQUIRE(std::abs(yield*r.w-1000.*std::abs(r.w))<1e-9);
          if(!r.sliding){REQUIRE(r.w==0.);REQUIRE(r.state.a==r.beta*incoming.a);}
          const double eps=1e-6;
          const auto rp=BC::update(p,incoming,strain,phi+eps,dt);
          const double da=(rp.state.a-r.state.a)/eps;
          REQUIRE(std::abs(da-r.da_dphi)/std::max(1e-5,std::abs(r.da_dphi))<.003);
          // Central phase differences use an incoming lower obstacle below
          // the test point; the physical history remains exactly the same.
          auto anchor=incoming;anchor.phi=phi*.5;
          const auto rm=BC::update(p,anchor,strain,phi-eps,dt);
          const double dq=(rp.state.q-rm.state.q)/(2*eps);
          REQUIRE(std::abs(dq-r.dq_dphi)/std::max(1.,std::abs(r.dq_dphi))<1e-7);
          auto ip=incoming,im=incoming;ip.T[0][1]+=.001/r.beta;im.T[0][1]-=.001/r.beta;
          const auto tp=BC::update(p,ip,strain,phi,dt),tm=BC::update(p,im,strain,phi,dt);
          REQUIRE(std::abs((tp.state.q-tm.state.q)/.002-r.dq_dt)<1e-7);
        }
    }
}

TEST_CASE("Bulk cohesive finite viscosity and cyclic signed histories", "[bulk_cohesive]")
{
  const BC::Parameters p{1e6,1e8,128.,1000.};
  BC::State state;state.phi=.1;BC::Tensor strain;
  strain[0][1]=.005;
  auto r=BC::update(p,state,strain,.1,.01);
  REQUIRE(r.w>0.);state=r.state;
  strain=0.;
  r=BC::update(p,state,strain,.1,1.);
  REQUIRE(r.w==0.);
  REQUIRE(std::abs(r.state.a-state.a/1.01)<1e-16);
  REQUIRE(r.state.signed_shear==state.signed_shear);
  state=r.state;strain[0][1]=-.01;
  r=BC::update(p,state,strain,.1,.01);
  REQUIRE(r.w<0.);
  REQUIRE(r.state.absolute_shear>std::abs(r.state.signed_shear));
  REQUIRE_THROWS(BC::update(p,r.state,strain,.09,.01));
}

TEST_CASE("Bulk cohesive observer retains profile residual energy and friction", "[bulk_cohesive]")
{
  const BC::Parameters p{1e6,1e8,128.,1000.};
  const std::vector<double> weights{.02,.08,.13,.2,.2,.13,.08,.02};
  std::vector<BC::Response> samples(weights.size());
  for(const bool nonuniform:{false,true})
    {
      for(unsigned int i=0;i<samples.size();++i)
        {
          auto &r=samples[i];r.state.phi=i==0 ? 0. : .05+.05*i;
          const double h=BC::compliance(r.state.phi,p.m).h;
          r.state.q=nonuniform ? 40.+13.*i : 100.;
          r.state.a=h*r.state.q/p.G;
          r.cohesive_energy=h*r.state.q*r.state.q/(2*p.G);
          r.cohesive_dissipation=h*r.state.q*r.state.q/p.eta;
          r.w=(nonuniform && i%2 ? -1. : 1.)*h*1e-5;
        }
      const auto m=BC::observe_profile(p,weights,samples);
      REQUIRE(m.normalized);
      REQUIRE(std::abs(m.residual_moment)<1e-16);
      REQUIRE(std::abs(m.bulk_cohesive-m.surface_cohesive-m.residual_cohesive)<1e-14);
      REQUIRE(std::abs(m.bulk_viscous-m.surface_viscous-m.residual_viscous)<1e-14);
      REQUIRE(m.friction_excess>=0.);
      if(nonuniform){REQUIRE(m.residual_cohesive>1e-5);REQUIRE(m.friction_excess>0.);}
      else {REQUIRE(m.residual_cohesive<1e-28);REQUIRE(m.friction_excess==0.);}
    }
}

TEST_CASE("Bulk cohesive continuous power and partial phase force", "[bulk_cohesive]")
{
  const BC::Parameters p{1e6,1e8,128.,1000.};
  const double phi=.2,q=100.,phase_rate=.02;
  const auto h=BC::compliance(phi,p.m);
  const double a=h.h*q/p.G;
  BC::Tensor S;S[0][1]=.5;
  for(const double w:{-.01,.01})
    {
      BC::Tensor T,D;
      T[0][0]=210.;T[1][1]=-80.;T[2][2]=-130.;
      T[0][1]=p.G*a+q+std::copysign(p.resistance,w);
      D[0][0]=.01;D[1][1]=-.003;D[2][2]=-.007;D[0][1]=.02;
      const auto stress=T-2*p.G*a*S;
      const double adot=w-p.G/p.eta*a;
      const auto stress_dot=2*p.G*D-p.G/p.eta*T-2*p.G*adot*S;
      const double Y=h.first*q*q/(2*p.G);
      const double stored_rate=(stress*stress_dot)/(2*p.G)+q*adot-Y*phase_rate;
      const double dissipation=(stress*stress)/(2*p.eta)+h.h*q*q/p.eta+p.resistance*std::abs(w);
      REQUIRE(std::abs(stored_rate-(stress*D)+dissipation+Y*phase_rate)<1e-12);
      const double epsilon=1e-7;
      const auto energy=[&](double x){return (stress*stress)/(4*p.G)+p.G*a*a/(2*BC::compliance(x,p.m).h);};
      REQUIRE(std::abs((energy(phi+epsilon)-energy(phi-epsilon))/(2*epsilon)+Y)/std::max(1.,Y)<1e-7);
    }
}
