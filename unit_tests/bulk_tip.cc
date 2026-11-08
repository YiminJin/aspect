/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/material_model/phase_field_fault/bulk_tip.h>
#include <aspect/material_model/phase_field_fault/bulk_cohesive_tensor.h>
#include "../contrib/catch/catch.hpp"
#include <cmath>
#include <deal.II/lac/vector.h>

namespace BT=aspect::MaterialModel::BulkCohesive;

TEST_CASE("Bulk tip constrained endpoints, reversal and independent power", "[bulk_tip]")
{
  const BT::Parameters p{1e6,1e8,128.,1000.};
  BT::Tensor S;S[0][1]=.5;
  for(const double old_phi:{0.,.01,.5,1.})
    for(const double phi:{old_phi,(1.+old_phi)/2.,1.})
      for(const double gamma:{-.02,0.,.02})
        {
          BT::Tip::State old;old.phi=old_phi;
          old.stress[0][1]=700.;old.stress[0][0]=300.;old.stress[2][2]=-300.;
          old.cohesive_strain=(old_phi==0.?0.:.002)*S;
          const double dt=.1,f=-200.;const auto increment=gamma*S;
          const auto r=BT::Tip::update(p,old,increment,phi,dt,S,f);
          const double beta=1./(1.+dt*p.G/p.eta);
          REQUIRE((r.state.stress-beta*(old.stress+2*p.G*(increment-dt*r.rate*S))).norm()<1e-10);
          REQUIRE((r.state.cohesive_strain-beta*(old.cohesive_strain+dt*r.rate*S)).norm()<1e-16);
          REQUIRE(std::abs((r.state.stress-r.Q)*S-f)<1e-11);
          REQUIRE(std::abs(BT::Tip::account(p,old,r,increment,dt,f).error)<1e-10);
          if(phi==0.){REQUIRE(r.state.cohesive_strain.norm()==0.);REQUIRE(r.rate==0.);}
          if(phi==1.){REQUIRE(r.Q.norm()==0.);REQUIRE(r.cohesive_energy==0.);REQUIRE(r.driving==0.);}
          REQUIRE(r.bulk_viscosity>=0.);REQUIRE(r.cohesive_viscosity>=0.);
        }
  // No Coulomb excess is imported into the smooth section law's intact limit.
  for(double t:{500.,1500.})
    {
      BT::Tip::State old;old.stress[0][1]=t*(1.+.01*p.G/p.eta);
      const auto r=BT::Tip::update(p,old,{},0.,.01,S,0.);
      REQUIRE(std::abs(r.driving-p.m*t*t/(2*p.G))<1e-12);
      REQUIRE((r.driving>64.)==(t>1000.));
    }
}

TEST_CASE("Bulk tip consistent phase, strain and resistance tangent", "[bulk_tip]")
{
  const BT::Parameters p{1e6,1e8,128.,1000.};BT::Tensor S;S[0][1]=.5;
  BT::Tip::State old;old.phi=0.;old.stress[0][1]=1500.;
  BT::Tensor increment;increment[0][1]=.001;increment[0][0]=.0002;increment[2][2]=-.0002;
  for(double phi:{0.,.03,.8,1.})for(unsigned int mode=0;mode<3;++mode)
    {
      const double dphi=mode==0?(phi==1.?-1.:1.):0.,df=mode==1?200.:0.;
      BT::Tensor de;if(mode==2)de=increment;
      const auto r=BT::Tip::update(p,old,increment,phi,.1,S,100.);
      const auto a=BT::Tip::derivative(p,r,de,dphi,.1,S,df);
      const double eps=1e-7;
      const auto trial=BT::Tip::update(p,old,increment+eps*de,phi+eps*dphi,.1,S,100.+eps*df);
      REQUIRE(((trial.state.stress-r.state.stress)/eps-a.stress).norm()<.02*std::max(1.,a.stress.norm()));
      REQUIRE(std::abs((trial.driving-r.driving)/eps-a.driving)<.002*std::max(1.,std::abs(a.driving)));
      REQUIRE(std::abs((trial.rate-r.rate)/eps-a.rate)<.002*std::max(1.,std::abs(a.rate)));
    }
}

TEST_CASE("Bulk tip direct storage equals retained tensor return on sliding branch", "[bulk_tip]")
{
  const BT::Parameters p{1e6,1e8,128.,1000.};BT::Tensor S;S[0][1]=.5;
  BT::TensorReturn::State old;old.phi=.02;
  for(unsigned int step=0;step<12;++step)
    {
      const auto de=.002*S;const double dt=.05,phi=.02+.002*step;
      const auto retained=BT::TensorReturn::update(p,old,de,phi,dt,&S);
      REQUIRE(retained.sliding);
      BT::Tip::State physical{old.T-2.*p.G*old.A,old.A,old.phi};
      const auto direct=BT::Tip::update(p,physical,de,phi,dt,S,p.resistance);
      REQUIRE((direct.state.stress-retained.stress).norm()<1e-10);
      REQUIRE((direct.state.cohesive_strain-retained.state.A).norm()<1e-14);
      REQUIRE(std::abs(direct.cohesive_energy-retained.cohesive_energy)<1e-10);
      REQUIRE(std::abs(direct.driving-retained.driving)<1e-10);
      old=retained.state;
    }
}

TEST_CASE("Bulk section signed power and transverse variance cannot be discarded", "[bulk_tip]")
{
  const double G=1e6,f=300.;const std::vector<double> dn{.1,.3,.2},h{2.,5.,1.},q{100.,300.,-200.},w{.02,-.003,.001};
  double V=0.,absolute=0.,power=0.,energy=0.,H=0.,elastic_slip=0.;
  for(unsigned int i=0;i<dn.size();++i)
    {V+=dn[i]*w[i];absolute+=dn[i]*std::abs(w[i]);power+=dn[i]*f*w[i];H+=dn[i]*h[i];elastic_slip+=dn[i]*h[i]*q[i]/G;energy+=dn[i]*h[i]*q[i]*q[i]/(2*G);}
  REQUIRE(std::abs(power-f*V)<1e-15);REQUIRE(absolute>std::abs(V));
  REQUIRE(energy>G*elastic_slip*elastic_slip/(2*H));
  double variance=0.;const double mean=G*elastic_slip/H;
  for(unsigned int i=0;i<dn.size();++i)variance+=dn[i]*h[i]*(q[i]-mean)*(q[i]-mean)/(2*G);
  REQUIRE(std::abs(energy-G*elastic_slip*elastic_slip/(2*H)-variance)<1e-16);
}

TEST_CASE("Bulk tip matched relaxation without fault slip", "[bulk_tip]")
{
  const BT::Parameters p{1e6,1e6,80.,0.};BT::Tensor S;S[0][1]=.5;
  for(double phi:{0.,.2,1.})
    {
      BT::Tip::State old;old.phi=phi;
      old.cohesive_strain=(phi==0.?0.:.002)*S;
      const double inverse=phi>0.?std::pow(1.-phi,2)/(p.m*phi*(1.+phi)):0.;
      old.stress=2.*p.G*inverse*old.cohesive_strain;
      old.stress[0][0]=300.;old.stress[2][2]=-300.;
      const double dt=.2,beta=1./(1.+dt*p.G/p.eta);
      const auto r=BT::Tip::update(p,old,{},phi,dt,S,0.);
      REQUIRE((r.state.stress-beta*old.stress).norm()<1e-11);
      REQUIRE((r.state.cohesive_strain-beta*old.cohesive_strain).norm()<1e-15);
      REQUIRE(std::abs(r.rate)<1e-15);
      REQUIRE(std::abs(BT::Tip::account(p,old,r,{},dt,0.).error)<1e-12);
    }
}

#include <aspect/phase_field/box_constraints.h>
#include <deal.II/lac/full_matrix.h>

TEST_CASE("Coupled box iteration releases incorrect lower and upper bounds", "[bulk_tip][bounds]")
{
  // Independently prescribed constrained minimizer of a coupled positive
  // quadratic. Free mechanics changes the sign of the initially wrong phase
  // multiplier. Mirroring covers release of both lower and upper constraints.
  for (const double sense : {1.,-1.})
    {
      dealii::FullMatrix<double> A(3,3);
      A(0,0)=2.; A(0,1)=A(1,0)=-1.; A(0,2)=A(2,0)=.5;
      A(1,1)=2.; A(2,2)=1.;
      dealii::Vector<double> exact(3),rhs(3),x(3),g(3);
      exact[0]=.2;exact[1]=.4;exact[2]=1.;A.vmult(rhs,exact);rhs[2]+=.3;
      x[0]=-5.;
      if(sense<0.)
        {
          dealii::Vector<double> shift(3),action(3);shift[1]=shift[2]=1.;A.vmult(action,shift);
          rhs*=-1.;rhs+=action;exact*=-1.;exact+=shift;x*=-1.;x+=shift;
        }
      std::array<int,3> previous{{0,0,0}};
      unsigned int releases=0;bool converged=false;
      for(unsigned int iteration=0;iteration<10;++iteration)
        {
          A.vmult(g,x);g-=rhs;std::array<int,3> active{{0,0,0}};
          for(unsigned int i=1;i<3;++i)
            {
              active[i]=aspect::PhaseField::box_constraint(x[i],g[i],A(i,i),0.,1.).status;
              releases+=previous[i]!=0 && previous[i]!=active[i];
            }
          if(iteration>0 && previous==active){converged=true;break;}
          auto restricted=A;auto load=rhs;
          for(unsigned int i=0;i<3;++i)
            if(active[i])
              {for(unsigned int j=0;j<3;++j)restricted(i,j)=0.;restricted(i,i)=1.;load[i]=active[i]>0?1.:0.;}
            else for(unsigned int j=1;j<3;++j)if(active[j])
              {load[i]-=A(i,j)*(active[j]>0?1.:0.);restricted(i,j)=0.;}
          restricted.gauss_jordan();restricted.vmult(x,load);previous=active;
        }
      REQUIRE(converged);REQUIRE(releases>0);
      x-=exact;REQUIRE(x.linfty_norm()<1e-14);
      REQUIRE(sense*g[2]<0.); // Correct signed multiplier on the retained bound.
    }
}

TEST_CASE("Bound displacement retains a tiny accepted obstacle", "[bulk_tip][bounds]")
{
  const double lower=1e-30,x=.01,increment=lower-x;
  REQUIRE(x+increment==0.); // The naive expression loses the physical obstacle.
  REQUIRE(aspect::PhaseField::box_trial_value(x,increment,lower,1.,1.)==lower);
  for(const double alpha:{.25,.5,1.})
    {
      const double trial=aspect::PhaseField::box_trial_value(x,increment,lower,1.,alpha);
      REQUIRE(trial>=lower);REQUIRE(trial<=x);
    }
  REQUIRE(aspect::PhaseField::box_trial_value(x,-.02,lower,1.,1.)<lower);
}
