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

TEST_CASE("Midpoint stage elimination satisfies the continuous discrete equations", "[bulk_tip][midpoint]")
{
  const BT::Parameters p{1e6,1e8,128.,0.};BT::Tensor S;S[0][1]=.5;
  for(double pn:{0.,.02,.4,1.})for(double p1:{pn,.5*(1.+pn),1.})for(double shear:{-.02,0.,.02})
    {
      BT::Tip::State old;old.phi=pn;old.stress[0][1]=700.;old.stress[0][0]=300.;old.stress[2][2]=-300.;
      old.cohesive_strain=(pn==0.?0.:.002)*S;
      const double dt=.1,f=-200.;const auto increment=shear*S;
      const auto m=BT::Tip::update(p,old,.5*increment,.5*(pn+p1),.5*dt,S,f);
      const auto end=BT::Tip::midpoint_endpoint(old,m);
      REQUIRE(((end.stress-old.stress)/(2.*p.G*dt)+m.state.stress/(2.*p.eta)-increment/dt+m.rate*S).norm()<1e-16);
      REQUIRE(((end.cohesive_strain-old.cohesive_strain)/dt+(p.G/p.eta)*m.state.cohesive_strain-m.rate*S).norm()<1e-16);
      REQUIRE(std::abs((m.state.stress-m.Q)*S-f)<1e-10);
      REQUIRE(std::abs(end.phi-p1)<2e-16);
      const auto ledger=BT::Tip::midpoint_account(p,old,m,increment,dt,f);
      REQUIRE(std::abs(ledger.error)<2e-10);
      REQUIRE(ledger.stress_loss==0.);REQUIRE(ledger.traction_loss==0.);
      REQUIRE(m.bulk_viscosity>=0.);REQUIRE(m.cohesive_viscosity>=0.);
      const double C=m.Q*(end.cohesive_strain-old.cohesive_strain)-m.driving*(p1-pn)
                     -(BT::Tip::cohesive_energy(p,end)-BT::Tip::cohesive_energy(p,old));
      REQUIRE(std::abs(ledger.compliance_work-m.driving*(p1-pn)-C)<2e-10);
    }
}

TEST_CASE("Midpoint analytic Maxwell forcing, relaxation and stiff sign reversal", "[bulk_tip][midpoint]")
{
  const BT::Parameters p{1e6,1e6,80.,0.};BT::Tensor S;S[0][1]=.5;
  for(double sign:{-1.,1.})for(double force:{0.,300.})
    {
      double previous=0.;
      for(unsigned int n:{10u,20u,40u})
        {
          BT::Tip::State state;state.stress[0][1]=sign*700.;
          const double dt=1./n;
          for(unsigned int i=0;i<n;++i)
            {
              // Intact material: prescribed smooth shear strain rate cos(t).
              const auto increment=sign*force/p.G*std::cos((i+.5)*dt)*dt*S;
              const auto stage=BT::Tip::update(p,state,.5*increment,0.,.5*dt,S,0.);
              state=BT::Tip::midpoint_endpoint(state,stage);
            }
          const double exact=sign*(700.*std::exp(-1.)+.5*force*(std::cos(1.)+std::sin(1.)-std::exp(-1.)));
          const double error=std::abs(state.stress[0][1]-exact);
          if(previous>0.) REQUIRE(previous/error>3.9);
          previous=error;
        }
    }
  BT::Tip::State old;old.stress[0][0]=300.;old.stress[2][2]=-300.;
  for(double z:{.2,2.,4.,20.})
    {
      const auto m=BT::Tip::update(p,old,{},0.,.5*z,S,0.);
      const auto end=BT::Tip::midpoint_endpoint(old,m);
      REQUIRE(std::abs(end.stress[0][0]-300.*(1.-z/2.)/(1.+z/2.))<1e-12);
      if(z>2.) REQUIRE(end.stress[0][0]<0.);
    }
}

TEST_CASE("Midpoint nonlinear cohesive chain defect converges without hiding work", "[bulk_tip][midpoint]")
{
  const BT::Parameters p{1e6,1e7,128.,0.};BT::Tensor S;S[0][1]=.5;
  // Independently prescribed A(t), phi(t). This is a chain-rule check, not a VI.
  auto phase=[](double t){return .1+.1*t;};
  auto strain=[&](double t){return (.001+.0003*std::sin(t))*S;};
  double previous=0.;
  for(unsigned int n:{8u,16u,32u})
    {
      double sum=0.,absolute=0.;
      for(unsigned int i=0;i<n;++i)
        {
          const double t=double(i)/n,t1=double(i+1)/n,phi=.5*(phase(t)+phase(t1));
          const auto A=.5*(strain(t)+strain(t1));
          const double c=p.m*phi*(1.+phi),d=(1.-phi)*(1.-phi);
          const auto Q=2.*p.G*d/c*A;
          const double Y=p.G*(A*A)*p.m*(1.+3.*phi)*(1.-phi)/(c*c);
          const double defect=Q*(strain(t1)-strain(t))-Y*(phase(t1)-phase(t))
            -(BT::Tip::cohesive_energy(p,{{},strain(t1),phase(t1)})-BT::Tip::cohesive_energy(p,{{},strain(t),phase(t)}));
          sum+=defect;absolute+=std::abs(defect);
        }
      REQUIRE(absolute>=std::abs(sum));REQUIRE(absolute>0.);
      if(previous>0.) REQUIRE(previous/absolute>3.9);
      previous=absolute;
    }
}

TEST_CASE("Midpoint physical state converges for prescribed smooth cohesive phase", "[bulk_tip][midpoint]")
{
  const BT::Parameters p{1e6,1e7,128.,0.};BT::Tensor S;S[0][1]=.5;
  auto phi=[](double t){return .05+.02*t;};
  auto fraction=[&](double v){const double c=p.m*v*(1.+v);return c/(c+(1.-v)*(1.-v));};
  for(double sign:{-1.,1.})
    {
      double previous=0.;
      for(unsigned int n:{10u,20u,40u})
        {
          // Z=tau+2GA obeys a scalar forced Maxwell ODE, independent of phi.
          const double Z0=sign*1200.,f=sign*200.,forcing=sign*300.,lambda=p.G/p.eta,dt=1./n;
          BT::Tip::State state;state.phi=phi(0.);
          state.cohesive_strain=fraction(state.phi)*(Z0-f)/p.G*S;
          state.stress=2.*Z0*S-2.*p.G*state.cohesive_strain;
          for(unsigned int i=0;i<n;++i)
            {
              const auto increment=forcing/p.G*dt*S;
              const double stage_phase=.5*(phi(double(i)/n)+phi(double(i+1)/n));
              const auto m=BT::Tip::update(p,state,.5*increment,stage_phase,.5*dt,S,f);
              state=BT::Tip::midpoint_endpoint(state,m);
            }
          const double Z=Z0*std::exp(-lambda)+forcing/lambda*(1.-std::exp(-lambda));
          const auto A=fraction(phi(1.))*(Z-f)/p.G*S;
          const auto tau=2.*Z*S-2.*p.G*A;
          const double error=(state.stress-tau).norm()+p.G*(state.cohesive_strain-A).norm();
          if(previous>0.) REQUIRE(previous/error>3.8);
          previous=error;
        }
    }
}
