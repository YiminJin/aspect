/* Copyright (C) 2026 by the authors of ASPECT. GPL-2.0-or-later. */
#include "common.h"
#include <aspect/reconstructed_fault/endpoint_phase_constraints.h>

TEST_CASE("Endpoint phase native master expansion and raw dual reduction", "[endpoint_phase_constraints]")
{
  using namespace dealii;
  using aspect::internal::EndpointPhaseConstraints;
  AffineConstraints<double> native;
  // A chained relation is resolved by native close, independently of the helper.
  native.add_line(12); native.add_entry(12,10,.5); native.add_entry(12,11,.5);
  native.add_line(14); native.add_entry(14,12,.5); native.add_entry(14,13,.5);
  native.close();
  EndpointPhaseConstraints c(native,10,5);
  REQUIRE(c.size()==3);
  Vector<double> psi(3);psi[0]=.2;psi[1]=.4;psi[2]=.8;
  const auto phi=c.expand(psi);
  REQUIRE(phi[0]==.2);REQUIRE(phi[1]==.4);REQUIRE(phi[2]==Approx(.3));
  REQUIRE(phi[3]==.8);REQUIRE(phi[4]==Approx(.55));
  Vector<double> independently_distributed(15);
  independently_distributed[10]=.2;independently_distributed[11]=.4;independently_distributed[13]=.8;
  native.distribute(independently_distributed);
  for(unsigned int i=0;i<5;++i)REQUIRE(phi[i]==independently_distributed[10+i]);
  // Explicit independently written C, including the closed chained row.
  FullMatrix<double> C(5,3);C(0,0)=1;C(1,1)=1;C(2,0)=C(2,1)=.5;
  C(3,2)=1;C(4,0)=C(4,1)=.25;C(4,2)=.5;
  Vector<double> g(5),expected(3);for(unsigned int i=0;i<5;++i)g[i]=.13*(i+1)*(i%2 ? -1:1);
  C.Tvmult(expected,g);
  const auto dual=c.transpose(g);
  for(unsigned int i=0;i<3;++i)REQUIRE(dual[i]==Approx(expected[i]));
  REQUIRE(psi*dual==Approx(phi*g));
  FullMatrix<double> H(5,5),HC(5,3),reduced(3,3);
  for(unsigned int i=0;i<5;++i)for(unsigned int j=0;j<5;++j)H(i,j)=(i==j?2.:0.)+.03*(i+1)*(j+1);
  H.mmult(HC,C);C.Tmmult(reduced,HC);
  const auto pulled=c.pull_back(H);
  for(unsigned int i=0;i<3;++i)for(unsigned int j=0;j<3;++j)REQUIRE(pulled(i,j)==Approx(reduced(i,j)));
  // Independent nonlinear energy: FD of E(C psi) exercises physical dependent
  // rows, gradient and Hessian together, rather than testing an identity action.
  const auto energy=[&](const Vector<double> &x){Vector<double> p(5);C.vmult(p,x);Vector<double> Hp(5);H.vmult(Hp,p);double e=.5*(p*Hp);for(double a:p)e+=a*a*a*a/4.;return e;};
  Vector<double> Hp(5);H.vmult(Hp,phi);
  FullMatrix<double> nonlinear(H);for(unsigned int i=0;i<5;++i){Hp[i]+=std::pow(phi[i],3);nonlinear(i,i)+=3*phi[i]*phi[i];}
  const auto grad=c.transpose(Hp);const auto hess=c.pull_back(nonlinear);
  const double eps=1e-5;
  for(unsigned int j=0;j<3;++j)
    {
      auto plus=psi,minus=psi;plus[j]+=eps;minus[j]-=eps;
      REQUIRE((energy(plus)-energy(minus))/(2*eps)==Approx(grad[j]).epsilon(1e-8));
      const auto derivative=[&](const Vector<double> &x){Vector<double> p(5),h(5),v(3);C.vmult(p,x);H.vmult(h,p);for(unsigned int i=0;i<5;++i)h[i]+=std::pow(p[i],3);C.Tvmult(v,h);return v;};
      const auto a=derivative(plus),b=derivative(minus);
      for(unsigned int i=0;i<3;++i)REQUIRE((a[i]-b[i])/(2*eps)==Approx(hess(i,j)).epsilon(1e-8));
    }
  auto increment=psi;increment*=.01;const auto physical_increment=c.expand(increment);
  for(double value:physical_increment)REQUIRE(value>=0.);
  REQUIRE(c.select(phi)==psi);
}

TEST_CASE("Endpoint phase identity and admission", "[endpoint_phase_constraints]")
{
  using namespace dealii;using aspect::internal::EndpointPhaseConstraints;
  AffineConstraints<double> empty;empty.close();EndpointPhaseConstraints identity(empty,7,4);
  Vector<double> v(4);for(unsigned int i=0;i<4;++i)v[i]=.1/(i+1);
  REQUIRE(identity.identity());REQUIRE(identity.expand(v)==v);REQUIRE(identity.select(v)==v);REQUIRE(identity.transpose(v)==v);
  deal_II_exceptions::disable_abort_on_exception();
  for(unsigned int kind=0;kind<4;++kind)
    {
      AffineConstraints<double> invalid;invalid.add_line(8);
      if(kind==0){invalid.add_entry(8,7,-.5);invalid.add_entry(8,9,1.5);}
      if(kind==1)invalid.add_entry(8,3,1.);
      if(kind==2){invalid.add_entry(8,7,1.);invalid.set_inhomogeneity(8,.2);}
      if(kind==3)invalid.add_entry(8,7,.5);
      invalid.close();REQUIRE_THROWS(EndpointPhaseConstraints(invalid,7,4));
    }
  deal_II_exceptions::enable_abort_on_exception();
}
