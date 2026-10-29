/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/material_model/phase_field_fault/bulk_cohesive_gradient.h>
#include <deal.II/base/quadrature_lib.h>
#include <iostream>
using namespace dealii;
namespace BC=aspect::MaterialModel::BulkCohesive;
namespace SG=BC::SlipGradient;
int main()
{
  std::vector<SG::Space::Sample> samples;const QGauss<2> quad(6);
  for(unsigned int j=0;j<4;++j)for(unsigned int i=0;i<4;++i)
    for(unsigned int q=0;q<quad.size();++q)
      samples.push_back({CellId(i+4*j,{}),q,Point<2>((i+quad.point(q)[0])*.0625,(j+quad.point(q)[1])*.0625),quad.weight(q)*.0625*.0625});
  SG::Space space(samples,Point<2>(),Point<2>(.25,.25));
  std::vector<BC::Tensor> K(space.points.size());
  for(auto &v:K){v[0][0]=.001;v[1][1]=-.001;v[0][1]=.002;}
  const double G=1e6,ls=.006;
  for(const auto &X:space.apply(K,G,ls))AssertThrow(X.norm()==0.,ExcInternalError());
  AssertThrow(space.energy(K,G,ls)==0.,ExcInternalError());
  for(unsigned int i=0;i<K.size();++i)
    {const auto x=space.points[i];K[i][0][0]=.001*x[0];K[i][1][1]=-K[i][0][0];K[i][0][1]=.002*x[1];}
  const double affine=2*G*ls*ls*(1e-6+4e-6)*.25*.25;
  const double affine_error=std::abs(space.energy(K,G,ls)/affine-1.);
  // Complete bilinear xy is represented, including in the extrapolated strips.
  for(unsigned int i=0;i<K.size();++i)
    {const auto x=space.points[i];K[i][0][0]=x[0]*x[1];K[i][1][1]=-K[i][0][0];K[i][0][1]=.2*x[0]-.4*x[1];}
  const double mixed=2*G*ls*ls*(2*std::pow(.25,4)/3.+(.04+.16)*.25*.25);
  const double mixed_error=std::abs(space.energy(K,G,ls)/mixed-1.);
  AssertThrow(affine_error<1e-12 && mixed_error<1e-12,ExcMessage("Independent analytic gradient-energy check failed."));
  auto D=K;for(unsigned int i=0;i<K.size();++i){D[i]*=.001*std::cos(i);K[i]+=D[i];}
  const auto X=space.apply(K,G,ls);long double work=0.;
  for(unsigned int i=0;i<K.size();++i)work+=space.mass[i]*(X[i]*D[i]);
  auto previous=K;for(unsigned int i=0;i<K.size();++i)previous[i]-=D[i];
  const double ledger=work-space.energy(K,G,ls)+space.energy(previous,G,ls)-space.energy(D,G,ls);
  AssertThrow(std::abs(ledger)<1e-12,ExcMessage("Gradient BE work identity failed."));
  BC::Parameters p{1e6,1e8,4.,2000.};
  BC::TensorReturn::State state;state.phi=.8;double identity=0.,fd=0.;
  for(unsigned int step=0;step<24;++step)
    {
      BC::Tensor de;de[0][1]=step<16?.0000625:0.;de[0][0]=step>=16?.000125:0.;de[1][1]=-de[0][0];
      const auto local=BC::CoulombReturn::update({p,.1,0.},state,de,.8,.00625,10000.);
      const auto r=SG::joint_point({p,.1,0.},state,de,local.point.state.K,{},.00625,10000.);
      identity=std::max(identity,r.residual.norm()/r.force_scale);
      if(step==18)
        {
          BC::Tensor dK,dX,dE;dK[0][0]=.0001;dK[1][1]=-.0001;dK[0][1]=-.0002;dX[0][1]=10.;dE[0][1]=.0003;
          const auto a=SG::joint_action({p,.1,0.},r,dE,dK,dX,.00625,300.);
          const double epsilon=1e-5;
          const auto plus=SG::joint_point({p,.1,0.},state,de+epsilon*dE,local.point.state.K+epsilon*dK,epsilon*dX,.00625,10000.+epsilon*300.);
          const auto minus=SG::joint_point({p,.1,0.},state,de-epsilon*dE,local.point.state.K-epsilon*dK,-epsilon*dX,.00625,10000.-epsilon*300.);
          fd=((plus.residual-minus.residual)/(2*epsilon)-a.residual).norm()/std::max(1.,a.residual.norm());
        }
      state=local.point.state;
    }
  AssertThrow(identity<=1e-12 && fd<=1e-6,ExcMessage("Native joint normal-map identity/action failed."));
  std::cout<<std::setprecision(17)<<"{\"affine_energy_error\":"<<affine_error<<",\"xy_energy_error\":"<<mixed_error
           <<",\"BE_energy_defect\":"<<ledger<<",\"homogeneous_force_error\":"<<identity<<",\"joint_FD_error\":"<<fd<<"}\n";
}
