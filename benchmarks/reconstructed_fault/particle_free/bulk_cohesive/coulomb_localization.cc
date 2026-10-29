// Material states and complete algorithmic derivatives; no native law change.
#include <aspect/material_model/phase_field_fault/bulk_cohesive_coulomb.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cmath>
namespace BC=aspect::MaterialModel::BulkCohesive;
namespace CR=BC::CoulombReturn;
namespace TR=BC::TensorReturn;
BC::Tensor strain(double x,double y,double shear)
{BC::Tensor a;a[0][0]=x-(x+y)/3.;a[1][1]=y-(x+y)/3.;a[2][2]=-(x+y)/3.;a[0][1]=shear/2;return a;}
int main(int argc,char **argv)
{
  if(argc!=2)return 2;
  try
    {
      std::ofstream out(argv[1]);out<<std::setprecision(17);
      out<<"kind,normal_sign,mu,phi,h,step,dt,active,oldTxx,oldTxy,oldAxx,oldAxy,oldQxx,oldQxy,Txx,Tyy,Tzz,Txy,Axx,Axy,Qxx,Qxy,tauxx,tauyy,tauzz,tauxy,Wxx,Wxy,gamma_plus,gamma_minus,theta_plus,theta_minus,P,Y,beta";
      for(unsigned int j=0;j<5;++j)for(const char *key:{"xx","yy","zz","xy","Y"})out<<",d"<<j<<'_'<<key;
      out<<'\n';
      const CR::Parameters parameters{{1e6,1e8,4.,1000.},.1,0.};
      double normal_sign=1.;
      auto record=[&](const CR::Parameters &law,const char *kind,const TR::State &old,const CR::Response &r,unsigned int step,double dt)
        {
          const auto &s=r.point.state;const auto &a=r.point;
          out<<kind<<','<<normal_sign<<','<<law.mu<<','<<s.phi<<','<<BC::compliance(s.phi,4.).h<<','<<step<<','<<dt<<','<<r.active;
          for(double v:{old.T[0][0],old.T[0][1],old.A[0][0],old.A[0][1],old.Q[0][0],old.Q[0][1],
                        s.T[0][0],s.T[1][1],s.T[2][2],s.T[0][1],s.A[0][0],s.A[0][1],s.Q[0][0],s.Q[0][1],
                        a.stress[0][0],a.stress[1][1],a.stress[2][2],a.stress[0][1],a.flow_rate[0][0],a.flow_rate[0][1],
                        r.gamma[0],r.gamma[1],r.planes[0].theta,r.planes[1].theta,r.pressure,a.driving,a.beta})out<<','<<v;
          for(unsigned int j=0;j<5;++j)
            {
              BC::Tensor de;if(j==0)de=strain(1,0,0);if(j==1)de=strain(0,1,0);if(j==2)de=strain(0,0,1);
              const auto d=CR::derivative(law,old,r,de,j==4?1.:0.,dt,j==3?1e6:0.).point;
              for(double value:{d.stress[0][0]/1e6,d.stress[1][1]/1e6,d.stress[2][2]/1e6,d.stress[0][1]/1e6,d.driving/8.})out<<','<<value;
            }
          out<<'\n';
        };
      for(double sign:{1.,-1.})
      for(double phi:{0.,.2,.6,.75,.8,.9,.95,.99})
        {
          normal_sign=sign;TR::State old;old.phi=phi;
          for(unsigned int step=1;step<=24;++step)
            {
              const BC::Tensor rate=step<=8?strain(0,0,.02):(step<=20?strain(sign*.02,-sign*.02,0):BC::Tensor());
              const auto r=CR::update(parameters,old,.0125*rate,phi,.0125,10000.);
              record(parameters,"path",old,r,step,.0125);
              if(step==5 || step==8 || step==9 || step==12 || step==16 || step==20)
                for(double dt:{.0125,.00625,.000125,.0000625,1e-8})
                  {
                    const auto probe=CR::update(parameters,r.point.state,dt*rate,phi,dt,10000.);
                    record(parameters,"forward",r.point.state,probe,step,dt);
                  }
              if(step==5 || step==8 || step==9 || step==12 || step==16 || step==20)
                {
                  auto control=parameters;control.mu=0.;control.residual_cohesion=1000.;
                  const auto reference=CR::update(control,r.point.state,.0125*rate,phi,.0125,10000.);
                  record(control,"constant-R",r.point.state,reference,step,.0125);
                }
              old=r.point.state;
            }
        }
      std::cout<<"Mixed-symbol constitutive states exported\n";
    }
  catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
}
