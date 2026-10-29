// Benchmark-only C interface to the actual experimental fixed-phase return.
#include "gradient_return.h"
#include <cstring>
#include <exception>
namespace BC=aspect::MaterialModel::BulkCohesive;
namespace GR=BC::GradientReturn;
BC::Tensor grad_tensor(double xx,double xy)
{BC::Tensor t;t[0][0]=xx;t[1][1]=-xx;t[0][1]=xy;return t;}
BC::Tensor grad_strain(double xx,double yy,double shear)
{BC::Tensor t;t[0][0]=xx-(xx+yy)/3;t[1][1]=yy-(xx+yy)/3;t[2][2]=-(xx+yy)/3;t[0][1]=shear/2;return t;}

extern "C" int gradient_projection(const double *input,const double *tau,double pressure,double mu,double cohesion,
                                   double *output,double *jacobian,unsigned int *active,char *error)
{
  try
    {
      BC::Tensor stress;stress[0][0]=tau[0];stress[1][1]=tau[1];stress[2][2]=-tau[0]-tau[1];stress[0][1]=tau[2];
      const auto r=GR::project(grad_tensor(input[0],input[1]),stress,pressure,mu,cohesion);
      output[0]=r.force[0][0];output[1]=r.force[0][1];*active=r.active;
      if(jacobian)
        for(unsigned int c=0;c<6;++c)
          {
            BC::Tensor dz,ds;
            if(c<2)dz=grad_tensor(c==0?1.:0.,c==1?1.:0.);
            if(c==2){ds[0][0]=1.;ds[2][2]=-1.;}
            if(c==3){ds[1][1]=1.;ds[2][2]=-1.;}
            if(c==4)ds[0][1]=1.;
            const auto d=GR::projection_action(r,dz,ds,c==5?1.:0.);
            jacobian[c]=d[0][0];jacobian[6+c]=d[0][1];
          }
      return 0;
    }
  catch(const std::exception &e){std::strncpy(error,e.what(),4095);error[4095]=0;return 1;}
}

// State: Txx,Tyy,Txy,Axx,Axy,Kxx,Kxy,Qxx,Qxy,L,phi.
// Output: state, stress xx/yy/xy, Wxx/Wxy, speed,Y,Eb,Ec,Db,Dc,Df,
// active,theta+/theta-,pn+/pn-,gamma+/gamma-,system_speed.
extern "C" int gradient_point(const double *parameters,const double *state,
                             const double *increment,double pressure,const double *X,
                             double *output,double *jacobian,char *error)
{
  try
    {
      const BC::CoulombReturn::Parameters p{{parameters[0],parameters[1],parameters[2],1000.},parameters[3],parameters[4]};
      const double dt=parameters[5];BC::TensorReturn::State old;
      old.T[0][0]=state[0];old.T[1][1]=state[1];old.T[2][2]=-state[0]-state[1];old.T[0][1]=state[2];
      old.A=grad_tensor(state[3],state[4]);old.K=grad_tensor(state[5],state[6]);old.Q=grad_tensor(state[7],state[8]);old.L=state[9];old.phi=state[10];
      const auto r=GR::update(p,old,grad_strain(increment[0],increment[1],increment[2]),dt,pressure,grad_tensor(X[0],X[1]));
      const auto &a=r.point;const auto &s=a.state;unsigned int i=0;
      for(double v:{s.T[0][0],s.T[1][1],s.T[0][1],s.A[0][0],s.A[0][1],s.K[0][0],s.K[0][1],s.Q[0][0],s.Q[0][1],s.L,s.phi,
                    a.stress[0][0],a.stress[1][1],a.stress[0][1],a.flow_rate[0][0],a.flow_rate[0][1],a.speed,a.driving,a.bulk_energy,a.cohesive_energy,
                    a.bulk_dissipation,a.cohesive_dissipation,a.friction_dissipation,double(r.active),r.planes[0].theta,r.planes[1].theta,
                    r.planes[0].normal,r.planes[1].normal,r.gamma[0],r.gamma[1],r.system_speed})output[i++]=v;
      if(jacobian)
        for(unsigned int c=0;c<6;++c)
          {
            BC::Tensor de,dX;if(c<3)de=grad_strain(c==0?1.:0.,c==1?1.:0.,c==2?1.:0.);
            if(c>=4)dX=grad_tensor(c==4?p.material.G:0.,c==5?p.material.G:0.);
            const auto v=GR::derivative(p,old,r,de,dt,c==3?p.material.G:0.,dX);const auto &d=v.point;unsigned int k=0;
            for(double value:{d.T[0][0],d.T[1][1],d.T[0][1],d.A[0][0],d.A[0][1],d.K[0][0],d.K[0][1],d.Q[0][0],d.Q[0][1],d.L,0.,
                              d.stress[0][0],d.stress[1][1],d.stress[0][1],d.flow_rate[0][0],d.flow_rate[0][1],d.speed,d.driving,d.bulk_energy,d.cohesive_energy,
                              d.bulk_dissipation,d.cohesive_dissipation,d.friction_dissipation,0.,v.theta[0],v.theta[1],v.normal[0],v.normal[1],v.gamma[0],v.gamma[1],v.system_speed})
              {jacobian[6*k+c]=value;++k;}
          }
      return 0;
    }
  catch(const std::exception &e){std::strncpy(error,e.what(),4095);error[4095]=0;return 1;}
}
