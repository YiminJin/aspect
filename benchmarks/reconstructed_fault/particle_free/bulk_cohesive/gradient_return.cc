/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include "gradient_return.h"
#include <deal.II/base/exceptions.h>
#include <algorithm>
#include <cmath>
#include <limits>

namespace aspect { namespace MaterialModel { namespace BulkCohesive { namespace GradientReturn
{
  namespace
  {
    using V=std::array<double,2>;
    V add(V a,V b){return {{a[0]+b[0],a[1]+b[1]}};}
    V sub(V a,V b){return {{a[0]-b[0],a[1]-b[1]}};}
    V mul(double s,V a){return {{s*a[0],s*a[1]}};}
    double dot(V a,V b){return a[0]*b[0]+a[1]*b[1];}
    double cross(V a,V b){return a[0]*b[1]-a[1]*b[0];}
    V J(V a){return {{-a[1],a[0]}};}
    double norm(V a){return std::hypot(a[0],a[1]);}
    V vec(const Tensor &a){return {{.5*(a[0][0]-a[1][1]),a[0][1]}};}
    Tensor tensor(V a){Tensor t;t[0][0]=a[0];t[1][1]=-a[0];t[0][1]=a[1];return t;}
    V z(V t,V q,V X,double h,double mu,int k)
    {return sub(mul(-k,J(sub(sub(t,X),mul(1+h,q)))),mul(mu,sub(t,mul(h,q))));}

    std::array<double,4> solve(double a[4][4],std::array<double,4> b,unsigned int n)
    {
      for(unsigned int i=0;i<n;++i)
        {
          unsigned int pivot=i;
          for(unsigned int j=i+1;j<n;++j)if(std::abs(a[j][i])>std::abs(a[pivot][i]))pivot=j;
          AssertThrow(std::isfinite(a[pivot][i]) && a[pivot][i]!=0.,
                      dealii::ExcMessage("Singular gradient Coulomb active derivative; no regularization of its Jacobian."));
          for(unsigned int j=0;j<n;++j)std::swap(a[i][j],a[pivot][j]);
          std::swap(b[i],b[pivot]);
          const double s=a[i][i];for(unsigned int j=i;j<n;++j)a[i][j]/=s;b[i]/=s;
          for(unsigned int j=0;j<n;++j)if(j!=i)
            {const double s=a[j][i];for(unsigned int k=i;k<n;++k)a[j][k]-=s*a[i][k];b[j]-=s*b[i];}
        }
      return b;
    }
  }

  Response update(const CoulombReturn::Parameters &p,const TensorReturn::State &old,
                  const Tensor &de,double dt,double pressure,const Tensor &X)
  {
    AssertThrow((X-TensorReturn::in_plane(X)).norm()<=1e-12*std::max(1.,X.norm()),
                dealii::ExcMessage("Gradient backstress must belong to the retained in-plane trace-free space."));
    const double G=p.material.G,h=compliance(old.phi,p.material.m).h,beta=1./(1.+dt*G/p.material.eta);
    Response result;result.backstress=X;
    if(X.norm()==0.)
      {
        static_cast<CoulombReturn::Response&>(result)=CoulombReturn::update(p,old,de,old.phi,dt,pressure);
        if(p.mu==0. && result.active)result.gamma[0]=norm(vec(result.point.state.Q-beta*old.Q));
        return result; // Exact retained homogeneous/zero-length identity.
      }
    AssertThrow(h>0.,dealii::ExcMessage("Nonzero gradient backstress in exactly intact material is outside the fixed-positive-phase prototype; phase-active use is disabled."));
    // Exact algebraic change of the LOCAL trial problem, not an alteration of
    // accepted history: t*=t+hX, q*=q+X and ref*=ref+X. Both physical tau and
    // its normal pressure are invariant. Reuse the qualified complete return.
    auto virtual_old=old;
    virtual_old.T+=h/beta*X;virtual_old.A+=h/(2*G*beta)*X;virtual_old.Q+=X/beta;
    static_cast<CoulombReturn::Response&>(result)=CoulombReturn::update(p,virtual_old,de,old.phi,dt,pressure);
    auto &r=result.point;
    r.state.T=beta*(old.T+2*G*de);r.state.Q-=X;
    // Reevaluate the selected branch in PHYSICAL variables. Subtracting X
    // from q* loses small Q at developed damage, then h amplifies the error.
    // The retained return selects the unique branch; this is its same exact
    // algebra without intermediate hX-sized operands or altered tolerances.
    const V t=vec(r.state.T),xx=vec(X),ref=mul(beta,vec(old.Q));
    const double l=1+h,b=p.mu*h,pbar=pressure-.5*(r.state.T[0][0]+r.state.T[1][1]);
    const double R=p.residual_cohesion+p.mu*pbar;
    if(result.active==1 || result.active==2)
      {
        const unsigned int i=result.active==1?0:1;const int k=i==0?1:-1;
        const V z0=z(t,ref,xx,h,p.mu,k);const double s=norm(z0);
        AssertThrow(s>R,dealii::ExcMessage("Gradient return branch disagrees with physical trial yield."));
        const double excess=(s-R)*(s+R),g=excess/(R*l+std::sqrt((R*l)*(R*l)+(l*l+b*b)*excess));
        const V v=mul(1/(s*s),add(mul(R+l*g,z0),mul(b*g*k,J(z0))));
        r.state.Q=tensor(add(ref,mul(g*k,J(v))));result.gamma={{0.,0.}};result.gamma[i]=g;
      }
    else if(result.active==3)
      {
        const V u=add(t,mul(h,xx));const double s=norm(u);
        if(s>0.)
          {
            const V v=mul(1/s,u);const double perpendicular=-cross(t,xx)/s;
            const double e=dot(sub(t,xx),v),d=dot(t,v),A=l*l+b*b,B=l*e+p.mu*p.mu*h*d,C=e*e+p.mu*p.mu*d*d-R*R;
            const double discriminant=B*B-A*C;
            AssertThrow(discriminant>=0.,dealii::ExcMessage("Gradient double-slip physical intersection is absent."));
            const double plus=(B+std::sqrt(discriminant))/A;
            const double minus=B+std::sqrt(discriminant)==0.?(B-std::sqrt(discriminant))/A:C/(B+std::sqrt(discriminant));
            const V q0=add(mul(plus,v),mul(perpendicular,J(v))),q1=add(mul(minus,v),mul(perpendicular,J(v)));
            const V chosen=norm(sub(q0,vec(r.state.Q)))<norm(sub(q1,vec(r.state.Q)))?q0:q1;r.state.Q=tensor(chosen);
          }
        const V q=vec(r.state.Q),zp=z(t,q,xx,h,p.mu,1),zm=z(t,q,xx,h,p.mu,-1);
        const V gp=mul(1/norm(zp),J(zp)),gm=mul(-1/norm(zm),J(zm)),d=sub(q,ref);
        const double determinant=cross(gp,gm);
        AssertThrow(determinant!=0.,dealii::ExcMessage("Degenerate gradient double-slip flow directions."));
        result.gamma={{cross(d,gm)/determinant,cross(gp,d)/determinant}};
      }
    r.state.A=r.sliding?h/(2*G)*r.state.Q:beta*old.A;
    r.flow_rate=r.sliding?(r.state.A-beta*old.A)/(beta*dt):Tensor();
    r.speed=std::sqrt(2.*(r.flow_rate*r.flow_rate));
    r.state.K=old.K+dt*r.flow_rate;r.state.L=old.L+dt*r.speed;
    r.stress=r.state.T-2*G*r.state.A;
    const auto H=compliance(old.phi,p.material.m);
    r.driving=H.first*(r.state.Q*r.state.Q)/(4*G);
    r.bulk_energy=(r.stress*r.stress)/(4*G);r.cohesive_energy=h*(r.state.Q*r.state.Q)/(4*G);
    r.bulk_dissipation=(r.stress*r.stress)/(2*p.material.eta);
    r.cohesive_dissipation=h*(r.state.Q*r.state.Q)/(2*p.material.eta);
    r.friction_dissipation=(r.stress-r.state.Q-X)*r.flow_rate;
    result.planes=CoulombReturn::planes(r.stress,r.state.Q+X,pressure,p.mu,p.residual_cohesion);
    if(p.mu==0. && result.active)result.gamma[0]=norm(vec(r.state.Q-beta*old.Q));
    result.system_speed=h*(result.gamma[0]+result.gamma[1])/(G*beta*dt);
    const double tolerance=1e-12*std::max({1.,norm(t),norm(ref),norm(xx),R});
    for(unsigned int i=0;i<2;++i)
      AssertThrow(result.planes[i].margin<=tolerance && result.gamma[i]>=-tolerance
                  && (!(result.active&(1u<<i)) || std::abs(result.planes[i].margin)<=tolerance),
                  dealii::ExcMessage("Gradient physical return fails the retained Coulomb force allowance."));
    AssertThrow(pbar>norm(vec(r.stress)),dealii::ExcMessage("Gradient return leaves physical closed contact; no pressure clipping."));
    AssertThrow(r.friction_dissipation>=0. && std::isfinite(r.driving+r.friction_dissipation),
                dealii::ExcMessage("Gradient Coulomb physical dissipation is invalid."));
    return result;
  }

  CoulombReturn::Variation derivative(const CoulombReturn::Parameters &p,const TensorReturn::State &old,
                  const Response &r,const Tensor &de,double dt,double dp,const Tensor &dX)
  {
    const auto &x=r.point;const double G=p.material.G,h=compliance(old.phi,p.material.m).h;
    if(h==0.)
      {
        AssertThrow(r.backstress.norm()==0. && dX.norm()==0.,
                    dealii::ExcMessage("Intact gradient control requires X=dX=0; no phase-active derivative is qualified."));
        return CoulombReturn::derivative(p,old,r,de,0.,dt,dp);
      }
    CoulombReturn::Variation result;auto &a=result.point;
    a.T=2*G*x.beta*de;
    const V t=vec(x.state.T),q=vec(x.state.Q),xx=vec(r.backstress);
    const V inputT=vec(a.T);const double dP=dp-.5*(a.T[0][0]+a.T[1][1]);
    V dq{{0.,0.}};auto gamma=r.gamma;
    // The retained mu=0 identity branch does not need individual multipliers;
    // reconstruct its single multiplier solely for the new X derivative.
    if(p.mu==0. && r.active)gamma[0]=norm(vec(x.state.Q-x.beta*old.Q));
    if(r.active)
      {
        std::array<unsigned int,2> indices{};unsigned int count=0;
        for(unsigned int i=0;i<2;++i)if(r.active&(1u<<i))indices[count++]=i;
        double M[4][4]{};M[0][0]=M[1][1]=1.;
        std::array<double,4> rhs{{dq[0],dq[1],0.,0.}};
        for(unsigned int j=0;j<count;++j)
          {
            const auto i=indices[j];const int k=i==0?1:-1;
            const V zz=z(t,q,xx,h,p.mu,k),v=mul(1/norm(zz),zz),g=mul(k,J(v));
            const auto dz=[&](V a){return add(mul(k*(1+h),J(a)),mul(p.mu*h,a));};
            const auto dg=[&](V a){return mul(k/norm(zz),J(sub(a,mul(dot(v,a),v))));};
            const V input=sub(mul(-k,J(sub(inputT,vec(dX)))),mul(p.mu,inputT)),gi=dg(input);
            rhs[0]+=gamma[i]*gi[0];rhs[1]+=gamma[i]*gi[1];rhs[2+j]=p.mu*dP-dot(v,input);
            for(unsigned int c=0;c<2;++c)
              {V unit{{0.,0.}};unit[c]=1.;const V zc=dz(unit),gc=dg(zc);M[0][c]-=gamma[i]*gc[0];M[1][c]-=gamma[i]*gc[1];M[2+j][c]=dot(v,zc);}
            M[0][2+j]=-g[0];M[1][2+j]=-g[1];
          }
        const auto solution=solve(M,rhs,2+count);dq={{solution[0],solution[1]}};
        for(unsigned int j=0;j<count;++j)result.gamma[indices[j]]=solution[2+j];
      }
    a.Q=tensor(dq);a.A=h/(2*G)*a.Q;
    a.stress=a.T-2*G*a.A;a.flow_rate=a.A/(x.beta*dt);a.K=dt*a.flow_rate;
    if(x.speed>0.)a.speed=2*(x.flow_rate*a.flow_rate)/x.speed;
    a.L=dt*a.speed;result.system_speed=h*(result.gamma[0]+result.gamma[1])/(G*x.beta*dt);
    a.driving=compliance(old.phi,p.material.m).first*(x.state.Q*a.Q)/(2*G);
    a.bulk_energy=(x.stress*a.stress)/(2*G);a.cohesive_energy=h*(x.state.Q*a.Q)/(2*G);
    a.bulk_dissipation=(x.stress*a.stress)/p.material.eta;a.cohesive_dissipation=h*(x.state.Q*a.Q)/p.material.eta;
    a.friction_dissipation=(a.stress-a.Q-dX)*x.flow_rate+(x.stress-x.state.Q-r.backstress)*a.flow_rate;
    for(unsigned int i=0;i<2;++i)if(r.planes[i].defined)
      {
        const int k=i==0?1:-1;const V zz=z(t,q,xx,h,p.mu,k),v=mul(1/norm(zz),zz);
        const V dzz=sub(mul(-k,J(vec(a.stress-a.Q-dX))),mul(p.mu,vec(a.stress)));
        const V dv=mul(1/norm(zz),sub(dzz,mul(dot(v,dzz),v)));
        result.theta[i]=.5*cross(v,dv);
        result.normal[i]=dP+dot(vec(a.stress),v)+dot(vec(x.stress),dv);
        result.shear[i]=dot(vec(a.stress-a.Q-dX),J(v))+dot(vec(x.stress-x.state.Q-r.backstress),J(dv));
        result.margin[i]=dot(v,dzz)-p.mu*dP;
      }
    return result;
  }
}}}}
