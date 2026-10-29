/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/material_model/phase_field_fault/bulk_cohesive_gradient.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <algorithm>
namespace aspect { namespace MaterialModel { namespace BulkCohesive { namespace SlipGradient
{
  using namespace dealii;
  namespace
  {
    using V=std::array<double,2>;
    V add(V a,V b){return {{a[0]+b[0],a[1]+b[1]}};}
    V sub(V a,V b){return {{a[0]-b[0],a[1]-b[1]}};}
    V mul(double s,V a){return {{s*a[0],s*a[1]}};}
    double dot(V a,V b){return a[0]*b[0]+a[1]*b[1];}
    V J(V a){return {{-a[1],a[0]}};}
    double norm(V a){return std::hypot(a[0],a[1]);}
    V vec(const Tensor &a){return {{.5*(a[0][0]-a[1][1]),a[0][1]}};}
    Tensor tensor(V a){Tensor t;t[0][0]=a[0];t[1][1]=-a[0];t[0][1]=a[1];return t;}
  }
  Projection project(const Tensor &input,const Tensor &stress,double pressure,double mu,double cohesion)
  {
    const V d=vec(stress),Z=vec(input),c=mul(mu,J(d));const double C=norm(c);
    const double pbar=pressure-.5*(stress[0][0]+stress[1][1]),R=cohesion+mu*pbar;
    AssertThrow(mu>=0. && cohesion>=0. && std::isfinite(R) && pbar>norm(d),
                dealii::ExcMessage("Joint Coulomb projection requires physical closed contact; pressure is not clipped."));
    Projection result;result.input=input;result.stress=stress;result.pressure=pressure;result.mu=mu;result.cohesion=cohesion;
    if(norm(sub(Z,c))<=R && norm(add(Z,c))<=R){result.force=input;return result;}
    if(R==0.){result.active=3;return result;}
    if(C==0.){result.force=tensor(mul(R/norm(Z),Z));result.active=1;return result;}
    double distance=std::numeric_limits<double>::infinity();
    const auto admit=[&](V force,unsigned int active,int sign)
      {const double candidate=norm(sub(Z,force));if(candidate<distance){distance=candidate;result.force=tensor(force);result.active=active;result.corner_sign=sign;}};
    for(unsigned int i=0;i<2;++i)
      {
        const V center=mul(i==0?1.:-1.,c),direction=sub(Z,center);const double radius=norm(direction);
        if(radius==0.)continue;
        const V force=add(center,mul(R/radius,direction));
        if(norm(add(force,center))<=R)admit(force,1u<<i,1);
      }
    if(C>0.)
      {
        const V axis=mul(1/C,J(c));const double radius=std::sqrt((R-C)*(R+C));
        admit(mul(radius,axis),3,1);admit(mul(-radius,axis),3,-1);
      }
    AssertThrow(std::isfinite(distance),dealii::ExcMessage("Coulomb lens projection has no candidate."));
    return result;
  }

  Tensor projection_action(const Projection &r,const Tensor &dinput,const Tensor &dstress,double dp)
  {
    if(r.active==0)return dinput;
    const V d=vec(r.stress),dd=vec(dstress),c=mul(r.mu,J(d)),dc=mul(r.mu,J(dd));
    const double C=norm(c),R=r.cohesion+r.mu*(r.pressure-.5*(r.stress[0][0]+r.stress[1][1]));
    const double dR=r.mu*(dp-.5*(dstress[0][0]+dstress[1][1]));
    if(R==0.)return Tensor();
    if(r.active!=3)
      {
        const int k=r.active==1?1:-1;const V center=mul(k,c),dcenter=mul(k,dc),q=sub(vec(r.input),center);
        const double radius=norm(q);const V v=mul(1/radius,q),delta=sub(vec(dinput),dcenter);
        return tensor(add(add(dcenter,mul(dR,v)),mul(R/radius,sub(delta,mul(dot(v,delta),v)))));
      }
    AssertThrow(C>0.,dealii::ExcMessage("Undefined coincident-circle corner derivative."));
    const V u=mul(1/C,J(c));const double b=std::sqrt((R-C)*(R+C));
    const double db=(R*dR-dot(c,dc))/b;
    const V du=sub(mul(1/C,J(dc)),mul(dot(c,dc)/(C*C),u));
    return tensor(mul(r.corner_sign,add(mul(db,u),mul(b,du))));
  }

  double experimental_length()
  {
    const char *value=std::getenv("ASPECT_BULK_SLIP_LENGTH");
    if(!value)return 0.;
    char *end=nullptr;const double length=std::strtod(value,&end);
    AssertThrow(end!=value && *end=='\0' && std::isfinite(length) && length>=0.,
                dealii::ExcMessage("Experimental slip length must be finite, nonnegative and in metres."));
    return length;
  }

  JointResponse joint_point(const CoulombReturn::Parameters &p,const TensorReturn::State &old,
                            const Tensor &de,const Tensor &K,const Tensor &X,double dt,double pressure)
  {
    JointResponse result;auto &c=result.local;auto &r=c.point;
    const double G=p.material.G;const auto H=compliance(old.phi,p.material.m);
    AssertThrow(H.h>0. && dt>0.,dealii::ExcMessage("The fixed-phase slip-gradient experiment requires positive phase and timestep."));
    const auto delta=K-old.K;r.beta=1./(1.+dt*G/p.material.eta);
    r.state=old;r.state.T=r.beta*(old.T+2*G*de);
    r.state.A=r.beta*(old.A+delta);r.state.Q=2*G/H.h*r.state.A;r.state.K=K;
    r.flow_rate=delta/dt;r.speed=std::sqrt(2.*(r.flow_rate*r.flow_rate));r.state.L=old.L+dt*r.speed;
    r.stress=r.state.T-2*G*r.state.A;
    const auto trial=r.state.T-2*G*r.beta*old.A;
    AssertThrow(pressure-.5*(trial[0][0]+trial[1][1])>TensorReturn::magnitude(trial),
                dealii::ExcMessage("Slip-gradient elastic trial leaves physical closed contact; no pressure clipping."));
    const auto F=TensorReturn::in_plane(r.stress)-r.state.Q-X;
    result.b=2*G*r.beta*(1.+1./H.h);
    result.projection=project(F+result.b*delta,r.stress,pressure,p.mu,p.residual_cohesion);
    result.residual=result.projection.active ? F-result.projection.force : -result.b*delta;
    result.force_scale=std::max({1.,r.state.T.norm(),r.state.Q.norm(),X.norm(),p.mu*pressure,p.residual_cohesion});
    c.pressure=pressure;c.active=result.projection.active;
    c.planes=CoulombReturn::planes(r.stress,r.state.Q+X,pressure,p.mu,p.residual_cohesion);
    r.sliding=r.speed>0.;r.intact=false;
    r.driving=H.first*(r.state.Q*r.state.Q)/(4*G);
    r.bulk_energy=(r.stress*r.stress)/(4*G);r.cohesive_energy=H.h*(r.state.Q*r.state.Q)/(4*G);
    r.bulk_dissipation=(r.stress*r.stress)/(2*p.material.eta);
    r.cohesive_dissipation=H.h*(r.state.Q*r.state.Q)/(2*p.material.eta);
    r.friction_dissipation=F*r.flow_rate;
    if(c.active && r.speed>0.)
      {
        V m[2];for(unsigned int i=0;i<2;++i)
          {const double sign=i==0?1.:-1.;m[i]={{-.5*sign*std::sin(2*c.planes[i].theta),.5*sign*std::cos(2*c.planes[i].theta)}};}
        const V w=vec(r.flow_rate);
        if(c.active==3)
          {
            const double det=m[0][0]*m[1][1]-m[0][1]*m[1][0];
            AssertThrow(det!=0.,dealii::ExcMessage("Degenerate simultaneous Coulomb directions."));
            c.gamma[0]=(w[0]*m[1][1]-w[1]*m[1][0])/det;
            c.gamma[1]=(m[0][0]*w[1]-m[0][1]*w[0])/det;
          }
        else {const unsigned int i=c.active==1?0:1;c.gamma[i]=dot(w,m[i])/dot(m[i],m[i]);}
        // This joint record reports physical system rates (1/s), rather than
        // the stress multipliers used by the eliminated local return.
        c.system_speed=c.gamma[0]+c.gamma[1];
      }
    return result;
  }

  JointAction joint_action(const CoulombReturn::Parameters &p,const JointResponse &r,
                           const Tensor &de,const Tensor &dK,const Tensor &dX,double,double dp)
  {
    JointAction result;const double G=p.material.G,beta=r.local.point.beta;
    const double h=compliance(r.local.point.state.phi,p.material.m).h;
    result.stress=2*G*beta*(de-dK);
    if(r.projection.active==0){result.residual=-r.b*dK;return result;}
    const auto dF=TensorReturn::in_plane(result.stress)-2*G*beta/h*dK-dX;
    result.residual=dF-projection_action(r.projection,dF+r.b*dK,result.stress,dp);
    return result;
  }

  Space::Space(const std::vector<Sample> &samples,const Point<2> &minimum,const Point<2> &maximum)
  {
    AssertThrow(!samples.empty(),dealii::ExcMessage("Empty slip-gradient history layout."));
    std::array<std::vector<double>,2> axis;
    const double tolerance=64*std::numeric_limits<double>::epsilon()*std::max({maximum.norm(),minimum.norm(),(maximum-minimum).norm()});
    for(unsigned int d=0;d<2;++d)
      {
        for(const auto &s:samples)axis[d].push_back(s.point[d]);
        std::sort(axis[d].begin(),axis[d].end());
        std::vector<double> distinct;
        for(const auto x:axis[d])if(distinct.empty() || x-distinct.back()>tolerance)distinct.push_back(x);
        axis[d]=std::move(distinct);axis_sizes[d]=axis[d].size();
        AssertThrow(axis_sizes[d]>=2 && axis[d].front()>minimum[d] && axis[d].back()<maximum[d],
                    dealii::ExcMessage("Slip-gradient space requires interior Cartesian quadrature axes."));
      }
    const unsigned int nx=axis_sizes[0],ny=axis_sizes[1],n=nx*ny;
    AssertThrow(n==samples.size(),dealii::ExcMessage("Slip-gradient history layout is not a complete tensor product; no coverage relaxation."));
    points.resize(n);mass.reinit(n);
    for(const auto &s:samples)
      {
        unsigned int index[2];
        for(unsigned int d=0;d<2;++d)
          {
            const auto it=std::lower_bound(axis[d].begin(),axis[d].end(),s.point[d]-tolerance);
            AssertThrow(it!=axis[d].end() && std::abs(*it-s.point[d])<=tolerance,dealii::ExcMessage("Unmatched slip-gradient point."));
            index[d]=it-axis[d].begin();coordinate_error=std::max(coordinate_error,std::abs(*it-s.point[d]));
          }
        const unsigned int i=index[0]+nx*index[1];
        AssertThrow(mass[i]==0. && s.weight>0.,dealii::ExcMessage("Duplicate or invalid slip-gradient quadrature record."));
        points[i]=Point<2>(axis[0][index[0]],axis[1][index[1]]);mass[i]=s.weight;
        auto &record=ids[s.cell];if(record.size()<=s.q)record.resize(s.q+1,numbers::invalid_unsigned_int);record[s.q]=i;
      }
    std::array<FullMatrix<double>,2> M,L;
    for(unsigned int d=0;d<2;++d)
      {
        const unsigned int count=axis_sizes[d];M[d].reinit(count,count);L[d].reinit(count,count);
        std::vector<double> ends{minimum[d]};ends.insert(ends.end(),axis[d].begin(),axis[d].end());ends.push_back(maximum[d]);
        const QGauss<1> quadrature(2);
        for(unsigned int interval=0;interval+1<ends.size();++interval)
          {
            const unsigned int left=interval==0?0:std::min(interval-1,count-2);
            const double h=axis[d][left+1]-axis[d][left],width=ends[interval+1]-ends[interval];
            for(unsigned int q=0;q<quadrature.size();++q)
              {
                const double x=ends[interval]+width*quadrature.point(q)[0],r=(x-axis[d][left])/h;
                const double N[2]={1-r,r},D[2]={-1/h,1/h};
                for(unsigned int a=0;a<2;++a)for(unsigned int b=0;b<2;++b)
                  {M[d](left+a,left+b)+=width*quadrature.weight(q)*N[a]*N[b];L[d](left+a,left+b)+=width*quadrature.weight(q)*D[a]*D[b];}
              }
          }
      }
    DynamicSparsityPattern dynamic(n,n);
    for(unsigned int i=0;i<n;++i)
      for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx)
        {const int x=int(i%nx)+dx,y=int(i/nx)+dy;if(x>=0 && x<int(nx) && y>=0 && y<int(ny))dynamic.add(i,x+nx*y);}
    sparsity.copy_from(dynamic);stiffness.reinit(sparsity);
    for(unsigned int i=0;i<n;++i)
      {
        double diagonal=0.;
        for(auto it=stiffness.begin(i);it!=stiffness.end(i);++it)
          if(it->column()!=i)
            {
              const unsigned int j=it->column();
              it->value()=L[0](i%nx,j%nx)*M[1](i/nx,j/nx)+M[0](i%nx,j%nx)*L[1](i/nx,j/nx);diagonal-=it->value();
            }
        stiffness.set(i,i,diagonal);
      }
  }
  std::vector<Tensor> Space::apply(const std::vector<Tensor> &K,double G,double length) const
  {
    AssertDimension(K.size(),points.size());std::vector<Tensor> result(K.size());
    for(unsigned int i=0;i<K.size();++i)
      {
        for(auto it=stiffness.begin(i);it!=stiffness.end(i);++it)
          if(it->column()!=i)result[i]+=it->value()*(K[it->column()]-K[i]);
        result[i]*=2*G*length*length/mass[i];
      }
    return result;
  }
  double Space::energy(const std::vector<Tensor> &K,double G,double length) const
  {
    AssertDimension(K.size(),points.size());long double result=0.;
    for(unsigned int i=0;i<K.size();++i)
      for(auto it=stiffness.begin(i);it!=stiffness.end(i);++it)if(it->column()>i)
        {const auto delta=K[it->column()]-K[i];result-=static_cast<long double>(it->value())*(delta*delta);}
    return G*length*length*result;
  }
}}}}
