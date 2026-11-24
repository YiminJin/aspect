// Benchmark-only one-dimensional split Maxwell background. No ASPECT state.
#pragma once
#include <array>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
namespace LayeredReference
{
  constexpr double G=10., bottom=-2., top=2., delta_u=2., width=.35;
  constexpr std::array<double,8> nodes= {{-.9602898564975363,-.7966664774136267,-.5255324099163290,-.1834346424956498,.1834346424956498,.5255324099163290,.7966664774136267,.9602898564975363}};
  constexpr std::array<double,8> weights= {{.1012285362903763,.2223810344533745,.3137066458778873,.3626837833783620,.3626837833783620,.3137066458778873,.2223810344533745,.1012285362903763}};
  struct Field
  {
    std::vector<double> a,xy,gamma;
    explicit Field(unsigned n=0):a(n),xy(n),gamma(n) {}
  };
  class Background
  {
    public:
      unsigned panels;
      double h;
      bool homogeneous;
      std::vector<double> y,eta;
      explicit Background(unsigned panels=64,bool homogeneous=false):panels(panels),h(4./panels),homogeneous(homogeneous)
      {
        for (unsigned p=0; p<panels; ++p)for (double x:nodes)
            {
              double point=bottom+h*(p+.5+.5*x);
              y.push_back(point);
              eta.push_back(homogeneous ? 10. : 5.5+4.5*std::tanh(point/width));
            }
      }
      double integral(const std::vector<double> &f) const
      {
        double s=0;
        for (unsigned i=0; i<f.size(); ++i)s+=.5*h*weights[i%8]*f[i];
        return s;
      }
      Field advance(const Field &parent,double dt,double &q) const
      {
        if (!(dt>0) || parent.a.size()!=y.size())throw std::runtime_error("Invalid background parent/dt");
        Field next(y.size());
        std::vector<double> b(y.size()),inv(y.size()),ratio(y.size());
        for (unsigned i=0; i<y.size(); ++i)
          {
            const double theta=parent.gamma[i]*dt,c=std::cos(theta),s=std::sin(theta);
            const double beta=std::exp(-G*dt/eta[i]),kappa=-eta[i]*std::expm1(-G*dt/eta[i]);
            next.a[i]=beta*(c*parent.a[i]+s*parent.xy[i]);
            b[i]=beta*(c*parent.xy[i]-s*parent.a[i]);
            inv[i]=1./kappa;
            ratio[i]=b[i]/kappa;
          }
        q=(delta_u+integral(ratio))/integral(inv);
        for (unsigned i=0; i<y.size(); ++i)
          {
            next.xy[i]=q;
            next.gamma[i]=(q-b[i])*inv[i];
          }
        return next;
      }
      double interpolate(const std::vector<double> &f,unsigned p,double x) const
      {
        double sum=0.;
        for (unsigned i=0; i<8; ++i)
          {
            double b=1.;
            for (unsigned j=0; j<8; ++j)if (i!=j)b*=(x-nodes[j])/(nodes[i]-nodes[j]);
            sum+=b*f[8*p+i];
          }
        return sum;
      }
      double value(const std::vector<double> &f,double point) const
      {
        const unsigned p=std::min(panels-1,unsigned(std::max(0.,std::floor((point-bottom)/h))));
        return interpolate(f,p,2*(point-bottom)/h-2*p-1);
      }
      std::vector<double> primitive(const std::vector<double> &f) const
      {
        std::vector<double> result(panels+1);
        for (unsigned p=0; p<panels; ++p)
          {
            result[p+1]=result[p];
            for (unsigned i=0; i<8; ++i)result[p+1]+=.5*h*weights[i]*f[8*p+i];
          }
        return result;
      }
      double velocity(const Field &field,const std::vector<double> &prefix,double point,double translation) const
      {
        const unsigned p=std::min(panels-1,unsigned(std::max(0.,std::floor((point-bottom)/h))));
        const double length=std::max(0.,std::min(h,point-bottom-p*h));
        double sum=0.;
        for (unsigned i=0; i<8; ++i)sum+=weights[i]*interpolate(field.gamma,p,-1.+length/h*(1.+nodes[i]));
        return translation-1.+prefix[p]+.5*length*sum;
      }
  };
}
