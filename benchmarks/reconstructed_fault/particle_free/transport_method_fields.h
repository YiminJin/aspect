/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#ifndef PF4_TRANSPORT_METHOD_FIELDS_H
#define PF4_TRANSPORT_METHOD_FIELDS_H
#include <aspect/global.h>
#include <array>
namespace aspect
{
  namespace PF4Methods
  {
    // Benchmark data only. No material law or production history selector.
    inline std::array<double,2> velocity_source(const double x,const double y)
    {
      return {{
          1.e-3*std::exp(-std::pow((x-.45)/.16,2)) *(y-.5),
          .6e-3*std::exp(-std::pow((x-.55)/.19,2)) *(std::pow(y-.5,2)+.3*(y-.5))
        }
      };
    }
    inline std::array<double,3> native_history(const double x,const double y,const unsigned int nx)
    {
      const double dx=1./nx,dy=.5/nx;
      const double ox=std::floor(x/dx)*dx,oy=std::floor(y/dy)*dy;
      const double r=(x-ox)/dx,s=(y-oy)/dy;
      const double l[3]= {(2*r-1) *(r-1),4*r *(1-r),r *(2*r-1)};
      const double m[3]= {(2*s-1) *(s-1),4*s *(1-s),s *(2*s-1)};
      const double dl[3]= {4*r-3,4-8*r,4*r-1},dm[3]= {4*s-3,4-8*s,4*s-1};
      double grad[2][2]= {};
      for (unsigned int i=0; i<3; ++i)
        for (unsigned int j=0; j<3; ++j)
          {
            const auto u=velocity_source(ox+.5*i*dx,oy+.5*j*dy);
            for (unsigned int c=0; c<2; ++c)
              {
                grad[c][0]+=u[c]*dl[i]*m[j]/dx;
                grad[c][1]+=u[c]*l[i]*dm[j]/dy;
              }
          }
      const double kappa=-1.e8*std::expm1(-.1*1.e6/1.e8);
      const double tr=(grad[0][0]+grad[1][1])/3.;
      return {{2*kappa *(grad[0][0]-tr),2*kappa *(grad[1][1]-tr),kappa *(grad[0][1]+grad[1][0])}};
    }
    inline double scalar(const unsigned int field,const double x,const double y,const double time,const unsigned int nx)
    {
      const double a=x-.4*time;
      if (field==0) return a>=.25 && a<.5 ? 1. : (a>=.5 && a<.75 ? -1. : 0.);
      if (field==1)
        {
          const double delta=.15625/std::log(9.);
          return .5*std::tanh((a-.25)/delta)-std::tanh((a-.5)/delta)+.5*std::tanh((a-.75)/delta);
        }
      return native_history(a,y,nx)[field-2];
    }
  }
}
#endif
