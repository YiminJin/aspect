#include "background.h"
#include <iostream>
#include <iomanip>
int main()
{
  using namespace LayeredReference;
  Background coarse(32),fine(64),finest(128),constant(64,true);
  Field a(coarse.y.size()),b(fine.y.size()),c(finest.y.size()),d(constant.y.size());
  double q,maximum=0.,constant_error=0.,velocity_error=0.;
  a.gamma=coarse.advance(a,.1,q).gamma;
  b.gamma=fine.advance(b,.1,q).gamma;
  c.gamma=finest.advance(c,.1,q).gamma;
  d.gamma=constant.advance(d,.1,q).gamma;
  double real=0.,imag=0.;
  for (unsigned step=0; step<128; ++step)
    {
      const double dt=.00625;
      a=coarse.advance(a,dt,q);
      b=fine.advance(b,dt,q);
      c=finest.advance(c,dt,q);
      d=constant.advance(d,dt,q);
      const double beta=std::exp(-dt),angle=.5*dt;
      double r=beta*(std::cos(angle)*real+std::sin(angle)*imag);
      imag=beta*(std::cos(angle)*imag-std::sin(angle)*real)-5.*std::expm1(-dt);
      real=r;
      velocity_error=std::max(velocity_error,std::abs(fine.velocity(b,fine.primitive(b.gamma),2.,2.5)-3.5));
      for (unsigned i=0; i<b.a.size(); ++i)
        {
          maximum=std::max({maximum,std::abs(b.a[i]-finest.value(c.a,fine.y[i])),std::abs(b.gamma[i]-finest.value(c.gamma,fine.y[i])),std::abs(b.a[i]-coarse.value(a.a,fine.y[i]))});
          constant_error=std::max({constant_error,std::abs(d.a[i]-real),std::abs(d.xy[i]-imag),std::abs(d.gamma[i]-.5)});
        }
    }
  std::cout<<std::setprecision(17)<<"{\"quadrature_interpolation_max\":"<<maximum<<",\"constant_reference_max\":"<<constant_error<<",\"velocity_integral_error\":"<<std::abs(fine.integral(b.gamma)-2.)<<",\"top_velocity_error\":"<<velocity_error<<"}\n";
  return maximum<1e-8 && constant_error<1e-12 && std::abs(fine.integral(b.gamma)-2.)<1e-12 ? 0 : 1;
}
