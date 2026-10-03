#ifndef ASPECT_BENCHMARK_BP3_MODEL_H
#define ASPECT_BENCHMARK_BP3_MODEL_H

#include "geometry.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace BP3
{
  // SEAS BP3-QD, 2021-10-01, Table 1 and equations (18), (25), (26).
  constexpr double rho = 2670., cs = 3464., G = rho*cs*cs;
  constexpr double damping = G/(2*cs), sigma0 = 50e6;
  constexpr double Vp = 1e-9, Vinit = 1e-9, Vref = 1e-6;
  constexpr double a0 = .010, amax = .025, b = .015, f0 = .6;
#ifdef ASPECT_BP3_LOCAL_BOTTOM_TEST
  extern double local_state_disturbance;
#endif
  inline const double tau0 = sigma0*amax*std::asinh(Vinit/(2*Vref)
    *std::exp((f0+b*std::log(Vref/Vinit))/amax))+damping*Vinit;

  // Configured before initial fields are evaluated; the transition remains 3 km.
  inline double fraction(double xd)
  {
#ifdef ASPECT_BP3_LOCAL_BOTTOM_TEST
    (void)xd;
    return 1.;
#else
    return std::clamp((xd-geometry().weakening_length)/3000., 0., 1.);
#endif
  }

  // Horizontal bulk VW/transition/VS interfaces. On the sharp fault this
  // equals true down-dip distance; it is not used for deep V constraints.
  inline double depth_fraction(double y)
  { return fraction((geometry().upper[1]-y)/geometry().sine); }

  inline double direct_effect(double xd)
  { return a0+(amax-a0)*fraction(xd); }

  inline double theta0(double xd, const double Dc)
  {
    const double a = direct_effect(xd);
    return Dc/Vref*std::exp(a/b*std::log(2*Vref/Vinit
             *std::sinh((tau0-damping*Vinit)/(a*sigma0)))-f0/b);
  }

  // Research variants obtain every friction parameter from the live material
  // law. The historical constants above remain only for old analytical tests.
  template <class Friction>
  double configured_initial_state(double xd, const Friction &law)
  {
    const double target=law.friction_coefficient({0.,1.}, Vinit,
                         law.get_characteristic_slip_distance()/Vinit);
    const double f=fraction(xd);
    const double steady=law.initial_state_for_friction_coefficient({1-f,f},Vinit,target);
#ifdef ASPECT_BP3_LOCAL_BOTTOM_TEST
    const double y=geometry().upper[1]-xd*geometry().sine;
    return steady*std::exp(local_state_disturbance*std::pow(std::sin(std::acos(-1.)*(y-geometry().origin[1])/geometry().extent[1]),2));
#else
    return steady;
#endif
  }

  // Independent benchmark check of the split aging law. Avoid subtracting
  // O(Dc/V) terms when the physical state is only O(Theta_old+dt).
  inline long double aging_state_reference(const double V,
                                           const double old_theta,
                                           const double dt,
                                           const double Dc)
  {
    const long double steady=static_cast<long double>(Dc)/V;
    const long double x=static_cast<long double>(V)*dt/Dc;
    return old_theta*std::exp(-x)-steady*std::expm1(-x);
  }

  // Coordinates follow the validated top-to-bottom chart, independent of
  // file ordering. Bulk material fractions retain the horizontal extension.
  inline double down_dip(double x, double y)
  { return geometry().down_dip(x,y); }

  inline double signed_normal(double x, double y)
  { return geometry().signed_normal(x,y); }

  inline double normal_distance(double x, double y)
  { return std::abs(signed_normal(x,y)); }

}
#endif
