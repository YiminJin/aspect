#ifndef ASPECT_BENCHMARK_BP3_MODEL_H
#define ASPECT_BENCHMARK_BP3_MODEL_H

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
  constexpr double Wf = 40000., core_phi = .6;
  constexpr double horizontal_sign = -1.;
#ifdef ASPECT_BP3_LOCAL_BOTTOM_TEST
  constexpr double box_size = 1000.;
  extern double local_state_disturbance;
#else
  constexpr double box_size = 50000.;
#endif
  inline const double sine = std::sqrt(3.)/2;
  constexpr double cosine = .5;
  inline const double dip = std::acos(cosine);
#ifdef ASPECT_BP3_LOCAL_BOTTOM_TEST
  inline const double trace_x = -box_size*cosine/sine;
#else
  inline const double trace_x = horizontal_sign < 0 ? 0. : .5*box_size*(1+cosine/sine);
#endif
  inline const double tau0 = sigma0*amax*std::asinh(Vinit/(2*Vref)
    *std::exp((f0+b*std::log(Vref/Vinit))/amax))+damping*Vinit;

  // Configured before initial fields are evaluated; the transition remains 3 km.
  extern double weakening_length;
  inline double fraction(double xd)
  {
#ifdef ASPECT_BP3_LOCAL_BOTTOM_TEST
    (void)xd;
    return 1.;
#else
    return std::clamp((xd-weakening_length)/3000., 0., 1.);
#endif
  }

  // Horizontal bulk VW/transition/VS interfaces. On the sharp fault this
  // equals true down-dip distance; it is not used for deep V constraints.
  inline double depth_fraction(double y)
  { return fraction((box_size-y)/sine); }

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
    const double y=box_size-xd*sine;
    return steady*std::exp(local_state_disturbance*std::pow(std::sin(std::acos(-1.)*y/box_size),2));
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

  // The maintained chart is a proper 180-degree rotation. The explicitly
  // restored right-dipping chart reflects x and selects manager shear sense -1
  // so V remains the positive thrust magnitude in either chart.
  inline double down_dip(double x, double y)
  { return horizontal_sign*(trace_x-x)*cosine+(box_size-y)*sine; }

  inline double signed_normal(double x, double y)
  { return horizontal_sign*(trace_x-x)*sine-(box_size-y)*cosine; }

  inline double normal_distance(double x, double y)
  { return std::abs(signed_normal(x,y)); }

}
#endif
