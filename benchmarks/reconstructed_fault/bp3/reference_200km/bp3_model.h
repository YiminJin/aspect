#ifndef ASPECT_BENCHMARK_BP3_MODEL_H
#define ASPECT_BENCHMARK_BP3_MODEL_H

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace BP3
{
  // SEAS BP3-QD, 2021-10-01, Table 1 and equations (18), (25), (26).
  constexpr double rho = 2670., cs = 3464., G = rho*cs*cs;
  constexpr double damping = G/(2*cs), sigma0 = 50e6;
  constexpr double Vp = 1e-9, Vinit = 1e-9, Vref = 1e-6, Dc = .008;
  constexpr double a0 = .010, amax = .025, b = .015, f0 = .6;
  constexpr double Wf = 40000., box_size = 100000., core_phi = .6;
  inline const double sine = std::sqrt(3.)/2;
  constexpr double cosine = .5;
  inline const double dip = std::acos(cosine);
  inline const double trace_x = .5*box_size*(1+cosine/sine);
  inline const double tau0 = sigma0*amax*std::asinh(Vinit/(2*Vref)
    *std::exp((f0+b*std::log(Vref/Vinit))/amax))+damping*Vinit;

  inline double fraction(double xd)
  { return std::clamp((xd-15000.)/3000., 0., 1.); }

  // Horizontal bulk VW/transition/VS interfaces. On the sharp fault this
  // equals true down-dip distance; it is not used for deep V constraints.
  inline double depth_fraction(double y)
  { return fraction((box_size-y)/sine); }

  inline double direct_effect(double xd)
  { return a0+(amax-a0)*fraction(xd); }

  inline double theta0(double xd)
  {
    const double a = direct_effect(xd);
    return Dc/Vref*std::exp(a/b*std::log(2*Vref/Vinit
             *std::sinh((tau0-damping*Vinit)/(a*sigma0)))-f0/b);
  }

  // Independent benchmark check of the split aging law. Avoid subtracting
  // O(Dc/V) terms when the physical state is only O(Theta_old+dt).
  inline long double aging_state_reference(const double V,
                                           const double old_theta,
                                           const double dt)
  {
    const long double steady=static_cast<long double>(Dc)/V;
    const long double x=static_cast<long double>(V)*dt/Dc;
    return old_theta*std::exp(-x)-steady*std::expm1(-x);
  }

  // ASPECT x=trace_x-x_BP3, y=box_size-z_BP3. This proper 180-degree
  // rotation keeps positive manager V equal to official positive thrust slip.
  inline double down_dip(double x, double y)
  { return (trace_x-x)*cosine+(box_size-y)*sine; }

  inline double normal_distance(double x, double y)
  { return std::abs((trace_x-x)*sine-(box_size-y)*cosine); }

  using Stress = std::array<double, 3>; // total effective xx, yy, xy; tensile positive

  class Airy
  {
    public:
      Airy(double angle, double normal, double shear)
      {
        const double sn=std::sin(2*angle), co=std::cos(2*angle);
        const double a11=2*(co-1), a12=2*(sn-2*angle);
        const double a21=2*sn, a22=2*(1-co), det=a11*a22-a12*a21;
        if (std::abs(det)<1e-12) throw std::runtime_error("Singular BP3 Airy wedge");
        C=(normal*a22-a12*shear)/det;
        D=(a11*shear-normal*a21)/det;
        A=-C; B=-2*D;
      }

      Stress stress(double theta, int side) const
      {
        // Phi=r^2 f(theta): equilibrium is exact away from the corner.
        // No old radial blending or cohesion-dependent change of Theta0.
        const double f=A+B*theta+C*std::cos(2*theta)+D*std::sin(2*theta);
        const double rr=2*f-4*C*std::cos(2*theta)-4*D*std::sin(2*theta);
        const double tt=2*f, rt=-B+2*C*std::sin(2*theta)-2*D*std::cos(2*theta);
        const double c=std::cos(theta), s=std::sin(theta);
        return {{rr*c*c+tt*s*s-2*rt*c*s,
                rr*s*s+tt*c*c+2*rt*c*s,
                side*((rr-tt)*c*s+rt*(c*c-s*s))}};
      }

      Stress radial_stress(double theta, int side, double integral_over_r,
                           double traction, double r_derivative) const
      {
        // For the unit-shear angular mode use Phi=r*integral(q dr)*f.
        // Its fault value f=0 preserves normal traction, while -f'=+/-1
        // prescribes q(r). Radial derivatives are retained: this is not stress blending.
        const double f=A+B*theta+C*std::cos(2*theta)+D*std::sin(2*theta);
        const double df=B-2*C*std::sin(2*theta)+2*D*std::cos(2*theta);
        const double ddf=-4*C*std::cos(2*theta)-4*D*std::sin(2*theta);
        const double rr=(integral_over_r+traction)*f+integral_over_r*ddf;
        const double tt=(2*traction+r_derivative)*f, rt=-traction*df;
        const double c=std::cos(theta), s=std::sin(theta);
        return {{rr*c*c+tt*s*s-2*rt*c*s, rr*s*s+tt*c*c+2*rt*c*s,
                side*((rr-tt)*c*s+rt*(c*c-s*s))}};
      }

    private:
      double A, B, C, D;
  };

  inline Stress stress(double x, double y, double cohesive)
  {
    const double xb=trace_x-x, z=box_size-y;
    if (std::hypot(xb,z)==0)
      throw std::runtime_error("BP3 Airy corner has directional traces, not a unique point stress");
    const bool plus = xb*sine-z*cosine>=0;
    const Airy wedge(plus ? dip : std::acos(-1.)-dip,
                     -sigma0, (plus ? 1 : -1)*(tau0+cohesive));
    return wedge.stress(std::atan2(z, plus ? xb : -xb), plus ? 1 : -1);
  }

  inline double shear(const Stress &s)
  { return sine*cosine*(s[1]-s[0])+(cosine*cosine-sine*sine)*s[2]; }

  inline double normal(const Stress &s)
  { return -sine*sine*s[0]+2*sine*cosine*s[2]-cosine*cosine*s[1]; }
}
#endif
