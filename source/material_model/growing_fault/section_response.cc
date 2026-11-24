/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#include <aspect/material_model/growing_fault/section_response.h>

namespace aspect::GrowingFault
{
  void SectionTransfer::add(const unsigned int k, const double dn, const double rate, const double m)
  {
    free_rate[k]+=dn*rate;
    mobility[k]+=dn*m;
  }

  void SectionTransfer::sum(const MPI_Comm comm)
  {
    Utilities::MPI::sum(free_rate,comm,free_rate);
    Utilities::MPI::sum(mobility,comm,mobility);
  }

  SectionRoot solve_section(const Friction &law, const double free_rate, const double mobility,
                            const double age, const double h, const double pressure)
  {
    const double Dc=law.slip_distance, a=law.direct_effect, b=law.evolution_effect;
    AssertThrow(pressure>0. && h>0. && age>0. && mobility>=0. && law.damping>=pressure*h*b/Dc,
                ExcMessage("Growing fault requires a monotone regularized section resistance at this duration."));
    const double scale=2.*law.reference_speed*std::exp(-(law.reference_coefficient+b*std::log((age+h)*law.reference_speed/Dc))/a);
    AssertThrow(scale>0. && std::isfinite(scale),ExcMessage("Section creep scale exceeds floating-point range."));
    if (free_rate==0.)
      {
        const long double compliance=scale/(static_cast<long double>(pressure)*a+law.damping*scale);
        const long double condensed=1.L/(mobility+compliance);
        AssertThrow(condensed<=std::numeric_limits<double>::max(),ExcMessage("Locked section tangent exceeds floating-point range."));
        return {0.,0.,static_cast<double>(condensed)};
      }
    const double target=std::abs(free_rate), sign=std::copysign(1.,free_rate);
    long double lo=0., hi=std::asinh(static_cast<long double>(target)/scale), y=.5L*hi;
    for (unsigned int iteration=0; iteration<100; ++iteration)
      {
        const double speed=mobility==0. ? target : double(scale*std::sinh(y));
        AssertThrow(speed>0.,ExcMessage("Section rate is below floating-point resolution."));
        const double theta=law.advance_age(speed,age,h);
        const double x=speed*h/Dc;
        const double kernel=x<1e-3 ? -.5+x*(1./3.+x*(-.125+x*(1./30.+x*(-1./144.+x/840.))))
                            : (std::expm1(-x)+x*std::exp(-x))/(x*x);
        const double age_derivative=h/Dc*(h*kernel-age*std::exp(-x));
        long double traction, condensed, step_scale;
        // At extreme creep speed df/dV may exceed DBL_MAX. V df/dV is
        // finite. Form both the Newton step and its Schur coefficient there.
        if (static_cast<long double>(speed)*std::numeric_limits<double>::max()<=pressure*a*2.)
          {
            const long double v=speed;
            const long double zlog=std::log(v/(2.L*law.reference_speed))
                                   +(law.reference_coefficient+b*std::log(theta*law.reference_speed/Dc))/static_cast<long double>(a);
            const long double small=std::exp(-std::abs(zlog));
            const long double friction=zlog>0 ? zlog+std::log1p(std::hypot(1.L,small)) : std::asinh(small);
            traction=pressure*a*friction+law.damping*v;
            const long double log_slope=pressure*a*(zlog>0 ? 1.L : small)/std::hypot(1.L,small)
                                        *(1.L+v*b*age_derivative/(a*theta))+v*law.damping;
            const long double denominator=v+mobility*log_slope;
            condensed=log_slope/denominator;
            step_scale=denominator*scale/v;
          }
        else
          {
            const auto r=law.resistance(speed,theta,pressure);
            traction=r.traction;
            const double slope=r.speed_derivative+(r.speed_derivative-law.damping)*speed*b/(a*theta)*age_derivative;
            if (mobility>1. && slope>std::numeric_limits<double>::max()/mobility)
              {
                condensed=1.L/(mobility+1.L/slope);
                step_scale=(1.L+static_cast<long double>(mobility)*slope)*scale;
              }
            else
              {
                condensed=slope/(1.+mobility*slope);
                step_scale=(1.+mobility*slope)*scale;
              }
          }
        const long double residual=static_cast<long double>(speed)+mobility*traction-target;
        if (std::abs(residual)<=32.*std::numeric_limits<double>::epsilon()*target)
          {
            AssertThrow(std::abs(condensed)<=std::numeric_limits<double>::max(),ExcMessage("Section tangent exceeds floating-point range."));
            return {sign*speed,static_cast<double>(sign*traction),static_cast<double>(condensed)};
          }
        if (residual>0.) hi=y;
        else lo=y;
        const long double trial=y-residual/(step_scale*std::cosh(y));
        y=trial>lo && trial<hi ? trial : .5L*(lo+hi);
      }
    AssertThrow(false,ExcMessage("Signed section RSF root failed its roundoff certificate."));
    return {};
  }

  double physical_phase(const double value)
  {
    constexpr double tolerance=128.*std::numeric_limits<double>::epsilon();
    AssertThrow(value>=-tolerance && value<=1.+tolerance,ExcMessage("Native phase is outside the cohesive domain."));
    return std::clamp(value,0.,1.);
  }

  double CohesiveMaxwell::relaxation(const double h) const
  {
    return 1./(1.+h*G/eta);
  }

  PointResponse CohesiveMaxwell::response(const PointHistory &old, const Stress &increment,
                                          const double phase, const double h, const Stress &S, const double f, const double traction_remainder) const
  {
    const double phi=physical_phase(phase);
    const double B=relaxation(h);
    const double c=m*phi*(1.+phi), d=(1.-phi)*(1.-phi), den=c+d;
    const double cp=m*(1.+2.*phi), dp=-2.*(1.-phi);
    PointResponse r;
    r.fraction=c/den;
    r.fraction_derivative=(cp*d-c*dp)/(den*den);
    r.driving_factor=m*(1.+3.*phi)*(1.-phi)/(den*den);
    r.driving_factor_derivative=m*(2.-6.*phi)/(den*den)-2.*r.driving_factor*(cp+dp)/den;
    const double old_e=2.*(old.cohesive_strain*S);
    const Stress predictor=B*(old.stress+2.*G*increment);
    r.force=(predictor*S+G*B*old_e-f)+traction_remainder;
    const double e=r.fraction*r.force/G;
    r.stress=predictor-2.*G*(e-B*old_e)*S;
    r.cohesive_strain=e*S;
    r.rate=(e/B-old_e)/h;
    r.driving=r.driving_factor*r.force*r.force/(2.*G);
    return r;
  }

  CohesiveMaxwell::Variation CohesiveMaxwell::derivative(const PointResponse &r, const Stress &increment,
                                                         const double dphi, const double h, const Stress &S, const double df) const
  {
    const double B=relaxation(h), dz=2.*G*B*(increment*S)-df;
    const double de=(r.fraction*dz+r.fraction_derivative*r.force*dphi)/G;
    return {2.*G *(B*increment-de*S),de/(B*h),
            (r.driving_factor_derivative *r.force *r.force*dphi+2.*r.driving_factor *r.force*dz)/(2.*G)
           };
  }

  double CohesiveMaxwell::cohesive_energy(const Stress &A, const double phase) const
  {
    const double phi=physical_phase(phase);
    if (phi==0.)
      {
        AssertThrow(A.norm()==0.,ExcMessage("Intact material point carries cohesive strain."));
        return 0.;
      }
    return G*(A*A)*(1.-phi)*(1.-phi)/(m*phi*(1.+phi));
  }
}
