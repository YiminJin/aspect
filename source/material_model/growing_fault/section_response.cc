/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#include <aspect/material_model/growing_fault/section_response.h>
#include <aspect/phase_field/degradation.h>

namespace aspect::GrowingFault
{
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
    const PhaseField::CohesiveDegradation degradation(phi,m);
    PointResponse r;
    r.fraction=degradation.fraction();
    r.fraction_derivative=degradation.fraction_derivative();
    r.driving_factor=degradation.driving_factor();
    r.driving_factor_derivative=degradation.driving_factor_derivative();
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
    const PhaseField::CohesiveDegradation degradation(phi,m);
    return degradation.cohesive_energy(G*(A*A));
  }
}
