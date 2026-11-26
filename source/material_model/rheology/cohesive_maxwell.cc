/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#include <aspect/material_model/rheology/cohesive_maxwell.h>

namespace aspect::GrowingFault
{
  double physical_phase(const double value)
  {
    constexpr double tolerance = 128.*std::numeric_limits<double>::epsilon();
    AssertThrow(value >= -tolerance && value <= 1. + tolerance, ExcMessage("Native phase is outside the cohesive domain."));
    return std::clamp(value, 0., 1.);
  }

  double CohesiveMaxwell::relaxation(const double half_duration) const
  {
    return 1. / (1. + half_duration * G / eta);
  }

  PhaseField::CohesiveDegradation CohesiveMaxwell::degradation(const double phase) const
  {
    return {physical_phase(phase), m};
  }

  PointResponse CohesiveMaxwell::response(const MaxwellState &parent,
                                          const Stress       &strain_increment,
                                          const double       phase,
                                          const double       half_duration,
                                          const Stress       &slip_plane,
                                          const double       resistance,
                                          const double       traction_remainder) const
  {
    const double B = relaxation(half_duration);
    const auto law = degradation(phase);
    PointResponse stage;
    stage.fraction = law.fraction();
    stage.fraction_derivative = law.fraction_derivative();
    stage.driving_factor = law.driving_factor();
    stage.driving_factor_derivative = law.driving_factor_derivative();
    const double parent_shear = 2.*(parent.cohesive_strain * slip_plane);
    const Stress predictor = B * (parent.stress + 2.*G * strain_increment);
    stage.force = (predictor * slip_plane + G * B * parent_shear - resistance) + traction_remainder;
    const double stage_shear = stage.fraction * stage.force / G;
    stage.stress = predictor - 2.*G * (stage_shear - B * parent_shear) * slip_plane;
    stage.cohesive_strain = stage_shear * slip_plane;
    stage.rate = (stage_shear / B - parent_shear) / half_duration;
    stage.driving = stage.driving_factor * stage.force * stage.force / (2.*G);
    return stage;
  }

  CohesiveMaxwell::Variation CohesiveMaxwell::derivative(const PointResponse &stage,
                                                         const Stress        &strain_variation,
                                                         const double        phase_variation,
                                                         const double        half_duration,
                                                         const Stress        &slip_plane,
                                                         const double        resistance_variation) const
  {
    const double B = relaxation(half_duration);
    const double traction_variation = 2.*G * B * (strain_variation * slip_plane) - resistance_variation;
    const double shear_variation = (stage.fraction * traction_variation + stage.fraction_derivative * stage.force * phase_variation) / G;
    return
    {
      2.*G *(B * strain_variation - shear_variation * slip_plane),
      shear_variation / (B * half_duration),
      (stage.driving_factor_derivative *stage.force *stage.force *phase_variation
      + 2.*stage.driving_factor *stage.force * traction_variation) / (2.*G)
    };
  }

  double CohesiveMaxwell::cohesive_energy(const Stress &cohesive_strain, const double phase) const
  {
    const double phi = physical_phase(phase);
    if (phi == 0.)
      {
        AssertThrow(cohesive_strain.norm() == 0., ExcMessage("Intact material point carries cohesive strain."));
        return 0.;
      }
    return degradation(phi).cohesive_energy(G * (cohesive_strain * cohesive_strain));
  }
}
