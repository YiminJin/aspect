/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#ifndef _aspect_material_model_rheology_cohesive_maxwell_h
#define _aspect_material_model_rheology_cohesive_maxwell_h
#include <aspect/global.h>
#include <aspect/phase_field/degradation.h>

namespace aspect::GrowingFault
{
  // Plane strain sets the out-of-plane strain increment to zero. Taking the
  // three-dimensional deviator still produces a zz stress and strain.
  using Stress = SymmetricTensor<2,3>;

  /** Local constitutive parent: tension-positive deviatoric stress in Pa and
   * recoverable cohesive strain, with zz retained in plane strain. */
  struct MaxwellState
  {
    Stress stress, cohesive_strain;
  };

  /** Evaluate the analytic endpoint for FE phase samples within roundoff of [0,1]. */
  double physical_phase(double value);

  struct PointResponse
  {
    Stress stress, cohesive_strain;
    double rate;                      //!< Signed native engineering shear rate (1/s).
    double driving;                   //!< Local phase driving force (Pa).
    double fraction, fraction_derivative;
    double force;                     //!< Shear predictor minus strip resistance (Pa).
    double driving_factor, driving_factor_derivative;
  };
  /** Evaluate stress (Pa), recoverable strain, shear rate (1/s) and damage
   * driving force (Pa) at the stage from the immutable material parent. */
  struct CohesiveMaxwell
  {
    double G;   //!< Shear modulus (Pa).
    double eta; //!< Maxwell viscosity (Pa s).
    double m;   //!< Dimensionless cohesive degradation magnitude.
    double relaxation(double half_duration) const;
    /** Energetic degradation used by the point update and prospective plane
     * search at this native phase. Handles roundoff at the physical limits. */
    PhaseField::CohesiveDegradation degradation(double phase) const;
    /**
     * Return the native response at the midpoint, starting from the accepted
     * endpoint tensors. The strain increment is the 3-D deviator of the
     * symmetric gradient of the half-step displacement. The slip plane is sym(t ⊗ n),
     * so contracting stress with it gives signed shear traction in Pa.
     * The resistance is the signed full-strip traction in Pa. At endpoint
     * recovery, traction_remainder restores the low part of that difference.
     */
    PointResponse response(const MaxwellState &parent,
                           const Stress       &strain_increment,
                           double             phase,
                           double             half_duration,
                           const Stress       &slip_plane,
                           double             resistance,
                           double             traction_remainder = 0.) const;
    struct Variation
    {
      Stress stress;
      double rate, driving;
    };
    /** Directional derivative at fixed parent, half-duration and slip plane. */
    Variation derivative(const PointResponse &stage,
                         const Stress        &strain_variation,
                         double              phase_variation,
                         double              half_duration,
                         const Stress        &slip_plane,
                         double              resistance_variation) const;
    double cohesive_energy(const Stress &cohesive_strain, double phase) const;
  };


}
#endif
