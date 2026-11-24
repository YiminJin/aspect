/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#ifndef _aspect_growing_fault_section_response_h
#define _aspect_growing_fault_section_response_h
#include <aspect/global.h>

namespace aspect::GrowingFault
{
  using Stress = SymmetricTensor<2,3>;

  /** A stage is the midpoint of a physical step. Tensors retain zz in plane
   * strain. This state belongs to one native point on the fixed material mesh. */
  struct PointHistory
  {
    Stress stress, cohesive_strain;
    double signed_shear=0., absolute_shear=0.;
    template <class Archive> void serialize(Archive &ar, const unsigned int)
    {
      ar &stress &cohesive_strain &signed_shear &absolute_shear;
    }
  };

  /** Evaluate the analytic endpoint for FE phase samples within roundoff of [0,1]. */
  double physical_phase(double value);

  struct PointResponse
  {
    Stress stress, cohesive_strain;
    double rate, driving, fraction, fraction_derivative, force, driving_factor, driving_factor_derivative;
  };
  /** Evaluate stress (Pa), recoverable strain, shear rate (1/s) and damage
   * driving force (Pa) at the stage from the immutable material parent. */
  struct CohesiveMaxwell
  {
    double G, eta, m;
    double relaxation(double h) const;
    /** Evaluate a native point at the fixed midpoint iterate. At publication
     * traction_remainder (Pa) recovers the low part of the driving traction
     * lost when subtracting the strip resistance. */
    PointResponse response(const PointHistory &, const Stress &increment, double phase,
                           double h, const Stress &plane, double resistance, double traction_remainder=0.) const;
    struct Variation
    {
      Stress stress;
      double rate, driving;
    };
    Variation derivative(const PointResponse &, const Stress &increment, double dphase,
                         double h, const Stress &plane, double dresistance) const;
    double cohesive_energy(const Stress &strain, double phase) const;
  };


}
#endif
