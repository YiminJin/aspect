/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#ifndef _aspect_growing_fault_section_response_h
#define _aspect_growing_fault_section_response_h
#include <aspect/global.h>
#include <aspect/material_model/rheology/fault_resistance.h>

namespace aspect::GrowingFault
{
  using Stress = SymmetricTensor<2,3>;
  using MaterialModel::Rheology::SectionRoot;

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

  /** Replicated, ordered material-strip history. Slip is signed and in metres. */
  struct SectionHistory
  {
    double slip=0., angle=0., rate=0., resistance=0.;
    bool formed=false;
    template <class Archive> void serialize(Archive &ar, const unsigned int)
    {
      ar &slip &angle &rate &resistance &formed;
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

  /** Rank-local weighted strip integrals followed by a collective sum. The
   * adjoint return at point q is the resistance of its unique material strip. */
  struct SectionTransfer
  {
    explicit SectionTransfer(unsigned int n) : free_rate(n), mobility(n) {}
    void add(unsigned int strip, double area_over_length, double rate, double point_mobility);
    void sum(MPI_Comm);
    std::vector<double> free_rate, mobility;
  };
}
#endif
