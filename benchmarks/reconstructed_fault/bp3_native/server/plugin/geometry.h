#ifndef ASPECT_BENCHMARK_BP3_GEOMETRY_H
#define ASPECT_BENCHMARK_BP3_GEOMETRY_H

#include <aspect/reconstructed_fault/utilities.h>
#include <aspect/simulator_access.h>

namespace BP3
{
  /** Validated stationary geometry, independent of prescribed vertex ordering.
   * The native manager retains the original anchors and resampling ownership. */
  struct Geometry
  {
    dealii::Point<2> origin, extent, upper, lower;
    dealii::Tensor<1,2> tangent, normal;
    double length, sine, cosine, dip, peak_phase, weakening_length;
    int shear_sense;
    aspect::PrescribedInitialFault<2> prescribed;
    std::string identity;

    double down_dip(double x, double y) const;
    double signed_normal(double x, double y) const;
    double minimum_cell_distance(const dealii::Point<2> &center,
                                 const dealii::Point<2> &half_width) const;
    std::vector<double> stations() const;
  };

  /** Pure validation/derivation, also used by focused geometry tests. */
  Geometry make_geometry(const std::vector<aspect::PrescribedInitialFault<2>> &faults,
                         const dealii::Point<2> &origin,
                         const dealii::Point<2> &extent,
                         double activation_threshold, double upper_phase_threshold,
                         double weakening_length, bool allow_truncated_transition);

  /** Every consuming plugin calls this during its own parameter parsing.
   * Reads once collectively, before field evaluation/refinement/constraints;
   * later callers only check the immutable configuration. */
  template <int dim>
  void configure_geometry(const aspect::SimulatorAccess<dim> &sim,
                          dealii::ParameterHandler &prm);

  const Geometry &geometry();
}
#endif
