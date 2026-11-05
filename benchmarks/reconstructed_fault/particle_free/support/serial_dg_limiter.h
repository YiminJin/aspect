/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#ifndef ASPECT_BENCHMARK_SERIAL_DG_LIMITER_H
#define ASPECT_BENCHMARK_SERIAL_DG_LIMITER_H
#include "../../../../tests/phase_field_fault_test_access.h"
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/material_model/phase_field_fault/tensor_history_internal.h>
#include <set>

namespace aspect
{
namespace ParticleFreeBenchmark
{
/** Retained PF4 serial experiment. No production material dependency. */
template <int dim> class SerialDGLimiter
{
  using Access = MaterialModel::internal::PhaseFieldFaultTestAccess<dim>;
  using Model = MaterialModel::PhaseFieldFault<dim>;
  const SimulatorAccess<dim> &context;
  const Model &model;

public:
  using DGHistory = typename Access::DGHistory;
  using DGTransport = typename Access::DGTransport;
  using DGField = typename Access::DGField;
  using DGComponents
      = std::array<Vector<double>,
                   SymmetricTensor<2, dim>::n_independent_components>;
  struct DGLimiter;
  struct DGLimiterStats;
  SerialDGLimiter (const SimulatorAccess<dim> &context, const Model &model)
      : context (context), model (model)
  {
  }
  std::shared_ptr<DGLimiter>
  make_dg_limiter (const DGTransport &, const LinearAlgebra::BlockVector &,
                   bool constrained_candidates = false) const;
  void limit_dg_transport (const DGLimiter &, const DGComponents &,
                           DGComponents &, const DGField &, double,
                           DGLimiterStats &) const;
  std::shared_ptr<DGHistory>
  advance_dg_transport (const DGTransport &, const DGHistory &,
                        const DGField &, const DGField &, double, double,
                        const DGLimiter *, DGLimiterStats *) const;
};
/** Fixed affine Cartesian stencil for the optional nonpublishing experiment.
 * Neighbor polynomials are extrapolated to every target support point. */
template <int dim> struct SerialDGLimiter<dim>::DGLimiter
{
  enum class Detector
  {
    fu_shu,
    kxrcf,
    force,
    scaling
  };
  struct Candidate
  {
    std::vector<types::global_dof_index> indices;
    FullMatrix<double> extrapolate;
    // Neighbor-basis averages on every cell in the compact patch and
    // precomputed KKT corrections, indexed by the useful-neighbor mask.
    std::vector<Vector<double>> patch_means;
    std::vector<FullMatrix<double>> constrained_corrections;
  };
  struct FacePoint
  {
    Point<dim> point;
    double weight;
    Vector<double> inside, outside;
    std::vector<types::global_dof_index> neighbor;
  };
  struct Cell
  {
    std::vector<Candidate> candidates;
    std::vector<FacePoint> inflow;
    double h_power;
    double measure;
  };
  std::shared_ptr<typename DGHistory::Space> space;
  std::vector<Cell> cells;
  std::vector<Point<dim>> mesh_vertices;
  FullMatrix<double> values;
  FullMatrix<double> scaling_values;
  std::array<FullMatrix<double>, 5> derivatives;
  Vector<double> mean;
  std::vector<double> quadrature_weights;
  Detector detector = Detector::fu_shu;
  double threshold = .1;
  // Diagnostic Q2 adaptation of constrained neighbor-cell fitting.
  bool constrained_candidates = false;
  // Diagnostic scalar envelope for the selected proportional tensor pulse.
  // No production stress bound: scalar = <A,reference>_3/|reference|_3^2.
  SymmetricTensor<2, dim> scaling_reference;
  double scaling_lower = -1., scaling_upper = 1.;
};

template <int dim> struct SerialDGLimiter<dim>::DGLimiterStats
{
  struct Indicator
  {
    double mean = 0., polynomial = 0., kxrcf = 0., mean_error = 0.;
    bool limited = false;
    std::vector<double> weights, beta;
    std::vector<unsigned int> useful_masks;
    std::vector<std::vector<SymmetricTensor<2, dim>>> candidate_values;
    SymmetricTensor<2, dim> physical_mean_before, physical_mean_after;
    double theta = 1.;
  };
  struct NormChange
  {
    double time = 0., before = 0., after = 0., change = 0.;
    unsigned int positive_cells = 0, negative_cells = 0;
    unsigned int means_outside_envelope = 0;
    double maximum_mean_excess = 0.;
    double maximum_physical_mean_change = 0.;
    SymmetricTensor<2, dim> integral_mean_change;
  };
  std::vector<Indicator> last;
  unsigned long long visits = 0, limited = 0, mean_flagged = 0,
                     safeguard_only = 0;
  double detector_seconds = 0., reconstruction_seconds = 0.,
         maximum_mean_error = 0.;
  // Optional diagnostic measurements; do not affect any reconstruction.
  bool measure_norm_changes = false, capture_first_call = false;
  double norm_measurement_seconds = 0.;
  std::vector<NormChange> norm_changes;
  DGComponents first_incoming, first_outgoing;
  bool measure_physical_means = false;
  std::set<unsigned int> trace_cells;
  // Called after all cells are reconstructed, so neighbor flux changes
  // use the complete separate output rather than partially written data.
  std::function<void (const DGComponents &, const DGComponents &, double,
                      const std::vector<Indicator> &)>
      stage_observer;
};
}
}
#endif
