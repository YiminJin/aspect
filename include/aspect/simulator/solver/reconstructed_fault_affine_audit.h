/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#ifndef ASPECT_RECONSTRUCTED_FAULT_AFFINE_AUDIT_H
#define ASPECT_RECONSTRUCTED_FAULT_AFFINE_AUDIT_H
#include <aspect/simulator.h>
#include <aspect/simulator/assemblers/reconstructed_fault_stokes.h>
#include <aspect/simulator/reconstructed_fault_residual_audit.h>
#include <aspect/simulator/solver/reconstructed_fault_nonlinear.h>
#include <aspect/simulator/solver/stokes_operators.h>
#include <iomanip>
#include <sstream>

namespace aspect
{
/** Observational affine/load audit, retaining original assembly order and
 * scratch restoration. It never supplies an accepted Newton candidate. */
template <int dim> struct Simulator<dim>::ReconstructedFaultAffineAudit
{
  using FaultVector = ReconstructedFaultVector;
  Simulator &sim;
  LinearAlgebra::BlockSparseMatrix audit_matrix;
  LinearAlgebra::BlockVector audit_rhs, audit_unknowns, audit_frozen;
  ReconstructedFaultAffineAudit (Simulator &sim, const bool enabled,
                                 const LinearAlgebra::BlockVector &working_x,
                                 const FaultVector &slip_rate,
                                 const ReconstructedFaultActiveSet &active_set)
      : sim (sim)
  {
    if (!enabled)
      return;

    audit_matrix.reinit (2, 2);
    for (unsigned int i = 0; i < 2; ++i)
      for (unsigned int j = 0; j < 2; ++j)
        audit_matrix.block (i, j).copy_from (sim.system_matrix.block (i, j));
    audit_matrix.collect_sizes ();
    audit_rhs = sim.system_rhs;
    audit_unknowns = audit_channel (
        working_x, slip_rate, internal::FaultResidualAuditChannel::unknowns);
    audit_frozen = audit_channel (working_x, slip_rate,
                                  internal::FaultResidualAuditChannel::frozen);
    sim.system_rhs = audit_rhs;
    unsigned int active = 0, total = 0;
    for (const auto &fault : active_set)
      for (const bool value : fault)
        {
          active += value;
          ++total;
        }
    sim.pcout << "      Affine audit active=" << active
              << " free=" << total - active << std::endl;
  }
  void
  restore_audit_matrix ()
  {

    for (unsigned int i = 0; i < 2; ++i)
      for (unsigned int j = 0; j < 2; ++j)
        sim.system_matrix.block (i, j).copy_from (audit_matrix.block (i, j));
  }
  LinearAlgebra::BlockVector
  audit_channel (const LinearAlgebra::BlockVector &x, const FaultVector &v,
                 const internal::FaultResidualAuditChannel channel)
  {

    internal::fault_residual_audit_channel = channel;
    try
      {
        sim.evaluate_reconstructed_fault_coupled_residual (x, v);
      }
    catch (...)
      {
        internal::fault_residual_audit_channel
            = internal::FaultResidualAuditChannel::normal;
        throw;
      }
    internal::fault_residual_audit_channel
        = internal::FaultResidualAuditChannel::normal;
    restore_audit_matrix ();
    return LinearAlgebra::BlockVector (sim.system_rhs);
  }
  void
  evaluate (const double step_length,
            const LinearAlgebra::BlockVector &owned_trial_x,
            const LinearAlgebra::BlockVector &trial_x,
            const LinearAlgebra::BlockVector &working_x,
            const LinearAlgebra::BlockVector &bulk_direction,
            const FaultVector &trial_slip_rate, const FaultVector &slip_rate)
  {

    // Use the represented, pressure-normalized update, with
    // homogeneous constraint rows removed and p in solver units.
    const LinearAlgebra::BlockVector fresh_rhs (sim.system_rhs);
    LinearAlgebra::BlockVector actual (owned_trial_x);
    actual = trial_x;
    LinearAlgebra::BlockVector owned_base (owned_trial_x);
    owned_base = working_x;
    actual -= owned_base;
    sim.current_constraints.set_zero (actual);
    LinearAlgebra::BlockVector dx (bulk_direction), action (bulk_direction);
    dx.block (0) = actual.block (sim.introspection.block_indices.velocities);
    dx.block (1) = actual.block (sim.introspection.block_indices.pressure);
    dx.block (1) /= sim.pressure_scaling;
    internal::StokesBlock (audit_matrix).vmult (action, dx);
    LinearAlgebra::BlockVector requested (bulk_direction);
    internal::StokesBlock (audit_matrix).vmult (requested, bulk_direction);
    requested *= step_length;
    FaultVector dv = trial_slip_rate;
    double v_representation_error = 0.0, dv_max = 0.0;
    for (unsigned int f = 0; f < dv.size (); ++f)
      for (unsigned int i = 0; i < dv[f].size (); ++i)
        {
          const double represented
              = slip_rate[f][i] + (trial_slip_rate[f][i] - slip_rate[f][i]);
          v_representation_error
              = std::max (v_representation_error,
                          std::abs (represented - trial_slip_rate[f][i]));
          dv[f][i] = represented - slip_rate[f][i];
          dv_max = std::max (dv_max, std::abs (dv[f][i]));
        }
    LinearAlgebra::BlockVector b_action (sim.system_rhs);
    sim.reconstructed_fault_stokes_coupling->apply_B (dv, b_action);
    const auto fresh_unknowns
        = audit_channel (trial_x, trial_slip_rate,
                         internal::FaultResidualAuditChannel::unknowns);
    const auto fresh_frozen = audit_channel (
        trial_x, trial_slip_rate, internal::FaultResidualAuditChannel::frozen);
    // A block action may nearly cancel (notably continuity).
    // Reuse the absolute-row-sum precision bound on this
    // represented direction, not the small cancelled action.
    const double action_precision
        = internal::reconstructed_fault_bulk_precision_scale (
            audit_matrix, dx, sim.mpi_communicator);
    // A near-equilibrium correction can be tiny while each
    // fresh residual subtracts large A*x and frozen loads.
    // Bound that assembly roundoff from the absolute matrix
    // operands at the base, as well as at the direction.
    LinearAlgebra::BlockVector base_operand (bulk_direction);
    base_operand.block (0)
        = working_x.block (sim.introspection.block_indices.velocities);
    base_operand.block (1)
        = working_x.block (sim.introspection.block_indices.pressure);
    base_operand.block (1) /= sim.pressure_scaling;
    const double base_precision
        = internal::reconstructed_fault_bulk_precision_scale (
            audit_matrix, base_operand, sim.mpi_communicator);
    for (unsigned int block = 0; block < 2; ++block)
      {
        // sim.system_rhs=-R: affine prediction is rhs-A dx+B dV.
        auto predicted = audit_rhs.block (block);
        predicted -= action.block (block);
        predicted += b_action.block (block);
        auto error = fresh_rhs.block (block);
        error -= predicted;
        auto representation = action.block (block);
        representation -= requested.block (block);
        auto unknowns_error = fresh_unknowns.block (block);
        unknowns_error -= audit_unknowns.block (block);
        unknowns_error += action.block (block);
        unknowns_error -= b_action.block (block);
        auto frozen_error = fresh_frozen.block (block);
        frozen_error -= audit_frozen.block (block);
        const double accuracy
            = 1e-10
                  * std::max ({ audit_rhs.block (block).l2_norm (),
                                action.block (block).l2_norm (),
                                b_action.block (block).l2_norm () })
              + 32.0 * (action_precision + base_precision)
              + 32.0 * std::numeric_limits<double>::epsilon ()
                    * audit_frozen.block (block).l2_norm ();
        AssertThrow (
            error.l2_norm () <= accuracy && frozen_error.l2_norm () == 0.0,
            ExcMessage ("The represented coupled increment failed "
                        "the affine residual consistency regression: block="
                        + std::to_string (block) + ", error="
                        + Utilities::to_string (error.l2_norm (), 17)
                        + ", bound=" + Utilities::to_string (accuracy, 17)
                        + ", frozen change="
                        + Utilities::to_string (frozen_error.l2_norm (), 17)));
        std::ostringstream report;
        report << std::setprecision (17)
               << "      Affine audit alpha=" << step_length
               << " block=" << block
               << " base=" << audit_rhs.block (block).l2_norm ()
               << " predicted=" << predicted.l2_norm ()
               << " fresh=" << fresh_rhs.block (block).l2_norm ()
               << " affine_error=" << error.l2_norm ()
               << " represented_action_error=" << representation.l2_norm ()
               << " unknowns_base=" << audit_unknowns.block (block).l2_norm ()
               << " frozen_base=" << audit_frozen.block (block).l2_norm ()
               << " unknowns_affine_error=" << unknowns_error.l2_norm ()
               << " frozen_load_change=" << frozen_error.l2_norm ()
               << " test_accuracy=" << accuracy
               << " B_dV=" << b_action.block (block).l2_norm ()
               << " dV_max=" << dv_max
               << " V_representation_error=" << v_representation_error;
        sim.pcout << report.str () << std::endl;
      }

    // A separate admissible V probe is never an accepted
    // candidate. It prevents an all-active solve from hiding
    // accidental freezing of BV in either accumulator.
    FaultVector probe_v = trial_slip_rate, probe_dv = trial_slip_rate;
    for (unsigned int f = 0; f < probe_v.size (); ++f)
      for (unsigned int i = 0; i < probe_v[f].size (); ++i)
        {
          probe_v[f][i] += 1e-12 * (1.0 + 0.1 * i);
          probe_dv[f][i]
              = (slip_rate[f][i] + (probe_v[f][i] - slip_rate[f][i]))
                - (slip_rate[f][i]
                   + (trial_slip_rate[f][i] - slip_rate[f][i]));
        }
    sim.evaluate_reconstructed_fault_coupled_residual (trial_x, probe_v);
    const LinearAlgebra::BlockVector probe_rhs (sim.system_rhs);
    sim.reconstructed_fault_stokes_coupling->apply_B (probe_dv, b_action);
    const auto probe_frozen = audit_channel (
        trial_x, probe_v, internal::FaultResidualAuditChannel::frozen);
    AssertThrow (
        b_action.l2_norm () > 0.0,
        ExcMessage ("The nonzero-V probe must have a nonzero weak action."));
    for (unsigned int block = 0; block < 2; ++block)
      {
        auto error = probe_rhs.block (block);
        error -= fresh_rhs.block (block);
        error -= b_action.block (block);
        auto frozen_change = probe_frozen.block (block);
        frozen_change -= fresh_frozen.block (block);
        const double accuracy
            = 1e-10 * b_action.block (block).l2_norm ()
              // Both fresh residuals contain the full A*x.
              // Their subtraction loses its assembly roundoff
              // even when the probe and frozen load are tiny.
              + 32.0 * (action_precision + base_precision)
              + 32.0 * std::numeric_limits<double>::epsilon ()
                    * fresh_frozen.block (block).l2_norm ();
        AssertThrow (
            error.l2_norm () <= accuracy && frozen_change.l2_norm () == 0.0,
            ExcMessage (
                "The nonzero-V probe changed the frozen load "
                "or disagreed with B: block="
                + std::to_string (block) + ", error="
                + Utilities::to_string (error.l2_norm (), 17) + ", bound="
                + Utilities::to_string (accuracy, 17) + ", frozen change="
                + Utilities::to_string (frozen_change.l2_norm (), 17)));
        std::ostringstream report;
        report << std::setprecision (17)
               << "      Affine nonzero-V probe: block=" << block
               << " B_dV=" << b_action.block (block).l2_norm ()
               << " error=" << error.l2_norm ()
               << " frozen_change=" << frozen_change.l2_norm ()
               << " test_accuracy=" << accuracy;
        sim.pcout << report.str () << std::endl;
      }
    // Shadow assembly is observational: restore the actual
    // production residual and the original linearization.
    sim.system_rhs = fresh_rhs;
  }
};
}
#endif
