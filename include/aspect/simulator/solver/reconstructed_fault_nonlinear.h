/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#ifndef _aspect_simulator_solver_reconstructed_fault_nonlinear_h
#define _aspect_simulator_solver_reconstructed_fault_nonlinear_h

#include <aspect/reconstructed_fault/surface_system.h>

#include <functional>
#include <string>

namespace aspect
{
  namespace internal
  {
    bool
    reconstructed_fault_locally_at_lower_bound(const double value,
                                                const double minimum);

    ReconstructedFaultActiveSet
    make_reconstructed_fault_inactive_set(
      const ReconstructedFaultVector &slip_rate);

    unsigned int
    update_reconstructed_fault_active_set(
      const ReconstructedFaultVector &slip_rate,
      const ReconstructedFaultVector &direction,
      const double minimum,
      ReconstructedFaultActiveSet &active_set);

    double
    reconstructed_fault_maximum_step_length(
      const ReconstructedFaultVector &slip_rate,
      const ReconstructedFaultVector &direction,
      const ReconstructedFaultActiveSet &active_set,
      const double minimum);

    /** Affine non-contact trial, with exact absolute V_min at bound contact. */
    double reconstructed_fault_trial_value(const double value,
                                          const double direction,
                                          const double step_length,
                                          const double minimum);

    double
    reconstructed_fault_residual_scale(const double initial_norm,
                                       const double reference_norm,
                                       const double floor_factor);

    /**
     * Bulk residual scale of a machine-precision perturbation of the first
     * physical iterate, expressed in solver coordinates (p/pressure_scaling).
     * Uses absolute row sums and block maxima, not an observed residual floor.
     * The vector must be owned, not ghosted; all ranks participate.
     */
    double
    reconstructed_fault_bulk_precision_scale(
      const LinearAlgebra::BlockSparseMatrix &matrix,
      const LinearAlgebra::BlockVector &solver_state,
      const MPI_Comm communicator);

    double
    normalized_reconstructed_fault_residual(const double residual_norm,
                                            const double scale,
                                            const std::string &block_name);

    struct ReconstructedFaultLineSearchResult
    {
      bool accepted = false;
      double step_length = 0.0;
      unsigned int rejected_candidates = 0;
    };

    ReconstructedFaultLineSearchResult
    reconstructed_fault_armijo_line_search(
      const double maximum_step_length,
      const unsigned int maximum_reductions,
      const double current_merit,
      const std::function<double (double)> &evaluate_trial_merit,
      const std::function<void (double)> &accept_trial);
  }
}

#endif
