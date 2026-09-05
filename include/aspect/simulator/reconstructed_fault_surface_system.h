/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#ifndef _aspect_simulator_reconstructed_fault_surface_system_h
#define _aspect_simulator_reconstructed_fault_surface_system_h

#include <aspect/simulator_access.h>

namespace aspect
{
  namespace MaterialModel
  {
    template <int dim>
    class PhaseFieldFault;
  }

  /** Surface residual assembled on the replicated reconstructed faults. */
  struct ReconstructedFaultSurfaceResidual
  {
    std::vector<std::vector<double>> values;
    double weighted_rms = numbers::signaling_nan<double>();
    std::vector<double> per_fault_weighted_rms;
  };


  /**
   * Assemble and solve the particle/Q1 reconstructed-fault surface system.
   * Bulk cell weak-form assembly is intentionally outside this class.
   */
  template <int dim>
  class ReconstructedFaultSurfaceSystem : public SimulatorAccess<dim>
  {
    public:
      using FaultVector = std::vector<std::vector<double>>;

      explicit ReconstructedFaultSurfaceSystem(const Simulator<dim> &simulator);
      ~ReconstructedFaultSurfaceSystem();

      ReconstructedFaultSurfaceResidual
      evaluate_surface_residual(const LinearAlgebra::BlockVector &bulk_state,
                                const FaultVector &slip_rate) const;

      /** Assemble and replace the current residual, K_V, and its factors. */
      const ReconstructedFaultSurfaceResidual &
      linearize_surface_system(const LinearAlgebra::BlockVector &bulk_state,
                               const FaultVector &slip_rate);

      /** Apply the inverse of the K_V most recently assembled above. */
      void
      solve_surface_jacobian(const FaultVector &rhs,
                             FaultVector &solution) const;

    private:
      struct SurfaceLinearization;
      const MaterialModel::PhaseFieldFault<dim> &phase_field_fault;
      std::unique_ptr<SurfaceLinearization> surface_linearization;
  };
}

#endif
