/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#ifndef _aspect_reconstructed_fault_surface_system_h
#define _aspect_reconstructed_fault_surface_system_h

#include <aspect/simulator_access.h>

#include <deal.II/grid/grid_tools_cache.h>

#include <memory>
#include <vector>

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


  using ReconstructedFaultVector = std::vector<std::vector<double>>;


  /** Bound-active reconstructed-fault vertices, indexed by fault and vertex. */
  using ReconstructedFaultActiveSet = std::vector<std::vector<bool>>;


  /** Semantic solve used by reconstructed-fault condensation. */
  template <int dim>
  class ReconstructedFaultSurfaceLinearSolve
  {
    public:
      virtual ~ReconstructedFaultSurfaceLinearSolve() = default;

      /** Overwrite @p solution with the solution of the current surface system. */
      virtual void
      solve(const ReconstructedFaultVector &rhs,
            ReconstructedFaultVector &solution) const = 0;
  };


  /**
   * Assemble and solve the particle/Q1 reconstructed-fault surface system.
   * Bulk cell weak-form assembly is intentionally outside this class.
   */
  template <int dim>
  class ReconstructedFaultSurfaceSystem : public SimulatorAccess<dim>,
    public ReconstructedFaultSurfaceLinearSolve<dim>
  {
    public:
      using FaultVector = ReconstructedFaultVector;

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
      solve(const FaultVector &rhs,
            FaultVector &solution) const override;

      /**
       * Build a semantic inverse of the principal free block of the current
       * K_V. Active rows and columns are disconnected, active right-hand-side
       * entries are ignored, and active solution entries are exactly zero.
       */
      std::unique_ptr<ReconstructedFaultSurfaceLinearSolve<dim>>
      create_restricted_linear_solve(
        const ReconstructedFaultActiveSet &active_set) const;

      /** Overwrite @p result with the current K_V applied to @p direction. */
      void
      apply_surface_jacobian(const FaultVector &direction,
                             FaultVector &result) const;

      /**
       * Return the consistent-Q1 RMS norm of a weak surface residual after
       * projecting out active vertices.
       */
      double
      surface_residual_rms(
        const ReconstructedFaultSurfaceResidual &residual,
        const ReconstructedFaultActiveSet &active_set) const;

      /**
       * Overwrite @p result with G applied to a full-system direction whose
       * pressure component is in physical units.
       */
      void
      apply_G(const LinearAlgebra::BlockVector &physical_bulk_direction,
              FaultVector &result) const;

      /** Generation of the current surface-system state. */
      unsigned int
      get_linearization_generation() const;

    private:
      struct SurfaceAssembly;
      struct SurfaceLinearization;

      SurfaceAssembly
      assemble_surface_system(const LinearAlgebra::BlockVector &bulk_state,
                              const FaultVector &slip_rate,
                              const bool assemble_jacobian) const;

      const MaterialModel::PhaseFieldFault<dim> &phase_field_fault;
      GridTools::Cache<dim> grid_cache;
      std::unique_ptr<SurfaceLinearization> surface_linearization;
      unsigned int linearization_generation = 0;
  };
}

#endif
