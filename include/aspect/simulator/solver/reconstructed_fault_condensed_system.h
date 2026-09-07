/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#ifndef _aspect_simulator_solver_reconstructed_fault_condensed_system_h
#define _aspect_simulator_solver_reconstructed_fault_condensed_system_h

#include <aspect/reconstructed_fault/surface_system.h>
#include <aspect/simulator_access.h>

#include <deal.II/lac/affine_constraints.h>

#include <memory>

namespace aspect
{
  namespace Assemblers
  {
    template <int dim>
    class ReconstructedFaultStokes;
  }

  namespace StokesSolver
  {
    /**
     * Solver-side exact condensation of the reconstructed-fault surface block.
     * This object references the canonical simulator-owned coupling components.
     */
    template <int dim>
    class ReconstructedFaultCondensedSystem : public SimulatorAccess<dim>
    {
      public:
        using FaultVector = ReconstructedFaultVector;

        /** A view whose validity is exactly one coupled linearization. */
        class Linearization
        {
          public:
            const ReconstructedFaultSurfaceResidual &
            surface_residual() const;

            /**
             * Return a view of this same A/B/G/K_V linearization using a
             * different semantic K_V inverse. This invalidates this view.
             */
            Linearization
            with_surface_solve(
              const ReconstructedFaultSurfaceLinearSolve<dim> &new_surface_solve) const;

            /** Overwrite @p result with (A-B K_V^{-1} G) @p direction. */
            void
            vmult(LinearAlgebra::BlockVector &result,
                  const LinearAlgebra::BlockVector &direction) const;

            /** Form -R_bulk+B K_V^{-1} R_Gamma. */
            void
            build_condensed_rhs(const LinearAlgebra::BlockVector &bulk_newton_rhs,
                                LinearAlgebra::BlockVector &result) const;

            /** Recover K_V^{-1}(R_Gamma+G delta_x) without changing fault state. */
            void
            recover_slip_rate_increment(
              const LinearAlgebra::BlockVector &bulk_increment,
              FaultVector &slip_rate_increment) const;

            /** Apply the complete uncondensed block Jacobian for verification. */
            void
            apply_uncondensed_jacobian(
              const LinearAlgebra::BlockVector &bulk_direction,
              const FaultVector &slip_rate_direction,
              LinearAlgebra::BlockVector &bulk_result,
              FaultVector &surface_result) const;

            /** Convert a homogeneous solver-scaled Stokes direction to a
             * constrained full-system direction with physical pressure. */
            LinearAlgebra::BlockVector
            make_physical_bulk_direction(
              const LinearAlgebra::BlockVector &solver_direction) const;

          private:
            friend class ReconstructedFaultCondensedSystem<dim>;

            Linearization(
              ReconstructedFaultCondensedSystem<dim> &owner,
              const LinearAlgebra::BlockSparseMatrix &bulk_matrix,
              const ReconstructedFaultSurfaceLinearSolve<dim> &surface_solve,
              const ReconstructedFaultSurfaceResidual &surface_residual,
              const unsigned int generation,
              const unsigned int surface_generation,
              const unsigned int B_generation,
              const std::shared_ptr<const AffineConstraints<double>> &system_constraints,
              const std::shared_ptr<const AffineConstraints<double>> &stokes_constraints);

            void assert_is_current() const;

            LinearAlgebra::BlockVector
            make_constrained_solver_direction(
              const LinearAlgebra::BlockVector &direction) const;

            ReconstructedFaultCondensedSystem<dim> &owner;
            const LinearAlgebra::BlockSparseMatrix &bulk_matrix;
            const ReconstructedFaultSurfaceLinearSolve<dim> &surface_solve;
            const ReconstructedFaultSurfaceResidual &residual;
            const unsigned int generation;
            const unsigned int surface_generation;
            const unsigned int B_generation;
            const std::shared_ptr<const AffineConstraints<double>> homogeneous_system_constraints;
            const std::shared_ptr<const AffineConstraints<double>> homogeneous_stokes_constraints;
        };

        explicit ReconstructedFaultCondensedSystem(const Simulator<dim> &simulator);

        /**
         * Build and publish one mutually consistent A/B/G/K_V linearization.
         * The bulk matrix and, when supplied, @p surface_solve must remain
         * alive and unchanged while the returned view is used. Starting any
         * new coupling-component linearization invalidates that view.
         */
        Linearization
        linearize(
          const LinearAlgebra::BlockSparseMatrix &bulk_matrix,
          const LinearAlgebra::BlockVector &physical_bulk_state,
          const FaultVector &slip_rate,
          const ReconstructedFaultSurfaceLinearSolve<dim> *surface_solve = nullptr);

      private:
        Linearization
        rebind_surface_solve(
          const Linearization &linearization,
          const ReconstructedFaultSurfaceLinearSolve<dim> &surface_solve);

        ReconstructedFaultSurfaceSystem<dim> &surface_system;
        Assemblers::ReconstructedFaultStokes<dim> &stokes_coupling;
        unsigned int active_generation = 0;
    };
  }
}

#endif
