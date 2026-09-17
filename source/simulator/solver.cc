/*
  Copyright (C) 2011 - 2024 by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.

  ASPECT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ASPECT; see the file LICENSE.  If not see
  <http://www.gnu.org/licenses/>.
*/


#include <aspect/simulator.h>
#include <aspect/global.h>
#include <aspect/melt.h>
#include <aspect/simulator/solver/block_stokes_preconditioner.h>
#include <aspect/simulator/solver/stokes_matrix_free.h>
#include <aspect/simulator/solver/stokes_matrix_free_local_smoothing.h>
#include <aspect/simulator/solver/stokes_direct.h>
#include <aspect/mesh_deformation/interface.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/plugins.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/reconstructed_fault/surface_system.h>
#include <aspect/simulator/solver/reconstructed_fault_condensed_system.h>
#include <aspect/simulator/solver/reconstructed_fault_nonlinear.h>
#include <aspect/simulator/solver/reconstructed_fault_linear.h>
#include <aspect/simulator/assemblers/reconstructed_fault_stokes.h>
#include "reconstructed_fault_residual_audit.h"
#include "reconstructed_fault_interface_preconditioner.h"
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>

#include <deal.II/base/signaling_nan.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/trilinos_vector.h>

namespace aspect
{
  namespace internal
  {
    /**
     * Implement multiplication with Stokes part of system matrix. In essence, this
     * object represents a 2x2 block matrix that corresponds to the top left
     * sub-blocks of the entire system matrix (i.e., the Stokes part)
     */
    class StokesBlock
    {
      public:
        /**
         * @brief Constructor
         *
         * @param S The entire system matrix
         */
        StokesBlock (const LinearAlgebra::BlockSparseMatrix  &S)
          : system_matrix(S) {}

        /**
         * Matrix vector product with Stokes block.
         */
        void vmult (LinearAlgebra::BlockVector       &dst,
                    const LinearAlgebra::BlockVector &src) const;

        void Tvmult (LinearAlgebra::BlockVector       &dst,
                     const LinearAlgebra::BlockVector &src) const;

        void vmult_add (LinearAlgebra::BlockVector       &dst,
                        const LinearAlgebra::BlockVector &src) const;

        void Tvmult_add (LinearAlgebra::BlockVector       &dst,
                         const LinearAlgebra::BlockVector &src) const;

        /**
         * Compute the residual with the Stokes block. In a departure from
         * the other functions, the #b variable may actually have more than
         * two blocks so that we can put it a global system_rhs vector. The
         * other vectors need to have 2 blocks only.
         */
        double residual (LinearAlgebra::BlockVector       &dst,
                         const LinearAlgebra::BlockVector &x,
                         const LinearAlgebra::BlockVector &b) const;


      private:

        /**
         * Reference to the system matrix object.
         */
        const LinearAlgebra::BlockSparseMatrix &system_matrix;
    };



    void StokesBlock::vmult (LinearAlgebra::BlockVector       &dst,
                             const LinearAlgebra::BlockVector &src) const
    {
      Assert (src.n_blocks() == 2, ExcInternalError());
      Assert (dst.n_blocks() == 2, ExcInternalError());

      system_matrix.block(0,0).vmult(dst.block(0), src.block(0));
      system_matrix.block(0,1).vmult_add(dst.block(0), src.block(1));

      system_matrix.block(1,0).vmult(dst.block(1), src.block(0));
      system_matrix.block(1,1).vmult_add(dst.block(1), src.block(1));
    }


    void StokesBlock::Tvmult (LinearAlgebra::BlockVector       &dst,
                              const LinearAlgebra::BlockVector &src) const
    {
      Assert (src.n_blocks() == 2, ExcInternalError());
      Assert (dst.n_blocks() == 2, ExcInternalError());

      system_matrix.block(0,0).Tvmult(dst.block(0), src.block(0));
      system_matrix.block(1,0).Tvmult_add(dst.block(0), src.block(1));

      system_matrix.block(0,1).Tvmult(dst.block(1), src.block(0));
      system_matrix.block(1,1).Tvmult_add(dst.block(1), src.block(1));
    }


    void StokesBlock::vmult_add (LinearAlgebra::BlockVector       &dst,
                                 const LinearAlgebra::BlockVector &src) const
    {
      Assert (src.n_blocks() == 2, ExcInternalError());
      Assert (dst.n_blocks() == 2, ExcInternalError());

      system_matrix.block(0,0).vmult_add(dst.block(0), src.block(0));
      system_matrix.block(0,1).vmult_add(dst.block(0), src.block(1));

      system_matrix.block(1,0).vmult_add(dst.block(1), src.block(0));
      system_matrix.block(1,1).vmult_add(dst.block(1), src.block(1));
    }


    void StokesBlock::Tvmult_add (LinearAlgebra::BlockVector       &dst,
                                  const LinearAlgebra::BlockVector &src) const
    {
      Assert (src.n_blocks() == 2, ExcInternalError());
      Assert (dst.n_blocks() == 2, ExcInternalError());

      system_matrix.block(0,0).Tvmult_add(dst.block(0), src.block(0));
      system_matrix.block(1,0).Tvmult_add(dst.block(0), src.block(1));

      system_matrix.block(0,1).Tvmult_add(dst.block(1), src.block(0));
      system_matrix.block(1,1).Tvmult_add(dst.block(1), src.block(1));
    }



    double StokesBlock::residual (LinearAlgebra::BlockVector       &dst,
                                  const LinearAlgebra::BlockVector &x,
                                  const LinearAlgebra::BlockVector &b) const
    {
      Assert (x.n_blocks() == 2, ExcInternalError());
      Assert (dst.n_blocks() == 2, ExcInternalError());

      // compute b-Ax where A is only the top left 2x2 block
      this->vmult (dst, x);
      dst.block(0).sadd (-1, 1, b.block(0));
      dst.block(1).sadd (-1, 1, b.block(1));

      // clear blocks we didn't want to fill
      for (unsigned int block=2; block<dst.n_blocks(); ++block)
        dst.block(block) = 0;

      return dst.l2_norm();
    }

    /**
     * Base class for Schur Complement operators.
     */
    class SchurComplementOperator
    {
      public:
        virtual ~SchurComplementOperator() = default;

        virtual void vmult(LinearAlgebra::Vector &dst,
                           const LinearAlgebra::Vector &src) const=0;
        virtual unsigned int n_iterations() const=0;

    };

    /**
     * This class approximates the Schur Complement inverse operator
     * by S^{-1} = (BC^{-1}B^T)^{-1}(BC^{-1}AD^{-1}B^T)(BD^{-1}B^T)^{-1},
     * which is known as the weighted BFBT method. Here,
     * C^{-1} and D^{-1} are chosen to be the inverse weighted lumped
     * velocity mass matrix.
     */
    template <class PreconditionerMp>
    class WeightedBFBT: public SchurComplementOperator
    {
      public:
        /**
         * Constructor.
         * @param pressure_laplace_matrix Laplace operator on the pressure space. This is how we choose to discretize (BC^{-1}B^T).
         * @param laplace_preconditioner The preconditioner for @p pressure_laplace_matrix
         * @param solver_tolerance The relative solver tolerance for the inner solve
         * @param inverse_lumped_mass_matrix Lumped mass matrix associated with the velocity block
         * @param system_matrix Sparse block matrix storing the Stokes system of the form
         * [A B^T
         *  B 0].
         */
        WeightedBFBT(const LinearAlgebra::SparseMatrix &pressure_laplace_matrix,
                     const PreconditionerMp &mp_preconditioner,
                     const double solver_tolerance,
                     const LinearAlgebra::Vector &inverse_lumped_mass_matrix,
                     const LinearAlgebra::BlockSparseMatrix &system_matrix);

        void vmult(LinearAlgebra::Vector &dst,
                   const LinearAlgebra::Vector &src) const override;

        unsigned int n_iterations() const override;

      private:
        mutable unsigned int n_iterations_;
        const LinearAlgebra::SparseMatrix &pressure_laplace_matrix;
        const PreconditionerMp &laplace_preconditioner;
        const double solver_tolerance;
        const LinearAlgebra::Vector &inverse_lumped_mass_matrix;
        const LinearAlgebra::BlockSparseMatrix &system_matrix;
    };

    template <class PreconditionerMp>
    WeightedBFBT<PreconditionerMp>::WeightedBFBT(
      const LinearAlgebra::SparseMatrix &pressure_laplace_matrix,
      const PreconditionerMp &laplace_preconditioner,
      const double solver_tolerance,
      const LinearAlgebra::Vector &inverse_lumped_mass_matrix,
      const LinearAlgebra::BlockSparseMatrix &system_matrix)
      : n_iterations_ (0),
        pressure_laplace_matrix(pressure_laplace_matrix),
        laplace_preconditioner (laplace_preconditioner),
        solver_tolerance (solver_tolerance),
        inverse_lumped_mass_matrix(inverse_lumped_mass_matrix),
        system_matrix (system_matrix)
    {}


    template <class PreconditionerMp>
    void WeightedBFBT<PreconditionerMp>::vmult(LinearAlgebra::Vector &dst,
                                               const LinearAlgebra::Vector &src) const
    {
      SolverControl solver_control(1000, src.l2_norm() * solver_tolerance);
      PrimitiveVectorMemory<LinearAlgebra::Vector> mem;
      SolverCG<LinearAlgebra::Vector> solver(solver_control, mem);

      try
        {
          LinearAlgebra::Vector utmp;
          utmp.reinit(inverse_lumped_mass_matrix);
          LinearAlgebra::Vector ptmp;
          ptmp.reinit(src);
          LinearAlgebra::Vector wtmp;
          wtmp.reinit(inverse_lumped_mass_matrix);
          {
            SolverControl solver_control(5000, 1e-6 * src.l2_norm(), false, true);
            SolverCG<LinearAlgebra::Vector> solver(solver_control);

            solver.solve(pressure_laplace_matrix,
                         ptmp,
                         src,
                         laplace_preconditioner);
            n_iterations_ += solver_control.last_step();
            system_matrix.block(0,1).vmult(utmp,ptmp);

            utmp.scale(inverse_lumped_mass_matrix);
            system_matrix.block(0,0).vmult(wtmp,utmp);
            wtmp.scale(inverse_lumped_mass_matrix);
            system_matrix.block(1,0).vmult(ptmp,wtmp);

            dst=0;
            solver_control.set_tolerance(1e-6*ptmp.l2_norm());
            solver.solve(pressure_laplace_matrix,
                         dst,
                         ptmp,
                         laplace_preconditioner);
            n_iterations_ += solver_control.last_step();
          }
        }
      // if the solver fails, report the error from processor 0 with some additional
      // information about its location, and throw a quiet exception on all other
      // processors
      catch (const std::exception &exc)
        {
          Utilities::throw_linear_solver_failure_exception("iterative (bottom right) solver",
                                                           "BlockSchurPreconditioner::vmult",
                                                           std::vector<SolverControl> {solver_control},
                                                           exc,
                                                           src.get_mpi_communicator());
        }
    }



    template <class PreconditionerMp>
    unsigned int WeightedBFBT<PreconditionerMp>::n_iterations() const
    {
      return n_iterations_;
    }



    /**
      * This class is used in the implementation of the right preconditioner.
      * Here, the Schur complement is approximated by
      * the pressure mass matrix weighted by the inverse of viscosity and
      * the inverse is computed with a CG solve preconditioned by
      * PreconditionerMp passed to the constructor.
      */
    template <class PreconditionerMp>
    class InverseWeightedMassMatrix: public SchurComplementOperator
    {
      public:
        /**
         * Constructor.
         * @param mp_matrix Matrix approximating S to be used in the inner solve
         * @param mp_preconditioner The preconditioner for @p mp_matrix
         * @param solver_tolerance The relative solver tolerance for the inner solve
         */
        InverseWeightedMassMatrix(const LinearAlgebra::SparseMatrix &mp_matrix,
                                  const PreconditionerMp &mp_preconditioner,
                                  const double solver_tolerance);

        void vmult(LinearAlgebra::Vector &dst,
                   const LinearAlgebra::Vector &src) const override;

        unsigned int n_iterations() const override;

      private:
        mutable unsigned int n_iterations_;
        const LinearAlgebra::SparseMatrix &mp_matrix;
        const PreconditionerMp &mp_preconditioner;
        const double solver_tolerance;
    };



    template <class PreconditionerMp>
    InverseWeightedMassMatrix<PreconditionerMp>::InverseWeightedMassMatrix(
      const LinearAlgebra::SparseMatrix &mp_matrix,
      const PreconditionerMp &mp_preconditioner,
      const double solver_tolerance)
      : n_iterations_ (0),
        mp_matrix (mp_matrix),
        mp_preconditioner (mp_preconditioner),
        solver_tolerance (solver_tolerance)
    {}



    template <class PreconditionerMp>
    void InverseWeightedMassMatrix<PreconditionerMp>::vmult(LinearAlgebra::Vector &dst,
                                                            const LinearAlgebra::Vector &src) const
    {
      // Trilinos reports a breakdown in case src=dst=0, even though it should return
      // convergence without iterating. We simply skip solving in this case.
      if (src.l2_norm() > 1e-50)
        {
          SolverControl solver_control(1000, src.l2_norm() * solver_tolerance);
          PrimitiveVectorMemory<LinearAlgebra::Vector> mem;
          SolverCG<LinearAlgebra::Vector> solver(solver_control, mem);
          try
            {
              dst = 0.0;
              solver.solve(mp_matrix,
                           dst,
                           src,
                           mp_preconditioner);
              n_iterations_ += solver_control.last_step();
            }
          // if the solver fails, report the error from processor 0 with some additional
          // information about its location, and throw a quiet exception on all other
          // processors
          catch (const std::exception &exc)
            {
              Utilities::throw_linear_solver_failure_exception("iterative (bottom right) solver",
                                                               "BlockSchurPreconditioner::vmult",
                                                               std::vector<SolverControl> {solver_control},
                                                               exc,
                                                               src.get_mpi_communicator());
            }
        }
    }



    template <class PreconditionerMp>
    unsigned int InverseWeightedMassMatrix<PreconditionerMp>::n_iterations() const
    {
      return n_iterations_;
    }

  }


  namespace
  {
    using FaultVector = ReconstructedFaultVector;


    template <int dim>
    FaultVector
    current_slip_rate(const ReconstructedFaultManager<dim> &fault_manager)
    {
      FaultVector values(fault_manager.get_faults().size());
      for (unsigned int fault = 0; fault < values.size(); ++fault)
        values[fault] = fault_manager.get_slip_rate(fault);
      return values;
    }


  }



  template <int dim>
  double Simulator<dim>::solve_advection (const AdvectionField &advection_field)
  {
    const unsigned int block_idx = advection_field.block_index(introspection);

    const std::string field_name = (advection_field.is_temperature()
                                    ?
                                    "temperature"
                                    :
                                    introspection.name_for_compositional_index(advection_field.compositional_variable) + " composition");

    const double advection_solver_tolerance = (advection_field.is_temperature()) ? (parameters.temperature_solver_tolerance) : (parameters.composition_solver_tolerance);

    const double tolerance = std::max(1e-50,
                                      advection_solver_tolerance*system_rhs.block(block_idx).l2_norm());

    SolverControl solver_control (1000, tolerance);

    solver_control.enable_history_data();

    SolverGMRES<LinearAlgebra::Vector> solver (solver_control,
                                               SolverGMRES<LinearAlgebra::Vector>::AdditionalData(parameters.advection_gmres_restart_length,true));

    // check if matrix and/or RHS are zero
    // note: to avoid a warning, we compare against numeric_limits<double>::min() instead of 0 here
    if (system_rhs.block(block_idx).l2_norm() <= std::numeric_limits<double>::min())
      {
        pcout << "   Skipping " + field_name + " solve because RHS is zero." << std::endl;
        solution.block(block_idx) = 0;

        // signal successful solver and signal residual of zero
        solver_control.check(0, 0.0);
        signals.post_advection_solver(*this,
                                      advection_field.is_temperature(),
                                      advection_field.compositional_variable,
                                      solver_control);

        return 0;
      }

    AssertThrow(system_matrix.block(block_idx,
                                    block_idx).linfty_norm() > std::numeric_limits<double>::min(),
                ExcMessage ("The " + field_name + " equation can not be solved, because the matrix is zero, "
                            "but the right-hand side is nonzero."));

    LinearAlgebra::PreconditionILU preconditioner;
    // first build without diagonal strengthening:
    build_advection_preconditioner(advection_field, preconditioner, 0.);

    computing_timer.enter_subsection(advection_field.is_temperature() ?
                                     "Solve temperature system" :
                                     "Solve composition system");

    if (advection_field.is_temperature())
      {
        pcout << "   Solving temperature system... " << std::flush;
      }
    else
      {
        pcout << "   Solving "
              << introspection.name_for_compositional_index(advection_field.compositional_variable)
              << " system "
              << "... " << std::flush;
      }

    // Create distributed vector (we need all blocks here even though we only
    // solve for the current block) because we only have an AffineConstraints object
    // for the whole system, current_linearization_point contains our initial guess.
    LinearAlgebra::BlockVector distributed_solution (
      introspection.index_sets.system_partitioning,
      mpi_communicator);
    distributed_solution.block(block_idx) = current_linearization_point.block (block_idx);

    // Temporary vector to hold the residual, we don't need a BlockVector here.
    LinearAlgebra::Vector temp (
      introspection.index_sets.system_partitioning[block_idx],
      mpi_communicator);

    current_constraints.set_zero(distributed_solution);

    // Compute the residual before we solve and return this at the end.
    // This is used in the nonlinear solver.
    const double initial_residual = system_matrix.block(block_idx,block_idx).residual
                                    (temp,
                                     distributed_solution.block(block_idx),
                                     system_rhs.block(block_idx));

    // solve the linear system:
    try
      {
        try
          {
            solver.solve (system_matrix.block(block_idx,block_idx),
                          distributed_solution.block(block_idx),
                          system_rhs.block(block_idx),
                          preconditioner);
          }
        catch (const std::exception &exc)
          {
            // Try rebuilding the preconditioner with diagonal strengthening. In general,
            // this increases the number of iterations needed, but helps in rare situations,
            // especially when SUPG is used.
            pcout << "retrying linear solve with different preconditioner..." << std::endl;
            build_advection_preconditioner(advection_field, preconditioner, 1e-5);
            solver.solve (system_matrix.block(block_idx,block_idx),
                          distributed_solution.block(block_idx),
                          system_rhs.block(block_idx),
                          preconditioner);
          }

      }
    // if the solver fails, report the error from processor 0 with some additional
    // information about its location, and throw a quiet exception on all other
    // processors
    catch (const std::exception &exc)
      {
        // signal unsuccessful solver
        signals.post_advection_solver(*this,
                                      advection_field.is_temperature(),
                                      advection_field.compositional_variable,
                                      solver_control);


        Utilities::throw_linear_solver_failure_exception("iterative advection solver",
                                                         "Simulator::solve_advection",
                                                         std::vector<SolverControl> {solver_control},
                                                         exc,
                                                         mpi_communicator,
                                                         parameters.output_directory+"solver_history.txt");
      }

    // signal successful solver
    signals.post_advection_solver(*this,
                                  advection_field.is_temperature(),
                                  advection_field.compositional_variable,
                                  solver_control);

    current_constraints.distribute (distributed_solution);
    solution.block(block_idx) = distributed_solution.block(block_idx);

    // print number of iterations and also record it in the
    // statistics file
    pcout << solver_control.last_step()
          << " iterations." << std::endl;

    if ((advection_field.is_discontinuous(introspection)
         &&
         (
           (advection_field.is_temperature() && parameters.use_limiter_for_discontinuous_temperature_solution)
           ||
           (!advection_field.is_temperature() && parameters.use_limiter_for_discontinuous_composition_solution[advection_field.compositional_variable])
         )))
      {
        apply_limiter_to_dg_solutions(advection_field);

        computing_timer.leave_subsection(advection_field.is_temperature() ?
                                         "Solve temperature system" :
                                         "Solve composition system");

        // by applying the limiter we have modified the solution to no longer
        // satisfy the equation. Therefore the residual is meaningless and cannot
        // converge to zero in nonlinear iterations. Disable residual computation
        // for this field.
        return 0.0;
      }

    computing_timer.leave_subsection(advection_field.is_temperature() ?
                                     "Solve temperature system" :
                                     "Solve composition system");

    return initial_residual;
  }



  template <int dim>
  std::pair<double,double>
  Simulator<dim>::solve_stokes (LinearAlgebra::BlockVector &solution_vector)
  {
    computing_timer.enter_subsection("Solve Stokes system");

    const std::string name = [&]() -> std::string
    {
      if (parameters.stokes_solver_type == Parameters<dim>::StokesSolverType::block_gmg)
        return stokes_matrix_free->name();
      if (parameters.use_direct_stokes_solver)
        return "direct";
      if (parameters.use_bfbt)
        return "AMG-BFBT";
      return "AMG";
    }();

    pcout << "   Solving Stokes system (" << name << ")... " << std::flush;

    StokesSolver::SolverOutputs outputs;

    if (parameters.stokes_solver_type == Parameters<dim>::StokesSolverType::block_gmg)
      {
        outputs = stokes_matrix_free->solve(system_matrix,
                                            system_rhs,
                                            assemble_newton_stokes_system,
                                            last_pressure_normalization_adjustment,
                                            solution_vector);
      }
    else if (parameters.use_direct_stokes_solver)
      {
        outputs = stokes_direct->solve(system_matrix,
                                       system_rhs,
                                       assemble_newton_stokes_system,
                                       last_pressure_normalization_adjustment,
                                       solution_vector);
      }
    else
      {
        // In the following, we will operate on a vector that contains only
        // the velocity and pressure DoFs, rather than on the full
        // system. Set such a reduced vector up, without any ghost elements.
        // (Worth noting: for direct solvers, this vector has one block,
        // whereas for the iterative solvers, the result has two blocks.)
        LinearAlgebra::BlockVector distributed_stokes_solution (introspection.index_sets.stokes_partitioning,
                                                                mpi_communicator);

        // We will need the Stokes block indices a lot below, shorten their names
        const unsigned int velocity_block_index = introspection.block_indices.velocities;
        const unsigned int pressure_block_index = (parameters.include_melt_transport) ?
                                                  introspection.variable("fluid pressure").block_index
                                                  : introspection.block_indices.pressure;
        (void) velocity_block_index;
        (void) pressure_block_index;

        // Create a view of all constraints that only pertains to the
        // Stokes subset of degrees of freedom. We can then use this later
        // to call constraints.distribute(), constraints.set_zero(), etc.,
        // on those block vectors that only have the Stokes components in
        // them.
        //
        // For the moment, assume that the Stokes degrees are first in the
        // overall vector, so that they form a contiguous range starting
        // at zero. The assertion checks this, but this could easily be
        // generalized if the Stokes block were not starting at zero.
#if DEAL_II_VERSION_GTE(9,6,0)
        {
          Assert (velocity_block_index == 0, ExcNotImplemented());
          if (parameters.use_direct_stokes_solver == false)
            Assert (pressure_block_index == 1, ExcNotImplemented());
        }

        IndexSet stokes_dofs (dof_handler.n_dofs());
        stokes_dofs.add_range (0, distributed_stokes_solution.size());
        const AffineConstraints<double> current_stokes_constraints
          = current_constraints.get_view (stokes_dofs);
#else
        const AffineConstraints<double> &current_stokes_constraints = current_constraints;
#endif

        Assert (distributed_stokes_solution.n_blocks() == 2, ExcInternalError());
        Assert(!parameters.include_melt_transport
               || introspection.variable("compaction pressure").block_index == 1,
               ExcNotImplemented());

        // Many parts of the solver depend on the block layout (velocity = 0,
        // pressure = 1). For example the linearized_stokes_initial_guess vector or the StokesBlock matrix
        // wrapper. Let us make sure that this holds:
        Assert(velocity_block_index == 0, ExcNotImplemented());
        Assert(pressure_block_index == 1, ExcNotImplemented());
        Assert(!parameters.include_melt_transport
               || introspection.variable("compaction pressure").block_index == 1,
               ExcNotImplemented());

        const internal::StokesBlock stokes_block(system_matrix);

        // create a completely distributed vector that will be used for
        // the scaled and denormalized solution and later used as a
        // starting guess for the linear solver
        LinearAlgebra::BlockVector linearized_stokes_initial_guess (introspection.index_sets.stokes_partitioning, mpi_communicator);

        // copy the velocity and pressure from current_linearization_point into
        // the vector linearized_stokes_initial_guess. We need to do the copy because
        // linearized_stokes_variables has a different
        // layout than current_linearization_point, which also contains all the
        // other solution variables.
        if (assemble_newton_stokes_system == false)
          {
            linearized_stokes_initial_guess.block (velocity_block_index) = current_linearization_point.block (velocity_block_index);
            linearized_stokes_initial_guess.block (pressure_block_index) = current_linearization_point.block (pressure_block_index);

            denormalize_pressure (this->last_pressure_normalization_adjustment,
                                  linearized_stokes_initial_guess);
          }
        else
          {
            // The Newton solver solves for updates to variables, for which our best guess is zero when
            // it isn't the first nonlinear iteration. When it is the first nonlinear iteration, we
            // have to assemble the full (non-defect correction) Picard, to get the boundary conditions
            // right in combination with being able to use the initial guess optimally. So we should never
            // end up here when it is the first nonlinear iteration.
            Assert(nonlinear_iteration != 0,
                   ExcMessage ("The Newton solver should not be active in the first nonlinear iteration."));

            linearized_stokes_initial_guess.block (velocity_block_index) = 0;
            linearized_stokes_initial_guess.block (pressure_block_index) = 0;
          }

        current_stokes_constraints.set_zero (linearized_stokes_initial_guess);
        linearized_stokes_initial_guess.block (pressure_block_index) /= pressure_scaling;

        double solver_tolerance = 0;
        if (assemble_newton_stokes_system == false)
          {
            // (ab)use the distributed solution vector to temporarily put a residual in
            // (we don't care about the residual vector -- all we care about is the
            // value (number) of the initial residual). The initial residual is returned
            // to the caller (for nonlinear computations). This value is computed before
            // the solve because we want to compute || A^{k+1} U^k - F^{k+1} ||, which is
            // the nonlinear residual. Because the place where the nonlinear residual is
            // checked against the nonlinear tolerance comes after the solve, the system
            // is solved one time too many in the case of a nonlinear Picard solver.
            outputs.initial_nonlinear_residual = stokes_block.residual (distributed_stokes_solution,
                                                                        linearized_stokes_initial_guess,
                                                                        system_rhs);

            // Note: the residual is computed with a zero velocity, effectively computing
            // || B^T p - g ||, which we are going to use for our solver tolerance.
            // We do not use the current velocity for the initial residual because
            // this would not decrease the number of iterations if we had a better
            // initial guess (say using a smaller timestep). But we need to use
            // the pressure instead of only using the norm of the rhs, because we
            // are only interested in the part of the rhs not balanced by the static
            // pressure (the current pressure is a good approximation for the static
            // pressure).
            const double velocity_residual = system_matrix.block(velocity_block_index,
                                                                 pressure_block_index).residual (distributed_stokes_solution.block(velocity_block_index),
                                                                     linearized_stokes_initial_guess.block(pressure_block_index),
                                                                     system_rhs.block(velocity_block_index));
            const double pressure_residual = system_rhs.block(pressure_block_index).l2_norm();

            solver_tolerance = parameters.linear_stokes_solver_tolerance *
                               std::sqrt(velocity_residual*velocity_residual+pressure_residual*pressure_residual);
          }
        else
          {
            // if we are solving for the Newton update, then the initial guess of the solution
            // vector is the zero vector, and the starting (nonlinear) residual is simply
            // the norm of the (Newton) right hand side vector
            const double velocity_residual = system_rhs.block(velocity_block_index).l2_norm();
            const double pressure_residual = system_rhs.block(pressure_block_index).l2_norm();
            solver_tolerance = parameters.linear_stokes_solver_tolerance *
                               std::sqrt(velocity_residual*velocity_residual+pressure_residual*pressure_residual);

            // as described in the documentation of the function, the initial
            // nonlinear residual for the Newton method is computed by just
            // taking the norm of the right hand side
            outputs.initial_nonlinear_residual = std::sqrt(velocity_residual*velocity_residual+pressure_residual*pressure_residual);
          }
        // Now overwrite the solution vector again with the current best guess
        // to solve the linear system
        distributed_stokes_solution = linearized_stokes_initial_guess;

        // extract Stokes parts of rhs vector
        LinearAlgebra::BlockVector distributed_stokes_rhs(introspection.index_sets.stokes_partitioning);

        distributed_stokes_rhs.block(velocity_block_index) = system_rhs.block(velocity_block_index);
        distributed_stokes_rhs.block(pressure_block_index) = system_rhs.block(pressure_block_index);

        PrimitiveVectorMemory<LinearAlgebra::BlockVector> mem;

        // create Solver controls for the cheap and expensive solver phase
        SolverControl solver_control_cheap (parameters.n_cheap_stokes_solver_steps,
                                            solver_tolerance);

        SolverControl solver_control_expensive (parameters.n_expensive_stokes_solver_steps,
                                                solver_tolerance);

        solver_control_cheap.enable_history_data();
        solver_control_expensive.enable_history_data();

        std::unique_ptr<internal::SchurComplementOperator> schur;
        if (parameters.use_bfbt)
          {
            schur = std::make_unique<internal::WeightedBFBT<LinearAlgebra::PreconditionBase>>(
                      system_preconditioner_matrix.block(pressure_block_index,pressure_block_index),
                      *Mp_preconditioner,
                      parameters.linear_solver_S_block_tolerance,
                      inverse_lumped_mass_matrix.block(velocity_block_index),
                      system_matrix);
          }
        else
          {
            schur = std::make_unique<internal::InverseWeightedMassMatrix<LinearAlgebra::PreconditionBase>>(
                      system_preconditioner_matrix.block(pressure_block_index,pressure_block_index),
                      *Mp_preconditioner,
                      parameters.linear_solver_S_block_tolerance);

          }

        // create a cheap preconditioner that consists of only a single V-cycle
        internal::InverseVelocityBlock<LinearAlgebra::PreconditionAMG, LinearAlgebra::Vector, LinearAlgebra::SparseMatrix> inverse_velocity_block_cheap(
          system_matrix.block(velocity_block_index,velocity_block_index),
          *Amg_preconditioner,
          /* do_solve_A = */ false,
          stokes_A_block_is_symmetric(),
          parameters.linear_solver_A_block_tolerance);
        const internal::BlockSchurPreconditioner<internal::InverseVelocityBlock<LinearAlgebra::PreconditionAMG, LinearAlgebra::Vector, LinearAlgebra::SparseMatrix>,
              internal::SchurComplementOperator, LinearAlgebra::SparseMatrix, LinearAlgebra::BlockVector>
              preconditioner_cheap (
                inverse_velocity_block_cheap,
                *schur,
                system_matrix.block(0,1));

        // create an expensive preconditioner that solves for the A block with CG
        internal::InverseVelocityBlock<LinearAlgebra::PreconditionAMG, LinearAlgebra::Vector, LinearAlgebra::SparseMatrix> inverse_velocity_block_expensive(
          system_matrix.block(velocity_block_index,velocity_block_index),
          *Amg_preconditioner,
          /* do_solve_A = */ true,
          stokes_A_block_is_symmetric(),
          parameters.linear_solver_A_block_tolerance);
        const internal::BlockSchurPreconditioner<internal::InverseVelocityBlock<LinearAlgebra::PreconditionAMG, LinearAlgebra::Vector, LinearAlgebra::SparseMatrix>,
              internal::SchurComplementOperator, LinearAlgebra::SparseMatrix, LinearAlgebra::BlockVector>
              preconditioner_expensive (
                inverse_velocity_block_expensive,
                *schur,
                system_matrix.block(0,1));
        // step 1a: try if the simple and fast solver
        // succeeds in n_cheap_stokes_solver_steps steps or less.
        try
          {
            // if this cheaper solver is not desired, then simply
            // short-cut the attempt at solving with the cheaper
            // preconditioner by throwing an exception right away,
            // which is equivalent to a 'goto' statement to the top of
            // the 'catch' block below
            if (parameters.n_cheap_stokes_solver_steps == 0)
              throw SolverControl::NoConvergence(0,0);

            SolverFGMRES<LinearAlgebra::BlockVector>
            solver(solver_control_cheap, mem,
                   SolverFGMRES<LinearAlgebra::BlockVector>::
                   AdditionalData(parameters.stokes_gmres_restart_length));

            solver.solve (stokes_block,
                          distributed_stokes_solution,
                          distributed_stokes_rhs,
                          preconditioner_cheap);

            // Success. Print all iterations to screen (0 expensive iterations).
            pcout << (solver_control_cheap.last_step() != numbers::invalid_unsigned_int ?
                      solver_control_cheap.last_step():
                      0)
                  << "+0"
                  << " iterations." << std::endl;

            outputs.final_linear_residual = solver_control_cheap.last_value();
          }

        // step 1b: take the stronger solver in case
        // the simple solver failed and attempt solving
        // it in n_expensive_stokes_solver_steps steps or less.
        catch (const SolverControl::NoConvergence &exc)
          {
            // The cheap solver failed or never ran.
            // Print the number of cheap iterations to screen to indicate we
            // try the expensive solver next.
            pcout << (solver_control_cheap.last_step() != numbers::invalid_unsigned_int ?
                      solver_control_cheap.last_step():
                      0) << '+' << std::flush;

            // use the value defined by the user
            // OR
            // at least a restart length of 100 for melt models
            const unsigned int number_of_temporary_vectors = (parameters.include_melt_transport == false ?
                                                              parameters.stokes_gmres_restart_length :
                                                              std::max(parameters.stokes_gmres_restart_length, 100U));

            try
              {
                // if no expensive steps allowed, we have failed, rethrow exception
                if (parameters.n_expensive_stokes_solver_steps == 0)
                  {
                    pcout << "0 iterations." << std::endl;
                    throw exc;
                  }

                SolverFGMRES<LinearAlgebra::BlockVector>
                solver(solver_control_expensive, mem,
                       SolverFGMRES<LinearAlgebra::BlockVector>::
                       AdditionalData(number_of_temporary_vectors));

                solver.solve (stokes_block,
                              distributed_stokes_solution,
                              distributed_stokes_rhs,
                              preconditioner_expensive);
                // Success. Print expensive iterations to screen.
                pcout << solver_control_expensive.last_step()
                      << " iterations." << std::endl;

                outputs.final_linear_residual = solver_control_expensive.last_value();
              }
            // if the solver fails, report the error from processor 0 with some additional
            // information about its location, and throw a quiet exception on all other
            // processors
            catch (const std::exception &exc)
              {
                signals.post_stokes_solver(*this,
                                           schur->n_iterations(),
                                           inverse_velocity_block_cheap.n_iterations()+inverse_velocity_block_expensive.n_iterations(),
                                           solver_control_cheap,
                                           solver_control_expensive);

                std::vector<SolverControl> solver_controls;
                if (parameters.n_cheap_stokes_solver_steps > 0)
                  solver_controls.push_back(solver_control_cheap);

                if (parameters.n_expensive_stokes_solver_steps > 0)
                  solver_controls.push_back(solver_control_expensive);

                // Exit with an exception that describes the underlying cause:
                Utilities::throw_linear_solver_failure_exception("iterative Stokes solver",
                                                                 "Simulator::solve_stokes",
                                                                 solver_controls,
                                                                 exc,
                                                                 mpi_communicator,
                                                                 parameters.output_directory+"solver_history.txt");
              }
          }

        // distribute hanging node and other constraints
        current_stokes_constraints.distribute (distributed_stokes_solution);

        // now rescale the pressure back to real physical units
        distributed_stokes_solution.block(pressure_block_index) *= pressure_scaling;

        // then copy back the solution from the temporary (non-ghosted) vector
        // into the ghosted one with all solution components
        solution_vector.block(velocity_block_index) = distributed_stokes_solution.block(velocity_block_index);
        solution_vector.block(pressure_block_index) = distributed_stokes_solution.block(pressure_block_index);

        // signal successful solver
        signals.post_stokes_solver(*this,
                                   schur->n_iterations(),
                                   inverse_velocity_block_cheap.n_iterations()+inverse_velocity_block_expensive.n_iterations(),
                                   solver_control_cheap,
                                   solver_control_expensive);

        // do some cleanup now that we have the solution
        remove_nullspace(solution_vector, distributed_stokes_solution);

        if (assemble_newton_stokes_system == false)
          outputs.pressure_normalization_adjustment = normalize_pressure(solution_vector);
      }

    last_pressure_normalization_adjustment = outputs.pressure_normalization_adjustment;

    // convert melt pressures:
    if (parameters.include_melt_transport)
      melt_handler->compute_melt_variables(system_matrix,solution_vector,system_rhs);

    computing_timer.leave_subsection("Solve Stokes system");

    return {outputs.initial_nonlinear_residual,
            outputs.final_linear_residual
           };
  }


  template <int dim>
  void
  Simulator<dim>::solve_reconstructed_fault_stokes ()
  {
    unsigned int total_fault_krylov_iterations = 0;
    double minimum_fault_accepted_alpha = 1.0;
    AssertThrow(dim == 2, ExcNotImplemented());
    AssertThrow(newton_handler != nullptr,
                ExcMessage("The coupled reconstructed-fault solver requires "
                           "the Newton solver handler."));

    auto &phase_field_fault =
      Plugins::get_plugin_as_type<MaterialModel::PhaseFieldFault<dim>>(
        *material_model);

    // Complete all frozen constitutive histories before opening the nonlinear
    // V lifecycle. Only a fresh timestep-zero model may initialize missing state.
    phase_field_fault.prepare_reconstructed_fault_mechanical_solve();

    ReconstructedFaultManager<dim> &fault_manager =
      *reconstructed_fault_manager;
    ReconstructedFaultSurfaceSystem<dim> &surface_system =
      *reconstructed_fault_surface_system;
    StokesSolver::ReconstructedFaultCondensedSystem<dim> condensed_system(*this);

    // Keep the production solution immutable during Newton. working_x is the
    // last accepted bulk iterate; only convergence publishes it to solution.
    const LinearAlgebra::BlockVector production_solution(solution);
    const LinearAlgebra::BlockVector saved_linearization_point(
      current_linearization_point);
    LinearAlgebra::BlockVector working_x(current_linearization_point);
    working_x = solution;

    // A pressure shift is a surface-equation gauge only in prescribed-pressure
    // mode. Keep the adjustment private, just like the accepted bulk iterate;
    // failed trials/solves must not alter published normalization bookkeeping.
    const bool normalize_fault_pressure =
      phase_field_fault.uses_adiabatic_friction_pressure();
    double working_pressure_adjustment = last_pressure_normalization_adjustment;

    // Coupled residual evaluation temporarily changes ASPECT assembly controls.
    // Snapshot them once so both success and every exception restore the caller.
    const bool saved_assemble_fault_terms =
      assemble_reconstructed_fault_stokes_terms;
    const bool saved_assemble_newton_system = assemble_newton_stokes_system;
    const bool saved_assemble_newton_matrix = assemble_newton_stokes_matrix;
    const bool saved_rebuild_matrix = rebuild_stokes_matrix;
    const bool saved_rebuild_preconditioner = rebuild_stokes_preconditioner;
    const double saved_derivative_scaling =
      newton_handler->parameters.newton_derivative_scaling_factor;
    const AffineConstraints<double> saved_current_constraints(current_constraints);

    const unsigned int max_nonlinear_iterations =
      (pre_refinement_step < parameters.initial_adaptive_refinement)
      ? std::min(parameters.max_nonlinear_iterations,
                 parameters.max_nonlinear_iterations_in_prerefinement)
      : parameters.max_nonlinear_iterations;
    SolverControl nonlinear_solver_control(max_nonlinear_iterations,
                                           parameters.nonlinear_tolerance);

    struct CoupledResidual
    {
      double bulk_norm;
      ReconstructedFaultSurfaceResidual surface;
    };

    bool nonlinear_state_is_active = false;
    bool terminal_commit_complete = false;
    auto restore_simulator_state = [&]()
    {
      assemble_reconstructed_fault_stokes_terms = saved_assemble_fault_terms;
      assemble_newton_stokes_system = saved_assemble_newton_system;
      assemble_newton_stokes_matrix = saved_assemble_newton_matrix;
      rebuild_stokes_matrix = saved_rebuild_matrix;
      rebuild_stokes_preconditioner = saved_rebuild_preconditioner;
      newton_handler->parameters.newton_derivative_scaling_factor =
        saved_derivative_scaling;
      current_constraints.copy_from(saved_current_constraints);
    };

    try
      {
        // Open manager-owned current/trial V state and freeze the fault-to-QP
        // geometry used throughout this mechanical solve.
        fault_manager.begin_slip_rate_nonlinear_solve();
        nonlinear_state_is_active = true;
        fault_manager.prepare_stokes_qp_projection_cache();

        assemble_reconstructed_fault_stokes_terms = true;
        assemble_newton_stokes_system = true;
        // The PhaseFieldFault bulk Maxwell law is linear in the current
        // strain rate. Its nonlinear fault derivative is represented by the
        // explicit B/K_V/G blocks, not by ASPECT's viscosity-derivative output.
        newton_handler->parameters.newton_derivative_scaling_factor = 0.0;
        set_assemblers();
        // Lift the base iterate with the current physical solution constraints,
        // including changed boundary loading. Never publish this lift before
        // convergence: a failed solve must restore the pre-solve bulk solution.
        assemble_newton_stokes_system = false;
        compute_current_constraints();
        assemble_newton_stokes_system = true;
        LinearAlgebra::BlockVector lifted_solution(
          introspection.index_sets.system_partitioning, mpi_communicator);
        lifted_solution = working_x;
        current_constraints.distribute(lifted_solution);
        working_x = lifted_solution;
        if (normalize_fault_pressure)
          working_pressure_adjustment = normalize_pressure(working_x);

        // Every subsequent assembly is a Newton residual/direction problem.
        // The physical lift is already present in working_x, so eliminate with
        // homogeneous Stokes constraints to avoid subtracting it a second time.
        const types::global_dof_index n_stokes_dofs =
          working_x.block(introspection.block_indices.velocities).size()
          + working_x.block(introspection.block_indices.pressure).size();
        for (const auto &line : current_constraints.get_lines())
          if (line.index < n_stokes_dofs)
            current_constraints.set_inhomogeneity(line.index, 0.0);
        pressure_scaling = compute_pressure_scaling_factor();

        auto evaluate_coupled_residual =
          [&](const LinearAlgebra::BlockVector &bulk_state,
              const FaultVector &slip_rate) -> CoupledResidual
        {
          // Use the very same absolute V in bulk and surface evaluation. A
          // subtract/add reconstruction could lose a small bound-contact value.
          fault_manager.begin_slip_rate_trial();
          bool trial_is_active = true;
          try
            {
              fault_manager.set_slip_rate_trial_values(slip_rate);
              current_linearization_point = bulk_state;
              assemble_newton_stokes_matrix = false;
              rebuild_stokes_preconditioner = false;
              rebuild_stokes_matrix =
                !boundary_velocity_manager
                   .get_prescribed_boundary_velocity_indicators().empty();
              assemble_stokes_system();

              const double velocity_residual =
                system_rhs.block(introspection.block_indices.velocities).l2_norm();
              const double pressure_residual =
                system_rhs.block(introspection.block_indices.pressure).l2_norm();
              CoupledResidual result;
              result.bulk_norm = std::sqrt(
                velocity_residual*velocity_residual
                + pressure_residual*pressure_residual);
              result.surface = surface_system.evaluate_surface_residual(
                bulk_state, slip_rate);
              fault_manager.rollback_slip_rate_trial();
              trial_is_active = false;
              return result;
            }
          catch (...)
            {
              if (trial_is_active)
                fault_manager.rollback_slip_rate_trial();
              throw;
            }
        };

        // Freeze separate dimensional normalization scales for the entire solve.
        // Their floors reuse existing bulk and K_V action scales rather than a
        // reconstructed-fault tuning parameter.
        const FaultVector initial_slip_rate = current_slip_rate(fault_manager);
        const CoupledResidual initial_residual =
          evaluate_coupled_residual(working_x, initial_slip_rate);

        LinearAlgebra::BlockVector bulk_reference(working_x);
        bulk_reference.block(introspection.block_indices.velocities) = 0.0;
        const double aspect_bulk_reference =
          evaluate_coupled_residual(bulk_reference, initial_slip_rate).bulk_norm;

        const double scale_floor_factor = std::max(
          parameters.linear_stokes_solver_tolerance,
          std::sqrt(std::numeric_limits<double>::epsilon()));
        const double bulk_scale =
          internal::reconstructed_fault_residual_scale(
          initial_residual.bulk_norm,
          aspect_bulk_reference,
          scale_floor_factor);
        double surface_scale = numbers::signaling_nan<double>();
        double bulk_precision = 0.0;
        double bulk_convergence_scale = bulk_scale;
        double fault_preconditioner_setup_seconds = 0.;

        auto solve_condensed_system =
          [&](const typename StokesSolver::ReconstructedFaultCondensedSystem<dim>
                      ::Linearization &linearization,
              const ReconstructedFaultActiveSet &active,
              const LinearAlgebra::BlockVector &rhs,
              LinearAlgebra::BlockVector &direction)
        {
          direction = 0.0;
          const double rhs_norm = rhs.l2_norm();
          if (rhs_norm == 0.0)
            return;

          TimerOutput::Scope linear_timer(computing_timer, "Fault: condensed linear solve");

          const double tolerance =
            parameters.linear_stokes_solver_tolerance*rhs_norm;
          const unsigned int budget = std::max(1U,
            parameters.n_cheap_stokes_solver_steps + parameters.n_expensive_stokes_solver_steps);
          PrimitiveVectorMemory<LinearAlgebra::BlockVector> memory;

          std::unique_ptr<internal::SchurComplementOperator> schur;
          if (parameters.use_bfbt)
            schur = std::make_unique<
              internal::WeightedBFBT<LinearAlgebra::PreconditionBase>>(
                system_preconditioner_matrix.block(1,1),
                *Mp_preconditioner,
                parameters.linear_solver_S_block_tolerance,
                inverse_lumped_mass_matrix.block(0),
                system_matrix);
          else
            schur = std::make_unique<
              internal::InverseWeightedMassMatrix<LinearAlgebra::PreconditionBase>>(
                system_preconditioner_matrix.block(1,1),
                *Mp_preconditioner,
                parameters.linear_solver_S_block_tolerance);

          const auto solve_with_velocity_preconditioner = [&](const auto &velocity_preconditioner)
          {
            internal::InverseVelocityBlock<
              std::decay_t<decltype(velocity_preconditioner)>,
              LinearAlgebra::Vector,
              LinearAlgebra::SparseMatrix> inverse_velocity(
                system_matrix.block(0,0),
                velocity_preconditioner,
                true,
                stokes_A_block_is_symmetric(),
                parameters.linear_solver_A_block_tolerance);
            const internal::BlockSchurPreconditioner<
              decltype(inverse_velocity),
              internal::SchurComplementOperator,
              LinearAlgebra::SparseMatrix,
              LinearAlgebra::BlockVector> preconditioner(
                inverse_velocity, *schur, system_matrix.block(0,1));

            // B and G are not assumed adjoints, so the condensed operator is
            // generally nonsymmetric and requires FGMRES rather than CG/MINRES.
            double right_null_error, left_null_error;
            const auto q = linearization.verified_pressure_nullspace(right_null_error, left_null_error);
            LinearAlgebra::BlockVector compatible_rhs(rhs), residual(rhs);
            // Compatibility noise must be both backward-small in the original
            // weak loads and below the unchanged nonlinear bulk target.
            const double compatibility_tolerance = std::min(
              100.*std::numeric_limits<double>::epsilon()
                * std::max({initial_residual.bulk_norm, aspect_bulk_reference, rhs_norm}),
              parameters.nonlinear_tolerance*bulk_scale);
            const double removed_rhs = internal::project_compatible_fault_rhs(
              q, compatibility_tolerance, compatible_rhs);
            const internal::FaultPressureComplementOperator<
              typename StokesSolver::ReconstructedFaultCondensedSystem<dim>::Linearization,
              LinearAlgebra::BlockVector> projected_operator{linearization, q};
            const internal::FaultPressureComplementOperator<
              decltype(preconditioner), LinearAlgebra::BlockVector>
              projected_preconditioner{preconditioner, q, true};
            const internal::FaultInterfacePreconditioner<dim,decltype(projected_preconditioner)>
              interface_preconditioner(projected_preconditioner,linearization,surface_system,active,rhs,pcout);
            const internal::FaultPressureComplementOperator<
              decltype(interface_preconditioner),LinearAlgebra::BlockVector>
              projected_interface{interface_preconditioner,q};

            unsigned int iterations = 0;
            while (iterations < budget)
              {
                SolverControl control(budget-iterations, tolerance);
                control.enable_history_data();
                SolverFGMRES<LinearAlgebra::BlockVector> solver(
                  control, memory,
                  typename SolverFGMRES<LinearAlgebra::BlockVector>::AdditionalData(
                    parameters.stokes_gmres_restart_length));
                bool solver_failed = false;
                try
                  {
                    internal::FaultLinearSection krylov_timer(internal::FaultLinearTiming::krylov_vectors);
                    solver.solve(projected_operator, direction, compatible_rhs, projected_interface);
                  }
                catch (const SolverControl::NoConvergence &)
                  {
                    solver_failed = true;
                  }
                iterations += std::max(1U, control.last_step());
                total_fault_krylov_iterations += std::max(1U, control.last_step());
                internal::project_fault_pressure(q, direction);

                // Arnoldi's residual estimate may disagree with the final vector.
                // Verify C*x-b afresh, retaining raw and null-component diagnostics.
                double raw_residual, residual_null_component;
                const double fresh = internal::fault_true_linear_residual(
                  linearization, q, direction, rhs, residual,
                  raw_residual, residual_null_component);
                std::ostringstream report;
                report << std::setprecision(17)
                       << "      Fault linear solve: iterations=" << iterations
                       << ", estimated=" << control.last_value() << ", fresh=" << fresh
                       << ", target=" << tolerance << ", raw=" << raw_residual
                       << ", rhs null=" << removed_rhs << ", residual null=" << residual_null_component
                       << ", compatibility bound=" << compatibility_tolerance
                       << ", pressure quotient=" << (q.l2_norm() > 0.)
                       << ", right null=" << right_null_error << ", left null=" << left_null_error;
                if (std::getenv("ASPECT_FAULT_NONLINEAR_DIAGNOSTIC"))
                  pcout << report.str() << std::endl;
                else
                  {
                    std::ostringstream progress;
                    progress << "      Fault linear solve: iterations=" << iterations
                             << std::scientific << std::setprecision(6)
                             << ", fresh=" << fresh << ", target=" << tolerance;
                    pcout << progress.str() << std::endl;
                  }
                AssertThrow(std::abs(residual_null_component) <= compatibility_tolerance,
                            ExcMessage("The full condensed residual has a significant pressure incompatibility."));
                if (fresh <= tolerance)
                  {
                    if (!signals.post_reconstructed_fault_linear_solver.empty())
                      signals.post_reconstructed_fault_linear_solver(
                        *this,
                        [&](auto &dst,const auto &src) { projected_operator.vmult(dst,src); },
                        [&](auto &dst,const auto &src) { projected_interface.vmult(dst,src); },
                        [&](auto &dst,const auto &src) { schur->vmult(dst,src); },
                        compatible_rhs,direction,tolerance,budget,fault_preconditioner_setup_seconds);
                    return;
                  }
                if (solver_failed || iterations >= budget)
                  throw SolverControl::NoConvergence(iterations, fresh);
                // Re-enter FGMRES from this vector with its freshly evaluated
                // residual, charging every restart to the same total budget.
              }
          };

          if (!std::getenv("ASPECT_FAULT_VELOCITY_GMG"))
            solve_with_velocity_preconditioner(*Amg_preconditioner);
          else
            {
              AssertThrow(parameters.stokes_velocity_degree==2 && stokes_A_block_is_symmetric(),
                          ExcMessage("The velocity-GMG prototype requires symmetric Q2 bulk Stokes."));
              StokesMatrixFreeHandlerLocalSmoothingImplementation<dim,2> gmg(*this,parameters);
              gmg.initialize_simulator(*this);
              gmg.initialize();
              gmg.with_velocity_preconditioner([&](const auto &cycle)
              {
                // Only adapt vector storage. The approximate velocity inverse
                // still applies the original assembled fine-level A matrix.
                struct Adapter
                {
                  const typename StokesMatrixFreeHandlerLocalSmoothingImplementation<dim,2>::VelocityCycle &cycle;
                  mutable dealii::LinearAlgebra::distributed::Vector<double> input,output;
                  void vmult(LinearAlgebra::Vector &dst,const LinearAlgebra::Vector &src) const
                  {
                    internal::ChangeVectorTypes::copy(input,src);
                    cycle.vmult(output,input);
                    internal::ChangeVectorTypes::copy(dst,output);
                  }
                } adapter{cycle,
                  dealii::LinearAlgebra::distributed::Vector<double>(rhs.block(0).locally_owned_elements(),mpi_communicator),
                  dealii::LinearAlgebra::distributed::Vector<double>(rhs.block(0).locally_owned_elements(),mpi_communicator)};
                solve_with_velocity_preconditioner(adapter);
              });
            }
        };

        for (nonlinear_iteration = 0;
             nonlinear_iteration < max_nonlinear_iterations;
             ++nonlinear_iteration)
          {
            // Assemble mutually consistent A, R_bulk, frozen B, G, K_V, and
            // R_Gamma at the current accepted pair (working_x,current V).
            current_linearization_point = working_x;
            assemble_newton_stokes_matrix = true;
            rebuild_stokes_matrix = true;
            rebuild_stokes_preconditioner = true;
            assemble_stokes_system();
            const auto preconditioner_start=internal::FaultLinearTiming::Clock::now();
            build_stokes_preconditioner();
            fault_preconditioner_setup_seconds=std::chrono::duration<double>(
              internal::FaultLinearTiming::Clock::now()-preconditioner_start).count();

            const FaultVector slip_rate = current_slip_rate(fault_manager);
            internal::FaultLinearProfile linear_profile(pcout, timestep_number, nonlinear_iteration);
            using CondensedLinearization =
              typename StokesSolver::ReconstructedFaultCondensedSystem<dim>
                ::Linearization;
            auto linearization = std::make_unique<CondensedLinearization>(
              condensed_system.linearize(system_matrix, working_x, slip_rate));

            ReconstructedFaultActiveSet active_set =
              fault_manager.prescribed_slip_rate_mask();
            std::unique_ptr<ReconstructedFaultSurfaceLinearSolve<dim>>
              restricted_surface_solve;
            // Prescribed rates are already lifted into the base iterate. Their
            // perturbations vanish, so condensation uses K_FF^{-1} from the
            // first solve, not after an unrestricted direction has been taken.
            bool has_prescribed_vertices = false;
            for (unsigned int f = 0; f < active_set.size(); ++f)
              for (unsigned int v = 0; v < active_set[f].size(); ++v)
                if (active_set[f][v])
                  {
                    has_prescribed_vertices = true;
                    AssertThrow(slip_rate[f][v] >= phase_field_fault.minimum_fault_slip_rate(),
                                ExcMessage("Prescribed V is below the material's minimum slip rate."));
                  }
            if (has_prescribed_vertices)
              {
                restricted_surface_solve = surface_system.create_restricted_linear_solve(active_set);
                linearization = std::make_unique<CondensedLinearization>(
                  linearization->with_surface_solve(*restricted_surface_solve));
              }
            LinearAlgebra::BlockVector bulk_rhs(
              introspection.index_sets.stokes_partitioning, mpi_communicator);
            LinearAlgebra::BlockVector bulk_direction(
              introspection.index_sets.stokes_partitioning, mpi_communicator);
            FaultVector slip_rate_direction;

            // Projected Newton solve: start free, add only at-bound vertices
            // whose direction is outward, and rebuild only K_FF^{-1} until stable.
            while (true)
              {
                linearization->build_condensed_rhs(system_rhs, bulk_rhs);
                solve_condensed_system(*linearization, active_set, bulk_rhs, bulk_direction);
                linearization->recover_slip_rate_increment(
                  bulk_direction, slip_rate_direction);

                const unsigned int n_new_active_vertices =
                  internal::update_reconstructed_fault_active_set(
                    slip_rate,
                    slip_rate_direction,
                    phase_field_fault.minimum_fault_slip_rate(),
                    active_set);
                if (n_new_active_vertices == 0)
                  break;

                auto new_surface_solve =
                  surface_system.create_restricted_linear_solve(active_set);
                auto new_linearization =
                  std::make_unique<CondensedLinearization>(
                    linearization->with_surface_solve(*new_surface_solve));
                linearization = std::move(new_linearization);
                restricted_surface_solve = std::move(new_surface_solve);
              }

            linear_profile.report();

            // Active residual entries do not participate in convergence or the
            // merit function; the bulk and free-surface blocks remain separate.
            const double current_velocity_norm =
              system_rhs.block(introspection.block_indices.velocities).l2_norm();
            const double current_pressure_norm =
              system_rhs.block(introspection.block_indices.pressure).l2_norm();
            const double current_bulk_norm = std::sqrt(
              current_velocity_norm*current_velocity_norm
              + current_pressure_norm*current_pressure_norm);
            const double current_surface_norm =
              surface_system.surface_residual_rms(
                linearization->surface_residual(), active_set);

            if (nonlinear_iteration == 0)
              {
                // Fix the attainable bulk accuracy from A and the represented
                // initial state, not from stalled residuals. The same mixed
                // absolute/relative scale is used by convergence and merit.
                LinearAlgebra::BlockVector solver_state(
                  introspection.index_sets.stokes_partitioning, mpi_communicator);
                solver_state.block(0) = working_x.block(introspection.block_indices.velocities);
                solver_state.block(1) = working_x.block(introspection.block_indices.pressure);
                solver_state.block(1) /= pressure_scaling;
                bulk_precision = internal::reconstructed_fault_bulk_precision_scale(
                  system_matrix, solver_state, mpi_communicator);
                bulk_convergence_scale = bulk_scale + bulk_precision/parameters.nonlinear_tolerance;

                // The surface scale is fixed from the first stabilized free set;
                // a characteristic K_V action supplies a physical traction scale.
                // Do not suppress it to roundoff: an initially balanced surface
                // still develops second-order residuals when bulk loading changes.
                FaultVector characteristic_slip_rate = slip_rate;
                for (auto &fault_values : characteristic_slip_rate)
                  for (double &value : fault_values)
                    value = std::max(phase_field_fault.minimum_fault_slip_rate(),
                                     std::abs(value));
                FaultVector characteristic_surface_action;
                surface_system.apply_surface_jacobian(
                  characteristic_slip_rate, characteristic_surface_action);
                ReconstructedFaultSurfaceResidual characteristic_residual;
                characteristic_residual.values =
                  std::move(characteristic_surface_action);
                const ReconstructedFaultActiveSet no_active_vertices =
                  fault_manager.prescribed_slip_rate_mask();
                const double surface_reference = std::max(
                  surface_system.surface_residual_rms(
                    linearization->surface_residual(), no_active_vertices),
                  surface_system.surface_residual_rms(
                    characteristic_residual, no_active_vertices));
                surface_scale = std::max(current_surface_norm, surface_reference);
              }

            const double relative_bulk_residual =
              internal::normalized_reconstructed_fault_residual(
                current_bulk_norm, bulk_convergence_scale, "bulk");
            const double relative_surface_residual =
              internal::normalized_reconstructed_fault_residual(
                current_surface_norm, surface_scale, "surface");
            {
              std::ostringstream progress;
              progress << "      Relative nonlinear residuals (bulk, fault) after "
                       << "nonlinear iteration " << std::setw(2) << nonlinear_iteration << ": "
                       << std::scientific << std::setprecision(6)
                       << relative_bulk_residual << ", " << relative_surface_residual;
              pcout << progress.str() << std::endl;
            }
            if (std::getenv("ASPECT_FAULT_NONLINEAR_DIAGNOSTIC"))
            {
              std::ostringstream report;
              report << std::setprecision(17)
                     << "      Fault nonlinear residual: bulk=" << current_bulk_norm
                     << ", bulk scale=" << bulk_scale << ", surface=" << current_surface_norm
                     << ", surface scale=" << surface_scale
                     << ", velocity=" << current_velocity_norm << ", scaled continuity=" << current_pressure_norm
                     << ", bulk precision=" << bulk_precision
                     << ", bulk target=" << parameters.nonlinear_tolerance*bulk_convergence_scale
                     << ", velocity correction=" << bulk_direction.block(0).linfty_norm();
              pcout << report.str() << std::endl;
            }

            const double maximum_step_length =
              internal::reconstructed_fault_maximum_step_length(
                slip_rate, slip_rate_direction, active_set,
                phase_field_fault.minimum_fault_slip_rate());

            // Observational bound probe: hold the current bulk iterate fixed
            // and set every unprescribed rate to V_min. This is a weak F(V_min)
            // diagnostic, not an independent scalar root or an accepted trial.
            const bool bound_audit = std::getenv("ASPECT_FAULT_NONLINEAR_DIAGNOSTIC") != nullptr;
            if (bound_audit)
              {
                const auto prescribed = fault_manager.prescribed_slip_rate_mask();
                FaultVector lower_rates = slip_rate;
                for (unsigned int f = 0; f < lower_rates.size(); ++f)
                  for (unsigned int v = 0; v < lower_rates[f].size(); ++v)
                    if (!prescribed[f][v])
                      lower_rates[f][v] = phase_field_fault.minimum_fault_slip_rate();
                const auto lower_residual = surface_system.evaluate_surface_residual(working_x, lower_rates);
                unsigned int free = 0, lower_active = 0, prefers_lower = 0;
                double minimum = std::numeric_limits<double>::max(), minimum_free = minimum;
                std::ofstream out;
                if (pcout.is_active())
                  {
                    out.open(parameters.output_directory + "nonlinear_bounds_"
                             + Utilities::int_to_string(timestep_number) + ".csv",
                             nonlinear_iteration == 0 ? std::ios::out : std::ios::app);
                    if (nonlinear_iteration == 0)
                      out << "iteration,fault,vertex,V,dV,prescribed,lower_active,Fmin_weak_density,alpha_max,bulk,surface\n";
                    out << std::setprecision(17);
                  }
                for (unsigned int f = 0; f < slip_rate.size(); ++f)
                  for (unsigned int v = 0; v < slip_rate[f].size(); ++v)
                    {
                      double mass = lower_residual.mass_diagonal[f][v];
                      if (v > 0) mass += lower_residual.mass_off_diagonal[f][v-1];
                      if (v+1 < slip_rate[f].size()) mass += lower_residual.mass_off_diagonal[f][v];
                      const double density = lower_residual.values[f][v]/mass;
                      if (!prescribed[f][v])
                        {
                          minimum = std::min(minimum, slip_rate[f][v]);
                          lower_active += active_set[f][v];
                          free += !active_set[f][v];
                          // R = shear - resistance; a negative F_min requests
                          // still lower rates when the local tangent is negative.
                          prefers_lower += density < 0.0;
                          if (!active_set[f][v]) minimum_free = std::min(minimum_free, slip_rate[f][v]);
                        }
                      if (pcout.is_active())
                        out << nonlinear_iteration << ',' << f << ',' << v << ',' << slip_rate[f][v]
                            << ',' << slip_rate_direction[f][v] << ',' << prescribed[f][v]
                            << ',' << (active_set[f][v] && !prescribed[f][v]) << ',' << density
                            << ',' << maximum_step_length << ',' << current_bulk_norm << ','
                            << current_surface_norm << '\n';
                    }
                pcout << "      Fault bound audit: free=" << free << ", lower-active=" << lower_active
                      << ", min V=" << minimum << ", min free V=" << minimum_free
                      << ", negative Fmin=" << prefers_lower << ", alpha_max=" << maximum_step_length
                      << std::endl;
              }

            if (relative_bulk_residual < parameters.nonlinear_tolerance
                && relative_surface_residual < parameters.nonlinear_tolerance)
              {
                // Observe accepted physical slip with the still-frozen history.
                // This opt-in standalone evaluation writes separate QP moments;
                // its residual is discarded and never enters the solve.
                if (std::getenv("ASPECT_FAULT_STRESS_SAMPLE_DIAGNOSTIC"))
                  {
                    LinearAlgebra::BlockVector diagnostic_residual(system_rhs);
                    reconstructed_fault_stokes_coupling->evaluate_slip_dependent_bulk_residual(
                      working_x, slip_rate, diagnostic_residual);
                  }


                // Allocate and validate the complete accepted publication
                // state before the first constitutive or kinematic write.
                LinearAlgebra::BlockVector accepted_solution(solution);
                accepted_solution.block(introspection.block_indices.velocities) =
                  working_x.block(introspection.block_indices.velocities);
                accepted_solution.block(introspection.block_indices.pressure) =
                  working_x.block(introspection.block_indices.pressure);
                LinearAlgebra::BlockVector accepted_linearization(working_x);
                fault_manager.validate_slip_rate_nonlinear_commit();
                nonlinear_solver_control.check(
                  nonlinear_iteration,
                  std::max(relative_bulk_residual,
                           relative_surface_residual));
                restore_simulator_state();

                // The history operation performs all failure-capable work
                // before its writes. Everything that follows is a fixed-size,
                // non-allocating terminal mutation of one accepted state.
                phase_field_fault.commit_reconstructed_fault_mechanical_history(
                  accepted_solution);
                fault_manager.commit_slip_rate_nonlinear_solve();
                solution.swap(accepted_solution);
                current_linearization_point.swap(accepted_linearization);
                last_pressure_normalization_adjustment = working_pressure_adjustment;
                nonlinear_state_is_active = false;
                terminal_commit_complete = true;
                signals.post_reconstructed_fault_solver(
                  nonlinear_iteration, total_fault_krylov_iterations,
                  minimum_fault_accepted_alpha, active_set);
                signals.post_nonlinear_solver(nonlinear_solver_control);
                return;
              }

            const double current_merit = 0.5*(
              relative_bulk_residual*relative_bulk_residual
              + relative_surface_residual*relative_surface_residual);

            // Limit only free downward directions and allow exact arrival at
            // V_min; active outward directions were already removed by K_FF.
            const LinearAlgebra::BlockVector physical_bulk_direction =
              linearization->make_physical_bulk_direction(bulk_direction);

            // Trial arithmetic uses owned vectors; residual point evaluation
            // receives a ghosted physical-pressure vector after the update.
            LinearAlgebra::BlockVector owned_physical_bulk_direction(
              introspection.index_sets.system_partitioning,
              mpi_communicator);
            owned_physical_bulk_direction = physical_bulk_direction;

            LinearAlgebra::BlockVector accepted_trial_x(working_x);
            double accepted_trial_pressure_adjustment = working_pressure_adjustment;
            FaultVector accepted_trial_slip_rate;

            // Bounded K1-style affine audit: freeze A before any residual-only
            // assembly can overwrite it. B/G/K_V retain their original caches.
            // Only iteration 0 is instrumented; no solver budget is changed.
            const bool audit = std::getenv("ASPECT_K1_FLOOR_AUDIT") != nullptr
                               && nonlinear_iteration == 0;
            LinearAlgebra::BlockSparseMatrix audit_matrix;
            LinearAlgebra::BlockVector audit_rhs, audit_unknowns, audit_frozen;
            auto restore_audit_matrix = [&]()
            {
              for (unsigned int i = 0; i < 2; ++i)
                for (unsigned int j = 0; j < 2; ++j)
                  system_matrix.block(i,j).copy_from(audit_matrix.block(i,j));
            };
            auto audit_channel = [&](const LinearAlgebra::BlockVector &x,
                                     const FaultVector &v,
                                     const internal::FaultResidualAuditChannel channel)
            {
              internal::fault_residual_audit_channel = channel;
              try
                {
                  evaluate_coupled_residual(x, v);
                }
              catch (...)
                {
                  internal::fault_residual_audit_channel =
                    internal::FaultResidualAuditChannel::normal;
                  throw;
                }
              internal::fault_residual_audit_channel =
                internal::FaultResidualAuditChannel::normal;
              restore_audit_matrix();
              return LinearAlgebra::BlockVector(system_rhs);
            };
            if (audit)
              {
                audit_matrix.reinit(2,2);
                for (unsigned int i = 0; i < 2; ++i)
                  for (unsigned int j = 0; j < 2; ++j)
                    audit_matrix.block(i,j).copy_from(system_matrix.block(i,j));
                audit_matrix.collect_sizes();
                audit_rhs = system_rhs;
                audit_unknowns = audit_channel(working_x, slip_rate,
                                               internal::FaultResidualAuditChannel::unknowns);
                audit_frozen = audit_channel(working_x, slip_rate,
                                             internal::FaultResidualAuditChannel::frozen);
                system_rhs = audit_rhs;
                unsigned int active = 0, total = 0;
                for (const auto &fault : active_set)
                  for (const bool value : fault)
                    {
                      active += value;
                      ++total;
                    }
                pcout << "      Affine audit active=" << active
                      << " free=" << total-active << std::endl;
              }
            const auto line_search_result =
              internal::reconstructed_fault_armijo_line_search(
                maximum_step_length,
                newton_handler->parameters.max_newton_line_search_iterations,
                current_merit,
                [&](const double step_length)
                {
                  // Every candidate is reconstructed from the same accepted base,
                  // and the stabilized active set remains fixed across the search.
                  LinearAlgebra::BlockVector owned_trial_x(
                    introspection.index_sets.system_partitioning,
                    mpi_communicator);
                  owned_trial_x = working_x;
                  owned_trial_x.block(introspection.block_indices.velocities).add(
                    step_length,
                    owned_physical_bulk_direction.block(
                      introspection.block_indices.velocities));
                  owned_trial_x.block(introspection.block_indices.pressure).add(
                    step_length,
                    owned_physical_bulk_direction.block(
                      introspection.block_indices.pressure));
                  owned_trial_x.compress(VectorOperation::insert);
                  LinearAlgebra::BlockVector trial_x(
                    introspection.index_sets.system_partitioning,
                    introspection.index_sets.system_relevant_partitioning,
                    mpi_communicator);
                  trial_x = owned_trial_x;

                  // Normalize the physical candidate before residual/merit
                  // evaluation, not the homogeneous Newton direction. This
                  // removes arbitrary pressure offsets from cancellation in
                  // bulk assembly; true-pressure friction is left unchanged.
                  const double trial_pressure_adjustment = normalize_fault_pressure
                    ? normalize_pressure(trial_x) : working_pressure_adjustment;

                  FaultVector trial_slip_rate = slip_rate;
                  for (unsigned int fault = 0; fault < slip_rate.size(); ++fault)
                    for (unsigned int vertex = 0;
                         vertex < slip_rate[fault].size(); ++vertex)
                      {
                        trial_slip_rate[fault][vertex] = internal::reconstructed_fault_trial_value(
                          slip_rate[fault][vertex], slip_rate_direction[fault][vertex],
                          step_length, phase_field_fault.minimum_fault_slip_rate());
                      }

                  const CoupledResidual trial_residual =
                    evaluate_coupled_residual(trial_x, trial_slip_rate);
                  if (audit)
                    {
                      // Use the represented, pressure-normalized update, with
                      // homogeneous constraint rows removed and p in solver units.
                      const LinearAlgebra::BlockVector fresh_rhs(system_rhs);
                      LinearAlgebra::BlockVector actual(owned_trial_x);
                      actual = trial_x;
                      LinearAlgebra::BlockVector owned_base(owned_trial_x);
                      owned_base = working_x;
                      actual -= owned_base;
                      current_constraints.set_zero(actual);
                      LinearAlgebra::BlockVector dx(bulk_direction), action(bulk_direction);
                      dx.block(0) = actual.block(introspection.block_indices.velocities);
                      dx.block(1) = actual.block(introspection.block_indices.pressure);
                      dx.block(1) /= pressure_scaling;
                      internal::StokesBlock(audit_matrix).vmult(action, dx);
                      LinearAlgebra::BlockVector requested(bulk_direction);
                      internal::StokesBlock(audit_matrix).vmult(requested, bulk_direction);
                      requested *= step_length;
                      FaultVector dv = trial_slip_rate;
                      double v_representation_error = 0.0, dv_max = 0.0;
                      for (unsigned int f = 0; f < dv.size(); ++f)
                        for (unsigned int i = 0; i < dv[f].size(); ++i)
                          {
                            const double represented = slip_rate[f][i]
                              + (trial_slip_rate[f][i]-slip_rate[f][i]);
                            v_representation_error = std::max(v_representation_error,
                              std::abs(represented-trial_slip_rate[f][i]));
                            dv[f][i] = represented-slip_rate[f][i];
                            dv_max = std::max(dv_max, std::abs(dv[f][i]));
                          }
                      LinearAlgebra::BlockVector b_action(system_rhs);
                      reconstructed_fault_stokes_coupling->apply_B(dv, b_action);
                      const auto fresh_unknowns = audit_channel(trial_x, trial_slip_rate,
                        internal::FaultResidualAuditChannel::unknowns);
                      const auto fresh_frozen = audit_channel(trial_x, trial_slip_rate,
                        internal::FaultResidualAuditChannel::frozen);
                      // A block action may nearly cancel (notably continuity).
                      // Reuse the absolute-row-sum precision bound on this
                      // represented direction, not the small cancelled action.
                      const double action_precision =
                        internal::reconstructed_fault_bulk_precision_scale(
                          audit_matrix, dx, mpi_communicator);
                      for (unsigned int block = 0; block < 2; ++block)
                        {
                          // system_rhs=-R: affine prediction is rhs-A dx+B dV.
                          auto predicted = audit_rhs.block(block);
                          predicted -= action.block(block);
                          predicted += b_action.block(block);
                          auto error = fresh_rhs.block(block);
                          error -= predicted;
                          auto representation = action.block(block);
                          representation -= requested.block(block);
                          auto unknowns_error = fresh_unknowns.block(block);
                          unknowns_error -= audit_unknowns.block(block);
                          unknowns_error += action.block(block);
                          unknowns_error -= b_action.block(block);
                          auto frozen_error = fresh_frozen.block(block);
                          frozen_error -= audit_frozen.block(block);
                          const double accuracy = 1e-10*std::max({
                            audit_rhs.block(block).l2_norm(), action.block(block).l2_norm(),
                            b_action.block(block).l2_norm()})
                            + 32.0*action_precision
                            + 32.0*std::numeric_limits<double>::epsilon()
                              * audit_frozen.block(block).l2_norm();
                          AssertThrow(error.l2_norm() <= accuracy
                                      && frozen_error.l2_norm() == 0.0,
                                      ExcMessage("The represented coupled increment failed "
                                                 "the affine residual consistency regression."));
                          std::ostringstream report;
                          report << std::setprecision(17)
                            << "      Affine audit alpha=" << step_length << " block=" << block
                            << " base=" << audit_rhs.block(block).l2_norm()
                            << " predicted=" << predicted.l2_norm()
                            << " fresh=" << fresh_rhs.block(block).l2_norm()
                            << " affine_error=" << error.l2_norm()
                            << " represented_action_error=" << representation.l2_norm()
                            << " unknowns_base=" << audit_unknowns.block(block).l2_norm()
                            << " frozen_base=" << audit_frozen.block(block).l2_norm()
                            << " unknowns_affine_error=" << unknowns_error.l2_norm()
                            << " frozen_load_change=" << frozen_error.l2_norm()
                            << " test_accuracy=" << accuracy
                            << " B_dV=" << b_action.block(block).l2_norm()
                            << " dV_max=" << dv_max
                            << " V_representation_error=" << v_representation_error;
                          pcout << report.str() << std::endl;
                        }

                      // A separate admissible V probe is never an accepted
                      // candidate. It prevents an all-active solve from hiding
                      // accidental freezing of BV in either accumulator.
                      FaultVector probe_v = trial_slip_rate, probe_dv = trial_slip_rate;
                      for (unsigned int f = 0; f < probe_v.size(); ++f)
                        for (unsigned int i = 0; i < probe_v[f].size(); ++i)
                          {
                            probe_v[f][i] += 1e-12*(1.0+0.1*i);
                            probe_dv[f][i] = (slip_rate[f][i]
                              + (probe_v[f][i]-slip_rate[f][i]))
                              - (slip_rate[f][i]
                                 + (trial_slip_rate[f][i]-slip_rate[f][i]));
                          }
                      evaluate_coupled_residual(trial_x, probe_v);
                      const LinearAlgebra::BlockVector probe_rhs(system_rhs);
                      reconstructed_fault_stokes_coupling->apply_B(probe_dv, b_action);
                      const auto probe_frozen = audit_channel(trial_x, probe_v,
                        internal::FaultResidualAuditChannel::frozen);
                      AssertThrow(b_action.l2_norm() > 0.0,
                                  ExcMessage("The nonzero-V probe must have a nonzero weak action."));
                      for (unsigned int block = 0; block < 2; ++block)
                        {
                          auto error = probe_rhs.block(block);
                          error -= fresh_rhs.block(block);
                          error -= b_action.block(block);
                          auto frozen_change = probe_frozen.block(block);
                          frozen_change -= fresh_frozen.block(block);
                          const double accuracy = 1e-10*b_action.block(block).l2_norm()
                            + 32.0*std::numeric_limits<double>::epsilon()
                              * fresh_frozen.block(block).l2_norm();
                          AssertThrow(error.l2_norm() <= accuracy
                                      && frozen_change.l2_norm() == 0.0,
                                      ExcMessage("The nonzero-V probe changed the frozen load "
                                                 "or disagreed with B."));
                          std::ostringstream report;
                          report << std::setprecision(17)
                            << "      Affine nonzero-V probe: block=" << block
                            << " B_dV=" << b_action.block(block).l2_norm()
                            << " error=" << error.l2_norm()
                            << " frozen_change=" << frozen_change.l2_norm()
                            << " test_accuracy=" << accuracy;
                          pcout << report.str() << std::endl;
                        }
                      // Shadow assembly is observational: restore the actual
                      // production residual and the original linearization.
                      system_rhs = fresh_rhs;
                    }
                  const double trial_relative_bulk =
                    internal::normalized_reconstructed_fault_residual(
                      trial_residual.bulk_norm, bulk_convergence_scale, "bulk");
                  const double trial_relative_surface =
                    internal::normalized_reconstructed_fault_residual(
                      surface_system.surface_residual_rms(
                        trial_residual.surface, active_set),
                      surface_scale,
                      "surface");
                  const double trial_merit = 0.5*(
                    trial_relative_bulk*trial_relative_bulk
                    + trial_relative_surface*trial_relative_surface);
                  if (audit || bound_audit)
                    {
                      std::ostringstream report;
                      report << std::setprecision(17)
                        << (audit ? "      Affine audit merit: alpha=" : "      Fault trial merit: alpha=") << step_length
                        << " bulk_relative=" << trial_relative_bulk
                        << " surface_relative=" << trial_relative_surface
                        << " current=" << current_merit << " trial=" << trial_merit;
                      pcout << report.str() << std::endl;
                    }
                  accepted_trial_x = trial_x;
                  accepted_trial_pressure_adjustment = trial_pressure_adjustment;
                  accepted_trial_slip_rate = std::move(trial_slip_rate);
                  return trial_merit;
                },
                [&](const double)
                {
                  // Accepting replaces manager current V and working_x; the
                  // timestep-committed V still changes only at convergence.
                  fault_manager.begin_slip_rate_trial();
                  fault_manager.set_slip_rate_trial_values(accepted_trial_slip_rate);
                  fault_manager.accept_slip_rate_trial();
                  working_x = accepted_trial_x;
                  working_pressure_adjustment = accepted_trial_pressure_adjustment;
                });

            if (!line_search_result.accepted)
              {
                pcout << "   Coupled reconstructed-fault Newton line search "
                      << "exhausted all admissible candidates." << std::endl;
                throw ExcNonlinearSolverNoConvergence();
              }
            pcout << "      Reconstructed-fault line search accepted after "
                  << line_search_result.rejected_candidates
                  << " rejected candidates; alpha=" << line_search_result.step_length << "." << std::endl;
            minimum_fault_accepted_alpha = std::min(minimum_fault_accepted_alpha,
                                                     line_search_result.step_length);
          }

        nonlinear_solver_control.check(max_nonlinear_iterations,
                                       std::numeric_limits<double>::max());
        AssertThrow(false, ExcNonlinearSolverNoConvergence());
      }
    catch (...)
      {
        if (terminal_commit_complete)
          throw;
        // Any failure restores both externally visible bulk state and committed
        // manager V; constitutive histories were not yet made mutable.
        if (nonlinear_state_is_active)
          fault_manager.rollback_slip_rate_nonlinear_solve();
        solution = production_solution;
        current_linearization_point = saved_linearization_point;
        restore_simulator_state();
        nonlinear_solver_control.check(max_nonlinear_iterations,
                                       std::numeric_limits<double>::max());
        signals.post_nonlinear_solver(nonlinear_solver_control);
        throw;
      }
  }

}



// explicit instantiation of the functions we implement in this file
namespace aspect
{
#define INSTANTIATE(dim) \
  template double Simulator<dim>::solve_advection (const AdvectionField &); \
  template std::pair<double,double> Simulator<dim>::solve_stokes (LinearAlgebra::BlockVector &solution_vector); \
  template void Simulator<dim>::solve_reconstructed_fault_stokes ();

  ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
}
