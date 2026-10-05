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


#ifndef aspect_internal_stokes_operators_h
#define aspect_internal_stokes_operators_h

#include <aspect/global.h>
#include <aspect/utilities.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/vector_memory.h>

#include <exception>
#include <memory>
#include <vector>

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
     * Construct the selected Schur wrapper around caller-owned data, which
     * must outlive the result. The velocity mass block is used only for BFBT.
     */
    std::unique_ptr<SchurComplementOperator>
    make_stokes_schur_preconditioner(
      const bool use_bfbt,
      const LinearAlgebra::SparseMatrix &pressure_matrix,
      const LinearAlgebra::PreconditionBase &pressure_preconditioner,
      const double solver_tolerance,
      const LinearAlgebra::BlockVector &inverse_lumped_mass_matrix,
      const unsigned int velocity_block_index,
      const LinearAlgebra::BlockSparseMatrix &system_matrix);

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
}

#endif
