/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#ifndef ASPECT_PHASE_FIELD_OBSTACLE_H
#define ASPECT_PHASE_FIELD_OBSTACLE_H
#include <aspect/global.h>

namespace aspect
{
  namespace PhaseField
  {
    /** Result of the sparse positive quadratic phase subproblem. Residual is
     * the diagonal-scaled projected gradient, in dual energy units. Bounds are
     * constraints of the solve, not a post-solve limiter. */
    struct ObstacleResult
    {
      unsigned int iterations = 0, linear_iterations = 0;
      double residual = 0., energy = 0.;
    };

    /** Minimize 1/2 x^T A x - b^T x under lower <= x <= upper.
     * A must be symmetric positive definite. Used for the DG endpoint search
     * subproblem and manufactured phase equilibrium. The full nonlinear physical
     * endpoint still supplies its own energy, tangent and acceptance criteria.
     * All vectors are uniquely owned; relevant contains readable matrix columns.
     * The caller supplies positive-basis coordinates when whole-cell bounds are
     * required. No global dense matrices or rank-zero direct factors are used. */
    ObstacleResult solve_obstacle(const LinearAlgebra::SparseMatrix &matrix,
                                  const LinearAlgebra::Vector &rhs,
                                  const LinearAlgebra::Vector &lower,
                                  const LinearAlgebra::Vector &upper,
                                  LinearAlgebra::Vector &state,
                                  const IndexSet &relevant, MPI_Comm communicator,
                                  double tolerance, unsigned int max_iterations,
                                  unsigned int max_linear_iterations);
  }
}
#endif
