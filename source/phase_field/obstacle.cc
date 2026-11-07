/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/phase_field/obstacle.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <cmath>
#include <limits>

namespace aspect
{
  namespace PhaseField
  {
    ObstacleResult solve_obstacle(const LinearAlgebra::SparseMatrix &matrix,
                                  const LinearAlgebra::Vector &rhs,
                                  const LinearAlgebra::Vector &lower,
                                  const LinearAlgebra::Vector &upper,
                                  LinearAlgebra::Vector &state,
                                  const IndexSet &relevant, const MPI_Comm communicator,
                                  const double tolerance, const unsigned int max_iterations,
                                  const unsigned int max_linear_iterations)
    {
      const auto owned = state.locally_owned_elements();
      LinearAlgebra::Vector gradient(owned, communicator), active(owned, communicator), fixed(owned, communicator),
                    load(owned, communicator), candidate(owned, communicator);
      LinearAlgebra::Vector active_ghost(owned, relevant, communicator), fixed_ghost(owned, relevant, communicator);
      unsigned int invalid = 0;
      for (const auto i : owned)
        {
          invalid |= !(lower[i] <= upper[i] && matrix.diag_element(i) > 0.);
          state[i] = std::max(lower[i], std::min(upper[i], static_cast<double>(state[i])));
        }
      AssertThrow(Utilities::MPI::max(invalid, communicator) == 0,
                  ExcMessage("Phase obstacle has inconsistent bounds or a nonpositive search diagonal."));
      state.compress(VectorOperation::insert);
      ObstacleResult result;
      for (unsigned int iteration = 0; iteration <= max_iterations; ++iteration)
        {
          matrix.vmult(gradient, state);
          gradient -= rhs;
          double defect = 0.;
          for (const auto i : owned)
            {
              const double diagonal = matrix.diag_element(i);
              const double prediction = state[i] - gradient[i] / diagonal;
              const double projected = std::max(lower[i], std::min(upper[i], prediction));
              defect = std::max(defect, std::abs(state[i] - projected) * diagonal);
              active[i] = prediction <= lower[i] ? -1. : prediction >= upper[i] ? 1. : 0.;
              fixed[i] = active[i] < 0. ? lower[i] : active[i] > 0. ? upper[i] : 0.;
            }
          result.residual = Utilities::MPI::max(defect, communicator);
          result.iterations = iteration;
          if (result.residual <= tolerance)
            {
              result.energy = .5 * (state * gradient) - .5 * (state * rhs);
              return result;
            }
          AssertThrow(iteration < max_iterations, ExcMessage("Native phase obstacle active-set iteration budget exhausted."));
          active.compress(VectorOperation::insert);
          fixed.compress(VectorOperation::insert);
          active_ghost = active;
          fixed_ghost = fixed;
          LinearAlgebra::SparseMatrix restricted;
          restricted.copy_from(matrix);
          load = rhs;
          for (const auto i : owned)
            {
              // Use insertion throughout: switching between vector insertion
              // and addition inside rank-local branches triggers implicit
              // Trilinos collectives on different ranks in different orders.
              double row_load = rhs[i];
              if (active[i] != 0.)
                {
                  restricted.clear_row(i, 1.);
                  row_load = fixed[i];
                }
              else
                for (auto entry = restricted.begin(i); entry != restricted.end(i); ++entry)
                  if (active_ghost[entry->column()] != 0.)
                    {
                      row_load -= entry->value() * fixed_ghost[entry->column()];
                      entry->value() = 0.;
                    }
              load[i] = row_load;
            }
          restricted.compress(VectorOperation::insert);
          load.compress(VectorOperation::insert);
          LinearAlgebra::PreconditionAMG preconditioner;
          LinearAlgebra::PreconditionAMG::AdditionalData data;
          data.elliptic = true;
          data.higher_order_elements = true;
          data.smoother_sweeps = 2;
          preconditioner.initialize(restricted, data);
          const double roundoff = 64 * std::numeric_limits<double>::epsilon() * load.l2_norm();
          SolverControl control(max_linear_iterations, std::max(.05 * tolerance, roundoff));
          SolverCG<LinearAlgebra::Vector> solver(control);
          candidate = state;
          solver.solve(restricted, candidate, load, preconditioner);
          result.linear_iterations += control.last_step();
          state = candidate;
          // No clipping here. Active-set classification and the final projected
          // residual must resolve every primal and dual violation themselves.
        }
      AssertThrow(false, ExcInternalError());
      return result;
    }
  }
}
