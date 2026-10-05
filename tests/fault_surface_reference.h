/* Copyright (C) 2026 by the authors of the ASPECT code.
 * SPDX-License-Identifier: GPL-2.0-or-later */
#ifndef ASPECT_TEST_FAULT_SURFACE_REFERENCE_H
#define ASPECT_TEST_FAULT_SURFACE_REFERENCE_H

#include <aspect/reconstructed_fault/surface_direct_internal.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <memory>

namespace aspect
{
  namespace Testing
  {
    // Independent inverse only for tests; production always uses pivoted LU.
    class FaultSurfaceInverse
    {
      public:
        FaultSurfaceInverse(const std::vector<double> &diagonal,
                            const std::vector<double> &upper,
                            const std::vector<bool> &active,
                            const unsigned int fault,
                            const bool pivoted,
                            const std::vector<double> &lower = {})
          : active(active)
        {
          if (pivoted)
            {
              direct = std::make_unique<internal::FaultSurfaceDirect>(
                diagonal, upper, active, fault, lower);
              return;
            }
          const auto &lo = lower.empty() ? upper : lower;
          dealii::DynamicSparsityPattern pattern(diagonal.size(), diagonal.size());
          for (unsigned int i=0; i<diagonal.size(); ++i)
            {
              pattern.add(i,i);
              if (i+1<diagonal.size() && !active[i] && !active[i+1])
                { pattern.add(i,i+1); pattern.add(i+1,i); }
            }
          sparsity.copy_from(pattern);
          matrix.reinit(sparsity);
          for (unsigned int i=0; i<diagonal.size(); ++i)
            {
              matrix.set(i,i,active[i] ? 1. : diagonal[i]);
              if (i+1<diagonal.size() && !active[i] && !active[i+1])
                { matrix.set(i,i+1,upper[i]); matrix.set(i+1,i,lo[i]); }
            }
          reference.initialize(matrix);
        }

        void solve(const std::vector<double> &rhs, std::vector<double> &result) const
        {
          if (direct)
            { direct->solve(rhs,result); return; }
          dealii::Vector<double> source(rhs.size()), solution(rhs.size());
          for (unsigned int i=0; i<rhs.size(); ++i) source[i]=active[i] ? 0. : rhs[i];
          reference.vmult(solution,source);
          result.resize(rhs.size());
          for (unsigned int i=0; i<rhs.size(); ++i) result[i]=active[i] ? 0. : solution[i];
        }

      private:
        const std::vector<bool> active;
        std::unique_ptr<internal::FaultSurfaceDirect> direct;
        dealii::SparsityPattern sparsity;
        dealii::SparseMatrix<double> matrix;
        dealii::SparseDirectUMFPACK reference;
    };
  }
}
#endif
