/* Copyright (C) 2026 by the authors of the ASPECT code.
 * SPDX-License-Identifier: GPL-2.0-or-later */
#ifndef _aspect_simulator_solver_reconstructed_fault_linear_h
#define _aspect_simulator_solver_reconstructed_fault_linear_h

#include <deal.II/base/exceptions.h>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace aspect
{
  namespace internal
  {
    /** Algebraic pressure-complement projection. q is unit or zero, with zero
     * constrained entries. This is not physical pressure normalization. */
    template <typename VectorType>
    void project_fault_pressure(const VectorType &q, VectorType &vector)
    {
      // Remove the roundoff remainder of a potentially large null component.
      for (unsigned int pass=0; pass<2; ++pass)
        vector.add(-(q*vector), q);
    }

    template <typename VectorType>
    double project_compatible_fault_rhs(const VectorType &q,
                                        const double compatibility_tolerance,
                                        VectorType &rhs)
    {
      const double component = q*rhs;
      if (!(std::abs(component) <= compatibility_tolerance))
        {
          std::ostringstream message;
          message << std::setprecision(17)
                  << "The condensed fault RHS has incompatible pressure component "
                  << component << ", exceeding the roundoff/nonlinear bound "
                  << compatibility_tolerance << '.';
          AssertThrow(false, dealii::ExcMessage(message.str()));
        }
      project_fault_pressure(q, rhs);
      return component;
    }

    /** P operator P, also used for the preconditioner. A zero q is identity. */
    template <typename Operator, typename VectorType>
    struct FaultPressureComplementOperator
    {
      const Operator &op;
      const VectorType &q;

      void vmult(VectorType &result, const VectorType &source) const
      {
        VectorType projected(source);
        project_fault_pressure(q, projected);
        op.vmult(result, projected);
        project_fault_pressure(q, result);
      }
    };

    /** Fresh residual of the original full operator, with raw and null norms
     * retained. Only independently verified compatibility noise is excluded. */
    template <typename Operator, typename VectorType>
    double fault_true_linear_residual(const Operator &op,
                                       const VectorType &q,
                                       const VectorType &solution,
                                       const VectorType &rhs,
                                       VectorType &residual,
                                       double &raw_norm,
                                       double &null_component)
    {
      op.vmult(residual, solution);
      residual -= rhs;
      raw_norm = residual.l2_norm();
      null_component = q*residual;
      project_fault_pressure(q, residual);
      return residual.l2_norm();
    }
  }
}
#endif
