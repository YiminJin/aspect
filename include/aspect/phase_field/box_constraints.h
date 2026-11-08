/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#ifndef ASPECT_PHASE_FIELD_BOX_CONSTRAINTS_H
#define ASPECT_PHASE_FIELD_BOX_CONSTRAINTS_H

#include <deal.II/base/exceptions.h>
#include <algorithm>

namespace aspect
{
  namespace PhaseField
  {
    /** One row of the natural-map VI, x=project_[l,u](x-g/m).
     * Negative/positive status denotes the lower/upper bound. Classification
     * uses the CURRENT primal and dual variables, so either bound can release.
     * A positive search metric changes scaling, never the physical obstacle. */
    struct BoxConstraint
    {
      int status;
      double projected_residual;
      double target;
    };

    /** Evaluate a computed bound displacement without cancelling a tiny
     * positive obstacle in x+(lower-x). This is affine interpolation to the
     * prescribed target, not projection of an arbitrary infeasible direction. */
    inline double box_trial_value(const double x, const double increment,
                                  const double lower, const double upper,
                                  const double fraction)
    {
      const double target=increment==lower-x ? lower : increment==upper-x ? upper : x+increment;
      return target>=x ? x+fraction*(target-x) : target+(1.-fraction)*(x-target);
    }

    inline BoxConstraint box_constraint(const double x, const double gradient,
                                        const double metric, const double lower,
                                        const double upper)
    {
      AssertThrow(metric > 0. && lower <= upper,
                  dealii::ExcMessage("Invalid box constraint or search metric."));
      const double prediction = x-gradient/metric;
      const double target = std::clamp(prediction, lower, upper);
      return {prediction <= lower ? -1 : prediction >= upper ? 1 : 0,
              x-target, target
             };
    }
  }
}
#endif
