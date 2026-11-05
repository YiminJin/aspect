/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#ifndef ASPECT_MATURE_STRESS_LAW_H
#define ASPECT_MATURE_STRESS_LAW_H
#include <aspect/utilities.h>
namespace aspect
{
  namespace MaterialModel
  {
    namespace Rheology
    {
      /** Frozen-coefficient endpoint forcing after advection and total-spin
       * rotation: beta*tau_bar + 2*kappa*D_3 - kappa*(2*localized_slip_rate).
       * The localization evaluator supplies the complete geometric field.
       * This is a physical stress value, before DG projection. */
      template<int dim> SymmetricTensor<2, dim> mature_stress_endpoint(
        const double beta, const double kappa, const SymmetricTensor<2, dim> &anchor,
        const SymmetricTensor<2, dim> &strain_rate,
        const SymmetricTensor<2, dim> &twice_localized_slip_rate)
      {
        return beta * anchor + 2 * kappa * Utilities::Tensors::consistent_deviator(strain_rate)
               - kappa * twice_localized_slip_rate;
      }
    }
  }
}
#endif
