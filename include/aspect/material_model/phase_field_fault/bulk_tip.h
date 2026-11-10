/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#ifndef ASPECT_MATERIAL_MODEL_BULK_TIP_H
#define ASPECT_MATERIAL_MODEL_BULK_TIP_H

#include <aspect/material_model/phase_field_fault/bulk_cohesive.h>

namespace aspect
{
  namespace MaterialModel
  {
    namespace BulkCohesive
    {
      /** Fixed-reference bulk energetic tip control. No surface damage, spin,
       * transport, or physical length inferred from the computational mesh.
       * All tensors use the full three-dimensional plane-strain metric. */
      namespace Tip
      {
        struct State
        {
          Tensor stress; ///< Physical deviatoric stress, Pa.
          Tensor cohesive_strain; ///< Physical recoverable strain, dimensionless.
          double phi = 0.; ///< Accepted physical damage, not a numerical mask.
        };

        struct Response
        {
          State state;
          Tensor Q; ///< Conjugate stress, including the intact reaction.
          double rate = 0.; ///< Signed engineering fault-deformation rate, 1/s.
          double driving = 0., bulk_energy = 0., cohesive_energy = 0.;
          double bulk_viscosity = 0., cohesive_viscosity = 0.;
          double fraction = 0., fraction_derivative = 0.;
          double force = 0., driving_factor = 0., driving_factor_derivative = 0.;
        };

        /** Endpoint elimination at prescribed work-conjugate section resistance
         * f (Pa). S:S=1/2, D_f=w S. The section solve, not this material point,
         * supplies f=RSF(integral w dn). The intact reaction is q=t-f; there is
         * no Coulomb sticking interval or positive damage floor. */
        Response update(const Parameters &parameters, const State &parent,
                        const Tensor &strain_increment, double phi, double dt,
                        const Tensor &direction, double resistance);

        /** Recover a physical endpoint from the half-duration algebraic stage. */
        State midpoint_endpoint(const State &parent, const Response &stage);
        double cohesive_energy(const Parameters &, const State &);
        /** Midpoint work: compliance_work includes Y_m Delta phi plus C_c.
         * No BE quadratic numerical losses are included. */
        StepAccounting midpoint_account(const Parameters &, const State &, const Response &,
                                        const Tensor &physical_increment, double dt, double resistance);

        struct Variation
        {
          Tensor stress, cohesive_strain, Q;
          double rate = 0., driving = 0.;
        };

        /** Consistent action with fixed parent, including the induced cohesive
         * state and the section-force variation. One-sided at phi=0 and 1. */
        Variation derivative(const Parameters &parameters, const Response &response,
                             const Tensor &strain_increment, double phase_increment,
                             double dt, const Tensor &direction, double resistance_increment);

        /** Independent endpoint work account. Compliance work is not Y Delta phi.
         * Physical friction is f*w, possibly signed locally for the nonlocal
         * section law; only its section integral is required nonnegative. */
        StepAccounting account(const Parameters &parameters, const State &parent,
                               const Response &response, const Tensor &strain_increment,
                               double dt, double resistance);
      }
    }
  }
}
#endif
