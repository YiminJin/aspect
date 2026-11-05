/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#ifndef ASPECT_MATERIAL_MODEL_BULK_COHESIVE_TENSOR_H
#define ASPECT_MATERIAL_MODEL_BULK_COHESIVE_TENSOR_H

#include <aspect/material_model/phase_field_fault/bulk_cohesive.h>

namespace aspect
{
  namespace MaterialModel
  {
    namespace BulkCohesive
    {
      namespace TensorReturn
      {
        /** Experimental point law only. T is a full 3-D plane-strain deviator.
         * A/K/Q are physical tensors in the in-plane trace-free shear space, never
         * scalars attached to a rotating director. A/K/L are dimensionless; Q is Pa.
         * The native bulk-cohesive adapter constructs this point state from its
         * own accepted records; this stateless law does not own publication.
         * Its member serialization serves the standalone point fixture. */
        struct State
        {
          Tensor T; ///< Trial full 3-D deviatoric stress (Pa).
          Tensor A; ///< Elastic inelastic-mode strain tensor.
          Tensor K; ///< Accumulated inelastic strain tensor.
          Tensor Q; ///< Cohesive tensor stress (Pa).
          double phi = 0.; ///< Physical phase/obstacle.
          double L = 0.; ///< Accumulated absolute inelastic strain magnitude.

          template<class Archive> void serialize(Archive &archive, const unsigned int)
          {
            archive &T &A &K &Q &phi &L;
          }
        };

        /** Unpublished tensor endpoint; energies J/m^3, dissipations W/m^3. */
        struct Response
        {
          State state;
          Tensor stress, flow_rate;
          double beta = 0., speed = 0., driving = 0.;
          double bulk_energy = 0., cohesive_energy = 0.;
          double bulk_dissipation = 0., cohesive_dissipation = 0., friction_dissipation = 0.;
          bool sliding = false, intact = false;
          bool direction_defined() const
          {
            return speed > 0.;
          }
        };

        /** Null fixed_direction selects the in-plane radial return. Otherwise
         * restrict to span(S), S:S=1/2, for the fixed-direction identity control;
         * incoming A/K/Q must belong to that same span. No history is transformed.
         * The increment is dev3(symgrad du). Exact yield ties select sticking. */
        Response update(const Parameters &parameters, const State &incoming, const Tensor &increment,
                        double phi, double dt, const Tensor *fixed_direction = nullptr);

        /** Directional action of Response with the incoming history held fixed. */
        struct Variation
        {
          Tensor T, A, K, Q, stress, flow_rate;
          double L = 0., speed = 0., driving = 0., bulk_energy = 0., cohesive_energy = 0.;
          double bulk_dissipation = 0., cohesive_dissipation = 0., friction_dissipation = 0.;
        };

        /** Complete current-step action with incoming history/dt fixed. Includes
         * the radial direction derivative. At h=0 use the admissible right phase
         * derivative; at a yield tie this returns the sticking generalized Jacobian,
         * not a classical derivative in every outward direction. */
        Variation derivative(const Parameters &parameters, const State &incoming, const Response &response,
                             const Tensor &increment_variation, double phase_variation,
                             double dt, const Tensor *fixed_direction = nullptr);

        /** Endpoint work balance for the full tensor law; no history mutation. */
        StepAccounting account(const Parameters &parameters, const State &incoming, const Response &response,
                               const Tensor &increment, double dt);

        /** Extract the in-plane trace-free part; preserve the supplied tensor. */
        Tensor in_plane(const Tensor &tensor);
        /** Shear-space magnitude sqrt((xx-yy)^2/4+xy^2). */
        double magnitude(const Tensor &tensor); // sqrt((xx-yy)^2/4+xy^2), not Frobenius norm
      }
    }
  }
}
#endif
