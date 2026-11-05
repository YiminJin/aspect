/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#ifndef ASPECT_MATERIAL_MODEL_BULK_COHESIVE_H
#define ASPECT_MATERIAL_MODEL_BULK_COHESIVE_H

#include <deal.II/base/symmetric_tensor.h>
#include <vector>

namespace aspect
{
  namespace MaterialModel
  {
    /** Default-off bulk constitutive experiment. All contractions are 3D;
     * a/k are engineering shear, w is a bulk rate (not surface slip rate).
     * This stateless module neither publishes history nor knows a surface. */
    namespace BulkCohesive
    {
      using Tensor = dealii::SymmetricTensor<2, 3>;

      struct Parameters
      {
        double G, eta, m, resistance;
      };

      struct Compliance
      {
        double h, first, second;
      };
      Compliance compliance(double phi, double m);

      struct State
      {
        Tensor T;
        double a = 0., signed_shear = 0., absolute_shear = 0.;
        double phi = 0., q = 0.;
      };

      struct Response
      {
        State state;
        Tensor stress;
        double w = 0., beta = 0.;
        double da_dt = 0., da_dphi = 0., dq_dt = 0., dq_dphi = 0.;
        double driving = 0., driving_dt = 0., driving_dphi = 0.;
        double bulk_energy = 0., cohesive_energy = 0.;
        double bulk_dissipation = 0., cohesive_dissipation = 0., friction_dissipation = 0.;
        bool sliding = false, intact = false;
      };

      /** Backward Euler from one immutable incoming history. The increment
       * is dev3 symgrad(du); the orientation is horizontal throughout bulk. */
      Response update(const Parameters &, const State &, const Tensor &strain_increment,
                      double phi, double dt);

      struct StepAccounting
      {
        double internal_work = 0., storage_change = 0., physical_dissipation = 0.;
        double stress_loss = 0., traction_loss = 0., compliance_work = 0.;
        double error = 0.;
      };
      StepAccounting account(const Parameters &, const State &, const Response &,
                             const Tensor &strain_increment, double dt);

      struct ProfileMoments
      {
        double I = 0., elastic_slip = 0., Z = 0., rate = 0., absolute_rate = 0.;
        double bulk_cohesive = 0., surface_cohesive = 0., residual_cohesive = 0.;
        double bulk_viscous = 0., surface_viscous = 0., residual_viscous = 0.;
        double bulk_friction = 0., surface_friction = 0., friction_excess = 0.;
        double residual_moment = 0.;
        bool normalized = false;
      };
      /** Observer only: positive native normal weights, never a projected
       * nodal I substituted for the raw integral. No physical field changes. */
      ProfileMoments observe_profile(const Parameters &, const std::vector<double> &weights,
                                     const std::vector<Response> &samples);
    }
  }
}
#endif
