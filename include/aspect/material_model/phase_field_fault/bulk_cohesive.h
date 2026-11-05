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

      /** Physical coefficients of the default-off bulk law (not surface RSF). */
      struct Parameters
      {
        double G; ///< Shear modulus (Pa).
        double eta; ///< Viscosity (Pa s).
        double m; ///< Dimensionless phase-compliance parameter.
        double resistance; ///< Bulk shear resistance (Pa).
      };

      /** Dimensionless compliance h(phi) and its phase derivatives. */
      struct Compliance
      {
        double h; ///< Compliance.
        double first; ///< dh/dphi.
        double second; ///< d2h/dphi2.
      };
      /** Evaluate compliance on its physical phase domain. */
      Compliance compliance(double phi, double m);

      /** Point history supplied by value; update() never mutates its parent. */
      struct State
      {
        Tensor T; ///< Full spatial 3-D trial deviatoric stress (Pa).
        double a = 0.; ///< Elastic engineering shear.
        double signed_shear = 0.; ///< Accumulated signed inelastic shear.
        double absolute_shear = 0.; ///< Accumulated absolute inelastic shear.
        double phi = 0.; ///< Physical phase/obstacle.
        double q = 0.; ///< Cohesive shear stress (Pa).
      };

      /** Endpoint and tangent data. Energy densities are J/m^3, dissipation
       * densities W/m^3; suffix dt means derivative with respect to trial shear
       * stress t, not time. This value object has no MPI/cache ownership. */
      struct Response
      {
        State state; ///< Unpublished endpoint.
        Tensor stress; ///< Physical stress T-G*a in the xy component (Pa).
        double w = 0.; ///< Inelastic bulk engineering shear rate (1/s).
        double beta = 0.; ///< Backward-Euler relaxation factor.
        double da_dt = 0., da_dphi = 0., dq_dt = 0., dq_dphi = 0.;
        double driving = 0., driving_dt = 0., driving_dphi = 0.;
        double bulk_energy = 0., cohesive_energy = 0.;
        double bulk_dissipation = 0., cohesive_dissipation = 0., friction_dissipation = 0.;
        bool sliding = false, intact = false;
      };

      /** Backward Euler from one immutable incoming history. The increment
       * is dev3 symgrad(du); the orientation is horizontal throughout bulk. */
      Response update(const Parameters &parameters, const State &incoming, const Tensor &strain_increment,
                      double phi, double dt);

      /** Endpoint work identity, all entries energy densities (J/m^3).
       * Keeps backward-Euler losses separate from physical dissipation. */
      struct StepAccounting
      {
        double internal_work = 0., storage_change = 0., physical_dissipation = 0.;
        double stress_loss = 0., traction_loss = 0., compliance_work = 0.;
        double error = 0.;
      };
      /** Evaluate the discrete work balance without modifying history. */
      StepAccounting account(const Parameters &parameters, const State &incoming, const Response &response,
                             const Tensor &strain_increment, double dt);

      /** Derived normal integrals only: I and elastic_slip in m, Z in Pa m,
       * rates in m/s; integrated energy/dissipation in J/m^2 and W/m^2.
       * Residual moments quantify failure of the common-profile reduction. */
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
      ProfileMoments observe_profile(const Parameters &parameters, const std::vector<double> &weights,
                                     const std::vector<Response> &samples);
    }
  }
}
#endif
