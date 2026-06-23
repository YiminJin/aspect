/*
  Copyright (C) 2025 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.

  ASPECT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ASPECT; see the file LICENSE.  If not see
  <http://www.gnu.org/licenses/>.
*/

#ifndef _aspect_material_model_rheology_rate_state_friction_h
#define _aspect_material_model_rheology_rate_state_friction_h

#include <aspect/global.h>
#include <aspect/simulator_access.h>

namespace aspect
{
  namespace MaterialModel
  {
    namespace Rheology
    {
      template <int dim>
      class RateStateFriction : public SimulatorAccess<dim>
      {
        public:
          /**
           * Update the slip state $\theta$ using the aging law:
           * @f[
           *  \dot\theta = 1 - \frac{V\theta}{D_c}.
           * @f]
           * Direct time integration gives
           * @f[
           *  \theta = \frac{D_c}{V}\left(1 - 
           *  \mathrm{e}^{-\frac{V\Delta t}{D_c}}\right) +
           *  \theta^{\text{old}}\mathrm{e}^{-\frac{V\Delta t}{D_c}},
           * @f]
           * where $\Delta t$ is the time step size, and 
           * $\theta^{\text{old}}$ is the slip state at the beginning of
           * the current time step.
           *
           * If @param is_transient is set to false, then the function
           * returns the steady slip state $\theta = D_c / V$. In this case,
           * parameters @param old_slip_state and @time_step are not used.
           */
          double slip_state (const double slip_rate,
                             const double old_slip_state,
                             const double time_step,
                             const bool is_transient) const;

          /**
           * Compute the friction coefficient $mu$ for the $j$-th composition.
           */
          double friction_coefficient (const std::vector<double> &volume_fractions,
                                       const double               slip_rate,
                                       const double               old_slip_state,
                                       const double               time_step,
                                       const bool                 is_transient = true,
                                       const bool                 regularized = true) const;

          double 
          friction_coefficient_derivative(const std::vector<double> &volume_fractions,
                                          const double               slip_rate,
                                          const double               old_slip_state,
                                          const double               time_step,
                                          const bool                 is_transient = true,
                                          const bool                 regularized = true) const;

          double solve(const std::vector<double> &volume_fractions,
                       const double               initial_guess,
                       const double               old_slip_state,
                       const double               shear_stress,
                       const double               normal_stress,
                       const double               time_step,
                       const bool                 is_transient = true,
                       const bool                 regularized = true) const;

          /**
           * Compute the derivative of the friction coefficient w.r.t. the
           * slip rate for the $j$-th composition.
           */
          double friction_coefficient_derivative(const unsigned int j,
                                                 const double slip_rate,
                                                 const double old_slip_state,
                                                 const double time_step,
                                                 const bool regularized = true) const;

          double get_reference_slip_rate() const;

          double get_minimum_slip_rate() const;

          double get_characteristic_slip_distance() const;

          static
          void
          declare_parameters (ParameterHandler &prm);

          void
          parse_parameters (ParameterHandler &prm);

        private:
          std::optional<double>
          solve_steady(const std::vector<double> &volume_fractions,
                       const double               initial_guess,
                       const double               shear_stress,
                       const double               normal_stress,
                       const bool                 regularized) const;

          std::optional<double>
          solve_transient(const std::vector<double> &volume_fractions,
                          const double               initial_guess,
                          const double               old_slip_state,
                          const double               shear_stress,
                          const double               normal_stress,
                          const double               time_step,
                          const bool                 regularized) const;

          double compute_mu(const double V,
                            const double theta_old,
                            const double mu0,
                            const double a,
                            const double b,
                            const double dt,
                            const bool is_transient,
                            const bool regularized) const;

          double compute_dmu_dV(const double V,
                                const double theta_old,
                                const double mu0,
                                const double a,
                                const double b,
                                const double dt,
                                const bool is_transient,
                                const bool regularized) const;

          double reference_slip_rate;
          double minimum_slip_rate;
          double maximum_slip_rate;
          double characteristic_slip_distance;
          std::vector<double> reference_friction_coefficients;
          std::vector<double> direct_effect_parameters;
          std::vector<double> evolution_effect_parameters;
          std::vector<double> radiation_damping_coefficients;
      };

      // Inline functions
      template <int dim>
      inline double 
      RateStateFriction<dim>::get_reference_slip_rate() const
      {
        return reference_slip_rate;
      }

      template <int dim>
      inline double
      RateStateFriction<dim>::get_minimum_slip_rate() const
      {
        return minimum_slip_rate;
      }

      template <int dim>
      inline double
      RateStateFriction<dim>::get_characteristic_slip_distance() const
      {
        return characteristic_slip_distance;
      }
    }
  }
}

#endif
