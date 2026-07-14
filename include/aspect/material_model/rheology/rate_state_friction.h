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
    template <int dim>
    class RSFAdditionalOutputs : public NamedAdditionalMaterialOutputs<dim>
    {
      public:
        RSFAdditionalOutputs(const unsigned int n_points);

        std::vector<double> get_nth_output(const unsigned int idx) const override;

        std::vector<double> friction_coefficients;
    };


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
           */
          double slip_state (const double slip_rate,
                             const double old_slip_state,
                             const double time_step) const;

          /**
           * Compute the friction coefficient $mu$ at the end of the current 
           * time step.
           */
          double friction_coefficient(const std::vector<double> &volume_fractions,
                                      const double               slip_rate,
                                      const double               slip_state) const;

          /**Compute the partial derivative of $\mu$ with respect to $V$.
           */
          double 
          friction_coefficient_derivative_wrt_slip_rate(const std::vector<double> &volume_fractions,
                                                        const double               slip_rate,
                                                        const double               slip_state) const;

          double get_reference_slip_rate() const;

          double get_minimum_slip_rate() const;

          double get_characteristic_slip_distance() const;

          static
          void
          declare_parameters (ParameterHandler &prm);

          void
          parse_parameters (ParameterHandler &prm);

        private:
          double V0;
          double Vmin;
          double Vmax;
          double Dc;
          bool   regularized;
          std::vector<double> mu0;
          std::vector<double> a;
          std::vector<double> b;
      };

      // Inline functions
      template <int dim>
      inline double 
      RateStateFriction<dim>::get_reference_slip_rate() const
      {
        return V0;
      }

      template <int dim>
      inline double 
      RateStateFriction<dim>::get_minimum_slip_rate() const
      {
        return Vmin;
      }

      template <int dim>
      inline double
      RateStateFriction<dim>::get_characteristic_slip_distance() const
      {
        return Dc;
      }
    }
  }
}

#endif
