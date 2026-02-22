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
           */
          double slip_state (const double slip_rate,
                             const double old_slip_state) const;

          /**
           * Compute the friction coefficient $mu$ for the $j$-th composition.
           * Note that @p slip_state is the slip state at the end of the
           * current time step, which is the output of function slip_state.
           */
          double friction_coefficient (const unsigned int j,
                                       const double slip_rate,
                                       const double slip_state,
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
          /**
           * The reference slip rate.
           */
          double V0;

          /**
           * The minimum slip rate.
           */
          double V_min;
          
          /**
           * The characteristic slip distance.
           */
          double Dc;

          /**
           * The reference friction coefficient for each chemical field.
           */
          std::vector<double> mu0;

          /**
           * The direct effect parameter for each chemical field.
           */
          std::vector<double> a;

          /**
           * The evolution effect parameter for each chemical field.
           */
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
        return V_min;
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
