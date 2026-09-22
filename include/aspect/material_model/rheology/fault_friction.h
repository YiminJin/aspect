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

#ifndef _aspect_material_model_rheology_fault_friction_h
#define _aspect_material_model_rheology_fault_friction_h

#include <aspect/global.h>
#include <aspect/simulator_access.h>

namespace aspect
{
  namespace MaterialModel
  {
    namespace Rheology
    {
      template <int dim>
      class FaultFriction : public SimulatorAccess<dim>
      {
        public:
          /** Return whether the selected fault-friction law owns state. */
          bool has_state_variable() const;

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
          double update_state (const double slip_rate,
                               const double old_state,
                               const double time_step) const;

          /**
           * Update the state on the material-mixture path used by surface
           * friction. The present aging law has a single global $D_c$, but
           * accepting the mixture keeps the constitutive state update tied to
           * the same surface state as the friction coefficient.
           */
          double update_state (const std::vector<double> &volume_fractions,
                               const double               slip_rate,
                               const double               old_state,
                               const double               time_step) const;

          /**
           * Compute the friction coefficient $mu$ at the end of the current 
           * time step.
           */
          double friction_coefficient(const std::vector<double> &volume_fractions,
                                      const double               slip_rate,
                                      const double               slip_state) const;

          /**
           * Compute the friction coefficient for a stateless friction law.
           */
          double friction_coefficient(const std::vector<double> &volume_fractions,
                                      const double               slip_rate) const;

          /**Compute the partial derivative of $\mu$ with respect to $V$.
           */
          double 
          friction_coefficient_derivative_wrt_slip_rate(const std::vector<double> &volume_fractions,
                                                        const double               slip_rate,
                                                        const double               slip_state) const;

          /**
           * Compute the partial derivative of $\mu$ with respect to $V$ for a
           * stateless friction law.
           */
          double
          friction_coefficient_derivative_wrt_slip_rate(const std::vector<double> &volume_fractions,
                                                        const double               slip_rate) const;

          /**
           * Compute the time step controlled by fault slip. The time step is
           * commonly given by
           * @f[
           *  \Delta t_{\text{RSF}}=C_{\text{RSF}}\min\frac{D_c}{V},
           * @f]
           * where $C_{\text{RSF}}\in(0, 1]$ serves as the CFL number. In
           * practice, if the slip state is updated in an operator-splitting
           * manner, then the time step should be scaled by $a/b$, i.e.
           * @f[
           *  \Delta t_{\text{RSF,OS}}=C_{\text{RSF}}\min\frac{a}{b}
           *    \frac{D_c}{V}.
           * @f]
           * The above expression implies that the time step should be reduced
           * to improve the stability if the material is slip-weakening.
           */
          double
          compute_time_step(const std::vector<double> &volume_fractions,
                            const double               slip_rate,
                            const double               cfl_number,
                            const bool                 use_operator_splitting) const;

          double get_reference_slip_rate() const;

          double get_minimum_slip_rate() const;

          double get_characteristic_slip_distance() const;

          /** Invert the configured stateful law for a positive initial state.
           * Parameters are mixed before inversion, as in friction_coefficient. */
          double initial_state_for_friction_coefficient(
            const std::vector<double> &volume_fractions,
            double slip_rate, double coefficient) const;

          /** Derivatives for noncommitting within-step state evaluation. */
          double update_state_derivative_wrt_slip_rate(double slip_rate,
                                                       double old_state,
                                                       double time_step) const;
          double friction_coefficient_derivative_wrt_state(
            const std::vector<double> &volume_fractions,
            double slip_rate, double slip_state) const;

          static
          void
          declare_parameters (ParameterHandler &prm);

          void
          parse_parameters (ParameterHandler &prm);

        private:
          /** Fault-friction laws supported by the common selector. */
          enum class FrictionLaw
          {
            rate_state,
            rate_dependent
          };

          FrictionLaw friction_law = FrictionLaw::rate_state;

          double V0;
          double Vmin;
          double Dc;
          bool   regularized;
          std::vector<double> mu0;
          std::vector<double> a;
          std::vector<double> b;
          std::vector<double> dynamic_friction_coefficients;
          std::vector<double> characteristic_weakening_slip_rates;
      };

      // Inline functions
      template <int dim>
      inline bool
      FaultFriction<dim>::has_state_variable() const
      {
        return friction_law == FrictionLaw::rate_state;
      }



      template <int dim>
      inline double 
      FaultFriction<dim>::get_reference_slip_rate() const
      {
        AssertThrow(friction_law == FrictionLaw::rate_state,
                    ExcMessage("The reference slip rate is only defined for "
                               "rate-and-state fault friction."));
        return V0;
      }

      template <int dim>
      inline double 
      FaultFriction<dim>::get_minimum_slip_rate() const
      {
        return Vmin;
      }

      template <int dim>
      inline double
      FaultFriction<dim>::get_characteristic_slip_distance() const
      {
        AssertThrow(friction_law == FrictionLaw::rate_state,
                    ExcMessage("The characteristic slip distance is only defined for "
                               "rate-and-state fault friction."));
        return Dc;
      }
    }
  }
}

#endif
