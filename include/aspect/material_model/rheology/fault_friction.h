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
          /** Time discretization of aging; independent of checkpoint spelling. */
          enum class AgingMethod
          {
            backward_euler, constant_rate_exact
          };

          struct AgeEvaluation
          {
            double age; ///< Age in seconds at the requested time location.
            double speed_derivative; ///< Partial derivative with respect to nonnegative speed (m/s).
          };

          /** Evaluate aging from immutable incoming age (s), speed (m/s), and
           * duration (s). No minimum-rate floor; zero speed and duration have
           * analytic limits. Does not mutate history. The caller supplies a
           * half-stage or full endpoint duration explicitly. Stateful law only. */
          AgeEvaluation evaluate_age(double speed, double incoming_age, double duration,
                                     AgingMethod method) const;

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

          /** Exact configured signed-law parameters for restart identity. */
          std::vector<double> signed_parameter_record() const;

          struct SignedEndpoint
          {
            double traction; ///< Total signed resistance (Pa).
            double derivative; ///< Total d traction/d signed rate (Pa s/m), including aging.
            double age; ///< Age at the requested time location (s), not a committed history.
            double friction, radiation; ///< Separate signed resistance contributions (Pa).
          };
          /** Integral over trial rate from zero to V (m/s) of the time-discrete
           * signed traction, with incoming age fixed. Uses the existing Gauss48
           * search quadrature. Not stored energy or physical fV dissipation;
           * does not mutate history. dt is the selected stage duration (s). */
          double section_primitive(const std::vector<double> &, double V, double old_age,
                                   double dt, double pressure, double radiation, AgingMethod aging = AgingMethod::backward_euler) const;

          struct SectionEndpoint
          {
            SignedEndpoint law;
            double rate=0., force_derivative=0.; ///< df/d(free_rate), with fixed mobility.
          };
          /** Eliminate V + mobility*f(V)=free_rate in a scaled asinh coordinate.
           * The admitted control requires a provably monotone endpoint law;
           * this never chooses an arbitrary velocity-weakening root. */
          SectionEndpoint section_endpoint(const std::vector<double> &fractions,
                                           double free_rate, double mobility,
                                           double old_age, double dt,
                                           double compression, double radiation, AgingMethod aging = AgingMethod::backward_euler) const;

          /** Evaluate signed regularized traction at V (m/s), allowing V=0.
           * Age is advanced from old_age over dt (s) using the explicit method.
           * compression is prescribed positive Pa, radiation is Pa s/m.
           * The returned derivative includes the aging chain rule, unlike the
           * fixed-state positive-rate API. No rate floor or history mutation. */
          SignedEndpoint signed_endpoint(const std::vector<double> &fractions,
                                         double velocity, double old_age, double dt,
                                         double compression, double radiation, AgingMethod aging = AgingMethod::backward_euler) const;

          double get_reference_slip_rate() const;

          double get_minimum_slip_rate() const;

          double get_characteristic_slip_distance() const;

          /** Invert the configured stateful law for a positive initial state.
           * Parameters are mixed before inversion, as in friction_coefficient. */
          double initial_state_for_friction_coefficient(
            const std::vector<double> &volume_fractions,
            double slip_rate, double coefficient) const;

          /** Partial derivative with respect to age (1/s) at fixed positive
           * speed. Stateful interface only; does not advance age or history. */
          double friction_coefficient_derivative_wrt_state(
            const std::vector<double> &volume_fractions,
            double slip_rate, double slip_state) const;

          static
          void
          declare_parameters (ParameterHandler &prm);

          void
          parse_parameters (ParameterHandler &prm);

        private:
          struct RateStateParameters
          {
            double reference = 0., direct = 0., evolution = 0.;
          };

          struct CoefficientEvaluation
          {
            double value, speed_partial, state_partial;
          };

          /** Mix configured stateful parameters once, skipping inactive materials.
           * Validate the law and all array sizes before indexing. */
          RateStateParameters rate_state_parameters(const std::vector<double> &) const;

          /** Positive-speed coefficient and partial derivatives at fixed age.
           * The regularized branch also admits zero speed. No aging or mutation. */
          CoefficientEvaluation rate_state_coefficient(const RateStateParameters &,
                                                       double speed, double age) const;

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
