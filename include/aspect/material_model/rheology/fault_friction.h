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

#include <optional>

namespace aspect
{
  namespace MaterialModel
  {
    namespace Rheology
    {
      /** Fault resistance and candidate aging for mature surfaces and bulk sections.
       * This rheology owns constitutive parameters, never accepted history. Callers
       * supply the accepted age and explicit trial or publication duration.
       */
      template <int dim>
      class FaultFriction : public SimulatorAccess<dim>
      {
        public:
          /** Declare the supported friction laws and their parameters. */
          static
          void
          declare_parameters (ParameterHandler &prm);

          /** Read and validate constitutive parameters; no history is initialized. */
          void
          parse_parameters (ParameterHandler &prm);

          /** Time discretization of aging; independent of checkpoint spelling. */
          enum class AgingMethod
          {
            backward_euler, constant_rate_exact
          };

          struct StateUpdate
          {
            double age; ///< Age in seconds at the requested time location.
            double derivative_wrt_speed; ///< Partial derivative with respect to nonnegative speed (m/s).
          };

          /** Compute candidate aging from immutable incoming age (s), speed (m/s), and
           * duration (s). No minimum-rate floor; zero speed and duration have
           * analytic limits. Inputs must be finite, age positive, speed and duration
           * nonnegative. Does not mutate or publish history. The caller supplies a
           * half-stage or full endpoint duration explicitly. Stateful law only. */
          StateUpdate update_state(double speed, double incoming_age, double duration,
                                   AgingMethod method) const;

          /** Return whether the selected fault-friction law owns state. */
          bool has_state_variable() const;

          /** Compute the dimensionless coefficient at finite speed (m/s), at or
           * above Vmin. Dispatches on this object's configured law. Supply age
           * (s) for a stateful law and std::nullopt when state is absent. A
           * stateful law requires finite positive age; a stateless law ignores
           * age if supplied. Does not advance age or mutate history. */
          double friction_coefficient(const std::vector<double> &volume_fractions,
                                      double slip_rate, std::optional<double> age) const;

          /** Partial derivative of the coefficient with respect to speed (s/m),
           * holding age fixed. Same speed and optional-age contract as
           * friction_coefficient; does not include aging's chain rule. */
          double friction_coefficient_derivative_wrt_slip_rate(
            const std::vector<double> &volume_fractions,
            double slip_rate, std::optional<double> age) const;

          /** Partial derivative of the coefficient with respect to age (1/s),
           * holding speed fixed. Same input contract as friction_coefficient.
           * Returns zero for a stateless law, with or without supplied age. */
          double friction_coefficient_derivative_wrt_state(
            const std::vector<double> &volume_fractions,
            double slip_rate, std::optional<double> age) const;

          /** Invert the configured stateful law for a positive initial state.
           * Parameters are mixed before inversion, as in friction_coefficient. */
          double initial_state_for_friction_coefficient(
            const std::vector<double> &volume_fractions,
            double slip_rate, double coefficient) const;

          struct ResistanceResponse
          {
            double resistance; ///< Total signed resistance (Pa).
            double derivative_wrt_slip_rate; ///< Total d traction/d signed rate (Pa s/m), including aging.
            double age; ///< Age at the requested time location (s), not a committed history.
            double friction, radiation; ///< Separate signed resistance contributions (Pa).
          };
          /** Evaluate signed regularized traction at V (m/s), allowing V=0.
           * Age is advanced from old_age over dt (s) using the explicit method.
           * compression is prescribed positive Pa, radiation is Pa s/m.
           * The returned derivative includes the aging chain rule, unlike the
           * fixed-state positive-rate API. No rate floor or history mutation. */
          ResistanceResponse evaluate_resistance(const std::vector<double> &fractions,
                                                 double velocity, double old_age, double dt,
                                                 double compression, double radiation, AgingMethod aging) const;

          struct SectionSlipSolution
          {
            ResistanceResponse resistance_response;
            double slip_rate = 0.; ///< Solved signed rate (m/s).
            double resistance_derivative_wrt_free_rate = 0.; ///< df/d free_rate (Pa s/m).
          };
          /** Eliminate V + mobility*f(V)=free_rate in a scaled asinh coordinate.
           * free_rate is signed m/s and mobility is nonnegative m/(Pa s).
           * old_age is the immutable accepted age (s); dt is the requested
           * positive stage duration (s). Compression (Pa), radiation (Pa s/m)
           * and explicit aging method have the evaluate_resistance contract.
           * Returns the solved signed rate, its response and df/d free_rate
           * with mobility fixed. Does not mutate or publish history.
           * The admitted control requires provably monotone time-discrete
           * resistance; this never chooses an arbitrary weakening root. */
          SectionSlipSolution solve_section_slip_rate(const std::vector<double> &fractions,
                                                      double free_rate, double mobility,
                                                      double old_age, double dt,
                                                      double compression, double radiation, AgingMethod aging) const;

          /** Integral over trial rate from zero to V (m/s) of the time-discrete
           * signed traction, with incoming age fixed. Uses the existing Gauss48
           * search quadrature. Not stored energy or physical fV dissipation;
           * does not mutate history. Returns Pa m/s. Pressure (Pa), radiation
           * (Pa s/m), positive old_age (s) and explicit method have the
           * evaluate_resistance contract. dt is the stage duration (s). */
          double compute_resistance_potential(const std::vector<double> &fractions, double V, double old_age,
                                              double dt, double pressure, double radiation, AgingMethod aging) const;

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

          double get_reference_slip_rate() const;

          double get_minimum_slip_rate() const;

          double get_characteristic_slip_distance() const;

        private:
          /** Effective stateless coefficients; mixed before evaluating the law. */
          struct RateDependentParameters
          {
            double static_coefficient = 0.;
            double dynamic_coefficient = 0.;
            double weakening_speed = 0.;
          };

          /** Mix only active components, preserving the stateless contract. */
          RateDependentParameters rate_dependent_parameters(const std::vector<double> &) const;

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

          /** Common public coefficient contract and law dispatch. Computes the
           * coefficient and fixed-state partials together; no aging or mutation.
           * Signed section resistance uses rate_state_coefficient directly to
           * admit zero speed without the mature Vmin restriction. */
          CoefficientEvaluation coefficient_at_state(const std::vector<double> &,
                                                     double speed, std::optional<double> age) const;

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
