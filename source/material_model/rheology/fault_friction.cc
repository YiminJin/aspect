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

#include <aspect/material_model/rheology/fault_friction.h>
#include <aspect/utilities.h>
#include <limits>
#include <deal.II/base/quadrature_lib.h>

namespace aspect
{
  namespace MaterialModel
  {
    namespace Rheology
    {
      template <int dim>
      void
      FaultFriction<dim>::
      declare_parameters(ParameterHandler &prm)
      {
        prm.declare_entry ("Friction law", "rate state",
                           Patterns::Selection("rate state|rate dependent"),
                           "Select the slip-rate-based fault friction law.");
        prm.declare_entry ("Reference slip rate", "1.e-6",
                           Patterns::Double(0.),
                           "The reference slip rate, $V_0$. Units: \\si{\\meter\\per\\second}.");
        prm.declare_entry ("Minimum slip rate", "1.e-20",
                           Patterns::Double(0.),
                           "The lower bound of slip rate. Units: \\si{\\miter\\per\\second}.");
        prm.declare_entry ("Characteristic slip distance", "0.04",
                           Patterns::Double(0.),
                           "The characteristic slip distance, $D_c$. Units: \\si{\\meter}.");
        prm.declare_entry ("Reference friction coefficients", "0.6",
                           Patterns::Anything(),
                           "List of the reference friction coefficients, $\\mu_0$, for "
                           "rate-and-state friction, or the static friction coefficients, "
                           "$\\mu_s$, for rate-dependent friction, "
                           "for background material and compositional fields, "
                           "for a total of N+1 values, where N is the number of all compositional fields or only "
                           "those corresponding to chemical compositions. Units: None.");
        prm.declare_entry ("Dynamic friction coefficients", "0.4",
                           Patterns::Anything(),
                           "List of the dynamic friction coefficients, $\\mu_d$, for "
                           "rate-dependent friction, for background material and compositional "
                           "fields. Units: None.");
        prm.declare_entry ("Characteristic weakening slip rates", "1.e-6",
                           Patterns::Anything(),
                           "List of characteristic weakening slip rates, $V_c$, for "
                           "rate-dependent friction, for background material and compositional "
                           "fields. Units: \\si{\\meter\\per\\second}.");
        prm.declare_entry ("Direct effect parameters", "0.025",
                           Patterns::Anything(),
                           "List of the direct effect parameters, $a$, "
                           "for background material and compositional fields, "
                           "for a total of N+1 values, where N is the number of all compositional fields or only "
                           "those corresponding to chemical compositions. Units: None.");
        prm.declare_entry ("Evolution effect parameters", "0.013",
                           Patterns::Anything(),
                           "List of the evolution effect parameters, $b$, "
                           "for background material and compositional fields, "
                           "for a total of N+1 values, where N is the number of all compositional fields or only "
                           "those corresponding to chemical compositions. Units: None.");
        prm.declare_entry ("Use regularized formulation", "true",
                           Patterns::Bool(),
                           "Whether to use the regularized formulation for the rate-and-state "
                           "friction coefficient.");
      }



      template <int dim>
      void
      FaultFriction<dim>::
      parse_parameters (ParameterHandler &prm)
      {
        const std::string friction_law_name = prm.get("Friction law");
        friction_law = (friction_law_name == "rate state"
                        ? FrictionLaw::rate_state
                        : FrictionLaw::rate_dependent);

        V0   = prm.get_double("Reference slip rate");
        Vmin = prm.get_double("Minimum slip rate");
        Dc   = prm.get_double("Characteristic slip distance");

        AssertThrow(numbers::is_finite(Vmin) && Vmin > 0.0,
                    ExcMessage("The minimum slip rate must be finite and positive."));

        // Retrieve the list of composition names
        std::vector<std::string> compositional_field_names = this->introspection().get_composition_names();

        // Retrieve the list of names of fields that represent chemical compositions
        std::vector<std::string> chemical_field_names = this->introspection().chemical_composition_field_names();

        // Establish that a background field is required here
        compositional_field_names.insert(compositional_field_names.begin(), "background");
        chemical_field_names.insert(chemical_field_names.begin(), "background");

        Utilities::MapParsing::Options options(chemical_field_names, "Reference friction coefficients");
        options.list_of_allowed_keys = compositional_field_names;

        mu0 = Utilities::MapParsing::parse_map_to_double_array(prm.get("Reference friction coefficients"),
                                                               options);

        if (friction_law == FrictionLaw::rate_state)
          {
            AssertThrow(numbers::is_finite(V0) && V0 > 0.0,
                        ExcMessage("The reference slip rate must be finite and positive."));
            AssertThrow(numbers::is_finite(Dc) && Dc > 0.0,
                        ExcMessage("The characteristic slip distance must be finite and positive."));

            options.property_name = "Direct effect parameters";
            a = Utilities::MapParsing::parse_map_to_double_array(prm.get("Direct effect parameters"),
                                                                 options);

            options.property_name = "Evolution effect parameters";
            b = Utilities::MapParsing::parse_map_to_double_array(prm.get("Evolution effect parameters"),
                                                                 options);

            regularized = prm.get_bool("Use regularized formulation");
            for (unsigned int j = 0; j < mu0.size(); ++j)
              {
                AssertThrow(numbers::is_finite(mu0[j]) && mu0[j] >= 0.0,
                            ExcMessage("Reference friction coefficients must be finite and nonnegative."));
                AssertThrow(numbers::is_finite(a[j])
                            && (regularized ? a[j] > 0.0 : a[j] >= 0.0),
                            ExcMessage("Direct effect parameters must be finite and nonnegative, "
                                       "and positive when using the regularized formulation."));
                AssertThrow(numbers::is_finite(b[j]) && b[j] >= 0.0,
                            ExcMessage("Evolution effect parameters must be finite and nonnegative."));
              }
          }
        else
          {
            options.property_name = "Dynamic friction coefficients";
            dynamic_friction_coefficients =
              Utilities::MapParsing::parse_map_to_double_array(
                prm.get("Dynamic friction coefficients"), options);

            options.property_name = "Characteristic weakening slip rates";
            characteristic_weakening_slip_rates =
              Utilities::MapParsing::parse_map_to_double_array(
                prm.get("Characteristic weakening slip rates"), options);

            for (unsigned int j = 0; j < mu0.size(); ++j)
              {
                AssertThrow(numbers::is_finite(mu0[j]) && mu0[j] >= 0.0,
                            ExcMessage("Static friction coefficients must be finite and nonnegative."));
                AssertThrow(numbers::is_finite(dynamic_friction_coefficients[j])
                            && dynamic_friction_coefficients[j] >= 0.0
                            && dynamic_friction_coefficients[j] <= mu0[j],
                            ExcMessage("Dynamic friction coefficients must be finite, nonnegative, "
                                       "and no greater than the corresponding static coefficient."));
                AssertThrow(numbers::is_finite(characteristic_weakening_slip_rates[j])
                            && characteristic_weakening_slip_rates[j] > 0.0,
                            ExcMessage("Characteristic weakening slip rates must be finite and positive."));
              }
          }
      }
      template <int dim>
      typename FaultFriction<dim>::StateUpdate
      FaultFriction<dim>::update_state(const double speed, const double old_age,
                                       const double dt, const AgingMethod method) const
      {
        AssertThrow(has_state_variable(), ExcMessage("The selected friction law has no state variable."));
        AssertThrow(std::isfinite(speed) && speed >= 0. && std::isfinite(old_age) && old_age > 0.
                    && std::isfinite(dt) && dt >= 0., ExcMessage("Invalid aging inputs."));
        if (dt == 0.) return {old_age, 0.};
        const double x = speed * dt / Dc;
        if (method == AgingMethod::backward_euler)
          {
            const double denominator = 1. + x;
            const double age = (old_age + dt) / denominator;
            return {age, -age *dt / (Dc * denominator)};
          }
        const double exponential = std::exp(-x);
        const double integral = x == 0. ? 1. : -std::expm1(-x) / x;
        const double age = old_age * exponential + dt * integral;
        // Differentiate the exact integral of exp(-speed*t/Dc). The two O(x)
        // terms cancel; use its Taylor limit without a speed substitution.
        const double kernel = x < 1e-3
                              ? -.5 + x * (1. / 3. + x * (-1. / 8. + x * (1. / 30. + x * (-1. / 144. + x / 840.))))
                              : (std::expm1(-x) + x * exponential) / (x * x);
        return {age, dt / Dc *(dt * kernel - old_age * exponential)};
      }

      template <int dim>
      typename FaultFriction<dim>::RateDependentParameters
      FaultFriction<dim>::rate_dependent_parameters(const std::vector<double> &volume_fractions) const
      {
        AssertThrow(friction_law == FrictionLaw::rate_dependent,
                    ExcMessage("Stateless parameters require rate-dependent friction."));
        AssertDimension(volume_fractions.size(), mu0.size());
        AssertDimension(dynamic_friction_coefficients.size(), mu0.size());
        AssertDimension(characteristic_weakening_slip_rates.size(), mu0.size());
        RateDependentParameters result;
        for (unsigned int j = 0; j < volume_fractions.size(); ++j)
          if (volume_fractions[j] > 0)
            {
              result.static_coefficient += volume_fractions[j] * mu0[j];
              result.dynamic_coefficient += volume_fractions[j] * dynamic_friction_coefficients[j];
              result.weakening_speed += volume_fractions[j] * characteristic_weakening_slip_rates[j];
            }
        return result;
      }

      template <int dim>
      typename FaultFriction<dim>::RateStateParameters
      FaultFriction<dim>::rate_state_parameters(const std::vector<double> &fractions) const
      {
        AssertThrow(has_state_variable(), ExcMessage("Rate-state parameters require a stateful friction law."));
        AssertThrow(fractions.size() == mu0.size() && a.size() == mu0.size() && b.size() == mu0.size(),
                    ExcMessage("Inconsistent rate-state material mixture sizes."));
        RateStateParameters result;
        for (unsigned int j = 0; j < fractions.size(); ++j)
          {
            AssertThrow(std::isfinite(fractions[j]) && fractions[j] >= 0.,
                        ExcMessage("Friction material fractions must be finite and nonnegative."));
            if (fractions[j] > 0.)
              {
                result.reference += fractions[j] * mu0[j];
                result.direct += fractions[j] * a[j];
                result.evolution += fractions[j] * b[j];
              }
          }
        return result;
      }

      template <int dim>
      typename FaultFriction<dim>::CoefficientEvaluation
      FaultFriction<dim>::rate_state_coefficient(const RateStateParameters &p,
                                                 const double speed, const double age) const
      {
        AssertThrow(std::isfinite(age) && age > 0. && std::isfinite(speed) && speed >= 0.,
                    ExcMessage("Invalid stateful friction inputs."));
        if (!regularized)
          {
            AssertThrow(speed > 0., ExcMessage("Logarithmic friction requires positive speed."));
            return {p.reference + p.direct *std::log(speed / V0) + p.evolution *std::log(age *V0 / Dc),
                    p.direct / speed, p.evolution / age
                   };
          }
        AssertThrow(p.direct > 0., ExcMessage("Regularized friction requires positive effective direct effect."));
        const double scale = std::exp((p.reference + p.evolution * std::log(age * V0 / Dc)) / p.direct) / (2.*V0);
        const double z = speed * scale;
        const double ratio = z / std::hypot(1., z);
        return {p.direct *std::asinh(z), speed > 0. ? p.direct *ratio / speed : p.direct * scale,
                p.evolution *ratio / age
               };
      }

      template <int dim>
      typename FaultFriction<dim>::CoefficientEvaluation
      FaultFriction<dim>::coefficient_at_state(const std::vector<double> &fractions,
                                               const double speed,
                                               const std::optional<double> age) const
      {
        AssertThrow(std::isfinite(speed) && speed >= Vmin,
                    ExcMessage("The slip rate is below the minimum admissible slip rate."));
        if (has_state_variable())
          {
            AssertThrow(age.has_value(), ExcMessage("Rate-and-state friction requires an age."));
            return rate_state_coefficient(rate_state_parameters(fractions), speed, *age);
          }

        const auto parameters = rate_dependent_parameters(fractions);
        const double mu_s = parameters.static_coefficient;
        const double mu_d = parameters.dynamic_coefficient;
        const double Vc = parameters.weakening_speed;
        return {mu_d + (mu_s - mu_d) *Vc / (Vc + speed),
                -(mu_s - mu_d) *Vc / ((Vc + speed) * (Vc + speed)),
                0.
               };
      }

      template <int dim>
      double FaultFriction<dim>::friction_coefficient(
        const std::vector<double> &fractions, const double speed,
        const std::optional<double> age) const
      {
        return coefficient_at_state(fractions, speed, age).value;
      }

      template <int dim>
      double FaultFriction<dim>::friction_coefficient_derivative_wrt_slip_rate(
        const std::vector<double> &fractions, const double speed,
        const std::optional<double> age) const
      {
        return coefficient_at_state(fractions, speed, age).speed_partial;
      }

      template <int dim>
      double FaultFriction<dim>::friction_coefficient_derivative_wrt_state(
        const std::vector<double> &fractions, const double speed,
        const std::optional<double> age) const
      {
        return coefficient_at_state(fractions, speed, age).state_partial;
      }

      template <int dim>
      double
      FaultFriction<dim>::
      initial_state_for_friction_coefficient(const std::vector<double> &fractions,
                                             const double V,
                                             const double coefficient) const
      {
        AssertThrow(std::isfinite(V) && V >= Vmin && std::isfinite(coefficient),
                    ExcMessage("Invalid initial friction target or slip rate."));
        const auto parameters = rate_state_parameters(fractions);
        const double f = parameters.reference, direct = parameters.direct, evolution = parameters.evolution;
        AssertThrow(evolution > 0. && (!regularized || coefficient > 0.),
                    ExcMessage("The requested friction target has no unique positive initial state."));
        double state_term = coefficient - f - direct * std::log(V / V0);
        if (regularized)
          {
            // log(2 sinh(z)) = z + log(1-exp(-2z)): avoid overflow
            // while retaining the small-target limit without cancellation.
            const double z = coefficient / direct;
            state_term = direct * (std::log(V0 / V) + z + std::log(-std::expm1(-2 * z))) - f;
          }
        const double theta = Dc / V0 * std::exp(state_term / evolution);
        AssertThrow(std::isfinite(theta) && theta > 0.,
                    ExcMessage("Initial friction state is outside the representable positive range."));
        return theta;
      }




      template <int dim>
      typename FaultFriction<dim>::ResistanceResponse
      FaultFriction<dim>::evaluate_resistance(const std::vector<double> &fractions,
                                              const double V, const double old_age,
                                              const double dt, const double pressure,
                                              const double radiation, const AgingMethod aging) const
      {
        AssertThrow(friction_law == FrictionLaw::rate_state && regularized
                    && old_age > 0. && dt > 0. && pressure > 0. && radiation >= 0.,
                    ExcMessage("Bulk sections require regularized RSF, positive age and prescribed compression."));
        const auto parameters = rate_state_parameters(fractions);
        const auto age = update_state(std::abs(V), old_age, dt, aging);
        const auto coefficient = rate_state_coefficient(parameters, std::abs(V), age.age);
        const double friction = std::copysign(pressure * coefficient.value, V);
        // For either sign, d[sign(V)*mu(|V|,theta(|V|))]/dV has the
        // positive-speed total derivative. The state partial vanishes at zero.
        const double slope = pressure * (coefficient.speed_partial
                                         + coefficient.state_partial * age.derivative_wrt_speed) + radiation;
        return {friction + radiation * V, slope, age.age, friction, radiation * V};
      }

      template <int dim>
      typename FaultFriction<dim>::SectionSlipSolution
      FaultFriction<dim>::solve_section_slip_rate(const std::vector<double> &fractions,
                                                  const double free_rate, const double mobility,
                                                  const double old_age, const double dt,
                                                  const double pressure, const double radiation, const AgingMethod aging) const
      {
        const auto zero = evaluate_resistance(fractions, 0., old_age, dt, pressure, radiation, aging);
        const auto parameters = rate_state_parameters(fractions);
        const double direct = parameters.direct;
        const double evolution = parameters.evolution;
        const double reference = parameters.reference;
        // R'(V)=p*ratio*(a/|V|-b/(Dc/dt+|V|))+eta_rad.
        // Exact aging is a positive mixture of exp(-|V|r/Dc), 0<=r<=dt,
        // so d(log theta)/d|V| >= -dt/Dc: R' >= eta_rad-p*b*dt/Dc.
        // For BE the negative part is bounded by -p*dt/Dc*(sqrt(b)-sqrt(a))_+^2.
        const double weakening = std::max(0., std::sqrt(evolution) - std::sqrt(direct));
        AssertThrow(mobility >= 0. && radiation >= pressure * dt / Dc * (aging == AgingMethod::constant_rate_exact ? evolution : weakening * weakening),
                    ExcMessage("Bulk section elimination requires a monotone endpoint RSF response. Multiple-root selection is not qualified."));
        if (free_rate == 0.)return {zero, 0., zero.derivative_wrt_slip_rate / (1. + mobility * zero.derivative_wrt_slip_rate)};
        if (mobility == 0.)
          {
            const auto response = evaluate_resistance(fractions, free_rate, old_age, dt, pressure, radiation, aging);
            return {response, free_rate, response.derivative_wrt_slip_rate};
          }
        const double scale = 2.*V0 * std::exp(-(reference + evolution * std::log((old_age + dt) * V0 / Dc)) / direct);
        AssertThrow(scale > 0. && std::isfinite(scale), ExcMessage("Section RSF rate scale is outside floating-point range."));
        const double target = std::abs(free_rate);
        const double sign = std::copysign(1., free_rate);
        long double lower = 0.;
        long double upper = std::asinh(static_cast<long double>(target) / scale);
        long double coordinate = .5L * upper;
        // The asinh coordinate resolves cold creep while retaining the original
        // safeguarded Newton bracket and roundoff-scaled root certificate.
        ResistanceResponse response = zero;
        double rate = 0.;
        bool converged = false;
        for (unsigned int iteration = 0; iteration < 100; ++iteration)
          {
            rate = static_cast<double>(scale * std::sinh(coordinate));
            response = evaluate_resistance(fractions, rate, old_age, dt, pressure, radiation, aging);
            const long double residual = static_cast<long double>(rate) + static_cast<long double>(mobility) * response.resistance - target;
            if (std::abs(residual) <= 32.*std::numeric_limits<double>::epsilon()*target)
              {
                converged = true;
                break;
              }
            if (residual > 0.)upper = coordinate;
            else lower = coordinate;
            const long double derivative = (1. + mobility * response.derivative_wrt_slip_rate) * scale * std::cosh(coordinate);
            const long double trial = coordinate - residual / derivative;
            coordinate = trial > lower && trial < upper ? trial : .5 * (lower + upper);
          }
        AssertThrow(converged, ExcMessage("Monotone section RSF elimination did not reach roundoff accuracy."));
        rate *= sign;
        response = evaluate_resistance(fractions, rate, old_age, dt, pressure, radiation, aging);
        return {response, rate, response.derivative_wrt_slip_rate / (1. + mobility * response.derivative_wrt_slip_rate)};
      }

      template <int dim>
      double FaultFriction<dim>::compute_resistance_potential(const std::vector<double> &fractions,
                                                              const double V, const double old_age,
                                                              const double dt, const double pressure,
                                                              const double radiation, const AgingMethod aging) const
      {
        if (V == 0.) return 0.;
        const auto zero = evaluate_resistance(fractions, 0., old_age, dt, pressure, radiation, aging);
        const double direct = rate_state_parameters(fractions).direct;
        const double scale = pressure * direct / (zero.derivative_wrt_slip_rate - radiation);
        const double limit = std::asinh(std::abs(V) / scale);
        // Logarithmic rate coordinates resolve cold creep and finite slip with
        // the same smooth quadrature. This is search quadrature, not a change
        // to constitutive/residual or physical work integration.
        static const QGauss<1> rule(48);
        double value = 0.;
        for (unsigned int q = 0; q < rule.size(); ++q)
          {
            const double u = limit * rule.point(q)[0];
            const double rate = scale * std::sinh(u);
            value += rule.weight(q) * evaluate_resistance(fractions, rate, old_age, dt,
                                                          pressure, radiation, aging).resistance * scale * std::cosh(u);
          }
        return limit * value;
      }

      template <int dim>
      double
      FaultFriction<dim>::
      compute_time_step(const std::vector<double> &volume_fractions,
                        const double               V_raw,
                        const double               cfl_number,
                        const bool                 use_operator_splitting) const
      {
        AssertDimension(volume_fractions.size(), mu0.size());

        if (friction_law == FrictionLaw::rate_dependent)
          return std::numeric_limits<double>::max();

        AssertThrow(std::isfinite(V_raw) && V_raw >= Vmin,
                    ExcMessage("The slip rate is below the minimum admissible slip rate."));
        const double V = V_raw;
        if (use_operator_splitting == false)
          return cfl_number * Dc / V;

        const auto parameters = rate_state_parameters(volume_fractions);
        const double a_eff = parameters.direct, b_eff = parameters.evolution;

        // Limit the time step only for rate-softening regions
        return (b_eff == 0. || a_eff > b_eff ?
                std::numeric_limits<double>::max() :
                cfl_number * (a_eff * Dc) / (b_eff * V));
      }



      template <int dim>
      std::vector<double>
      FaultFriction<dim>::signed_parameter_record() const
      {
        AssertThrow(friction_law == FrictionLaw::rate_state && regularized,
                    ExcMessage("Signed section law requires regularized rate and state friction."));
        std::vector<double> result {V0, Dc};
        for (const auto *values :
        {
          &mu0, &a, &b
        })
        result.insert(result.end(), values->begin(), values->end());
        return result;
      }


    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace MaterialModel
  {
    namespace Rheology
    {
#define INSTANTIATE(dim) \
  template class FaultFriction<dim>;

      ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
    }
  }
}
