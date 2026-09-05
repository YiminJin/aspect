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

namespace aspect
{
  namespace MaterialModel
  {
    namespace Rheology
    {
      template <int dim>
      double
      FaultFriction<dim>::
      update_state(const double V_raw,
                   const double theta_old,
                   const double dt) const
      {
        AssertThrow(friction_law == FrictionLaw::rate_state,
                    ExcMessage("The selected fault-friction law has no state variable."));
        AssertThrow(theta_old > 0, ExcMessage("The slip state is non-positive."));
        AssertThrow(dt >= 0, ExcMessage("Time step is negative."));

        AssertThrow(std::isfinite(V_raw) && V_raw >= Vmin,
                    ExcMessage("The slip rate is below the minimum admissible slip rate."));
        const double V = V_raw;
        const double x = V * dt / Dc;
        return -Dc / V * std::expm1(-x) + theta_old * std::exp(-x);
      }



      template <int dim>
      double
      FaultFriction<dim>::
      friction_coefficient(const std::vector<double> &volume_fractions,
                           const double               V_raw,
                           const double               theta) const
      {
        AssertThrow(friction_law == FrictionLaw::rate_state,
                    ExcMessage("The stateful friction-coefficient interface requires "
                               "the rate-and-state fault-friction law."));
        AssertDimension(volume_fractions.size(), mu0.size());
        AssertThrow(theta > 0, ExcMessage("The slip state is non-positive."));

        double mu0_eff = 0, a_eff = 0, b_eff = 0;
        for (unsigned int j = 0; j < volume_fractions.size(); ++j)
          if (volume_fractions[j] > 0)
            {
              mu0_eff += volume_fractions[j] * mu0[j];
              a_eff   += volume_fractions[j] * a[j];
              b_eff   += volume_fractions[j] * b[j];
            }

        AssertThrow(std::isfinite(V_raw) && V_raw >= Vmin,
                    ExcMessage("The slip rate is below the minimum admissible slip rate."));
        const double V = V_raw;

        return (regularized
                ?
                a_eff * std::asinh(V / (2. * V0) * std::exp((mu0_eff + b_eff * std::log(theta * V0 / Dc)) / a_eff))
                :
                mu0_eff + a_eff * std::log(V / V0) + b_eff * std::log(theta * V0 / Dc));
      }



      template <int dim>
      double
      FaultFriction<dim>::
      friction_coefficient(const std::vector<double> &volume_fractions,
                           const double               V_raw) const
      {
        AssertThrow(friction_law == FrictionLaw::rate_dependent,
                    ExcMessage("The stateless friction-coefficient interface requires "
                               "the rate-dependent fault-friction law."));
        AssertDimension(volume_fractions.size(), mu0.size());

        double mu_s = 0, mu_d = 0, Vc = 0;
        for (unsigned int j = 0; j < volume_fractions.size(); ++j)
          if (volume_fractions[j] > 0)
            {
              mu_s += volume_fractions[j] * mu0[j];
              mu_d += volume_fractions[j] * dynamic_friction_coefficients[j];
              Vc   += volume_fractions[j] * characteristic_weakening_slip_rates[j];
            }

        AssertThrow(std::isfinite(V_raw) && V_raw >= Vmin,
                    ExcMessage("The slip rate is below the minimum admissible slip rate."));
        const double V = V_raw;
        return mu_d + (mu_s - mu_d) * Vc / (Vc + V);
      }



      template <int dim>
      double
      FaultFriction<dim>::
      friction_coefficient_derivative_wrt_slip_rate(const std::vector<double> &volume_fractions,
                                                    const double               V_raw,
                                                    const double               theta) const
      {
        AssertThrow(friction_law == FrictionLaw::rate_state,
                    ExcMessage("The stateful friction-derivative interface requires "
                               "the rate-and-state fault-friction law."));
        AssertDimension(volume_fractions.size(), mu0.size());
        AssertThrow(theta > 0, ExcMessage("The slip state is non-positive."));

        double mu0_eff = 0, a_eff = 0, b_eff = 0;
        for (unsigned int j = 0; j < volume_fractions.size(); ++j)
          if (volume_fractions[j] > 0)
            {
              mu0_eff += volume_fractions[j] * mu0[j];
              a_eff   += volume_fractions[j] * a[j];
              b_eff   += volume_fractions[j] * b[j];
            }

        AssertThrow(std::isfinite(V_raw) && V_raw >= Vmin,
                    ExcMessage("The slip rate is below the minimum admissible slip rate."));
        const double V = V_raw;

        double dmu_dV = a_eff / V;
        if (regularized)
          {
            const double Z = V / (2. * V0) * std::exp((mu0_eff + b_eff * std::log(V0 * theta / Dc)) / a_eff);
            if (Z < 1.e6)
              dmu_dV *= Z / (std::sqrt(1. + Z * Z));
          }

        return dmu_dV;
      }



      template <int dim>
      double
      FaultFriction<dim>::
      friction_coefficient_derivative_wrt_slip_rate(
        const std::vector<double> &volume_fractions,
        const double               V_raw) const
      {
        AssertThrow(friction_law == FrictionLaw::rate_dependent,
                    ExcMessage("The stateless friction-derivative interface requires "
                               "the rate-dependent fault-friction law."));
        AssertDimension(volume_fractions.size(), mu0.size());

        double mu_s = 0, mu_d = 0, Vc = 0;
        for (unsigned int j = 0; j < volume_fractions.size(); ++j)
          if (volume_fractions[j] > 0)
            {
              mu_s += volume_fractions[j] * mu0[j];
              mu_d += volume_fractions[j] * dynamic_friction_coefficients[j];
              Vc   += volume_fractions[j] * characteristic_weakening_slip_rates[j];
            }

        AssertThrow(std::isfinite(V_raw) && V_raw >= Vmin,
                    ExcMessage("The slip rate is below the minimum admissible slip rate."));
        const double V = V_raw;
        return -(mu_s - mu_d) * Vc / ((Vc + V) * (Vc + V));
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

        double a_eff = 0, b_eff = 0;
        for (unsigned int j = 0; j < volume_fractions.size(); ++j)
          if (volume_fractions[j] > 0)
            {
              a_eff += volume_fractions[j] * a[j];
              b_eff += volume_fractions[j] * b[j];
            }

        // Limit the time step only for rate-softening regions
        return (a_eff > b_eff ? 
                std::numeric_limits<double>::max() :
                cfl_number * (a_eff * Dc) / (b_eff * V));
      }



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
