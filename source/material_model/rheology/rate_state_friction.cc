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

#include <aspect/material_model/rheology/rate_state_friction.h>
#include <aspect/utilities.h>

namespace aspect
{
  namespace MaterialModel
  {
    namespace Rheology
    {
      template <int dim>
      double
      RateStateFriction<dim>::
      slip_state(const double V_raw,
                 const double theta_old,
                 const double dt) const
      {
        AssertThrow(theta_old > 0, ExcMessage("The slip state is non-positive."));
        AssertThrow(dt >= 0, ExcMessage("Time step is negative."));

        const double V = std::clamp(V_raw, Vmin, Vmax);
        const double x = V * dt / Dc;
        return -Dc / V * std::expm1(-x) + theta_old * std::exp(-x);
      }



      template <int dim>
      double
      RateStateFriction<dim>::
      friction_coefficient(const std::vector<double> &volume_fractions,
                           const double V_raw,
                           const double theta_old,
                           const double dt) const
      {
        AssertDimension(volume_fractions.size(), mu0.size());

        double mu0_eff = 0, a_eff = 0, b_eff = 0;
        for (unsigned int j = 0; j < volume_fractions.size(); ++j)
          if (volume_fractions[j] > 0)
            {
              mu0_eff += volume_fractions[j] * mu0[j];
              a_eff   += volume_fractions[j] * a[j];
              b_eff   += volume_fractions[j] * b[j];
            }

        const double V = std::clamp(V_raw, Vmin, Vmax);
        const double theta = slip_state(V, theta_old, dt);

        return (regularized
                ?
                a_eff * std::asinh(V / (2. * V0) * std::exp((mu0_eff + b_eff * std::log(theta * V0 / Dc)) / a_eff))
                :
                mu0_eff + a_eff * std::log(V / V0) + b_eff * std::log(theta * V0 / Dc));
      }



      template <int dim>
      double
      RateStateFriction<dim>::
      friction_coefficient_derivative(const std::vector<double> &volume_fractions,
                                      const double               V_raw,
                                      const double               theta_old,
                                      const double               dt) const
      {
        AssertIndexRange(volume_fractions.size(), mu0.size());
        AssertThrow(theta_old > 0, ExcMessage("The slip state is non-positive."));
        AssertThrow(dt > 0, ExcMessage("Time step is non-positive."));

        double mu0_eff = 0, a_eff = 0, b_eff = 0;
        for (unsigned int j = 0; j < volume_fractions.size(); ++j)
          if (volume_fractions[j] > 0)
            {
              mu0_eff += volume_fractions[j] * mu0[j];
              a_eff   += volume_fractions[j] * a[j];
              b_eff   += volume_fractions[j] * b[j];
            }

        const double V = std::clamp(V_raw, Vmin, Vmax);
        const double x = V * dt / Dc;
        const double E = std::exp(-x);
        const double Em1 = std::expm1(-x);
        const double theta = -Dc / V * Em1 + theta_old * E;
        const double dtheta_dV = Dc / (V * V) * Em1 + (dt / V - theta_old * dt / Dc) * E;

        double dmu_dV = a_eff / V + b_eff / theta * dtheta_dV;
        if (regularized)
          {
            const double Z = V / (2. * V0) * std::exp((mu0_eff + b_eff * std::log(V0 * theta / Dc)) / a_eff);
            if (Z < 1.e6)
              dmu_dV *= Z / (std::sqrt(1. + Z * Z));
          }

        return dmu_dV;
      }



      template <int dim>
      void
      RateStateFriction<dim>::
      declare_parameters(ParameterHandler &prm)
      {
        prm.declare_entry ("Reference slip rate", "1.e-6",
                           Patterns::Double(0.),
                           "The reference slip rate, $V_0$. Units: \\si{\\meter\\per\\second}.");
        prm.declare_entry ("Minimum slip rate", "1.e-20",
                           Patterns::Double(0.),
                           "The lower bound of slip rate. Units: \\si{\\miter\\per\\second}.");
        prm.declare_entry ("Maximum slip rate", "100",
                           Patterns::Double(0.),
                           "The upper bound of slip rate. Units: \\si{\\miter\\per\\second}.");
        prm.declare_entry ("Characteristic slip distance", "0.04",
                           Patterns::Double(0.),
                           "The characteristic slip distance, $D_c$. Units: \\si{\\meter}.");
        prm.declare_entry ("Reference friction coefficients", "0.6",
                           Patterns::List(Patterns::Double(0.)),
                           "List of the reference friction coefficients, $\\mu_0$, "
                           "for background material and compositional fields, "
                           "for a total of N+1 values, where N is the number of all compositional fields or only "
                           "those corresponding to chemical compositions. Units: None.");
        prm.declare_entry ("Direct effect parameters", "0.025",
                           Patterns::List(Patterns::Double(0.)),
                           "List of the direct effect parameters, $a$, "
                           "for background material and compositional fields, "
                           "for a total of N+1 values, where N is the number of all compositional fields or only "
                           "those corresponding to chemical compositions. Units: None.");
        prm.declare_entry ("Evolution effect parameters", "0.013",
                           Patterns::List(Patterns::Double(0.)),
                           "List of the evolution effect parameters, $b$, "
                           "for background material and compositional fields, "
                           "for a total of N+1 values, where N is the number of all compositional fields or only "
                           "those corresponding to chemical compositions. Units: None.");
      }



      template <int dim>
      void
      RateStateFriction<dim>::
      parse_parameters (ParameterHandler &prm)
      {
        V0   = prm.get_double("Reference slip rate");
        Vmin = prm.get_double("Minimum slip rate");
        Vmax = prm.get_double("Maximum slip rate");
        Dc   = prm.get_double("Characteristic slip distance");

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

        options.property_name = "Direct effect parameters";
        a = Utilities::MapParsing::parse_map_to_double_array(prm.get("Direct effect parameters"),
                                                             options);

        options.property_name = "Evolution effect parameters";
        b = Utilities::MapParsing::parse_map_to_double_array(prm.get("Evolution effect parameters"),
                                                             options);
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
      template class RateStateFriction<dim>;

      ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
    }
  }
}
