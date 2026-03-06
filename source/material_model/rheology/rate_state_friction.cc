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
      slip_state (const double V,
                  const double theta_old) const
      {
        AssertThrow(V >= 0, ExcMessage("The slip rate is negative."));
        AssertThrow(theta_old > 0, ExcMessage("The slip state is non-positive."));

        if (this->get_timestep_number() == 0)
          return theta_old;

        const double V_eff = std::max(V, V_min);
        const double dt    = this->get_timestep();
        const double ratio = std::exp(-V * dt / Dc);

        return Dc / V_eff * (1.0 - ratio) + theta_old * ratio;
      }



      template <int dim>
      double
      RateStateFriction<dim>::
      friction_coefficient (const unsigned int j,
                            const double V,
                            const double theta,
                            const bool regularized) const
      {
        AssertIndexRange(j, mu0.size());
        AssertThrow(V >= 0, ExcMessage("The slip rate is negative."));
        AssertThrow(theta > 0, ExcMessage("The slip state is non-positive."));

        const double V_eff = std::max(V, V_min);
        double mu = numbers::signaling_nan<double>();
        if (regularized)
          mu = a[j] * std::asinh(V_eff / (2.0 * V0) * std::exp((mu0[j] + b[j] * std::log(theta * V0 / Dc)) / a[j]));
        else
          mu = mu0[j] + a[j] * std::log(V_eff / V0) + b[j] * std::log(theta * V0 / Dc);

        return mu;
      }



      template <int dim>
      void
      RateStateFriction<dim>::
      declare_parameters(ParameterHandler &prm)
      {
        prm.declare_entry ("Reference slip rate", "1.e-6",
                           Patterns::Double(0.),
                           "The reference slip rate, $V_0$. Units: \\si{\\meter\\per\\second}.");
        prm.declare_entry ("Minimum slip rate", "1.e-9",
                           Patterns::Double(0.),
                           "The lower bound of slip rate. Units: \\si{\\miter\\per\\second}.");
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
        V0    = prm.get_double("Reference slip rate");
        V_min = prm.get_double("Minimum slip rate");
        Dc    = prm.get_double("Characteristic slip distance");

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
