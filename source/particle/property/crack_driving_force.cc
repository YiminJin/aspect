/*
  Copyright (C) 2025 - 2024 by the authors of the ASPECT code.

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

#include <aspect/particle/property/crack_driving_force.h>
#include <aspect/phase_field.h>

namespace aspect
{
  namespace Particle
  {
    namespace Property
    {
      template <int dim>
      void CrackDrivingForce<dim>::initialize()
      {
        // Check if the material model is PhaseFieldRSF
        AssertThrow(dynamic_cast<const MaterialModel::PhaseFieldModel<dim>*>(&this->get_material_model()),
                    ExcMessage("Particle property 'crack driving force' only works when the material model is derived from "
                               "'MaterialModel::PhaseFieldModel'."));
      }



      template <int dim>
      void
      CrackDrivingForce<dim>::
      initialize_one_particle_property(const Point<dim> &position,
                                       std::vector<double> &data) const
      {
        // Initialize the crack driving force to the steady state with zero phase-field
        const auto &material_model = dynamic_cast<const MaterialModel::PhaseFieldModel<dim>&>(this->get_material_model());

        std::vector<double> initial_composition(this->introspection().n_compositional_fields);
        for (unsigned int j = 0; j < initial_composition.size(); ++j)
          initial_composition[j] = this->get_initial_composition_manager().initial_composition(position, j);

        const std::vector<double> volume_fractions = MaterialModel::MaterialUtilities::compute_only_composition_fractions(
          initial_composition, this->introspection().chemical_composition_field_indices());

        const double Hc = MaterialModel::MaterialUtilities::average_value(volume_fractions,
                                                                          material_model.get_critical_crack_driving_forces(),
                                                                          MaterialModel::MaterialUtilities::arithmetic);
        data.push_back(Hc);
      }



      template <int dim>
      std::vector<std::pair<std::string, unsigned int>>
      CrackDrivingForce<dim>::get_property_information() const
      {
        const std::vector<std::pair<std::string,unsigned int>> 
        property_information(1, std::make_pair("crack_driving_force", 1));

        return property_information;
      }
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace Particle
  {
    namespace Property
    {
      ASPECT_REGISTER_PARTICLE_PROPERTY(CrackDrivingForce,
                                        "crack driving force",
                                        "Implementation of a plugin in which the particle "
                                        "property is defined as the crack driving force at "
                                        "this position.")
    }
  }
}
