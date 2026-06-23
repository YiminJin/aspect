/*
  Copyright (C) 2015 - 2023 by the authors of the ASPECT code.

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

#include <aspect/particle/property/initial_composition.h>
#include <aspect/initial_composition/interface.h>

namespace aspect
{
  namespace Particle
  {
    namespace Property
    {
      template <int dim>
      void
      InitialComposition<dim>::initialize_one_particle_property(const Point<dim> &position,
                                                                std::vector<double> &data) const
      {
        for (const unsigned int index : compositional_field_indices)
          data.push_back(this->get_initial_composition_manager().initial_composition(position,index));
      }



      template <int dim>
      InitializationModeForLateParticles
      InitialComposition<dim>::late_initialization_mode () const
      {
        return interpolate_respect_boundary;
      }



      template <int dim>
      AdvectionField
      InitialComposition<dim>::advection_field_for_boundary_initialization (const unsigned int property_component) const
      {
        Assert (property_component < this->n_compositional_fields(),
                ExcInternalError());

        return AdvectionField::composition(property_component);
      }



      template <int dim>
      std::vector<std::pair<std::string, unsigned int>>
      InitialComposition<dim>::get_property_information() const
      {
        std::vector<std::pair<std::string,unsigned int>> property_information;

        for (const unsigned int index : compositional_field_indices)
          {
            if (this->get_parameters().mapped_particle_properties.size() > 0)
              {
                const auto it = this->get_parameters().mapped_particle_properties.find(index);
                AssertThrow(it != this->get_parameters().mapped_particle_properties.end(),
                            ExcInternalError());
                property_information.emplace_back(it->second.first, 1);
              }
            else
              {
                std::ostringstream field_name;
                field_name << "initial " << this->introspection().name_for_compositional_index(index);
                property_information.emplace_back(field_name.str(), 1);
              }
          }

        return property_information;
      }



      template <int dim>
      void
      InitialComposition<dim>::declare_parameters(ParameterHandler &prm)
      {
        prm.enter_subsection("Initial composition");
        {
          prm.declare_entry("List of field names", "",
                            Patterns::List(Patterns::Anything()),
                            "A comma separated list of names denoting those "
                            "compositional fields to be handled by this particle "
                            "property plugin. Each of the names listed here must "
                            "be one of those declared in 'Compositional fields/"
                            "Names of fields'. If this parameter is left empty, "
                            "then it will be assumed that all the compositional "
                            "fields advected by particles are to be handled by "
                            "this plugin.");
        }
        prm.leave_subsection();
      }



      template <int dim>
      void
      InitialComposition<dim>::parse_parameters(ParameterHandler &prm)
      {
        prm.enter_subsection("Initial composition");
        {
          compositional_field_indices.clear();
          const std::vector<std::string> field_names = Utilities::split_string_list(prm.get("List of field names"));
          if (field_names.size() == 0)
            {
              AssertThrow(this->n_compositional_fields() > 0,
                          ExcMessage("You have requested the particle property <initial "
                                     "composition>, but the number of compositional fields is 0. "
                                     "Please add compositional fields to your model, or remove "
                                     "this particle property."));
              for (unsigned int index = 0; index < this->n_compositional_fields(); ++index)
                if (this->introspection().compositional_field_methods[index]
                    == Parameters<dim>::AdvectionFieldMethod::particles)
                  compositional_field_indices.push_back(index);
            }
          else
            {
              for (const auto &name : field_names)
                {
                  const unsigned int index = this->introspection().compositional_index_for_name(name);
                  AssertThrow(this->introspection().compositional_field_methods[index]
                              == Parameters<dim>::AdvectionFieldMethod::particles,
                              ExcMessage("Compositional field '" + name + "' is included in the list of "
                                         "fields to be handled by particle property 'initial composition', "
                                         "by this field is not advected by particles."));
                  compositional_field_indices.push_back(index);
                }
            }
        }
        prm.leave_subsection();
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
      ASPECT_REGISTER_PARTICLE_PROPERTY(InitialComposition,
                                        "initial composition",
                                        "Implementation of a plugin in which the particle "
                                        "property is given as the initial composition "
                                        "at the particle's initial position. The particle "
                                        "gets as many properties as there are "
                                        "compositional fields.")
    }
  }
}
