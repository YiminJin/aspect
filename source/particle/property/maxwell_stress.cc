/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include <aspect/particle/property/maxwell_stress.h>
#include <aspect/material_model/phase_field_rsf.h>
#include <aspect/initial_composition/interface.h>

#include <algorithm>

namespace aspect
{
  namespace Particle
  {
    namespace Property
    {
      template <int dim>
      void
      MaxwellStress<dim>::initialize()
      {
        AssertThrow(dynamic_cast<const MaterialModel::PhaseFieldRSF<dim> *>(
                      &this->get_material_model()) != nullptr,
                    ExcMessage("Particle property 'maxwell stress' can only be used with "
                               "the 'phase field rsf' material model."));
        AssertThrow(this->get_parameters().mapped_particle_properties.size() > 0,
                    ExcMessage("Particle property 'maxwell stress' requires an explicit map "
                               "between particle-advected compositional fields and particle "
                               "property components."));

        initial_stress_field_indices.fill(numbers::invalid_unsigned_int);
        const std::vector<unsigned int> &stress_field_indices =
          this->introspection().get_indices_for_fields_of_type(
            CompositionalFieldDescription::stress);

        for (const auto &field_and_property :
             this->get_parameters().mapped_particle_properties)
          if (field_and_property.second.first == "maxwell stress")
            {
              const unsigned int field_index = field_and_property.first;
              const unsigned int component = field_and_property.second.second;

              AssertThrow(component < initial_stress_field_indices.size(),
                          ExcMessage("Compositional field '"
                                     + this->introspection().name_for_compositional_index(field_index)
                                     + "' maps to component " + Utilities::int_to_string(component)
                                     + " of particle property 'maxwell stress', but this property has only "
                                     + Utilities::int_to_string(initial_stress_field_indices.size())
                                     + " components."));
              AssertThrow(initial_stress_field_indices[component]
                          == numbers::invalid_unsigned_int,
                          ExcMessage("More than one compositional field maps to component "
                                     + Utilities::int_to_string(component)
                                     + " of particle property 'maxwell stress'."));
              AssertThrow(std::find(stress_field_indices.begin(),
                                    stress_field_indices.end(),
                                    field_index) != stress_field_indices.end(),
                          ExcMessage("Compositional field '"
                                     + this->introspection().name_for_compositional_index(field_index)
                                     + "' initializes particle property 'maxwell stress' but is not "
                                     "declared with compositional field type 'stress'."));

              initial_stress_field_indices[component] = field_index;
            }

        for (unsigned int component = 0;
             component < initial_stress_field_indices.size();
             ++component)
          AssertThrow(initial_stress_field_indices[component]
                      != numbers::invalid_unsigned_int,
                      ExcMessage("No particle-advected compositional field maps to component "
                                 + Utilities::int_to_string(component)
                                 + " of particle property 'maxwell stress'. Every independent "
                                 "stress component must be mapped explicitly."));
      }



      template <int dim>
      void
      MaxwellStress<dim>::initialize_one_particle_property(
        const Point<dim> &position,
        std::vector<double> &particle_properties) const
      {
        for (const unsigned int field_index : initial_stress_field_indices)
          particle_properties.push_back(
            this->get_initial_composition_manager().initial_composition(position,
                                                                         field_index));
      }



      template <int dim>
      std::vector<std::pair<std::string, unsigned int>>
      MaxwellStress<dim>::get_property_information() const
      {
        return {{"maxwell stress",
                 SymmetricTensor<2,dim>::n_independent_components}};
      }
    }
  }
}

namespace aspect
{
  namespace Particle
  {
    namespace Property
    {
      ASPECT_REGISTER_PARTICLE_PROPERTY(MaxwellStress,
                                        "maxwell stress",
                                        "Store the committed symmetric Maxwell stress history used "
                                        "by the reconstructed-fault phase-field RSF material model. "
                                        "Every independent tensor component is initialized from an "
                                        "explicitly mapped compositional field. The property applies "
                                        "no automatic update or objective rotation.")
    }
  }
}
