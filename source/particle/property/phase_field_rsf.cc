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

#include <aspect/particle/property/phase_field_rsf.h>
#include <aspect/material_model/phase_field_rsf.h>

namespace aspect
{
  namespace Particle
  {
    namespace Property
    {
      template <int dim>
      void PhaseFieldRSF<dim>::initialize()
      {
        // Check if the material model is PhaseFieldRSF
        AssertThrow(Plugins::plugin_type_matches<const MaterialModel::PhaseFieldRSF<dim>>(this->get_material_model()),
                    ExcMessage("Particle property 'phase field rsf' only works when the material model is also set to "
                               "'phase field rsf'."));

        // Make sure that the slip rate, the fault normal and the stress components are 
        // associated with compositional fields
        for (const auto &key_and_value : this->get_parameters().mapped_particle_properties)
          {
            // Check for slip rate
            if (key_and_value.second.first == "slip_rate")
              compositional_indices.slip_rate = key_and_value.first;

            // Check for fault normal
            for (unsigned int d = 0; d < dim; ++d)
              if (key_and_value.second.first == "normal" &&
                  key_and_value.second.second == d)
                compositional_indices.normal[d] = key_and_value.first;

            // Check for stress
            for (unsigned int c = 0; c < SymmetricTensor<2, dim>::n_independent_components; ++c)
              if (key_and_value.second.first == "stress" &&
                  key_and_value.second.second == c)
                compositional_indices.stress[c] = key_and_value.first;
          }

        AssertThrow(compositional_indices.slip_rate != numbers::invalid_unsigned_int,
                    ExcMessage("Particle property plugin 'phase field rsf' requires property "
                               "'slip_rate' to be associated with a compositional field."));

        for (unsigned int d = 0; d < dim; ++d)
          AssertThrow(compositional_indices.normal[d] != numbers::invalid_unsigned_int, 
                      ExcMessage("Particle property plugin 'phase field rsf' requires property "
                                 "'normal[d]' (d = 1, ..., dim) to be associated with a compositional field."));

        for (unsigned int c = 0; c < SymmetricTensor<2, dim>::n_independent_components; ++c)
          AssertThrow(compositional_indices.stress[c] != numbers::invalid_unsigned_int,
                      ExcMessage("Particle property plugin 'phase field rsf' requires property "
                                 "'stress[c]' (c = 1, ..., dim(dim+1)/2) to be associated with "
                                 "a compositional field."));

        // Initialize the data position cache
        data_position_cache.crack_driving_force = this->data_position;
        data_position_cache.slip_rate           = this->data_position + 1;
        data_position_cache.slip_state          = this->data_position + 2;
        data_position_cache.normal              = this->data_position + 3;
        data_position_cache.slip_direction      = this->data_position + 3 + dim;
        data_position_cache.stress              = this->data_position + 3 + dim + SymmetricTensor<2, dim>::n_independent_components;

        const auto &particle_data_info = this->get_phase_field_handler().get_associated_particle_manager().get_property_manager().get_data_info();
        data_position_cache.chemical_fields.clear();
        for (const unsigned int index : this->introspection().chemical_composition_field_indices())
          data_position_cache.chemical_fields.push_back(
            particle_data_info.get_position_by_field_name(this->get_parameters().mapped_particle_properties.find(index)->second.first));
      }



      template <int dim>
      void
      PhaseFieldRSF<dim>::initialize_one_particle_property(const Point<dim> &position,
                                                           std::vector<double> &data) const
      {
        const MaterialModel::PhaseFieldRSF<dim> &material_model = dynamic_cast<const MaterialModel::PhaseFieldRSF<dim>&>(this->get_material_model());
        const MaterialModel::Rheology::RateStateFriction<dim> &rsf_model = material_model.get_rate_state_friction_model();

        // The initial crack driving force is set to $H_t$
        std::vector<double> initial_composition(this->introspection().n_compositional_fields);
        for (unsigned int j = 0; j < initial_composition.size(); ++j)
          initial_composition[j] = this->get_initial_composition_manager().initial_composition(position, j);
        const std::vector<double> volume_fractions = MaterialModel::MaterialUtilities::compute_only_composition_fractions(
          initial_composition, this->introspection().chemical_composition_field_indices());
        const std::vector<double> Ht = material_model.get_threshold_crack_driving_forces();
        data.push_back(MaterialModel::MaterialUtilities::average_value(volume_fractions, Ht, MaterialModel::MaterialUtilities::arithmetic));

        // The initial slip rate is determined by the initial composition manager
        data.push_back(std::max(this->get_initial_composition_manager().initial_composition(position, compositional_indices.slip_rate),
                       rsf_model.get_minimum_slip_rate()));

        // The initial slip state is set to $D_c / V_{ref}$, i.e. the static slip state
        data.push_back(rsf_model.get_characteristic_slip_distance() / data.back());

        // The fault normal vector and the stress tensor are both determined by the initial composition manager
        for (unsigned int d = 0; d < dim; ++d)
          data.push_back(this->get_initial_composition_manager().initial_composition(position, compositional_indices.normal[d]));
        for (unsigned int c = 0; c < SymmetricTensor<2, dim>::n_independent_components; ++c)
          data.push_back(this->get_initial_composition_manager().initial_composition(position, compositional_indices.stress[c]));
      }



      template <int dim>
      void
      PhaseFieldRSF<dim>::update_particle_properties(const ParticleUpdateInputs<dim> &inputs,
                                                     typename ParticleHandler<dim>::particle_iterator_range &particles) const
      {
        const MaterialModel::PhaseFieldRSF<dim> &material_model 
          = dynamic_cast<const MaterialModel::PhaseFieldRSF<dim>&>(this->get_material_model());

        unsigned int p = 0;
        for (auto &particle : particles)
          {
            const ArrayView<double> particle_properties = particle.get_properties();

            // The crack driving force, slip rate, slip state and stress are
            // ought to be updated by the material model. Here we only need to
            // update the fault normal vector and the slip direction tensor
            Tensor<1, dim> n;
            for (unsigned int d = 0; d < dim; ++d)
              n[d] = particle_properties[data_position_cache.normal + d];

            if (n.norm_square() < 0.5)
              continue;

            // Get the velocity gradient
            Tensor<2, dim> L;
            for (unsigned int d = 0; d < dim; ++d)
              L[d] = inputs.gradients[p][d];

            // Rotate the fault normal if the time step number is not 0
            if (this->get_timestep_number() > 0)
              {
                n -= (transpose(L) * n) * this->get_timestep();
                n /= n.norm();

                for (unsigned int d = 0; d < dim; ++d)
                  particle_properties[data_position_cache.normal + d] = n[d];
              }

            // Compute the bulk stress
            std::vector<double> chemical_field_values;
            for (const unsigned int position : data_position_cache.chemical_fields)
              chemical_field_values.push_back(particle_properties[position]);
            const std::vector<double> volume_fractions = MaterialModel::MaterialUtilities::compute_composition_fractions(chemical_field_values);

            const SymmetricTensor<2, dim> tau_old = Utilities::Tensors::to_symmetric_tensor<dim>(
              &particle_properties[data_position_cache.stress],
              &particle_properties[data_position_cache.stress] + SymmetricTensor<2, dim>::n_independent_components);

            const double T = inputs.solution[p][this->introspection().component_indices.temperature];
            const SymmetricTensor<2, dim> tau_b = material_model.calculate_bulk_stress(T, symmetrize(L), tau_old, volume_fractions);

            // Compute the normal and shear stress on the fault plane
            const Tensor<1, dim> t = tau_b * n;
            const Tensor<1, dim> t_s = t - (t * n) * n;
            const double p_bar = this->get_adiabatic_conditions().pressure(particle.get_location());
            if (t_s.norm() < p_bar * 1.e-20)
              {
                // The shear stress is too small: impossible to slip
                // (this is only to avoid division by 0, so we do not have to compute
                // the friction coefficient)
                Utilities::Tensors::unroll_symmetric_tensor_into_array(
                  numbers::signaling_nan<SymmetricTensor<2, dim>>(),
                  &particle_properties[data_position_cache.stress],
                  &particle_properties[data_position_cache.stress] + SymmetricTensor<2, dim>::n_independent_components);
              }
            else
              {
                const Tensor<1, dim> s = t_s / t_s.norm();
                Utilities::Tensors::unroll_symmetric_tensor_into_array(
                  symmetrize(outer_product(n, s)),
                  &particle_properties[data_position_cache.stress],
                  &particle_properties[data_position_cache.stress] + SymmetricTensor<2, dim>::n_independent_components);
              }
          }
      }



      template <int dim>
      UpdateTimeFlags
      PhaseFieldRSF<dim>::need_update() const
      {
        return update_time_step;
      }



      template <int dim>
      UpdateFlags
      PhaseFieldRSF<dim>::get_update_flags(const unsigned int component) const
      {
        if (this->introspection().component_masks.velocities[component] == true)
          return update_gradients;
        
        return update_default;
      }




      template <int dim>
      std::vector<std::pair<std::string, unsigned int>>
      PhaseFieldRSF<dim>::get_property_information() const
      {
        std::vector<std::pair<std::string, unsigned int>> property_information;

        property_information.emplace_back("crack_driving_force", 1);
        property_information.emplace_back("slip_rate", 1);
        property_information.emplace_back("slip_state", 1);
        property_information.emplace_back("normal", dim);
        property_information.emplace_back("slip_direction", SymmetricTensor<2, dim>::n_independent_components);
        property_information.emplace_back("stress", SymmetricTensor<2, dim>::n_independent_components);

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
      ASPECT_REGISTER_PARTICLE_PROPERTY(PhaseFieldRSF,
                                        "phase field rsf",
                                        "")
    }
  }
}
