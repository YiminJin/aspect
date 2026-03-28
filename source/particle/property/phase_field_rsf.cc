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

        // Retrieve the indices of the compositional fields associated with particle properties
        for (const auto &key_and_value : this->get_parameters().mapped_particle_properties)
          {
            if (key_and_value.second.first == "crack_driving_force")
              compositional_indices.crack_driving_force = key_and_value.first;

            if (key_and_value.second.first == "slip_rate")
              compositional_indices.slip_rate = key_and_value.first;

            if (key_and_value.second.first == "slip_state")
              compositional_indices.slip_state = key_and_value.first;

            if (key_and_value.second.first == "normal_direction")
              {
                AssertThrow(key_and_value.second.second < dim,
                            ExcMessage("The component indices of normal direction exceed the range of [0, dim-1]."));
                compositional_indices.normal_direction[key_and_value.second.second] = key_and_value.first;
              }

            if (key_and_value.second.first == "slip_direction")
              {
                AssertThrow(key_and_value.second.second < dim,
                            ExcMessage("The component indices of slip direction exceed the range of [0, dim-1]."));
                compositional_indices.slip_direction[key_and_value.second.second] = key_and_value.first;
              }

            if (key_and_value.second.first == "bulk_stress")
              {
                AssertThrow((key_and_value.second.second < SymmetricTensor<2, dim>::n_independent_components),
                            ExcMessage("The component indices of bulk stress exceed the range of [0, dim(dim+1)/2-1]."));
                compositional_indices.bulk_stress[key_and_value.second.second] = key_and_value.first;
              }
          }

        // The normal direction and bulk stress must be associated with compositional fields
        for (unsigned int d = 0; d < dim; ++d)
          AssertThrow(compositional_indices.normal_direction[d] != numbers::invalid_unsigned_int,
                      ExcMessage("Particle property 'normal_direction[d]' (d = 0, ..., dim-1) needs to be "
                                 "associated with a compositional field."));

        for (unsigned int c = 0; c < SymmetricTensor<2, dim>::n_independent_components; ++c)
          AssertThrow(compositional_indices.bulk_stress[c] != numbers::invalid_unsigned_int,
                      ExcMessage("Particle property 'bulk_stress[c]' (c = 0, ..., dim(dim+1)/2-1) needs to be "
                                 "associated with a compositional field."));

        // If the slip rate is associated with a compositional field, then the initial slip rate,
        // crack driving force and slip direction will be determined by the initial composition model,
        // and the initial interface stress will be computed accordingly; otherwise, the initial slip 
        // rate will set to the lower limit (stick mode), the initial crack driving force will be set
        // to the threshold crack driving force (representing material cohesion), and the initial slip 
        // direction and interface stress will be set to invalid numbers
        start_with_slip = (compositional_indices.slip_rate != numbers::invalid_unsigned_int);
        if (start_with_slip == true)
          {
            AssertThrow(compositional_indices.crack_driving_force != numbers::invalid_unsigned_int,
                        ExcMessage("The particle property 'slip_rate' is associated with a compositional field, "
                                   "which means that the initial slip rate is prescribed by the initial composition "
                                   "model. In this case, the particle property 'crack_driving_force' must be "
                                   "associated with a compositional field to be initialized too, otherwise "
                                   "the initial conditions are incomplete."));

            for (unsigned int d = 0; d < dim; ++d)
              AssertThrow(compositional_indices.slip_direction[d] != numbers::invalid_unsigned_int,
                          ExcMessage("The particle property 'slip_rate' is associated with a compositional field, "
                                     "which means that the initial slip rate is prescribed by the initial composition "
                                     "model. In this case, the particle properties 'slip_direction[d]' (d = 0, ..., dim-1) "
                                     "must be associated with a compositional field to be initialized too, otherwise "
                                     "the initial conditions are incomplete."));
          }

        // If the slip state is associated with a compositional field, then the initial slip state
        // will be determined by the initial compositional model; otherwise, it will be computed
        // according to the initial slip rate under the steady state assumption
        start_with_steady_state = (compositional_indices.slip_state == numbers::invalid_unsigned_int);

        const auto &variable = this->introspection().variable("phase_field");
        phase_field_component_index = variable.first_component_index;
        phase_field_base_index      = variable.base_index;

        grid_cache = std::make_unique<GridTools::Cache<dim>>(this->get_triangulation(), this->get_mapping());
        grid_cache->mark_for_update(GridTools::update_vertex_to_cell_map);

        // Initialize the data position cache when it is sure that the phase field handler is initialized
        this->get_signals().post_simulator_initialization.connect(
          [&](const SimulatorAccess<dim> &)
        {
          this->initialize_data_position_cache();
        });
      }



      template <int dim>
      void
      PhaseFieldRSF<dim>::initialize_data_position_cache()
      {
        unsigned int position = this->data_position;
        data_position_cache.crack_driving_force = position++;
        data_position_cache.slip_rate           = position++;
        data_position_cache.slip_state          = position++;
        data_position_cache.normal_direction    = position;
        position += dim;
        data_position_cache.slip_direction      = position;
        position += dim;
        data_position_cache.bulk_stress         = position;
        position += SymmetricTensor<2, dim>::n_independent_components;
        data_position_cache.interface_stress    = position;

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

        // Function Particle::Manager::setup_initial_state is connected to signal post_setup_initial_state,
        // so we can get phase field values from the solution vector
        const std::pair<typename Triangulation<dim>::active_cell_iterator, Point<dim>>
        cell_and_point = GridTools::find_active_cell_around_point(this->get_mapping(), this->get_triangulation(), position);
        const double pf = get_phase_field_value(this->get_solution(), cell_and_point.first, cell_and_point.second);
        const bool is_fractured = material_model.is_fractured(pf);

        // If the user chooses to start with slip, then the initial crack driving force is determined by the
        // initial composition; otherwise, the initial crack driving force is set to $H_t$
        std::vector<double> initial_composition(this->introspection().n_compositional_fields);
        for (unsigned int j = 0; j < initial_composition.size(); ++j)
          initial_composition[j] = this->get_initial_composition_manager().initial_composition(position, j);
        const std::vector<double> volume_fractions = MaterialModel::MaterialUtilities::compute_only_composition_fractions(
          initial_composition, this->introspection().chemical_composition_field_indices());
        const double Ht = MaterialModel::MaterialUtilities::average_value(volume_fractions,
                                                                          material_model.get_threshold_crack_driving_forces(),
                                                                          MaterialModel::MaterialUtilities::arithmetic);

        double H = Ht;
        if (start_with_slip)
          H = std::max(Ht, this->get_initial_composition_manager().initial_composition(position, compositional_indices.crack_driving_force));

        data.push_back(H);

        // If the user chooses to start with slip, then the initial slip rate is determined by the initial composition;
        // otherwise, the initial slip rate is set to the lower bound
        const double V = std::max(rsf_model.get_minimum_slip_rate(),
                                  ((start_with_slip && is_fractured) ?
                                   this->get_initial_composition_manager().initial_composition(position, compositional_indices.slip_rate) :
                                   0.));
        data.push_back(V);

        // If the user chooses to start with steady state, then the initial slip state is calculated by $D_c / V_init$;
        // otherwise, the initial slip state is determined by the initial composition
        const double theta = (start_with_steady_state ?
                              rsf_model.get_characteristic_slip_distance() / V :
                              this->get_initial_composition_manager().initial_composition(position, compositional_indices.slip_state));
        data.push_back(theta);

        // The initial normal direction is determined by the initial composition
        Tensor<1, dim> n = numbers::signaling_nan<Tensor<1, dim>>();
        if (is_fractured)
          {
            for (unsigned int d = 0; d < dim; ++d)
              n[d] = this->get_initial_composition_manager().initial_composition(position, compositional_indices.normal_direction[d]);

            n /= n.norm();
          }

        for (unsigned int d = 0; d < dim; ++d)
          data.push_back(n[d]);

        // If the user chooses to start with slip, then the initial slip direction is determined by the initial composition;
        // otherwise, the initial slip direction is set to an invalid value
        Tensor<1, dim> s = numbers::signaling_nan<Tensor<1, dim>>();
        if (start_with_slip && is_fractured)
          {
            for (unsigned int d = 0; d < dim; ++d)
              s[d] = this->get_initial_composition_manager().initial_composition(position, compositional_indices.slip_direction[d]);

            s -= (s * n) * n;
            s /= s.norm();
          }

        for (unsigned int d = 0; d < dim; ++d)
          data.push_back(s[d]);

        //The initial bulk stress is determined by the initial composition
        for (unsigned int c = 0; c < SymmetricTensor<2, dim>::n_independent_components; ++c)
          data.push_back(this->get_initial_composition_manager().initial_composition(position, compositional_indices.bulk_stress[c]));

        // Finally, initialize the interface stress
        SymmetricTensor<2, dim> tau_f = numbers::signaling_nan<SymmetricTensor<2, dim>>();
        if (start_with_slip && is_fractured)
          {
            // If the user chooses to start with slip, then make the interface stress consistent with
            // the prescribed slip rate, i.e.
            // $\boldsymbol{\tau}_f^0 = 2(\mu\bar{p} + \eta_d V)\boldsymbol S$
            const SymmetricTensor<2, dim> S = symmetrize(outer_product(n, s));
            const double friction_strength = material_model.calculate_friction_strength(position, V, theta, volume_fractions);
            tau_f = (2. * friction_strength) * S;
          }

        for (unsigned int c = 0; c < SymmetricTensor<2, dim>::n_independent_components; ++c)
          data.push_back(tau_f.access_raw_entry(c));
      }



      template <int dim>
      std::vector<double>
      PhaseFieldRSF<dim>::
      initialize_late_particle(const Point<dim> &particle_location,
                               const typename Triangulation<dim>::active_cell_iterator &cell_hint) const
      {
        const Particle::Manager<dim> &particle_manager = this->get_phase_field_handler().get_associated_particle_manager();
        const Particles::ParticleHandler<dim> &particle_handler = particle_manager.get_particle_handler();
        const MaterialModel::PhaseFieldRSF<dim> &material_model = dynamic_cast<const MaterialModel::PhaseFieldRSF<dim>&>(this->get_material_model());

        const auto &data_info = particle_manager.get_property_manager().get_data_info();
        std::vector<double> particle_properties(data_info.n_components(), numbers::signaling_nan<double>());

        typename Triangulation<dim>::active_cell_iterator host_cell = cell_hint;
        if (host_cell->state() != IteratorState::valid)
          host_cell = GridTools::find_active_cell_around_point(this->get_mapping(), this->get_triangulation(), particle_location).first;

        // The crack driving force, slip state and bulk stress are interpolated by the user-specified interpolator
        std::vector<bool> component_mask(data_info.n_components(), false);
        component_mask[data_position_cache.crack_driving_force] = true;
        component_mask[data_position_cache.slip_state] = true;
        for (unsigned int c = 0; c < SymmetricTensor<2, dim>::n_independent_components; ++c)
          component_mask[data_position_cache.bulk_stress + c] = true;

        const std::vector<std::vector<double>> interpolated_properties
          = particle_manager.get_interpolator().properties_at_points(particle_handler,
                                                                     std::vector<Point<dim>>(1, particle_location),
                                                                     ComponentMask(component_mask),
                                                                     host_cell);

        particle_properties[data_position_cache.crack_driving_force] = interpolated_properties[0][data_position_cache.crack_driving_force];
        particle_properties[data_position_cache.slip_state] = interpolated_properties[0][data_position_cache.slip_state];
        for (unsigned int c = 0; c < SymmetricTensor<2, dim>::n_independent_components; ++c)
          particle_properties[data_position_cache.bulk_stress + c] = interpolated_properties[0][data_position_cache.bulk_stress + c];

        const double my_pf = get_phase_field_value(this->get_solution(), host_cell, this->get_mapping().transform_real_to_unit_cell(host_cell, particle_location));
        if (material_model.is_fractured(my_pf))
          {
            // The particle is fractured. Interpolate the slip rate, normal direction, slip direction
            // and interface stress from the fractured neighbor particles with distance-weighted averaging
            // method
            const double interpolation_range = 0.5 * host_cell->diameter();
            const double epsilon = interpolation_range * 0.1;

            const auto &vertex_to_cell_map = grid_cache->get_vertex_to_cell_map();
            small_vector<typename Triangulation<dim>::active_cell_iterator> cell_patch;
            for (const auto v : host_cell->vertex_indices())
              {
                const unsigned int vertex_index = host_cell->vertex_index(v);
                cell_patch.insert(cell_patch.end(),
                                  vertex_to_cell_map[vertex_index].begin(),
                                  vertex_to_cell_map[vertex_index].end());
              }
            std::sort(cell_patch.begin(), cell_patch.end());
            cell_patch.erase(std::unique(cell_patch.begin(), cell_patch.end()), cell_patch.end());
 
            double slip_rate = 0.;
            Tensor<1, dim> normal_direction, slip_direction;
            SymmetricTensor<2, dim> interface_stress;
            double integrated_weight = 0.;
            for (const auto &cell : cell_patch)
              {
                const auto &particles_in_cell = particle_handler.particles_in_cell(cell);
                for (auto particle = particles_in_cell.begin(); particle != particles_in_cell.end(); ++particle)
                  {
                    const Point<dim> neighbor_location = particle->get_location();
                    const double neighbor_pf = get_phase_field_value(this->get_solution(), cell, particle->get_reference_location());
                    const double distance = neighbor_location.distance(particle_location);
                    if (material_model.is_fractured(neighbor_pf) && distance < interpolation_range)
                      {
                        const ArrayView<const double> neighbor_properties = particle->get_properties();

                        // Use the modified Shephard method
                        const double weight = Utilities::fixed_power<2, double>((1.0 - (distance * distance / (interpolation_range * interpolation_range))))
                                              / (distance * distance + epsilon * epsilon);

                        slip_rate += weight * neighbor_properties[data_position_cache.slip_rate];
                        for (unsigned int d = 0; d < dim; ++d)
                          {
                            normal_direction[d] += weight * neighbor_properties[data_position_cache.normal_direction + d];
                            slip_direction[d]   += weight * neighbor_properties[data_position_cache.slip_direction + d];
                          }
                        for (unsigned int c = 0; c < SymmetricTensor<2, dim>::n_independent_components; ++c)
                          interface_stress.access_raw_entry(c) += weight * neighbor_properties[data_position_cache.interface_stress + c];

                        integrated_weight += weight;
                      }
                  }
              }

            AssertThrow(integrated_weight > 0., ExcMessage("No fractured neighbor particle found!"));
            slip_rate        /= integrated_weight;
            normal_direction /= integrated_weight;
            slip_direction   /= integrated_weight;
            interface_stress /= integrated_weight;

            normal_direction /= normal_direction.norm();
            slip_direction -= (slip_direction * normal_direction) * normal_direction;
            slip_direction /= slip_direction.norm();

            particle_properties[data_position_cache.slip_rate] = slip_rate;
            for (unsigned int d = 0; d < dim; ++d)
              {
                particle_properties[data_position_cache.normal_direction + d] = normal_direction[d];
                particle_properties[data_position_cache.slip_direction + d] = slip_direction[d];
              }
            for (unsigned int c = 0; c < SymmetricTensor<2, dim>::n_independent_components; ++c)
              particle_properties[data_position_cache.interface_stress + c] = interface_stress.access_raw_entry(c);
          }
        else
          {
            // The particle is intact. Interpolate the slip state, bulk stress and chemical fields
            // with the user-specified interpolator
            std::vector<bool> component_mask(data_info.n_components(), false);
            component_mask[data_position_cache.slip_state] = true;
            for (unsigned int c = 0; c < SymmetricTensor<2, dim>::n_independent_components; ++c)
              component_mask[data_position_cache.bulk_stress + c] = true;

            const unsigned int n_chemical_fields = data_position_cache.chemical_fields.size();
            for (unsigned int j = 0; j < n_chemical_fields; ++j)
              component_mask[data_position_cache.chemical_fields[j]] = true;

            const std::vector<std::vector<double>> interpolated_properties
              = particle_manager.get_interpolator().properties_at_points(particle_handler,
                                                                         std::vector<Point<dim>>(1, particle_location),
                                                                         ComponentMask(component_mask),
                                                                         host_cell);

            particle_properties[data_position_cache.slip_state] = interpolated_properties[0][data_position_cache.slip_state];
            for (unsigned int c = 0; c < SymmetricTensor<2, dim>::n_independent_components; ++c)
              particle_properties[data_position_cache.bulk_stress + c] = interpolated_properties[0][data_position_cache.bulk_stress + c];

            // The crack drving force is initialized to Ht
            std::vector<double> chemical_field_values(n_chemical_fields);
            for (unsigned int j = 0; j < n_chemical_fields; ++j)
              chemical_field_values[j] = interpolated_properties[0][data_position_cache.chemical_fields[j]];
            const std::vector<double> volume_fractions = MaterialModel::MaterialUtilities::compute_composition_fractions(chemical_field_values);

            const std::vector<double> Ht = material_model.get_threshold_crack_driving_forces();
            particle_properties[data_position_cache.crack_driving_force] 
              = MaterialModel::MaterialUtilities::average_value(volume_fractions, Ht, MaterialModel::MaterialUtilities::arithmetic);

            // The slip rate is initialized to the lower bound
            const MaterialModel::Rheology::RateStateFriction<dim> &rsf_model = material_model.get_rate_state_friction_model();
            particle_properties[data_position_cache.slip_rate] = rsf_model.get_minimum_slip_rate();
          }

        return particle_properties;
      }



      template <int dim>
      double
      PhaseFieldRSF<dim>::
      get_phase_field_value(const LinearAlgebra::BlockVector &solution,
                            const typename Triangulation<dim>::active_cell_iterator &cell,
                            const Point<dim> &reference_location) const
      {
        const typename DoFHandler<dim>::active_cell_iterator dof_cell(*cell, &this->get_dof_handler());

        Vector<double> dof_values(this->get_fe().dofs_per_cell);
        dof_cell->get_dof_values(solution, dof_values);

        const FiniteElement<dim> &fe_pf = this->get_fe().base_element(phase_field_base_index);
        double pf = 0.;
        for (unsigned int i_pf = 0; i_pf < fe_pf.dofs_per_cell; ++i_pf)
          {
            const unsigned int i = this->get_fe().component_to_system_index(phase_field_component_index, i_pf);
            pf += fe_pf.shape_value(i_pf, reference_location) * dof_values[i];
          }

        return pf;
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
      InitializationModeForLateParticles
      PhaseFieldRSF<dim>::late_initialization_mode() const
      {
        return custom;
      }



      template <int dim>
      std::vector<std::pair<std::string, unsigned int>>
      PhaseFieldRSF<dim>::get_property_information() const
      {
        std::vector<std::pair<std::string, unsigned int>> property_information;

        property_information.emplace_back("crack_driving_force", 1);
        property_information.emplace_back("slip_rate", 1);
        property_information.emplace_back("slip_state", 1);
        property_information.emplace_back("normal_direction", dim);
        property_information.emplace_back("slip_direction", dim);
        property_information.emplace_back("bulk_stress", SymmetricTensor<2, dim>::n_independent_components);
        property_information.emplace_back("interface_stress", SymmetricTensor<2, dim>::n_independent_components);

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
