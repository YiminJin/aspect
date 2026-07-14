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
                            ExcMessage("The component indices of normal_direction exceed the range of [0, dim)."));
                compositional_indices.normal_direction[key_and_value.second.second] = key_and_value.first;
              }

            if (key_and_value.second.first == "slip_direction")
              {
                AssertThrow(key_and_value.second.second < dim,
                            ExcMessage("The component indices of slip_direction exceed the range of [0, dim)."));
                compositional_indices.slip_direction[key_and_value.second.second] = key_and_value.first;
              }

            if (key_and_value.second.first == "ve_stress")
              {
                AssertThrow((key_and_value.second.second < SymmetricTensor<2, dim>::n_independent_components),
                            ExcMessage("The component indices of ve_stress exceed the range of [0, dim(dim+1)/2)."));
                compositional_indices.ve_stress[key_and_value.second.second] = key_and_value.first;
              }
          }

        // The slip rate, slip state and viscoelastic stress must be associated with compositional fields
        AssertThrow(compositional_indices.slip_rate != numbers::invalid_unsigned_int,
                    ExcMessage("Particle property 'slip_rate' must be associated with a compositional field."));
        AssertThrow(compositional_indices.slip_state != numbers::invalid_unsigned_int,
                    ExcMessage("Particle property 'slip_state' must be associated with a compositional field."));
        for (unsigned int c = 0; c < SymmetricTensor<2, dim>::n_independent_components; ++c)
          AssertThrow(compositional_indices.ve_stress[c] != numbers::invalid_unsigned_int,
                      ExcMessage("Particle property 've_stress[" + Utilities::int_to_string(c)
                                 + "] must be associated with a compositional field."));

        // If there are pre-existing cracks (i.e. crack_driving_force is associated with a compositional field),
        // then the normal direction and slip direction must be associated with compositional fields too
        has_preexisting_crack = (compositional_indices.crack_driving_force != numbers::invalid_unsigned_int);
        if (has_preexisting_crack == true)
          {
            for (unsigned int d = 0; d < dim; ++d)
              {
                AssertThrow(compositional_indices.normal_direction[d] != numbers::invalid_unsigned_int
                            && compositional_indices.slip_direction[d] != numbers::invalid_unsigned_int,
                            ExcMessage("The particle property 'crack_driving_force' is associated with a compositional field, "
                                       "which means that there are pre-existing cracks in the model. In this case, the "
                                       "particle properties 'normal_direction[d]' and 'slip_direction[d]' (d = 0, ..., dim-1) "
                                       "must be associated with compositional fields to be initialized too, otherwise the "
                                       "initial conditions are incomplete."));
              }
          }

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

        // When initializing the particle properties, the slip-related properties at intact regions
        // need special treatment, but the phase field has not been initialized at that stage.
        // We shall catch up on the cleanup work after solving the phase field system
        this->get_signals().post_restore_particles.connect(
          [&](Particle::Manager<dim> &)
        {
          this->do_cleanup_after_initialization();
        });
      }



      template <int dim>
      void
      PhaseFieldRSF<dim>::initialize_data_position_cache()
      {
        unsigned int position = this->data_position;
        data_position_cache.crack_driving_force = position++;
        data_position_cache.cohesive_force      = position++;
        data_position_cache.slip_rate           = position++;
        data_position_cache.slip_state          = position++;
        data_position_cache.normal_direction    = position;
        position += dim;
        data_position_cache.slip_direction      = position;
        position += dim;
        data_position_cache.ve_stress           = position;

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
        const MaterialModel::PhaseFieldRSF<dim> &material_model = 
          Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldRSF<dim>>(this->get_material_model());
        const MaterialModel::Rheology::RateStateFriction<dim> &rsf_model = material_model.get_rate_state_friction_model();

        // If there are pre-existing cracks, the initial crack driving force is determined by the initial value
        // of the corresponding compositional field; otherwise, the crack driving force is initialized to its
        // lower limit (critical crack driving force)
        std::vector<double> initial_composition(this->introspection().n_compositional_fields);
        for (unsigned int j = 0; j < initial_composition.size(); ++j)
          initial_composition[j] = this->get_initial_composition_manager().initial_composition(position, j);
        const std::vector<double> volume_fractions = MaterialModel::MaterialUtilities::compute_only_composition_fractions(
          initial_composition, this->introspection().chemical_composition_field_indices());
        const double Hc = MaterialModel::MaterialUtilities::average_value(volume_fractions,
                                                                          material_model.get_critical_crack_driving_forces(),
                                                                          MaterialModel::MaterialUtilities::arithmetic);

        double H = Hc;
        if (has_preexisting_crack)
          H = std::max(Hc, this->get_initial_composition_manager().initial_composition(position, compositional_indices.crack_driving_force));

        data.push_back(H);

        // Initialize the cohesive force by 0
        data.push_back(0.);

        // Initialize the slip rate by the initial value of the corresponding compositional field
        const double V = std::max(rsf_model.get_minimum_slip_rate(),
                                  (this->get_initial_composition_manager().initial_composition(position, compositional_indices.slip_rate)));
        data.push_back(V);

        // If the slip state is associated with a compositional field, then initialize it by the initial
        // composition model; otherwise, initialize it by $D_c / V_init$, which implies a steady state
        if (compositional_indices.slip_state != numbers::invalid_unsigned_int)
          data.push_back(this->get_initial_composition_manager().initial_composition(position, compositional_indices.slip_state));
        else
          data.push_back(rsf_model.get_characteristic_slip_distance() / V);

        // If there are pre-existing cracks, then the normal direction and the slip direction are initialized by
        // the initial values of the corresponding compositional fields; otherwise, they are initialized to NaN
        Tensor<1, dim> n = numbers::signaling_nan<Tensor<1, dim>>();
        Tensor<1, dim> s = numbers::signaling_nan<Tensor<1, dim>>();
        if (has_preexisting_crack)
          {
            for (unsigned int d = 0; d < dim; ++d)
              {
                n[d] = this->get_initial_composition_manager().initial_composition(position, compositional_indices.normal_direction[d]);
                s[d] = this->get_initial_composition_manager().initial_composition(position, compositional_indices.slip_direction[d]);
              }

            const double n_norm = n.norm();
            const double s_norm = s.norm();
            if (n_norm < std::numeric_limits<double>::epsilon() ||
                s_norm < std::numeric_limits<double>::epsilon() ||
                n * s > n_norm * s_norm * 0.5)
              {
                std::stringstream position_as_string;
                position_as_string << position;

                AssertThrow(n_norm < std::numeric_limits<double>::epsilon(),
                            ExcMessage("The norm of normal direction at point (" + position_as_string.str() + ") is zero."));
                AssertThrow(s_norm < std::numeric_limits<double>::epsilon(),
                            ExcMessage("The norm of slip direction at point (" + position_as_string.str() + ") is zero."));
                AssertThrow(n_norm < std::numeric_limits<double>::epsilon(),
                            ExcMessage("The normal direction and slip direction at point (" + position_as_string.str() 
                                       + ") are far from perpendicular."));
              }

            n /= n_norm;
            s -= (s * n) * n;
            s /= s.norm();
          }

        for (unsigned int d = 0; d < dim; ++d)
          data.push_back(n[d]);
        for (unsigned int d = 0; d < dim; ++d)
          data.push_back(s[d]);

        // Initialize the viscoelastic stress to zero
        for (unsigned int c = 0; c < SymmetricTensor<2, dim>::n_independent_components; ++c)
          data.push_back(0);
      }



      template <int dim>
      void PhaseFieldRSF<dim>::do_cleanup_after_initialization()
      {
        // Only do the cleanup in the first time step
        if (this->get_timestep_number() > 0)
          return;

        const MaterialModel::PhaseFieldRSF<dim> &material_model = 
          Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldRSF<dim>>(this->get_material_model());

        Particles::ParticleHandler<dim> &particle_handler = const_cast<Particles::ParticleHandler<dim>&>(
          this->get_phase_field_handler().get_associated_particle_manager().get_particle_handler());

        // We need a point evaluator to calculate the phase field values at particle locations
        FEPointEvaluation<1, dim> evaluator(this->get_mapping(), 
                                            this->get_fe(), 
                                            update_values,
                                            phase_field_component_index);

        for (const auto &cell : this->get_dof_handler().active_cell_iterators())
          if (cell->is_locally_owned())
            {
              const auto particles_in_cell = particle_handler.particles_in_cell(cell);

              // Collect the particle locations
              small_vector<Point<dim>> reference_locations;
              for (auto particle = particles_in_cell.begin(); particle != particles_in_cell.end(); ++particle)
                reference_locations.push_back(particle->get_reference_location());

              // Collect the DoF values
              small_vector<double, 500> dof_values(this->get_fe().dofs_per_cell);
              cell->get_dof_values(this->get_solution(), dof_values.begin(), dof_values.end());

              // Update the point evaluator and evaluate the phase field values at particle locations
              evaluator.reinit(cell, {reference_locations.data(), reference_locations.size()});
              evaluator.evaluate({dof_values.data(), dof_values.size()}, EvaluationFlags::values);

              small_vector<double> particle_solution_values(this->introspection().n_components);

              for (auto particle = particles_in_cell.begin(); particle != particles_in_cell.end(); ++particle)
                {
                  const ArrayView<double> particle_properties = particle->get_properties();
                  const unsigned int p = std::distance(particles_in_cell.begin(), particle);

                  const double pf = evaluator.get_value(p);

                  if (material_model.is_fractured(pf) == false)
                    {
                      // Set the cohesive force, normal direction and slip direction to NaN
                      particle_properties[data_position_cache.cohesive_force] = numbers::signaling_nan<double>();
                      for (unsigned int d = 0; d < dim; ++d)
                        {
                          particle_properties[data_position_cache.normal_direction + d] = numbers::signaling_nan<double>();
                          particle_properties[data_position_cache.slip_direction + d]   = numbers::signaling_nan<double>();
                        }
                    }
                }
            }
      }



      template <int dim>
      std::vector<double>
      PhaseFieldRSF<dim>::
      initialize_late_particle(const Point<dim> &particle_location,
                               const typename Triangulation<dim>::active_cell_iterator &cell_hint) const
      {
        const Particle::Manager<dim> &particle_manager = this->get_phase_field_handler().get_associated_particle_manager();
        const Particles::ParticleHandler<dim> &particle_handler = particle_manager.get_particle_handler();
        const MaterialModel::PhaseFieldRSF<dim> &material_model = 
          Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldRSF<dim>>(this->get_material_model());

        const auto &data_info = particle_manager.get_property_manager().get_data_info();
        std::vector<double> particle_properties(data_info.n_components(), numbers::signaling_nan<double>());

        typename Triangulation<dim>::active_cell_iterator host_cell = cell_hint;
        if (host_cell->state() != IteratorState::valid)
          host_cell = GridTools::find_active_cell_around_point(this->get_mapping(), this->get_triangulation(), particle_location).first;

        // The crack driving force, cohesive force, slip state and viscoelastic stress are interpolated 
        // by the user-specified interpolator
        std::vector<bool> component_mask(data_info.n_components(), false);
        component_mask[data_position_cache.crack_driving_force] = true;
        component_mask[data_position_cache.cohesive_force] = true;
        component_mask[data_position_cache.slip_state] = true;
        for (unsigned int c = 0; c < SymmetricTensor<2, dim>::n_independent_components; ++c)
          component_mask[data_position_cache.ve_stress + c] = true;

        const std::vector<std::vector<double>> interpolated_properties
          = particle_manager.get_interpolator().properties_at_points(particle_handler,
                                                                     std::vector<Point<dim>>(1, particle_location),
                                                                     ComponentMask(component_mask),
                                                                     host_cell);

        particle_properties[data_position_cache.crack_driving_force] = interpolated_properties[0][data_position_cache.crack_driving_force];
        particle_properties[data_position_cache.cohesive_force] = interpolated_properties[0][data_position_cache.cohesive_force];
        particle_properties[data_position_cache.slip_state] = interpolated_properties[0][data_position_cache.slip_state];
        for (unsigned int c = 0; c < SymmetricTensor<2, dim>::n_independent_components; ++c)
          particle_properties[data_position_cache.ve_stress + c] = interpolated_properties[0][data_position_cache.ve_stress + c];

        const double my_pf = get_phase_field_value(this->get_solution(), host_cell, this->get_mapping().transform_real_to_unit_cell(host_cell, particle_location));
        if (material_model.is_fractured(my_pf))
          {
            // The particle is fractured. Interpolate the slip rate, normal direction and slip direction
            // from the fractured neighbor particles with distance-weighted averaging method
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

                        integrated_weight += weight;
                      }
                  }
              }

            AssertThrow(integrated_weight > 0., ExcMessage("No fractured neighbor particle found!"));
            slip_rate        /= integrated_weight;
            normal_direction /= integrated_weight;
            slip_direction   /= integrated_weight;

            normal_direction /= normal_direction.norm();
            slip_direction -= (slip_direction * normal_direction) * normal_direction;
            slip_direction /= slip_direction.norm();

            particle_properties[data_position_cache.slip_rate] = slip_rate;
            for (unsigned int d = 0; d < dim; ++d)
              {
                particle_properties[data_position_cache.normal_direction + d] = normal_direction[d];
                particle_properties[data_position_cache.slip_direction + d] = slip_direction[d];
              }
          }
        else
          {
            // The particle is intact. Initialize the slip rate to the lower bound
            const MaterialModel::Rheology::RateStateFriction<dim> &rsf_model = material_model.get_rate_state_friction_model();
            particle_properties[data_position_cache.slip_rate] = rsf_model.get_minimum_slip_rate();

            // Initialize the normal direction and slip direction to NaN
            for (unsigned int d = 0; d < dim; ++d)
              {
                particle_properties[data_position_cache.normal_direction + d] = numbers::signaling_nan<double>();
                particle_properties[data_position_cache.slip_direction + d] = numbers::signaling_nan<double>();
              }
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
        property_information.emplace_back("cohesive_force", 1);
        property_information.emplace_back("slip_rate", 1);
        property_information.emplace_back("slip_state", 1);
        property_information.emplace_back("normal_direction", dim);
        property_information.emplace_back("slip_direction", dim);
        property_information.emplace_back("ve_stress", SymmetricTensor<2, dim>::n_independent_components);

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
