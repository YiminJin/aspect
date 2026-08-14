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
#include <aspect/postprocess/visualization.h>
#include <aspect/postprocess/particles.h>

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

        phase_field_base_element = this->introspection().variable("phase_field").base_index;
      }



      template <int dim>
      void PhaseFieldRSF<dim>::update()
      {
        if (maximum_slip_distance_between_outputs == 0)
          return;

        // Check if the maximum slip increment reaches the threshold
        Particles::ParticleHandler<dim> &particle_handler = const_cast<Particles::ParticleHandler<dim>&>(
          this->get_phase_field_handler().get_associated_particle_manager().get_particle_handler());

        const unsigned int slip_increment_index =
          Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldRSF<dim>>(this->get_material_model())
            .get_index_cache().particle_properties.slip_increment;

        double local_max_slip_increment = 0;

        for (const auto &cell : this->get_triangulation().active_cell_iterators())
          if (cell->is_locally_owned())
            for (const auto &particle : particle_handler.particles_in_cell(cell))
              {
                const double V_inc = particle.get_properties()[slip_increment_index];
                if (!numbers::is_nan(V_inc))
                  local_max_slip_increment = std::max(local_max_slip_increment, V_inc);
              }

        const double max_slip_increment = Utilities::MPI::max(local_max_slip_increment, this->get_mpi_communicator());
        
        if (max_slip_increment >= maximum_slip_distance_between_outputs)
          {
            // Send output request to the solution-based and particle-based visualizers
            this->get_postprocess_manager().template get_matching_active_plugin<Postprocess::Visualization<dim>>().request_output();
            this->get_postprocess_manager().template get_matching_active_plugin<Postprocess::Particles<dim>>().request_output();

            // Reset the slip increments
            for (const auto &cell : this->get_triangulation().active_cell_iterators())
              if (cell->is_locally_owned())
                for (auto &particle : particle_handler.particles_in_cell(cell))
                  {
                    const ArrayView<double> particle_properties = particle.get_properties();
                    const double V_inc = particle_properties[slip_increment_index];
                    if (!numbers::is_nan(V_inc))
                      particle_properties[slip_increment_index] = 0;
                  }
          }
      }



      template <int dim>
      void
      PhaseFieldRSF<dim>::initialize_one_particle_property(const Point<dim> &position,
                                                           std::vector<double> &data) const
      {
        const MaterialModel::PhaseFieldRSF<dim> &material_model = 
          Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldRSF<dim>>(this->get_material_model());
        const typename MaterialModel::PhaseFieldRSF<dim>::IndexCache &index_cache = material_model.get_index_cache();
        const MaterialModel::Rheology::RateStateFriction<dim> &rsf_model = material_model.get_rate_state_friction_model();

        const bool has_preexisting_crack = (index_cache.compositional_fields.crack_driving_force != numbers::invalid_unsigned_int);

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
          H = std::max(Hc, this->get_initial_composition_manager().initial_composition(position, index_cache.compositional_fields.crack_driving_force));

        data.push_back(H);

        // Initialize the cohesive force to 0
        data.push_back(0);

        // If there are pre-existing cracks, then the initial slip rate is determined by the initial compositional model; otherwise,
        // the slip rate is initialized to Vmin
        double V = rsf_model.get_minimum_slip_rate();
        if (has_preexisting_crack)
          V = std::max(V, this->get_initial_composition_manager().initial_composition(position, index_cache.compositional_fields.slip_rate));
        data.push_back(V);

        // The initial slip state is determined by the initial composition model
        const double theta = this->get_initial_composition_manager().initial_composition(position, index_cache.compositional_fields.slip_state);
        data.push_back(theta);

        // If there are pre-existing cracks, then the initial values of the direction vectors are determined by the initial composition model
        if (has_preexisting_crack)
          {
            Tensor<1, dim> n, s;
            for (unsigned int d = 0; d < dim; ++d)
              {
                n[d] = this->get_initial_composition_manager().initial_composition(position, index_cache.compositional_fields.normal_direction[d]);
                s[d] = this->get_initial_composition_manager().initial_composition(position, index_cache.compositional_fields.slip_direction[d]);
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

            for (unsigned int d = 0; d < dim; ++d)
              data.push_back(n[d]);
            for (unsigned int d = 0; d < dim; ++d)
              data.push_back(s[d]);
          }
        else
          {
            // There is no pre-existing crack. Initialize the direction vectors to NaN
            for (unsigned d = 0; d < 2 * dim; ++d)
              data.push_back(std::numeric_limits<double>::quiet_NaN());
          }

        // The initial viscoelastic stress is determined by the initial composition model
        for (unsigned int c = 0; c < SymmetricTensor<2, dim>::n_independent_components; ++c)
          data.push_back(this->get_initial_composition_manager().initial_composition(position, index_cache.compositional_fields.ve_stress[c]));

        // Initialize the friction coefficient if requested
        if (output_friction_coefficient)
          data.push_back(rsf_model.friction_coefficient(volume_fractions, V, theta));

        // Initialize the slip increment if requested
        if (output_slip_increment)
          data.push_back(0);
      }



      template <int dim>
      std::vector<double>
      PhaseFieldRSF<dim>::
      initialize_late_particle(const Point<dim> &particle_location,
                               const typename Triangulation<dim>::active_cell_iterator &cell_hint) const
      {
        const PhaseFieldHandler<dim> &phase_field_handler = this->get_phase_field_handler();
        const Particle::Manager<dim> &particle_manager = phase_field_handler.get_associated_particle_manager();
        const Particles::ParticleHandler<dim> &particle_handler = particle_manager.get_particle_handler();

        const MaterialModel::PhaseFieldRSF<dim> &material_model = 
          Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldRSF<dim>>(this->get_material_model());
        const typename MaterialModel::PhaseFieldRSF<dim>::IndexCache &index_cache = material_model.get_index_cache();

        const auto &data_info = particle_manager.get_property_manager().get_data_info();
        std::vector<double> particle_properties(data_info.n_components(), numbers::signaling_nan<double>());

        typename Triangulation<dim>::active_cell_iterator host_cell = cell_hint;
        if (host_cell->state() != IteratorState::valid)
          host_cell = GridTools::find_active_cell_around_point(this->get_mapping(), 
                                                               this->get_triangulation(), 
                                                               particle_location).first;

        // The crack driving force, cohesive force, slip state and viscoelastic stress are 
        // defined in the entire domain, while the rest are restricted in the crack zone
        // TODO: the slip state is discontinuous at the edge of crack zone, so it is better
        // to treat it separately
        std::vector<bool> is_unrestricted(data_info.n_components(), false);
        is_unrestricted[index_cache.particle_properties.crack_driving_force] = true;
        is_unrestricted[index_cache.particle_properties.cohesive_force] = true;
        is_unrestricted[index_cache.particle_properties.slip_state] = true;
        for (unsigned int c = 0; c < SymmetricTensor<2, dim>::n_independent_components; ++c)
          is_unrestricted[index_cache.particle_properties.ve_stress + c] = true;

        std::vector<bool> is_restricted(data_info.n_components(), false);
        is_restricted[index_cache.particle_properties.slip_rate] = true;
        for (unsigned int d = 0; d < dim; ++d)
          {
            is_restricted[index_cache.particle_properties.normal_direction + d] = true;
            is_restricted[index_cache.particle_properties.slip_direction + d]   = true;
          }
        if (output_friction_coefficient)
          is_restricted[index_cache.particle_properties.friction_coefficient] = true;
        if (output_slip_increment)
          is_restricted[index_cache.particle_properties.slip_increment] = true;

        // Interpolate the unrestricted properties by the user-defined interpolator
        const std::vector<std::vector<double>> unrestricted_properties
          = particle_manager.get_interpolator().properties_at_points(particle_handler,
                                                                     std::vector<Point<dim>>(1, particle_location),
                                                                     ComponentMask(is_unrestricted),
                                                                     host_cell);

        particle_properties[index_cache.particle_properties.crack_driving_force] = 
          unrestricted_properties[0][index_cache.particle_properties.crack_driving_force];
        particle_properties[index_cache.particle_properties.cohesive_force] =
          unrestricted_properties[0][index_cache.particle_properties.cohesive_force];
        particle_properties[index_cache.particle_properties.slip_state] =
          unrestricted_properties[0][index_cache.particle_properties.slip_state];
        for (unsigned int c = 0; c < SymmetricTensor<2, dim>::n_independent_components; ++c)
          particle_properties[index_cache.particle_properties.ve_stress + c] = 
            unrestricted_properties[0][index_cache.particle_properties.ve_stress + c];

        // Check if the particle is in the crack zone. Instead of using FEValues or FEPointEvaluation,
        // we evaluate the phase-field value at the particle location manually for efficiency
        const Point<dim> reference_location = this->get_mapping().transform_real_to_unit_cell(host_cell, particle_location);
        const typename DoFHandler<dim>::active_cell_iterator dof_cell(*host_cell, &this->get_dof_handler());
        const FiniteElement<dim> &fe_phi = this->get_fe().base_element(phase_field_base_element);

        double phi = 0;
        for (const unsigned int v : host_cell->vertex_indices())
          {
            const types::global_dof_index dof = dof_cell->vertex_dof_index(v, index_cache.components.phase_field);
            phi += fe_phi.shape_value(v, reference_location) * this->get_solution()(dof);
          }
        
        if (material_model.is_intact(phi))
          {
            // The particle is intact. Initialize the restricted properties to NaN
            for (unsigned int i = 0; i < data_info.n_components(); ++i)
              if (is_restricted[i])
                particle_properties[i] = std::numeric_limits<double>::quiet_NaN();
          }
        else
          {
            // The particle is damaged. Interpolate the restricted properties from the part of
            // surrounding particles that are inside the crack zone
            std::vector<std::pair<double, std::vector<double>>> proportions_and_values =
              PhaseFieldUtilities::interpolate_from_particles_in_crack_zone(particle_handler,
                                                                            std::vector<Point<dim>>(1, particle_location),
                                                                            ComponentMask(is_restricted),
                                                                            host_cell,
                                                                            phase_field_handler.get_grid_cache(),
                                                                            [&] (const ArrayView<const double> &properties) -> bool
                                                                            {
                                                                              return !numbers::is_nan(properties[
                                                                                index_cache.particle_properties.normal_direction]);
                                                                            });

            AssertThrow(proportions_and_values[0].first > 0,
                        ExcMessage("None of the surrounding particles are inside the crack zone!"));

            std::vector<double> &restricted_properties = proportions_and_values[0].second;

            // Orthogonalize and normalize the slip direction and the normal direction
            Tensor<1, dim> n, s;
            for (unsigned int d = 0; d < dim; ++d)
              {
                n[d] = restricted_properties[index_cache.particle_properties.normal_direction + d];
                s[d] = restricted_properties[index_cache.particle_properties.slip_direction + d];
              }

            n /= n.norm();
            s -= (s * n) * n;
            s /= s.norm();

            for (unsigned int d = 0; d < dim; ++d)
              {
                restricted_properties[index_cache.particle_properties.normal_direction + d] = n[d];
                restricted_properties[index_cache.particle_properties.slip_direction + d]   = s[d];
              }

            for (unsigned int i = 0; i < data_info.n_components(); ++i)
              if (is_restricted[i])
                particle_properties[i] = restricted_properties[i];
          }

        return particle_properties;
      }



      template <int dim>
      UpdateTimeFlags
      PhaseFieldRSF<dim>::need_update() const
      {
        // The update is handled by the material model
        return update_never;
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

        if (output_friction_coefficient)
          property_information.emplace_back("friction_coefficient", 1);

        if (output_slip_increment)
          property_information.emplace_back("slip_increment", 1);

        return property_information;
      }



      template <int dim>
      void
      PhaseFieldRSF<dim>::declare_parameters(ParameterHandler &prm)
      {
        prm.enter_subsection("Phase field RSF");
        {
          prm.declare_entry("Output friction coefficient", "false",
                            Patterns::Bool(),
                            "Whether to output the friction coefficient. If set to true, "
                            "then the friction coefficient will be stored in particles as "
                            "a passive particle property.");
          prm.declare_entry("Output slip distance", "false",
                            Patterns::Bool(),
                            "Whether to output the slip distance. If set to true, "
                            "then the slip distance will be stored in particles as "
                            "a passive particle property.");
          prm.declare_entry("Slip distance between graphical output", "0",
                            Patterns::Double(0.),
                            "The maximum slip distance between each generation of output "
                            "files. The default value is 0, which indicates that the output "
                            "is not controlled by the slip distance. If a positive value "
                            "is set, then a property named 'slip_increment' will be stored "
                            "in particles. Once the maximum slip increment reaches the "
                            "specified value, then an output request will be sent to both "
                            "the solution-based and particle-based visualizers, and the slip "
                            "increment will be reset to 0.");
        }
        prm.leave_subsection();
      }



      template <int dim>
      void
      PhaseFieldRSF<dim>::parse_parameters(ParameterHandler &prm)
      {
        prm.enter_subsection("Phase field RSF");
        {
          output_friction_coefficient           = prm.get_bool("Output friction coefficient");
          maximum_slip_distance_between_outputs = prm.get_double("Slip distance between graphical output");

          if (maximum_slip_distance_between_outputs > 0)
            output_slip_increment = true;
          else
            output_slip_increment = false;
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
      ASPECT_REGISTER_PARTICLE_PROPERTY(PhaseFieldRSF,
                                        "phase field rsf",
                                        "")
    }
  }
}
