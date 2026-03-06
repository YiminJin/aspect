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
          }

        // The normal direction must be associated with a compositional field
        for (unsigned int d = 0; d < dim; ++d)
          AssertThrow(compositional_indices.normal_direction[d] != numbers::invalid_unsigned_int,
                      ExcMessage("Particle property 'normal_direction[d]' (d = 0, ..., dim-1) needs to be "
                                 "associated with a compositional field."));

        // If the user chooses to start with slip, then the slip rate and slip direction
        // must be associated with compositional fields
        if (start_with_slip == true)
          {
            AssertThrow(compositional_indices.slip_rate != numbers::invalid_unsigned_int,
                        ExcMessage("When the parameter <Start with slip> is set to true, the particle property "
                                   "'slip_rate' needs to be associated with a compositional field."));
            for (unsigned int d = 0; d < dim; ++d)
              AssertThrow(compositional_indices.slip_direction[d] != numbers::invalid_unsigned_int,
                          ExcMessage("When the parameter <Start with slip> is set to true, the particle property "
                                     "'slip_direction[d]' (d = 0, ..., dim-1) needs to be associated with a "
                                     "compositional field."));
          }

        // If the user chooses not to start with steady state, then the slip state must be
        // associated with a compositional field
        if (start_with_steady_state == false)
          AssertThrow(compositional_indices.slip_state != numbers::invalid_unsigned_int,
                      ExcMessage("When the parameter <Start with steady state> is set to false, the particle "
                                 "property 'slip_state' needs to be associated with a compositional field."));

        // Initialize the data position cache
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

        const FEVariable<dim> &pf_variable = this->introspection().variable("phase_field");
        phase_field_component_index = pf_variable.first_component_index;
        phase_field_base_index      = pf_variable.base_index;
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
        const typename DoFHandler<dim>::active_cell_iterator dof_cell(*cell_and_point.first, &this->get_dof_handler());
        const double pf = get_phase_field_value(this->get_solution(), dof_cell, cell_and_point.second);
        const bool is_inside_fault_band = material_model.is_inside_fault_band(pf);

        // The initial crack driving force is set to $H_t$
        std::vector<double> initial_composition(this->introspection().n_compositional_fields);
        for (unsigned int j = 0; j < initial_composition.size(); ++j)
          initial_composition[j] = this->get_initial_composition_manager().initial_composition(position, j);
        const std::vector<double> volume_fractions = MaterialModel::MaterialUtilities::compute_only_composition_fractions(
          initial_composition, this->introspection().chemical_composition_field_indices());
        const std::vector<double> Ht = material_model.get_threshold_crack_driving_forces();
        data.push_back(MaterialModel::MaterialUtilities::average_value(volume_fractions, Ht, MaterialModel::MaterialUtilities::arithmetic));

        // If the user chooses to start with slip, then the initial slip rate is determined by the initial composition;
        // otherwise, the initial slip rate is set to the lower bound
        const double V = std::max(rsf_model.get_minimum_slip_rate(),
                                  ((start_with_slip && is_inside_fault_band) ?
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
        if (is_inside_fault_band)
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
        if (start_with_slip && is_inside_fault_band)
          {
            for (unsigned int d = 0; d < dim; ++d)
              s[d] = this->get_initial_composition_manager().initial_composition(position, compositional_indices.slip_direction[d]);

            s -= (s * n) * n;
            s /= s.norm();
          }

        for (unsigned int d = 0; d < dim; ++d)
          data.push_back(s[d]);

        //The initial bulk stress is set to 0
        for (unsigned int c = 0; c < SymmetricTensor<2, dim>::n_independent_components; ++c)
          data.push_back(0.);

        // Finally, initialize the interface stress
        SymmetricTensor<2, dim> tau_f = numbers::signaling_nan<SymmetricTensor<2, dim>>();
        if (start_with_slip && is_inside_fault_band)
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
                               const typename Triangulation<dim>::active_cell_iterator &cell) const
      {
        const Particle::Manager<dim> &particle_manager = this->get_phase_field_handler().get_associated_particle_manager();
        const auto &data_info = particle_manager.get_property_manager().get_data_info();
        std::vector<double> particle_properties(data_info.n_components(), numbers::signaling_nan<double>());

        // The properties to be interpolated depend on the phase field value
        typename Triangulation<dim>::active_cell_iterator host_cell = cell;
        if (host_cell->state() != IteratorState::valid)
          host_cell = GridTools::find_active_cell_around_point(this->get_mapping(), this->get_triangulation(), particle_location).first;
        const typename DoFHandler<dim>::active_cell_iterator dof_cell(*cell, &this->get_dof_handler());
        const double pf = get_phase_field_value(this->get_solution(), dof_cell, this->get_mapping().transform_real_to_unit_cell(host_cell, particle_location));

        const MaterialModel::PhaseFieldRSF<dim> &material_model = dynamic_cast<const MaterialModel::PhaseFieldRSF<dim>&>(this->get_material_model());
        if (material_model.is_inside_fault_band(pf))
          {

          }

        return particle_properties;
      }



      template <int dim>
      void
      PhaseFieldRSF<dim>::update_particle_properties(const ParticleUpdateInputs<dim> &inputs,
                                                     typename ParticleHandler<dim>::particle_iterator_range &particles) const
      {
        const MaterialModel::PhaseFieldRSF<dim> &material_model 
          = dynamic_cast<const MaterialModel::PhaseFieldRSF<dim>&>(this->get_material_model());

        // Vector storing the chemical field values, which is required for computing volume fractions
        std::vector<double> chemical_field_values(data_position_cache.chemical_fields.size());

        unsigned int p = 0;
        for (auto &particle : particles)
          {
            const ArrayView<double> particle_properties = particle.get_properties();

            // The crack driving force, slip rate, slip state and stress are 
            // ought to be updated by the material model. Here we only need to 
            // rotate the fault normal direction and the slip direction if
            // the particle is inside the fault band
            const double pf = get_phase_field_value(this->get_solution(), inputs.current_cell, particle.get_reference_location());
            if (material_model.is_inside_fault_band(pf) == false)
              continue;

            Tensor<1, dim> n;
            for (unsigned int d = 0; d < dim; ++d)
              n[d] = particle_properties[data_position_cache.normal_direction + d];
            // The normal direction should have been updated by the material model
            AssertThrow(n[0] != numbers::signaling_nan<double>(), ExcInternalError());

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
                  particle_properties[data_position_cache.normal_direction + d] = n[d];
              }

            // Now update the slip direction according to the normal direction and
            // the trial stress
            for (unsigned int j = 0; j < chemical_field_values.size(); ++j)
              chemical_field_values[j] = particle_properties[data_position_cache.chemical_fields[j]];
            const std::vector<double> volume_fractions = MaterialModel::MaterialUtilities::compute_composition_fractions(chemical_field_values);

            // Get the interface stress
            const SymmetricTensor<2, dim> tau_f = Utilities::Tensors::to_symmetric_tensor<dim>(
              &particle_properties[data_position_cache.interface_stress],
              &particle_properties[data_position_cache.interface_stress] + SymmetricTensor<2, dim>::n_independent_components);

            // Compute the normal and shear stress on the fault plane
            const Tensor<1, dim> t = tau_f * n;
            const Tensor<1, dim> t_s = t - (t * n) * n;
            const double p_bar = this->get_adiabatic_conditions().pressure(particle.get_location());
            // Here we only need to avoid division by 0, so there is no need to 
            // compute the friction coefficient
            if (t_s.norm() < p_bar * 1.e-20)
              {
                // The shear stress is too small: impossible to slip
                for (unsigned int d = 0; d < dim; ++d)
                  particle_properties[data_position_cache.slip_direction + d] = numbers::signaling_nan<double>();
              }
            else
              {
                const Tensor<1, dim> s = t_s / t_s.norm();
                for (unsigned int d = 0; d < dim; ++d)
                  particle_properties[data_position_cache.slip_direction + d] = s[d];
              }
          }
      }



      template <int dim>
      double
      PhaseFieldRSF<dim>::
      get_phase_field_value(const LinearAlgebra::BlockVector &solution,
                            const typename DoFHandler<dim>::active_cell_iterator &cell,
                            const Point<dim> &reference_location) const
      {
        Vector<double> dof_values(this->get_fe().dofs_per_cell);
        cell->get_dof_values(solution, dof_values);

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



      template <int dim>
      void
      PhaseFieldRSF<dim>::declare_parameters(ParameterHandler &prm)
      {
        prm.enter_subsection("Phase field RSF");
        {
          prm.declare_entry("Start with slip", "false",
                            Patterns::Bool(),
                            "If set to true, then the particle properties 'slip_rate' and 'slip_direction' "
                            "should be associated with compositional fields, and the program will initialize "
                            "the slip rate and slip direction with the corresponding initial composition "
                            "values, and make the initial stress consistent with the prescribed slip rate. "
                            "Otherwise, the initial slip rate and stress are set to 0, which means that "
                            "the fault (if prescribed) is initially in stick mode.");

          prm.declare_entry("Start with steady state", "true",
                            Patterns::Bool(),
                            "If set to true, then the initial slip state is assumed to be steady, which "
                            "leads to $\\theta_{\\text{init}} = D_c / V_{\\text{init}}$. Otherwise, the "
                            "particle property 'slip_state' should be associated with a compositional "
                            "field, and the program will initialze the slip state with the corresponding "
                            "initial composition value.");
        }
        prm.leave_subsection();
      }



      template <int dim>
      void
      PhaseFieldRSF<dim>::parse_parameters(ParameterHandler &prm)
      {
        prm.enter_subsection("Phase field RSF");
        {
          start_with_slip         = prm.get_bool("Start with slip");
          start_with_steady_state = prm.get_bool("Start with steady state");
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
