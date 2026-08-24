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

#include <aspect/material_model/phase_field_rsf.h>
#include <aspect/phase_field.h>
#include <aspect/particle/manager.h>
#include <aspect/newton.h>
#include <aspect/simulator.h>
#include <aspect/postprocess/visualization.h>
#include <aspect/postprocess/particles.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_cartesian.h>

namespace aspect
{
  namespace MaterialModel
  {
    template <int dim>
    void PhaseFieldRSF<dim>::IndexCache::
    initialize(const Introspection<dim>     &introspection,
               const Parameters<dim>        &parameters,
               const PhaseFieldHandler<dim> &phase_field_handler)
    {
      // Initialize the particle property indices
      const Particle::Manager<dim> &particle_manager = phase_field_handler.get_associated_particle_manager();
      const auto &particle_data_info = particle_manager.get_property_manager().get_data_info();

      particle_properties.crack_driving_force = particle_data_info.get_position_by_field_name("crack_driving_force");
      particle_properties.cohesive_force      = particle_data_info.get_position_by_field_name("cohesive_force");
      particle_properties.slip_rate           = particle_data_info.get_position_by_field_name("slip_rate");
      particle_properties.slip_state          = particle_data_info.get_position_by_field_name("slip_state");
      particle_properties.normal_direction    = particle_data_info.get_position_by_field_name("normal_direction");
      particle_properties.slip_direction      = particle_data_info.get_position_by_field_name("slip_direction");
      particle_properties.ve_stress           = particle_data_info.get_position_by_field_name("ve_stress");

      if (particle_data_info.fieldname_exists("friction_coefficient"))
        particle_properties.friction_coefficient = particle_data_info.get_position_by_field_name("friction_coefficient");

      if (particle_data_info.fieldname_exists("slip_increment"))
        particle_properties.slip_increment = particle_data_info.get_position_by_field_name("slip_increment");

      particle_properties.chemical_fields.clear();
      for (const unsigned int index : introspection.chemical_composition_field_indices())
        particle_properties.chemical_fields.push_back(
          particle_data_info.get_position_by_field_name(parameters.mapped_particle_properties.find(index)->second.first));

      // Initialize the compositional indices
      for (const auto &key_and_value : parameters.mapped_particle_properties)
        {
          if (key_and_value.second.first == "crack_driving_force")
            compositional_fields.crack_driving_force = key_and_value.first;

          if (key_and_value.second.first == "slip_rate")
            compositional_fields.slip_rate = key_and_value.first;

          if (key_and_value.second.first == "slip_state")
            compositional_fields.slip_state = key_and_value.first;

          if (key_and_value.second.first == "normal_direction")
            {
              AssertThrow(key_and_value.second.second < dim,
                          ExcMessage("The component indices of normal_direction exceed the range of [0, dim)."));
              compositional_fields.normal_direction[key_and_value.second.second] = key_and_value.first;
            }

          if (key_and_value.second.first == "ve_stress")
            {
              AssertThrow((key_and_value.second.second < SymmetricTensor<2, dim>::n_independent_components),
                          ExcMessage("The component indices of ve_stress exceed the range of [0, dim(dim+1)/2)."));
              compositional_fields.ve_stress[key_and_value.second.second] = key_and_value.first;
            }
        }

      // The slip state and components of viscoelastic stress must be associated with compositional fields
      AssertThrow(compositional_fields.slip_state != numbers::invalid_unsigned_int,
                  ExcMessage("Particle property 'slip_state' must be associated with a compositional field."));
      for (unsigned int c = 0; c < SymmetricTensor<2, dim>::n_independent_components; ++c)
        AssertThrow(compositional_fields.ve_stress[c] != numbers::invalid_unsigned_int,
                    ExcMessage("Particle property 've_stress[" + Utilities::int_to_string(c)
                               + "] must be associated with a compositional field."));

      // If there are pre-existing cracks (i.e. crack_driving_force is associated with a compositional field),
      // then the slip rate and the normal direction must be associated with compositional fields too
      if (compositional_fields.crack_driving_force != numbers::invalid_unsigned_int)
        {
          bool is_consistent = (compositional_fields.slip_rate != numbers::invalid_unsigned_int);
          for (unsigned int d = 0; d < dim; ++d)
            if (compositional_fields.normal_direction[d] == numbers::invalid_unsigned_int)
              {
                is_consistent = false;
                break;
              }

          AssertThrow(is_consistent,
                      ExcMessage("The particle property 'crack_driving_force' is associated with a compositional fields, "
                                 "which means that there are pre-existing cracks in the model. In this case, the particle "
                                 "properties 'slip_rate' and 'normal_direction[d]' (d = 0, ..., dim-1) should also be "
                                 "associated with compositional fields to get initial values from the initial composition "
                                 "model, otherwise the initial conditions are incomplete."));
        }

      // Initialize the component indices
      components.phase_field      = introspection.variable("phase_field").first_component_index;
      components.core_phase_field = introspection.variable("core_phase_field").first_component_index;
    }



    template <int dim>
    void PhaseFieldRSF<dim>::initialize()
    {
      AssertThrow(this->get_parameters().enable_phase_field == true,
                  ExcMessage("The phase field PSF model requires phase field to be included in "
                             "the system formulation. Please set <Formulation/Include phase field> to true."));

      AssertThrow(this->get_parameters().need_slip_rate == true,
                  ExcMessage("The phase field RSF model requires the slip rate to calculate the "
                             "friction coefficient. Please set <Formulation/Need slip rate> to true."));

      AssertThrow(this->get_parameters().enable_elasticity == false,
                  ExcMessage("In the phase field RSF model, the elastic rheology is not handled by "
                             "MaterialModel::Rheology::Elasticity. Please set <Formulation/Enable elasticity> "
                             "to false."));

      AssertThrow(this->get_parameters().use_implicit_constitutive_model == true,
                  ExcMessage("The phase field RSF model requires the Stokes assemblers to support implicit "
                             "constitutive model. Please set <Formulation/Use implicit constitutive model> "
                             "to true."));

      // Do the actual initialization when it is sure that the phase field handler has been initialized
      this->get_signals().post_simulator_initialization.connect(
        [&](const SimulatorAccess<dim> &)
      {
        this->index_cache.initialize(this->introspection(),
                                     this->get_parameters(),
                                     this->get_phase_field_handler());
      });

      // Perform return mapping before assembling the Stokes system
      this->get_signals().pre_assemble_stokes_system.connect(
        [&](const SimulatorAccess<dim> &)
      {
        this->perform_return_mapping();
      });

      // Update the history states (crack driving force, cohesive force, slip state and
      // viscoelastic stress) after the nonlinear iterations
      this->get_signals().post_nonlinear_solver.connect(
        [&](const SolverControl &nonlinear_solver_control)
      {
        this->update_history_states(nonlinear_solver_control);
      });

      // Update particles in the emerging crack zone before extending the 
      // core phase-field
      this->get_phase_field_handler().pre_extend_core_phase_field.connect(
        [&](const SimulatorAccess<dim> &)
      {
        this->update_particles_in_emerging_crack_zone();
      });
    }



    template <int dim>
    void PhaseFieldRSF<dim>::update()
    {
      // Work around a memory leak in deal.II that is fixed in 9.8.0-pre
#if !DEAL_II_VERSION_GTE(9,8,0)
      solution_evaluator.reset();
      solution_evaluator = construct_solution_evaluator(*this, update_values | update_gradients);
#endif
    }



    template <int dim>
    void
    PhaseFieldRSF<dim>::
    evaluate(const MaterialModel::MaterialModelInputs<dim> &in,
             MaterialModel::MaterialModelOutputs<dim> &out) const
    {
      EquationOfStateOutputs<dim> eos_outputs(this->introspection().n_chemical_composition_fields() + 1);

      const double dt = (this->get_timestep_number() > 0 ? this->get_timestep() : 0);

      for (unsigned int i = 0; i < in.n_evaluation_points(); ++i)
        {
          const std::vector<double> volume_fractions = MaterialUtilities::compute_only_composition_fractions(
            in.composition[i], this->introspection().chemical_composition_field_indices());

          // Fill in the equation-of-state outputs
          equation_of_state.evaluate(in, i, eos_outputs);

          out.densities[i] = MaterialUtilities::average_value(volume_fractions, eos_outputs.densities, MaterialUtilities::arithmetic);
          out.thermal_expansion_coefficients[i] = MaterialUtilities::average_value(volume_fractions, eos_outputs.thermal_expansion_coefficients, MaterialUtilities::arithmetic);
          out.specific_heat[i] = MaterialUtilities::average_value(volume_fractions, eos_outputs.specific_heat_capacities, MaterialUtilities::arithmetic);
          out.thermal_conductivities[i] = MaterialUtilities::average_value(volume_fractions, thermal_conductivities, MaterialUtilities::arithmetic);
          out.compressibilities[i] = MaterialUtilities::average_value(volume_fractions, eos_outputs.compressibilities, MaterialUtilities::arithmetic);
          out.entropy_derivative_pressure[i] = MaterialUtilities::average_value(volume_fractions, eos_outputs.entropy_derivative_pressure, MaterialUtilities::arithmetic);
          out.entropy_derivative_temperature[i] = MaterialUtilities::average_value(volume_fractions, eos_outputs.entropy_derivative_temperature, MaterialUtilities::arithmetic);

          if (in.requests_property(MaterialProperties::viscosity))
            {
              // Set the output viscosity to be the viscoelastic viscosity (It will not be used in the assemblers,
              // but might be requested by some other functions, like Simulator::compute_pressure_scaling_factor()).
              const double G = MaterialUtilities::average_value(volume_fractions, elastic_shear_moduli, viscosity_averaging);
              const double eta = calculate_creep_viscosity(in.temperature[i], volume_fractions);
              const double one_minus_beta = calculate_stress_relaxation_factor(eta, G);
              out.viscosities[i] = eta * one_minus_beta;
            }
        }

      const std::shared_ptr<MaterialModel::ImplicitConstitutiveOutputs<dim>> implicit_constitutive_outputs
        = out.template get_additional_output_object<MaterialModel::ImplicitConstitutiveOutputs<dim>>();

      // If the implicit constitutive outputs is requested, then we map the current states
      // from particles to quadrature points with MLS method
      if (implicit_constitutive_outputs != nullptr)
        {
          const Particles::ParticleHandler<dim> &particle_handler
            = this->get_phase_field_handler().get_associated_particle_manager().get_particle_handler();
          const unsigned int n_particles_in_cell = particle_handler.n_particles_in_cell(in.current_cell);
          const auto particles_in_cell = particle_handler.particles_in_cell(in.current_cell);

          implicit_constitutive_outputs->resize(n_particles_in_cell);

          // Collect the particle locations
          small_vector<Point<dim>> reference_locations;
          for (auto particle = particles_in_cell.begin(); particle != particles_in_cell.end(); ++particle)
            reference_locations.push_back(particle->get_reference_location());

          // Collect the DoF values
          small_vector<double, 500> dof_values(this->get_fe().dofs_per_cell);
          in.current_cell->get_dof_values(this->get_solution(), dof_values.begin(), dof_values.end());

          // Solution values/gradients to be evaluated at particle locations:
          // phase-field value;
          // core phase-field value;
          // temperature value;
          std::vector<EvaluationFlags::EvaluationFlags> evaluation_flags(this->introspection().n_components, 
                                                                         EvaluationFlags::nothing);
          evaluation_flags[index_cache.components.phase_field] = EvaluationFlags::values;
          evaluation_flags[index_cache.components.core_phase_field] = EvaluationFlags::values;
          evaluation_flags[this->introspection().component_indices.temperature] = EvaluationFlags::values;

          solution_evaluator->reinit(in.current_cell, {reference_locations.data(), reference_locations.size()});
          solution_evaluator->evaluate({dof_values.data(), dof_values.size()}, evaluation_flags);

          small_vector<double>         particle_solution_values(this->introspection().n_components);
          small_vector<Tensor<1, dim>> particle_solution_gradients(this->introspection().n_components);

          // Vector storing the chemical field values, which is required for computing volume fractions
          std::vector<double> chemical_field_values(this->introspection().n_chemical_composition_fields());

          for (auto particle = particles_in_cell.begin(); particle != particles_in_cell.end(); ++particle)
            {
              const ArrayView<const double> particle_properties = particle->get_properties();
              const unsigned int p = std::distance(particles_in_cell.begin(), particle);

              solution_evaluator->get_solution(p, {particle_solution_values.data(), particle_solution_values.size()}, evaluation_flags);

              // Compute the volume fractions
              for (unsigned int j = 0; j < chemical_field_values.size(); ++j)
                chemical_field_values[j] = particle_properties[index_cache.particle_properties.chemical_fields[j]];
              const std::vector<double> volume_fractions = MaterialUtilities::compute_composition_fractions(chemical_field_values);

              // Compute the reference viscosity
              const double T   = particle_solution_values[this->introspection().component_indices.temperature];
              const double eta = calculate_creep_viscosity(T, volume_fractions);
              const double G   = MaterialUtilities::average_value(volume_fractions, elastic_shear_moduli, viscosity_averaging);

              const double one_minus_beta = calculate_stress_relaxation_factor(eta, G);
              const double beta = 1. - one_minus_beta;
              const double eta_ve = eta * one_minus_beta;

              // Compute the trial stress
              const SymmetricTensor<2, dim> tau_old = Utilities::Tensors::to_symmetric_tensor<dim>(
                &particle_properties[index_cache.particle_properties.ve_stress],
                &particle_properties[index_cache.particle_properties.ve_stress] + SymmetricTensor<2, dim>::n_independent_components);

              // Check if the point is damaged
              const double phi = particle_solution_values[index_cache.components.phase_field];
              if (phi > phase_field_activation_threshold)
                {
                  // Compute the slip direction tensor
                  const Tensor<1, dim> n(ArrayView<const double>(&particle_properties[index_cache.particle_properties.normal_direction], dim));
                  const Tensor<1, dim> s(ArrayView<const double>(&particle_properties[index_cache.particle_properties.slip_direction], dim));
                  const SymmetricTensor<2, dim> S = symmetrize(outer_product(n, s));

                  // Compute the nonlinear stress
                  const double V = particle_properties[index_cache.particle_properties.slip_rate];
                  AssertThrow(numbers::is_finite(V), ExcInternalError());

                  const double phi_hat = particle_solution_values[index_cache.components.core_phase_field];
                  const double g   = this->get_phase_field_handler().energetic_degradation(volume_fractions, phi);
                  const double chi = this->get_phase_field_handler().slip_rate_localization_factor(volume_fractions, g, phi_hat);

                  implicit_constitutive_outputs->nonlinear_stresses[p] = -(2. * eta_ve * chi * V) * S + beta * tau_old;

                  // Compute the nonlinear tangent modulus if requested
                  if (in.requests_property(MaterialProperties::tangent_modulus))
                    {
                      const double theta_old = particle_properties[index_cache.particle_properties.slip_state];
                      const double theta  = rsf_rheology.slip_state(V, theta_old, dt);
                      const double dmu_dV = rsf_rheology.friction_coefficient_derivative_wrt_slip_rate(volume_fractions, V, theta);

                      const double sigma_n = this->get_adiabatic_conditions().pressure(particle->get_location());
                      const double eta_d = MaterialUtilities::average_value(volume_fractions, radiation_damping_coefficients, MaterialUtilities::arithmetic);

                      const double softening_factor = (2. * chi * eta_ve) / (chi * eta_ve / (1. - g) + dmu_dV * sigma_n + eta_d);

                      implicit_constitutive_outputs->nonlinear_tangent_moduli[p] = -(2. * eta_ve * softening_factor) * outer_product(S, S);
                    }
                }
              else // phi <= phase_field_activation_threshold
                {
                  implicit_constitutive_outputs->nonlinear_stresses[p] = beta * tau_old;
                  if (in.requests_property(MaterialProperties::tangent_modulus))
                    implicit_constitutive_outputs->nonlinear_tangent_moduli[p] = 0;
                }
            }
        }
    }



    namespace
    {
      struct ConvergenceHistory
      {
        std::vector<double> V;
        std::vector<double> F;
        std::vector<double> dF_dV;
        bool success;

        ConvergenceHistory()
          : success(true)
        {}

        void clear()
        {
          V.clear();
          F.clear();
          dF_dV.clear();
          success = true;
        }
      };



      void
      output_convergence_history(const std::string        &filename,
                                 const ConvergenceHistory &history)
      {
        std::ofstream f(filename);
        AssertThrow(f.is_open(), ExcMessage("Cannot open file <" + filename + ">."));

        // Output the Newton iterations
        f << std::setprecision(6) << std::left;
        f << "# Newton iterations:" << std::endl
          << "# Step    V             F             dF/dV" << std::endl;
        for (unsigned int n = 0; n < history.F.size(); ++n)
          f << "  "
            << std::setw(8) << n
            << std::setw(14) << history.V[n]
            << std::setw(14) << history.F[n]
            << std::setw(14) << history.dF_dV[n]
            << std::endl;

        f.close();
      }
    }



    template <int dim>
    void PhaseFieldRSF<dim>::perform_return_mapping()
    {
      const Introspection<dim> &introspection = this->introspection();
      const PhaseFieldHandler<dim> &phase_field_handler = this->get_phase_field_handler();
      Particles::ParticleHandler<dim> &particle_handler = const_cast<Particles::ParticleHandler<dim>&>(
        phase_field_handler.get_associated_particle_manager().get_particle_handler());

      // Vector storing the chemical field values, which is required for computing volume fractions
      std::vector<double> chemical_field_values(introspection.n_chemical_composition_fields());

      // Solution values/gradients to be evaluated at particle locations:
      // velocity gradient;
      // phase-field value;
      // core phase-field value;
      // temperature value;
      std::vector<EvaluationFlags::EvaluationFlags> evaluation_flags(introspection.n_components,
                                                                     EvaluationFlags::nothing);
      for (unsigned int d = 0; d < dim; ++d)
        evaluation_flags[d] = EvaluationFlags::gradients;
      evaluation_flags[index_cache.components.phase_field] = EvaluationFlags::values;
      evaluation_flags[index_cache.components.core_phase_field] = EvaluationFlags::values;
      evaluation_flags[introspection.component_indices.temperature] = EvaluationFlags::values;

      // The upper limit of Newton iterations for solving the nonlinear equation for V
      constexpr unsigned int max_newton_iterations = 30;

      // Record of the convergence behavoir
      ConvergenceHistory convergence_history;

      const double dt = (this->get_timestep_number() > 0 ? this->get_timestep() : 0);

      // Perform the local return-mapping
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            const auto particles_in_cell = particle_handler.particles_in_cell(cell);
            const unsigned int n_particles_in_cell = particle_handler.n_particles_in_cell(cell);
            AssertThrow(n_particles_in_cell > 0, ExcInternalError());

            // Collect the particle locations
            small_vector<Point<dim>> reference_locations;
            for (auto particle = particles_in_cell.begin(); particle != particles_in_cell.end(); ++particle)
              reference_locations.push_back(particle->get_reference_location());

            // Collect the DoF values
            small_vector<double, 500> dof_values(this->get_fe().dofs_per_cell);
            cell->get_dof_values(this->get_solution(), dof_values.begin(), dof_values.end());

            // Update the solution_evaluator and evaluate the solution values and gradients 
            // at particle locations
            solution_evaluator->reinit(cell, {reference_locations.data(), reference_locations.size()});
            solution_evaluator->evaluate({dof_values.data(), dof_values.size()}, evaluation_flags);

            small_vector<double>         particle_solution_values(introspection.n_components);
            small_vector<Tensor<1, dim>> particle_solution_gradients(introspection.n_components);

            for (auto particle = particles_in_cell.begin(); particle != particles_in_cell.end(); ++particle)
              {
                const unsigned int p = std::distance(particles_in_cell.begin(), particle);
                const ArrayView<double> particle_properties = particle->get_properties();

                solution_evaluator->get_solution(p, {particle_solution_values.data(), particle_solution_values.size()}, evaluation_flags);
                solution_evaluator->get_gradients(p, {particle_solution_gradients.data(), particle_solution_gradients.size()}, evaluation_flags);

                // Check if the particle is damaged
                const double phi = particle_solution_values[index_cache.components.phase_field];
                if (phi > phase_field_activation_threshold)
                  {
                    // Compute the volume fractions
                    for (unsigned int j = 0; j < chemical_field_values.size(); ++j)
                      chemical_field_values[j] = particle_properties[index_cache.particle_properties.chemical_fields[j]];
                    const std::vector<double> volume_fractions = MaterialUtilities::compute_composition_fractions(chemical_field_values);

                    if (this->get_timestep_number() == 0 && this->get_nonlinear_iteration() == 0)
                      {
                        // If it is the first nonlinear iteration of the first time step, the slip rate is prescribed,
                        // but the cohesive force has not been initialized (because it depends on the core phase-field).
                        // Initialize the cohesive force here.
                        const double phi_hat = particle_solution_values[index_cache.components.core_phase_field];

                        const double H = phase_field_handler.stationary_crack_driving_force(volume_fractions, phi, phi_hat);
                        const double g = phase_field_handler.energetic_degradation(volume_fractions, phi);
                        const double G = MaterialUtilities::average_value(volume_fractions, elastic_shear_moduli, viscosity_averaging);

                        particle_properties[index_cache.particle_properties.cohesive_force] = g * std::sqrt(2. * G * H);

                        continue;
                      }

                    // Get the ingredients for return-mapping
                    Tensor<2, dim> grad_u;
                    for (unsigned int d = 0; d < dim; ++d)
                      grad_u[d] = particle_solution_gradients[d];
                    const SymmetricTensor<2, dim> epsilon = symmetrize(grad_u);

                    const double T   = particle_solution_values[introspection.component_indices.temperature];
                    const double eta = calculate_creep_viscosity(T, volume_fractions);
                    const double G   = MaterialUtilities::average_value(volume_fractions, elastic_shear_moduli, viscosity_averaging);

                    const double one_minus_beta = calculate_stress_relaxation_factor(eta, G);
                    const double beta = 1. - one_minus_beta;
                    const double eta_ve = eta * one_minus_beta;

                    const double phi_hat = particle_solution_values[index_cache.components.core_phase_field];
                    const double g   = phase_field_handler.energetic_degradation(volume_fractions, phi);
                    const double chi = phase_field_handler.slip_rate_localization_factor(volume_fractions, g, phi_hat);

                    const Tensor<1, dim> n(ArrayView<const double>(&particle_properties[index_cache.particle_properties.normal_direction], dim));
                    const Tensor<1, dim> s(ArrayView<const double>(&particle_properties[index_cache.particle_properties.slip_direction], dim));

                    const SymmetricTensor<2, dim> S = symmetrize(outer_product(n, s));

                    const SymmetricTensor<2, dim> tau_old = Utilities::Tensors::to_symmetric_tensor<dim>(
                      &particle_properties[index_cache.particle_properties.ve_stress],
                      &particle_properties[index_cache.particle_properties.ve_stress] + SymmetricTensor<2, dim>::n_independent_components);

                    const double h_tau_coh_old = particle_properties[index_cache.particle_properties.cohesive_force];

                    const double sigma_n = this->get_adiabatic_conditions().pressure(particle->get_location());
                    const double eta_d = MaterialUtilities::average_value(volume_fractions,
                                                                          radiation_damping_coefficients,
                                                                          MaterialUtilities::arithmetic);

                    const double V_prev    = particle_properties[index_cache.particle_properties.slip_rate];
                    const double theta_old = particle_properties[index_cache.particle_properties.slip_state];

                    // If the particle was intact in the previous time step, then its slip rate has not
                    // been initialized.
                    AssertThrow(numbers::is_finite(V_prev) || this->get_nonlinear_iteration() == 0,
                                ExcInternalError());

                    const double theta = (numbers::is_finite(V_prev) ?
                                          rsf_rheology.slip_state(V_prev, theta_old, dt) :
                                          theta_old);

                    // Lambda functions that evaluate the value and derivative of the RSF yield function
                    auto F_value = [&](const double V_current)
                    {
                      const double v = V_current * chi;
                      const double tau_S = eta_ve * (2. * (epsilon * S) - v) + beta * (tau_old * S);

                      const double mu = rsf_rheology.friction_coefficient(volume_fractions, V_current, theta);
                      const double tau_fric = mu * sigma_n;

                      const double h = 1. / g - 1.;
                      const double tau_coh = (eta_ve * v + beta * h_tau_coh_old) / h;

                      const double tau_dpr = eta_d * V_current;

                      return tau_S - tau_fric - tau_coh - tau_dpr;
                    };

                    auto F_derivative = [&](const double V_current)
                    {
                      const double dmu_dV = rsf_rheology.friction_coefficient_derivative_wrt_slip_rate(volume_fractions, V_current, theta);
                      return -(chi * eta_ve / (1. - g) + dmu_dV * sigma_n + eta_d);
                    };

                    // Use Newton-Raphson method to solve the nonlinear equation for V.
                    const double V_init = rsf_rheology.get_minimum_slip_rate();
                    const double F_init = F_value(V_init);

                    // Check if |F| > (mu * sigma_n + eta_d * V) * 1e-6
                    const double mu_init = rsf_rheology.friction_coefficient(volume_fractions, V_init, theta);
                    if (std::abs(F_init) > (mu_init * sigma_n + eta_d * V_init) * 1.e-6)
                      {
                        convergence_history.clear();

                        convergence_history.V.push_back(V_init);
                        convergence_history.F.push_back(F_init);

                        const double tol = std::abs(F_init) * 1.e-8;
                        
                        double V = V_init;
                        double F = F_init;

                        unsigned int n_iter = 0;

                        while (std::abs(F) > tol)
                          {
                            const double dF_dV = F_derivative(V);
                            convergence_history.dF_dV.push_back(dF_dV);
                            if (std::abs(dF_dV) < std::numeric_limits<double>::epsilon())
                              break;

                            V -= F / dF_dV;
                            F = F_value(V);

                            ++n_iter;
                            if (n_iter > max_newton_iterations)
                              break;

                            convergence_history.V.push_back(V);
                            convergence_history.F.push_back(F);
                          }

                        // If solution failed, then exit all loops at once
                        if (std::abs(F) > tol)
                          {
                            convergence_history.success = false;
                            goto convergence_check;
                          }

                        // Update the slip rate
                        particle_properties[index_cache.particle_properties.slip_rate] = std::max(rsf_rheology.get_minimum_slip_rate(), V);
                      }
                  }
              }
          }

convergence_check:
      const int local_fail_flag = (convergence_history.success ? 0 : 1);
      const int global_fail_flag = Utilities::MPI::max(local_fail_flag, this->get_mpi_communicator());

      if (global_fail_flag)
        {
          // Report the error message for the first faling rank
          const int my_fail_rank = (convergence_history.success ? 
                                    std::numeric_limits<int>::max() :
                                    Utilities::MPI::this_mpi_process(this->get_mpi_communicator()));
          const int first_fail_rank = Utilities::MPI::min(my_fail_rank, this->get_mpi_communicator());

          if (my_fail_rank == first_fail_rank)
            {
              const std::string output_filename =
                this->get_parameters().output_directory + "convergence_history.txt";

              output_convergence_history(output_filename, convergence_history);

              AssertThrow(false,
                          ExcMessage("The nonlinear solver failed to find a root for the RSF "
                                     "consistent equation when performing local return-mapping. "
                                     "See <" + output_filename + "> for the convergence history."));
            }
        }      
    }



    namespace
    {
      /**
       * In 2D, the spin tensor w can be expressed as
       *
       *       /  0  -w0 \
       *   w = |         |
       *       \  w0  0  /,
       * where
       *   w0 = (l21 - l12) / 2.
       *
       * If w is nonzero, then the spinning axis t is given by
       *   t = a[0, 0, 1]^T,
       * where
       *   a = |w0| / w0.
       *
       * Otherwise, return an invalid vector.
       */
      Tensor<1, 3>
      rotation_axis(const Tensor<2, 2> &L)
      {
        const double w = (L[1][0] - L[0][1]) * 0.5;

        Tensor<1, 3> t = numbers::signaling_nan<Tensor<1, 3>>();
        if (std::abs(w) > std::numeric_limits<double>::epsilon())
          t[2] = (w > 0.0 ? 1.0 : -1.0);

        return t;
      }



      /**
       * In 3D, the spin tensor w can be expressed as
       *
       *       /  0  -w3  w2 \
       *       |             |
       *   w = |  w3  0  -w1 |
       *       |             |
       *       \ -w2  w1  0  /,
       * where
       *   w1 = (l32 - l23) / 2,
       *   w2 = (l13 - l31) / 2,
       *   w3 = (l21 - l12) / 2.
       *
       * If w is nonzero, then spinning axis t is given by
       *   t = a[w1  w2  w3]^T,
       * where
       *   a = 1 / sqrt(w1^2 + w2^2 + w3^2).
       *
       * Otherwise, return an invalid vector.
       */
      Tensor<1, 3>
      rotation_axis(const Tensor<2, 3> &L)
      {
        Tensor<1, 3> w;
        w[0] = (L[2][1] - L[1][2]) * 0.5;
        w[1] = (L[0][2] - L[2][0]) * 0.5;
        w[2] = (L[1][0] - L[0][1]) * 0.5;

        const double norm = w.norm();
        Tensor<1, 3> t = numbers::signaling_nan<Tensor<1, 3>>();
        if (norm > std::numeric_limits<double>::epsilon())
          t = w / w.norm();
        
        return t;
      }
      


      template <int dim>
      Tensor<1, dim>
      crack_surface_normal(const SymmetricTensor<2, dim> &stress,
                           const Tensor<2, dim> &velocity_gradient,
                           const double friction_angle)
      {
        const std::array<std::pair<double, Tensor<1, dim>>, dim> eigenvalues_and_vectors = eigenvectors(stress);

        // Compute the directions of the major principal stress (a1) and the 
        // intermediate principal stress (a2) in compression
        Tensor<1, 3> a1, a2;
        for (unsigned int d = 0; d < dim; ++d)
          a1[d] = eigenvalues_and_vectors[dim - 1].second[d];
        // In 2D, the intermediate principal stress is assumed to be perpendicular
        // to the plane
        if constexpr (dim == 2)
          a2[2] = 1.;
        else
          for (unsigned int d = 0; d < dim; ++d)
            a2[d] = eigenvalues_and_vectors[1].second[d];

        a1 /= a1.norm();
        a2 /= a2.norm();

        // There are two potential fault planes. Pick the one consistent with the vorticity
        double sign = 1.;
        const Tensor<1, 3> w = rotation_axis(velocity_gradient);
        if (numbers::is_finite(w[0]))
          sign = (w * a2 > 0 ? 1. : -1.);

        const Tensor<1, 3> n_3d = a1 * std::sin(friction_angle) + cross_product_3d(a2, a1) * (sign * std::cos(friction_angle));

        Tensor<1, dim> n;
        for (unsigned int d = 0; d < dim; ++d)
          n[d] = n_3d[d];

        return n / n.norm();
      }
    }



    template <int dim>
    void 
    PhaseFieldRSF<dim>::
    update_history_states(const SolverControl &nonlinear_solver_control)
    {
      // This function is connected to signal 'post_nonlinear_solver'. If 
      // the nonlinear solver failed to converge and the failure strategy 
      // is not 'continue_with_next_timestep', then do not update the 
      // history states.
      if (nonlinear_solver_control.last_check() == SolverControl::failure 
          &&
          this->get_parameters().nonlinear_solver_failure_strategy != 
            Parameters<dim>::NonlinearSolverFailureStrategy::continue_with_next_timestep)
        return;

      const Introspection<dim> &introspection = this->introspection();
      const PhaseFieldHandler<dim> &phase_field_handler = this->get_phase_field_handler();
      Particles::ParticleHandler<dim> &particle_handler = const_cast<Particles::ParticleHandler<dim>&>(
        phase_field_handler.get_associated_particle_manager().get_particle_handler());

      // Vector storing the chemical field values, which is required for computing volume fractions
      std::vector<double> chemical_field_values(introspection.n_chemical_composition_fields());

      // Solution values/gradients to be evaluated at particle locations:
      // velocity gradient;
      // phase-field value;
      // core phase-field value;
      // temperature value
      std::vector<EvaluationFlags::EvaluationFlags> evaluation_flags(introspection.n_components,
                                                                     EvaluationFlags::nothing);
      for (unsigned int d = 0; d < dim; ++d)
        evaluation_flags[d] = EvaluationFlags::gradients;
      evaluation_flags[index_cache.components.phase_field] = EvaluationFlags::values;
      evaluation_flags[index_cache.components.core_phase_field] = EvaluationFlags::values;
      evaluation_flags[introspection.component_indices.temperature] = EvaluationFlags::values;

      const double dt = (this->get_timestep_number() > 0 ? this->get_timestep() : 0);

      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            const auto particles_in_cell = particle_handler.particles_in_cell(cell);
            const unsigned int n_particles_in_cell = particle_handler.n_particles_in_cell(cell);
            AssertThrow(n_particles_in_cell > 0, ExcInternalError());

            // Collect the particle locations
            small_vector<Point<dim>> reference_locations;
            for (auto particle = particles_in_cell.begin(); particle != particles_in_cell.end(); ++particle)
              reference_locations.push_back(particle->get_reference_location());

            // Collect the DoF values
            small_vector<double, 500> dof_values(this->get_fe().dofs_per_cell);
            cell->get_dof_values(this->get_solution(), dof_values.begin(), dof_values.end());

            // Update the solution_evaluator and evaluate the solution values and gradients
            // at particle locations
            solution_evaluator->reinit(cell, {reference_locations.data(), reference_locations.size()});
            solution_evaluator->evaluate({dof_values.data(), dof_values.size()}, evaluation_flags);

            small_vector<double>         particle_solution_values(introspection.n_components);
            small_vector<Tensor<1, dim>> particle_solution_gradients(introspection.n_components);

            // For each particle, update all the particle properties held by Particle::Property::PhaseFieldRSF
            for (auto particle = particles_in_cell.begin(); particle != particles_in_cell.end(); ++particle)
              {
                const unsigned int p = std::distance(particles_in_cell.begin(), particle);
                const ArrayView<double> particle_properties = particle->get_properties();

                solution_evaluator->get_solution(p, {particle_solution_values.data(), particle_solution_values.size()}, evaluation_flags);
                solution_evaluator->get_gradients(p, {particle_solution_gradients.data(), particle_solution_gradients.size()}, evaluation_flags);

                // Compute the strain rate
                Tensor<2, dim> grad_u;
                for (unsigned int d = 0; d < dim; ++d)
                  grad_u[d] = particle_solution_gradients[d];
                const SymmetricTensor<2, dim> epsilon = symmetrize(grad_u);

                // Compute the volume fractions
                for (unsigned int j = 0; j < chemical_field_values.size(); ++j)
                  chemical_field_values[j] = particle_properties[index_cache.particle_properties.chemical_fields[j]];
                const std::vector<double> volume_fractions = MaterialUtilities::compute_composition_fractions(chemical_field_values);

                // Compute the viscoelastic viscosity
                const double T = particle_solution_values[introspection.component_indices.temperature];
                const double eta = calculate_creep_viscosity(T, volume_fractions);
                const double G   = MaterialUtilities::average_value(volume_fractions, elastic_shear_moduli, viscosity_averaging);

                const double one_minus_beta = calculate_stress_relaxation_factor(eta, G);
                const double beta = 1. - one_minus_beta;
                const double eta_ve = eta * one_minus_beta;

                const SymmetricTensor<2, dim> tau_old = Utilities::Tensors::to_symmetric_tensor<dim>(
                  &particle_properties[index_cache.particle_properties.ve_stress],
                  &particle_properties[index_cache.particle_properties.ve_stress] + SymmetricTensor<2, dim>::n_independent_components);

                // Check if the material point is damaged
                const double phi = particle_solution_values[index_cache.components.phase_field];
                if (phi > phase_field_activation_threshold)
                  {
                    // The particle is damaged. Update the slip state
                    const double V         = particle_properties[index_cache.particle_properties.slip_rate];
                    const double theta_old = particle_properties[index_cache.particle_properties.slip_state];

                    const double theta = rsf_rheology.slip_state(V, theta_old, dt);
                    particle_properties[index_cache.particle_properties.slip_state] = theta;

                    // Update the viscoelastic stress
                    const double phi_hat = particle_solution_values[index_cache.components.core_phase_field];
                    const double g   = phase_field_handler.energetic_degradation(volume_fractions, phi);
                    const double chi = phase_field_handler.slip_rate_localization_factor(volume_fractions, g, phi_hat);

                    const Tensor<1, dim> n(ArrayView<const double>(&particle_properties[index_cache.particle_properties.normal_direction], dim));
                    const Tensor<1, dim> s(ArrayView<const double>(&particle_properties[index_cache.particle_properties.slip_direction], dim));
                    const SymmetricTensor<2, dim> S = symmetrize(outer_product(n, s));

                    const SymmetricTensor<2, dim> epsilon_ve = epsilon - (chi * V) * S;
                    const SymmetricTensor<2, dim> tau = 2. * eta_ve * epsilon_ve + beta * tau_old;

                    Utilities::Tensors::unroll_symmetric_tensor_into_array<dim>(
                      tau,
                      &particle_properties[index_cache.particle_properties.ve_stress],
                      &particle_properties[index_cache.particle_properties.ve_stress] + SymmetricTensor<2, dim>::n_independent_components);

                    // Update the cohesive force
                    const double h_tau_coh_old = particle_properties[index_cache.particle_properties.cohesive_force];
                    const double v = chi * V;
                    const double h_tau_coh = eta_ve * v + beta * h_tau_coh_old;
                    particle_properties[index_cache.particle_properties.cohesive_force] = h_tau_coh;

                    // Update the slip direction
                    Tensor<1, dim> t = tau * n;
                    Tensor<1, dim> s_new = t - (t * n) * n;
                    const double s_norm = s_new.norm();
                    if (s_norm / t.norm() >= std::numeric_limits<double>::epsilon())
                      {
                        s_new /= s_norm;
                        for (unsigned int d = 0; d < dim; ++d)
                          particle_properties[index_cache.particle_properties.slip_direction + d] = s_new[d];
                      }

                    // Update the crack driving force if the phase-field is to be evolved
                    if (evolve_phase_field)
                      {
                        const double H_old = particle_properties[index_cache.particle_properties.crack_driving_force];
                        const double H_new = (h_tau_coh * h_tau_coh - h_tau_coh_old * h_tau_coh_old) / (2. * G * (1. - g) * (1. - g));
                        particle_properties[index_cache.particle_properties.crack_driving_force] = std::max(H_old, H_new);
                      }

                    // Update the friction coefficient if requested
                    if (index_cache.particle_properties.friction_coefficient != numbers::invalid_unsigned_int)
                      particle_properties[index_cache.particle_properties.friction_coefficient] = rsf_rheology.friction_coefficient(volume_fractions, V, theta);

                    // Update the slip increment if requested
                    if (index_cache.particle_properties.slip_increment != numbers::invalid_unsigned_int)
                      particle_properties[index_cache.particle_properties.slip_increment] += V * dt;
                  }
                else
                  {
                    // The particle is intact. Update the viscoelastic stress
                    const SymmetricTensor<2, dim> tau = 2. * eta_ve * epsilon + beta * tau_old;
                    Utilities::Tensors::unroll_symmetric_tensor_into_array<dim>(
                      tau,
                      &particle_properties[index_cache.particle_properties.ve_stress],
                      &particle_properties[index_cache.particle_properties.ve_stress] + SymmetricTensor<2, dim>::n_independent_components);

                    // Check if the particle reaches the material strength
                    const double mu = MaterialUtilities::average_value(volume_fractions, initial_friction_coefficients, MaterialUtilities::arithmetic);
                    const double c  = MaterialUtilities::average_value(volume_fractions, cohesions, MaterialUtilities::arithmetic);

                    Tensor<1, dim> n;
                    if (evolve_phase_field)
                      {
                        const double coulomb_angle = (numbers::PI - 2. * std::atan(mu)) * 0.25;
                        n = crack_surface_normal(tau, grad_u, coulomb_angle);
                      }
                    else
                      {
                        for (unsigned int d = 0; d < dim; ++d)
                          n[d] = particle_properties[index_cache.particle_properties.normal_direction + d];
                      }

                    const double sigma_n = this->get_adiabatic_conditions().pressure(particle->get_location());

                    const Tensor<1, dim> t = tau * n;
                    const Tensor<1, dim> t_s = t - (t * n) * n;
                    const double tau_diff = t_s.norm() - mu * sigma_n;
                    if (tau_diff > c)
                      {
                        // The particle is at the turning point. Update the crack driving force if 
                        // the phase-field is to be evolved
                        if (evolve_phase_field)
                          particle_properties[index_cache.particle_properties.crack_driving_force] = (tau_diff * tau_diff) / (2. * G);
                      }
                  }
              }
          }

      // Check if the maximum slip increment reaches the threshold for graphical output
      if (maximum_slip_increment_between_outputs == 0)
        return;

      double local_max_slip_increment = 0;

      for (const auto &cell : this->get_triangulation().active_cell_iterators())
        if (cell->is_locally_owned())
          for (const auto &particle : particle_handler.particles_in_cell(cell))
            {
              const double slip_increment = particle.get_properties()[index_cache.particle_properties.slip_increment];
              if (!numbers::is_nan(slip_increment))
                local_max_slip_increment = std::max(local_max_slip_increment, slip_increment);
            }

      const double max_slip_increment = Utilities::MPI::max(local_max_slip_increment, this->get_mpi_communicator());
      if (max_slip_increment >= maximum_slip_increment_between_outputs)
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
                  if (!numbers::is_nan(particle_properties[index_cache.particle_properties.slip_increment]))
                    particle_properties[index_cache.particle_properties.slip_increment] = 0;
                }
        }
    }



    template <int dim>
    void PhaseFieldRSF<dim>::update_particles_in_emerging_crack_zone()
    {
      const Introspection<dim> &introspection = this->introspection();
      const PhaseFieldHandler<dim> &phase_field_handler = this->get_phase_field_handler();
      Particles::ParticleHandler<dim> &particle_handler = const_cast<Particles::ParticleHandler<dim>&>(
        phase_field_handler.get_associated_particle_manager().get_particle_handler());

      // Vector storing the chemical field values, which is required for computing volume fractions
      std::vector<double> chemical_field_values(introspection.n_chemical_composition_fields());

      // We need the velocity gradients, phase-field values and core phase-field values at particle locations
      std::vector<EvaluationFlags::EvaluationFlags> evaluation_flags(introspection.n_components,
                                                                     EvaluationFlags::nothing);
      for (unsigned int d = 0; d < dim; ++d)
        evaluation_flags[d] = EvaluationFlags::gradients;
      evaluation_flags[index_cache.components.phase_field] = EvaluationFlags::values;
      evaluation_flags[index_cache.components.core_phase_field] = EvaluationFlags::values;

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

            // Update the solution_evaluator and evaluate the solution values and gradients
            // at particle locations
            solution_evaluator->reinit(cell, {reference_locations.data(), reference_locations.size()});
            solution_evaluator->evaluate({dof_values.data(), dof_values.size()}, evaluation_flags);

            small_vector<double>         particle_solution_values(introspection.n_components);
            small_vector<Tensor<1, dim>> particle_solution_gradients(introspection.n_components);

            for (auto particle = particles_in_cell.begin(); particle != particles_in_cell.end(); ++particle)
              {
                const unsigned int p = std::distance(particles_in_cell.begin(), particle);
                const ArrayView<double> particle_properties = particle->get_properties();

                solution_evaluator->get_solution(p, {particle_solution_values.data(), particle_solution_values.size()}, evaluation_flags);
                solution_evaluator->get_gradients(p, {particle_solution_gradients.data(), particle_solution_gradients.size()}, evaluation_flags);

                // Compute the strain rate
                Tensor<2, dim> grad_u;
                for (unsigned int d = 0; d < dim; ++d)
                  grad_u[d] = particle_solution_gradients[d];

                // Compute the volume fractions
                for (unsigned int j = 0; j < chemical_field_values.size(); ++j)
                  chemical_field_values[j] = particle_properties[index_cache.particle_properties.chemical_fields[j]];
                const std::vector<double> volume_fractions = MaterialUtilities::compute_composition_fractions(chemical_field_values);

                // The viscoelastic stress has been updated at the end of the previous time step
                const SymmetricTensor<2, dim> tau = Utilities::Tensors::to_symmetric_tensor<dim>(
                  &particle_properties[index_cache.particle_properties.ve_stress],
                  &particle_properties[index_cache.particle_properties.ve_stress] + SymmetricTensor<2, dim>::n_independent_components);

                // Check if the material point is damaged
                const double phi = particle_solution_values[index_cache.components.phase_field];
                if (phi > phase_field_activation_threshold)
                  {
                    // The particle is damaged. Check if the direction vectors are valid
                    if (numbers::is_finite(particle_properties[index_cache.particle_properties.normal_direction]))
                      {
                        // The direction vectors are valid. Check if the direction vectors need update
                        if (evolve_phase_field && phi < phase_field_normal_lock_threshold)
                          {
                            // Update the direction vectors
                            Tensor<1, dim> n_old;
                            for (unsigned int d = 0; d < dim; ++d)
                              n_old[d] = particle_properties[index_cache.particle_properties.normal_direction + d];

                            // The slip state has been updated at the end of the previous time step
                            const double theta = particle_properties[index_cache.particle_properties.slip_state];
                            const double V     = particle_properties[index_cache.particle_properties.slip_rate];

                            const double mu = rsf_rheology.friction_coefficient(volume_fractions, V, theta);
                            const double angle = (numbers::PI - 2. * std::atan(mu)) * 0.25;
                            const Tensor<1, dim> n_new = crack_surface_normal(tau, grad_u, angle);

                            // For robustness, average the projector (structure tensor) instead of vector
                            const SymmetricTensor<2, dim> projector_old = symmetrize(outer_product(n_old, n_old));
                            const SymmetricTensor<2, dim> projector_new = symmetrize(outer_product(n_new, n_new));
                            const double fraction = (phi - phase_field_activation_threshold) / (phase_field_normal_lock_threshold - phase_field_activation_threshold);
                            const SymmetricTensor<2, dim> projector_avg = fraction * projector_old + (1. - fraction) * projector_new;

                            const std::array<std::pair<double, Tensor<1, dim>>, dim> eigenvalues_and_vectors = eigenvectors(projector_avg);
                            Tensor<1, dim> n = eigenvalues_and_vectors[0].second;
                            n /= n.norm();

                            const Tensor<1, dim> t = tau * n;
                            Tensor<1, dim> s = t - (t * n) * n;
                            s /= s.norm();

                            for (unsigned int d = 0; d < dim; ++d)
                              {
                                particle_properties[index_cache.particle_properties.normal_direction + d] = n[d];
                                particle_properties[index_cache.particle_properties.slip_direction + d] = s[d];
                              }
                          }
                      }
                    else
                      {
                        // The direction vectors are invalid. This means that the particle was intact in the
                        // previous time step.
                        AssertThrow(this->get_timestep_number() > 0, ExcInternalError());

                        // Initialize the direction vectors
                        const double mu = MaterialUtilities::average_value(volume_fractions, initial_friction_coefficients, MaterialUtilities::arithmetic);
                        const double angle = (numbers::PI - 2. * std::atan(mu)) * 0.25;
                        const Tensor<1, dim> n = crack_surface_normal(tau, grad_u, angle);

                        const Tensor<1, dim> t = tau * n;
                        const Tensor<1, dim> t_s = t - (t * n) * n;
                        const Tensor<1, dim> s = t_s / t_s.norm();

                        for (unsigned int d = 0; d < dim; ++d)
                          {
                            particle_properties[index_cache.particle_properties.normal_direction + d] = n[d];
                            particle_properties[index_cache.particle_properties.slip_direction + d]   = s[d];
                          }

                        // Initialize the slip increment if requested
                        if (index_cache.particle_properties.slip_increment != numbers::invalid_unsigned_int)
                          particle_properties[index_cache.particle_properties.slip_increment] = 0;
                      }
                  }
                else
                  {
                    // The particle is intact. If it is the first time step, set the slip rate, direction vectors,
                    // friction coefficient and slip increment to NaN
                    if (this->get_timestep_number() == 0)
                      {
                        particle_properties[index_cache.particle_properties.slip_rate] = std::numeric_limits<double>::quiet_NaN();

                        if (evolve_phase_field)
                          for (unsigned int d = 0; d < dim; ++d)
                            {
                              particle_properties[index_cache.particle_properties.normal_direction + d] = std::numeric_limits<double>::quiet_NaN();
                              particle_properties[index_cache.particle_properties.slip_direction + d]   = std::numeric_limits<double>::quiet_NaN();
                            }

                        if (index_cache.particle_properties.friction_coefficient != numbers::invalid_unsigned_int)
                          particle_properties[index_cache.particle_properties.friction_coefficient] = std::numeric_limits<double>::quiet_NaN();

                        if (index_cache.particle_properties.slip_increment != numbers::invalid_unsigned_int)
                          particle_properties[index_cache.particle_properties.slip_increment] = std::numeric_limits<double>::quiet_NaN();
                      }
                  }
              }
          }
    }



    template <int dim>
    bool PhaseFieldRSF<dim>::is_compressible() const
    {
      return equation_of_state.is_compressible();
    }



    template <int dim>
    std::vector<double>
    PhaseFieldRSF<dim>::get_critical_crack_driving_forces() const
    {
      const unsigned int n_comp = elastic_shear_moduli.size();
      std::vector<double> critical_crack_driving_forces(n_comp);
      for (unsigned int j = 0; j < n_comp; ++j)
        critical_crack_driving_forces[j] = cohesions[j] * cohesions[j] / (2.0 * elastic_shear_moduli[j]);

      return critical_crack_driving_forces;
    }


    
    template <int dim>
    std::vector<double>
    PhaseFieldRSF<dim>::get_critical_energy_release_rates() const
    {
      return critical_energy_release_rates;
    }



    template <int dim>
    std::pair<double, double>
    PhaseFieldRSF<dim>::get_phase_field_range() const
    {
      return std::make_pair(phase_field_activation_threshold, 0.99);
    }



    template <int dim>
    double
    PhaseFieldRSF<dim>::
    calculate_creep_viscosity(const double               temperature,
                              const std::vector<double> &volume_fractions) const
    {
      const unsigned int n_compositions = volume_fractions.size();
      AssertDimension(n_compositions, reference_viscosities.size());

      const double dT_over_Tref = (temperature - reference_temperature) / reference_temperature;
      std::vector<double> composition_viscosities(n_compositions);
      for (unsigned int j = 0; j < n_compositions; ++j)
        composition_viscosities[j] = std::max(minimum_viscosity,
                                              std::min(maximum_viscosity,
                                                       reference_viscosities[j] * std::exp(-thermal_viscosity_exponents[j] * dT_over_Tref)));

      return MaterialUtilities::average_value(volume_fractions, composition_viscosities, viscosity_averaging);
    }



    template <int dim>
    double
    PhaseFieldRSF<dim>::
    calculate_stress_relaxation_factor(const double creep_viscosity,
                                       const double shear_modulus) const
    {
      const double time_step = (this->get_timestep_number() > 0 ? this->get_timestep() : initial_time_step);
      return -std::expm1(-time_step * shear_modulus / creep_viscosity);
    }



    template <int dim>
    const Rheology::RateStateFriction<dim> &
    PhaseFieldRSF<dim>::get_rate_state_friction_model() const
    {
      return rsf_rheology;
    }



    template <int dim>
    const typename PhaseFieldRSF<dim>::IndexCache &
    PhaseFieldRSF<dim>::get_index_cache() const
    {
      return index_cache;
    }



    template <int dim>
    bool
    PhaseFieldRSF<dim>::is_intact(const double phase_field) const
    {
      return phase_field <= phase_field_activation_threshold;
    }



    template <int dim>
    void
    PhaseFieldRSF<dim>::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Phase field RSF");
        {
          EquationOfState::MulticomponentIncompressible<dim>::declare_parameters(prm);
          Rheology::RateStateFriction<dim>::declare_parameters(prm);

          // Equation of state parameters
          prm.declare_entry("Thermal conductivities", "3.0",
                            Patterns::List(Patterns::Double(0)),
                            "List of thermal conductivities, for background material and compositional fields, "
                            "for a total of N+1 values, where N is the number of all compositional fields or only "
                            "those corresponding to chemical compositions. "
                            "If only one value is given, then all use the same value. "
                            "Units: \\si{\\watt\\per\\meter\\per\\kelvin}.");

          // Reference and minimum/maximum values
          prm.declare_entry("Reference temperature", "293",
                            Patterns::Double(0),
                            "The reference temperature $T_0$ in the power-law viscosity formula. "
                            "Units: \\si{\\kelvin}.");

          prm.declare_entry("Maximum viscosity", "1.e25",
                            Patterns::Double(0),
                            "Upper cutoff for the power-law viscosity. Units: \\si{\\pascal\\second}.");

          prm.declare_entry("Minimum viscosity", "1.e17",
                            Patterns::Double(0),
                            "Lower cutoff for the power-law viscosity. Units: \\si{\\pascal\\second}.");

          prm.declare_entry("Viscosity averaging scheme", "harmonic",
                            Patterns::Selection("arithmetic|harmonic|geometric|maximum composition"),
                            "When more than one compositional field is present at a point "
                            "with different viscosities, we need to come up with an average "
                            "viscosity at that point. Select a weighted harmonic, arithmetic, "
                            "geometric, or maximum composition.");

          prm.declare_entry("Phase field activation threshold", "0.1",
                            Patterns::Double(0, 1),
                            "Value of the phase-field damage variable above which frictional slip and "
                            "rate-and-state fault physics become active. Material points with damage "
                            "below this threshold are treated as intact and the fault friction law is "
                            "not applied. This parameter is used to avoid numerical noise when the "
                            "phase-field variable is small and the fracture is not yet fully developed. "
                            "The value of this parameter should be between 0 and 1.");

          prm.declare_entry("Initial time step", "1.",
                            Patterns::Double(0),
                            "The initial time step size. It is used for evolving the stress at the "
                            "zeroth time step. Note that if an initial distribution of slip rate is "
                            "provided, then it will be assumed that the modeling starts with steady "
                            "slip state, in which case it is recommended to set the initial time step "
                            "to a very large value to be consistent with the slip state. "
                            "Otherwise, it would be easier for the local return-mapping to fail. "
                            " Units: years if the 'Use years instead of seconds' "
                            "parameter is set; seconds otherwise.");

          // Rheological parameters
          prm.declare_entry("Reference viscosities", "1.e24",
                            Patterns::List(Patterns::Double(0)),
                            "List of the reference viscosity, $\\eta_0$, "
                            "for background material and compositional fields, "
                            "for a total of N+1 values, where N is the number of all compositional fields or only "
                            "those corresponding to chemical compositions. "
                            "If only one value is given, then all use the same value. "
                            "Units: \\si{\\pascal}.");

          prm.declare_entry("Thermal viscosity exponents", "0.0",
                            Patterns::List(Patterns::Double(0)),
                            "List of the temperature dependences of viscosity, $\\beta$, "
                            "for background material and compositional fields, "
                            "for a total of N+1 values, where N is the number of all compositional fields or only "
                            "those corresponding to chemical compositions. "
                            "If only one value is given, then all use the same value. "
                            "Units: none.");

          prm.declare_entry("Elastic shear moduli", "1e10",
                            Patterns::List(Patterns::Double(0)),
                            "List of elastic shear moduli, $G$, "
                            "for background material and compositional fields, "
                            "for a total of N+1 values, where N is the number of all compositional fields or only "
                            "those corresponding to chemical compositions. "
                            "If only one value is given, then all use the same value. "
                            "Units: \\si{\\pascal}.");

          prm.declare_entry("Cohesions", "1.e7",
                            Patterns::List(Patterns::Double(0)),
                            "List of cohesions, $C$, for background material and compositional fields, "
                            "for a total of N+1 values, where N is the number of all compositional fields or only "
                            "those corresponding to chemical compositions. Units: \\si{\\pascal}.");

          prm.declare_entry("Initial friction coefficients", "0.6",
                            Patterns::List(Patterns::Double(0)),
                            "List of the initial friction coefficients, $\\mu_{\\text{init}}$, "
                            "for background material and compositional fields, "
                            "for a total of N+1 values, where N is the number of all compositional fields or only "
                            "those corresponding to chemical compositions. "
                            "If only one value is given, then all use the same value. "
                            "Units: none.");

          prm.declare_entry("Critical energy release rates", "1.e5",
                            Patterns::List(Patterns::Double(0)),
                            "List of the critical energy release rates, $G_c$, "
                            "for background material and compositional fields, "
                            "for a total of N+1 values, where N is the number of all compositional fields or only "
                            "those corresponding to chemical compositions. "
                            "If only one value is given, then all use the same value. "
                            "Units: \\si{\\joule\\per\\square\\meter}.");

          prm.declare_entry("Radiation damping coefficients", "",
                            Patterns::List(Patterns::Double(0)),
                            "List of the rediation damping coefficients, $\\eta^d$, "
                            "for background material and compositional fields, "
                            "for a total of N+1 values, where N is the number of all compositional fields or only "
                            "those corresponding to chemical compositions. "
                            "If only one value is given, then all use the same value. "
                            "Units: \\si{\\pascal\\second\\per\\meter}.");

          prm.declare_entry("Phase field normal lock threshold", "0.5",
                            Patterns::Double(0, 1),
                            "Value of the phase-field damage variable above which the fault normal "
                            "vector is considered fully developed and its orientation is frozen. "
                            "Below this threshold the fault normal may still evolve according to the "
                            "local stress state, while above this value the stored normal direction "
                            "is used to define the slip plane. This parameter helps stabilize the "
                            "fault geometry once the fracture is sufficiently developed. The value "
                            "should be between 0 and 1.");

          prm.declare_entry("Evolve phase field", "true",
                            Patterns::Bool(),
                            "Whether to evolve the phase field during the simulation. If set to "
                            "false, then the crack driving force and the direction vectors will be "
                            "frozen after initialization. This is useful when conducting benchmarks "
                            "with pre-existing faults.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    template <int dim>
    void
    PhaseFieldRSF<dim>::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Phase field RSF");
        {
          // Equation of state parameters
          equation_of_state.initialize_simulator(this->get_simulator());
          equation_of_state.parse_parameters(prm);

          // RSF parameters
          rsf_rheology.initialize_simulator(this->get_simulator());
          rsf_rheology.parse_parameters(prm);

          // Reference and minimum/maximum values
          reference_temperature = prm.get_double("Reference temperature");
          maximum_viscosity     = prm.get_double("Maximum viscosity");
          minimum_viscosity     = prm.get_double("Minimum viscosity");

          viscosity_averaging = MaterialUtilities::parse_compositional_averaging_operation("Viscosity averaging scheme", prm);

          phase_field_activation_threshold  = prm.get_double("Phase field activation threshold");
          phase_field_normal_lock_threshold = prm.get_double("Phase field normal lock threshold");
          AssertThrow(phase_field_activation_threshold <= phase_field_normal_lock_threshold,
                      ExcMessage("The phase field normal lock threshold must be greater than or equal to "
                                 "the phase field activation threshold."));

          initial_time_step = prm.get_double("Initial time step");

          evolve_phase_field = prm.get_bool("Evolve phase field");

          // Make options file for parsing maps to double arrays
          std::vector<std::string> compositional_field_names = this->introspection().get_composition_names();
          compositional_field_names.insert(compositional_field_names.begin(), "background");

          std::vector<std::string> chemical_field_names = this->introspection().chemical_composition_field_names();
          chemical_field_names.insert(chemical_field_names.begin(), "background");

          Utilities::MapParsing::Options options(chemical_field_names, "Thermal conductivities");
          options.list_of_allowed_keys = compositional_field_names;

          thermal_conductivities = Utilities::MapParsing::parse_map_to_double_array(prm.get("Thermal conductivities"), options);

          options.property_name = "Reference viscosities";
          reference_viscosities = Utilities::MapParsing::parse_map_to_double_array(prm.get("Reference viscosities"), options);

          options.property_name = "Thermal viscosity exponents";
          thermal_viscosity_exponents = Utilities::MapParsing::parse_map_to_double_array(prm.get("Thermal viscosity exponents"), options);

          options.property_name = "Elastic shear moduli";
          elastic_shear_moduli = Utilities::MapParsing::parse_map_to_double_array(prm.get("Elastic shear moduli"), options);

          options.property_name = "Cohesions";
          cohesions = Utilities::MapParsing::parse_map_to_double_array(prm.get("Cohesions"), options);

          options.property_name = "Initial friction coefficients";
          initial_friction_coefficients = Utilities::MapParsing::parse_map_to_double_array(prm.get("Initial friction coefficients"), options);

          options.property_name = "Critical energy release rates";
          critical_energy_release_rates = Utilities::MapParsing::parse_map_to_double_array(prm.get("Critical energy release rates"), options);

          options.property_name = "Radiation damping coefficients";
          radiation_damping_coefficients = Utilities::MapParsing::parse_map_to_double_array(prm.get("Radiation damping coefficients"), options);
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      // Get the slip increment threshold for graphical output from the particle subsection
      prm.enter_subsection("Particles");
      {
        prm.enter_subsection("Phase field RSF");
        {
          maximum_slip_increment_between_outputs = prm.get_double("Slip distance between graphical output");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }
  }
}

// explicit instantiation
namespace aspect
{
namespace MaterialModel
  {
    ASPECT_REGISTER_MATERIAL_MODEL(PhaseFieldRSF,
                                   "phase field rsf",
                                   "")
  }
}
