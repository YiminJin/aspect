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
#include <aspect/particle/particle_domain.h>
#include <aspect/newton.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_cartesian.h>

namespace aspect
{
  namespace MaterialModel
  {
    template <int dim>
    void PhaseFieldRSF<dim>::initialize()
    {
      AssertThrow(this->get_parameters().enable_phase_field == true,
                  ExcMessage("The phase field PSF model requires phase field to be included in "
                             "the system formulation. Please set <Formulation/Include phase field> to true."));

      AssertThrow(this->get_parameters().enable_elasticity == false,
                  ExcMessage("The phase field RSF model assumes viscoelastic rheology for intact regions, but "
                             "the elasticity is not handled by MaterialModel::Rheology::Elasticity. Please set "
                             "<Formulation/Enable elasticity> to false."));

      phase_field_component_index = this->introspection().component_indices.compositional_fields[
        this->introspection().compositional_index_for_name("phase_field")];

      // Initialize the solution evaluator and evaluation flags
      evaluator = construct_solution_evaluator(*this, update_values | update_gradients);

      // We need the velocity gradients, temperature values, phase field values and phase field gradients at particle locations
      evaluation_flags.resize(this->introspection().n_components, EvaluationFlags::nothing);
      for (unsigned int d = 0; d < dim; ++d)
        evaluation_flags[d] = EvaluationFlags::gradients;
      evaluation_flags[this->introspection().component_indices.temperature] = EvaluationFlags::values;
      evaluation_flags[phase_field_component_index] = EvaluationFlags::values | EvaluationFlags::gradients;

      // Initialize the particle data information when it is sure that the phase field handler is initialized
      this->get_signals().post_simulator_initialization.connect(
        [&](const SimulatorAccess<dim> &)
      {
        this->initialize_particle_data_info();
      });

      // Perform return mapping before assembling the Stokes system
      this->get_signals().pre_assemble_stokes_system.connect(
        [&](const SimulatorAccess<dim> &)
      {
        this->perform_return_mapping();
      });

      // Update the history states (slip state, fault normal and stress) after the
      // nonlinear iterations
      this->get_signals().post_nonlinear_solver.connect(
        [&](const SolverControl &)
      {
        this->update_history_states();
      });
    }



    template <int dim>
    void
    PhaseFieldRSF<dim>::initialize_particle_data_info()
    {
      const Particle::Manager<dim> &particle_manager = this->get_phase_field_handler().get_associated_particle_manager();
      const auto &particle_data_info = particle_manager.get_property_manager().get_data_info();

      particle_data_positions.crack_driving_force = particle_data_info.get_position_by_field_name("crack_driving_force");
      particle_data_positions.slip_rate           = particle_data_info.get_position_by_field_name("slip_rate");
      particle_data_positions.slip_state          = particle_data_info.get_position_by_field_name("slip_state");
      particle_data_positions.normal_direction    = particle_data_info.get_position_by_field_name("normal_direction");
      particle_data_positions.slip_direction      = particle_data_info.get_position_by_field_name("slip_direction");
      particle_data_positions.bulk_stress         = particle_data_info.get_position_by_field_name("bulk_stress");
      particle_data_positions.interface_stress    = particle_data_info.get_position_by_field_name("interface_stress");

      particle_data_positions.chemical_fields.clear();
      for (const unsigned int index : this->introspection().chemical_composition_field_indices())
        particle_data_positions.chemical_fields.push_back(
          particle_data_info.get_position_by_field_name(this->get_parameters().mapped_particle_properties.find(index)->second.first));
    }



    template <int dim>
    void
    PhaseFieldRSF<dim>::
    evaluate(const MaterialModel::MaterialModelInputs<dim> &in,
             MaterialModel::MaterialModelOutputs<dim> &out) const
    {
      EquationOfStateOutputs<dim> eos_outputs(this->introspection().n_chemical_composition_fields() + 1);

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
              out.viscosities[i] = calculate_viscoelastic_viscosity(eta, G);
            }
        }

      const std::shared_ptr<MaterialModel::ImplicitConstitutiveOutputs<dim>> implicit_constitutive_outputs
        = out.template get_additional_output_object<MaterialModel::ImplicitConstitutiveOutputs<dim>>();

      // If the implicit constitutive outputs is requested, then we map the current states
      // from particles to quadrature points with MLS method
      if (implicit_constitutive_outputs != nullptr)
        {
          const bool require_equivalent_viscosity = (implicit_constitutive_outputs->equivalent_viscosities.size() > 0);

          const Particles::ParticleHandler<dim> &particle_handler
            = this->get_phase_field_handler().get_associated_particle_manager().get_particle_handler();
          const unsigned int n_particles_in_cell = particle_handler.n_particles_in_cell(in.current_cell);
          const auto particles_in_cell = particle_handler.particles_in_cell(in.current_cell);

          // Collect the particle locations
          small_vector<Point<dim>> reference_locations;
          for (auto particle = particles_in_cell.begin(); particle != particles_in_cell.end(); ++particle)
            reference_locations.push_back(particle->get_reference_location());

          // Collect the DoF values
          small_vector<double, 500> dof_values(this->get_fe().dofs_per_cell);
          in.current_cell->get_dof_values(this->get_solution(), dof_values.begin(), dof_values.end());

          // Update the solution evaluator
          evaluator->reinit(in.current_cell, {reference_locations.data(), reference_locations.size()});
          evaluator->evaluate({dof_values.data(), dof_values.size()}, evaluation_flags);

          small_vector<double>         particle_solution_values(this->introspection().n_components);
          small_vector<Tensor<1, dim>> particle_solution_gradients(this->introspection().n_components);

          // Vector storing the chemical field values, which is required for computing volume fractions
          std::vector<double> chemical_field_values(this->introspection().n_chemical_composition_fields());

          // Compute the current stress, the linearized stress and the equivalent viscosity on particles
          small_vector<SymmetricTensor<2, dim>> particle_stresses(n_particles_in_cell);
          small_vector<SymmetricTensor<2, dim>> particle_slip_directions(n_particles_in_cell);
          small_vector<double> particle_softening_factors(n_particles_in_cell);
          small_vector<double> particle_bulk_viscosities(n_particles_in_cell);
          small_vector<double> particle_equivalent_viscosities(n_particles_in_cell);

          for (auto particle = particles_in_cell.begin(); particle != particles_in_cell.end(); ++particle)
            {
              const ArrayView<const double> particle_properties = particle->get_properties();
              const unsigned int p = std::distance(particles_in_cell.begin(), particle);

              // Get the solution values and gradients
              evaluator->get_solution(p, {particle_solution_values.data(), particle_solution_values.size()}, evaluation_flags);
              evaluator->get_gradients(p, {particle_solution_gradients.data(), particle_solution_gradients.size()}, evaluation_flags);

              // Compute the strain rate
              Tensor<2, dim> grad_u;
              for (unsigned int d = 0; d < dim; ++d)
                grad_u[d] = particle_solution_gradients[d];
              const SymmetricTensor<2, dim> epsilon = symmetrize(grad_u);

              // Compute the volume fractions
              for (unsigned int j = 0; j < chemical_field_values.size(); ++j)
                chemical_field_values[j] = particle_properties[particle_data_positions.chemical_fields[j]];
              const std::vector<double> volume_fractions = MaterialUtilities::compute_composition_fractions(chemical_field_values);

              // Compute the bulk viscosity
              const double T      = particle_solution_values[this->introspection().component_indices.temperature];
              const double eta    = calculate_creep_viscosity(T, volume_fractions);
              const double G      = MaterialUtilities::average_value(volume_fractions, elastic_shear_moduli, viscosity_averaging);
              const double eta_ve = calculate_viscoelastic_viscosity(eta, G);
              particle_bulk_viscosities[p] = eta_ve;

              // Compute the bulk stress
              const SymmetricTensor<2, dim> tau_b_old = Utilities::Tensors::to_symmetric_tensor<dim>(
                &particle_properties[particle_data_positions.bulk_stress],
                &particle_properties[particle_data_positions.bulk_stress] + SymmetricTensor<2, dim>::n_independent_components);
              const SymmetricTensor<2, dim> tau_b = calculate_deviatoric_stress(epsilon, tau_b_old, eta, G);

              // Check if the point is fractured
              const double pf = particle_solution_values[phase_field_component_index];
              if (pf > phase_field_kinetic_threshold)
                {
                  // Compute the slip direction tensor
                  const Tensor<1, dim> n(ArrayView<const double>(&particle_properties[particle_data_positions.normal_direction], dim));
                  const Tensor<1, dim> s(ArrayView<const double>(&particle_properties[particle_data_positions.slip_direction], dim));
                  particle_slip_directions[p] = symmetrize(outer_product(n, s));

                  // Compute the interface stress
                  const SymmetricTensor<2, dim> tau_f_old = Utilities::Tensors::to_symmetric_tensor<dim>(
                    &particle_properties[particle_data_positions.interface_stress],
                    &particle_properties[particle_data_positions.interface_stress] + SymmetricTensor<2, dim>::n_independent_components);
                  const Tensor<1, dim> &grad_pf = particle_solution_gradients[phase_field_component_index];
                  const double gamma = this->get_phase_field_handler().crack_surface_density(pf, grad_pf);
                  const double V  = particle_properties[particle_data_positions.slip_rate];
                  const SymmetricTensor<2, dim> epsilon_eff = epsilon - V * gamma * particle_slip_directions[p];
                  const SymmetricTensor<2, dim> tau_f = calculate_deviatoric_stress(epsilon_eff, tau_f_old, eta, G);

                  // Compute the full stress
                  const double g = this->get_phase_field_handler().energetic_degradation(pf, volume_fractions);
                  particle_stresses[p] = g * tau_b + (1. - g) * tau_f;

                  // Compute the softening factor
                  const double theta = particle_properties[particle_data_positions.slip_state];
                  const double dmu_over_dV = calculate_friction_coefficient_derivative(V, theta, volume_fractions);
                  const double p_bar = this->get_adiabatic_conditions().pressure(particle->get_location());
                  const double eta_d = MaterialUtilities::average_value(volume_fractions, radiation_damping_coefficients, MaterialUtilities::arithmetic);
                  particle_softening_factors[p] = 2. * (1. - g) * gamma * eta_ve / (gamma * eta_ve + dmu_over_dV * p_bar + eta_d);

                  // Check if we need to apply the PD stabilization
                  if ((require_equivalent_viscosity && this->get_newton_handler().parameters.preconditioner_stabilization & Newton::Parameters::PD) ||
                      (!require_equivalent_viscosity && this->get_newton_handler().parameters.velocity_block_stabilization & Newton::Parameters::PD))
                    particle_softening_factors[p] = std::min(2. * this->get_newton_handler().parameters.SPD_safety_factor,
                                                             particle_softening_factors[p]);

                  if (require_equivalent_viscosity)
                    {
                      constexpr unsigned int N = (dim == 2 ? 3 : 5);
                      particle_equivalent_viscosities[p] = eta_ve * (1. - particle_softening_factors[p] / (2. * N - (N - 1.) * particle_softening_factors[p]));
                    }
                }
              else // pf <= phase_field_kinetic_threshold
                {
                  particle_stresses[p]               = tau_b;
                  particle_slip_directions[p]        = numbers::signaling_nan<SymmetricTensor<2, dim>>();
                  particle_softening_factors[p]      = 0.;
                  particle_equivalent_viscosities[p] = eta_ve;
                }
            }

          // We need the symmetric gradients of the velocity shape functions at particle locations
          // when assembling the linearized stress terms. To avoid constructing and destructing a
          // FEValues object for each cell, we map the shape gradients from reference cell to real cell
          // with the inverse Jacobian calculated by function Mapping::fill_mapping_data_for_generic_points
          dealii::internal::FEValuesImplementation::MappingRelatedData<dim> mapping_data;
          const MappingCartesian<dim> &mapping = dynamic_cast<const MappingCartesian<dim>&>(this->get_mapping());
          mapping.fill_mapping_data_for_generic_points(in.current_cell,
                                                       {reference_locations.data(), reference_locations.size()},
                                                       update_inverse_jacobians,
                                                       mapping_data);

          // For Cartesian mapping, all the particles in one cell share the same Jacobian.
          const Tensor<2, dim> &J_inv = mapping_data.inverse_jacobians[0];

          // Prepare for MLS interpolation
          const unsigned int n_basis = dim + 1;
          AssertThrow(n_particles_in_cell >= n_basis,
                      ExcMessage("Material model 'phase field rsf' maps the particle state onto quadrature points "
                                 "with the moving least squares method, which requires the number of particles in "
                                 "each cell to be greater than or equal to dim + 1."));

          const double cell_diameter = in.current_cell->diameter();

          for (unsigned int i = 0; i < in.n_evaluation_points(); ++i)
            {
              // Compute the coefficients of the linear MLS approximation:
              // $\alpha_p = w_p\phi_q^T M^{-1}\phi_p$,
              // where
              // $\phi_p = [1, x_p - x_q, y_p - y_q, z_p - z_q]^T$,
              // $M = \sum_p w_p\phi_p\phi_p^T$,
              // $w_p = (1 - s)^4 (1 + 4s)$ (s is the distance between p and q)
              small_vector<double> w(n_particles_in_cell);
              small_vector<Vector<double>> phi(n_particles_in_cell, Vector<double>(n_basis));
              FullMatrix<double> M(n_basis, n_basis);

              for (auto particle = particles_in_cell.begin(); particle != particles_in_cell.end(); ++particle)
                {
                  const unsigned int p = std::distance(particles_in_cell.begin(), particle);
                  const Tensor<1, dim> reference_location = particle->get_location() - in.position[i];
                  
                  phi[p][0] = 1.;
                  for (unsigned int d = 0; d < dim; ++d)
                    phi[p][d + 1] = reference_location[d];

                  const double s = reference_location.norm() / cell_diameter;
                  w[p] = Utilities::fixed_power<4, double>(1. - s) * (1. + 4. * s);

                  FullMatrix<double> Mp(n_basis, n_basis);
                  Mp.outer_product(phi[p], phi[p]);
                  M.add(w[p], Mp);
                }

              M.gauss_jordan();

              // Finally, fill in the implicit constitutive outputs
              implicit_constitutive_outputs->deviatoric_stresses[i] = 0.;
              implicit_constitutive_outputs->equivalent_viscosities[i] = 0.;
              std::fill(implicit_constitutive_outputs->linearized_stress_terms[i].begin(),
                        implicit_constitutive_outputs->linearized_stress_terms[i].end(),
                        0.);
              for (unsigned int p = 0; p < n_particles_in_cell; ++p)
                {
                  double alpha = 0.;
                  for (unsigned int m = 0; m < n_basis; ++m)
                    alpha += w[p] * M[0][m] * phi[p][m];

                  implicit_constitutive_outputs->deviatoric_stresses[i] += alpha * particle_stresses[p];
                  implicit_constitutive_outputs->equivalent_viscosities[i] += alpha * particle_equivalent_viscosities[p];

                  for (unsigned int I = 0; I < implicit_constitutive_outputs->linearized_stress_terms[i].size(); ++I)
                    {
                      const std::pair<unsigned int, unsigned int> component_and_index = this->get_fe().system_to_component_index(I);
                      if (component_and_index.first < dim)
                        {
                          const Tensor<1, dim> grad_hat = this->get_fe().base_element(0).shape_grad(component_and_index.second, reference_locations[p]);
                          const Tensor<1, dim> grad_x = transpose(J_inv) * grad_hat;

                          Tensor<2, dim> grad_phi_u;
                          grad_phi_u[component_and_index.first] = grad_x;
                          const SymmetricTensor<2, dim> symgrad_phi_u = symmetrize(grad_phi_u);

                          SymmetricTensor<2, dim> linearized_stress = (2. * particle_bulk_viscosities[p]) * symgrad_phi_u;
                          if (particle_softening_factors[p] > 0.)
                            linearized_stress -= (2. * particle_bulk_viscosities[p] 
                                                  * particle_softening_factors[p]
                                                  * (particle_slip_directions[p] * symgrad_phi_u))
                                                 * particle_slip_directions[p];

                          implicit_constitutive_outputs->linearized_stress_terms[i][I] += alpha * linearized_stress;
                        }
                    }
                }
            }
        }
    }



    template <int dim>
    void PhaseFieldRSF<dim>::perform_return_mapping()
    {
      const Introspection<dim> &introspection = this->introspection();
      const PhaseFieldHandler<dim> &phase_field_handler = this->get_phase_field_handler();
      Particles::ParticleHandler<dim> &particle_handler = particle_manager->get_particle_handler();

      // Vector storing the chemical field values, which is required for computing volume fractions
      std::vector<double> chemical_field_values(introspection.n_chemical_composition_fields());

      // Perform return-mapping at locally-owned and ghost particles
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (!cell->is_artificial())
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

            // Update the evaluator and evaluate the temperature values at the particle locations
            evaluator->reinit(cell, {reference_locations.data(), reference_locations.size()});
            evaluator->evaluate({dof_values.data(), dof_values.size()}, evaluation_flags);

            small_vector<double>         particle_solution_values(introspection.n_components);
            small_vector<Tensor<1, dim>> particle_solution_gradients(introspection.n_components);

            for (auto particle = particles_in_cell.begin(); particle != particles_in_cell.end(); ++particle)
              {
                const unsigned int p = std::distance(particles_in_cell.begin(), particle);
                const ArrayView<double> particle_properties = particle->get_properties();

                evaluator->get_solution(p, {particle_solution_values.data(), particle_solution_values.size()}, evaluation_flags);
                evaluator->get_gradients(p, {particle_solution_gradients.data(), particle_solution_gradients.size()}, evaluation_flags);

                // Check if the material point is fractured
                const double pf = particle_solution_values[phase_field_component_index];
                if (pf > phase_field_kinetic_threshold)
                  {
                    // Compute the volume fractions
                    for (unsigned int j = 0; j < chemical_field_values.size(); ++j)
                      chemical_field_values[j] = particle_properties[particle_data_positions.chemical_fields[j]];
                    const std::vector<double> volume_fractions = MaterialUtilities::compute_composition_fractions(chemical_field_values);

                    const double p_bar = this->get_adiabatic_conditions().pressure(particle->get_location());
                    const double eta_d = MaterialUtilities::average_value(volume_fractions,
                                                                          radiation_damping_coefficients,
                                                                          MaterialUtilities::arithmetic);

                    // Compute the strain rate
                    Tensor<2, dim> grad_u;
                    for (unsigned int d = 0; d < dim; ++d)
                      grad_u[d] = particle_solution_gradients[d];
                    const SymmetricTensor<2, dim> epsilon = symmetrize(grad_u);

                    // Compute the trial state (predictor)
                    const double T = particle_solution_values[introspection.component_indices.temperature];
                    const double eta = calculate_creep_viscosity(T, volume_fractions);
                    const double G   = MaterialUtilities::average_value(volume_fractions, elastic_shear_moduli, viscosity_averaging);

                    const SymmetricTensor<2, dim> tau_b_old = Utilities::Tensors::to_symmetric_tensor<dim>(
                      &particle_properties[particle_data_positions.bulk_stress],
                      &particle_properties[particle_data_positions.bulk_stress] + SymmetricTensor<2, dim>::n_independent_components);
                    const SymmetricTensor<2, dim> tau_b = calculate_deviatoric_stress(epsilon, tau_b_old, eta, G);

                    const SymmetricTensor<2, dim> tau_f_old = Utilities::Tensors::to_symmetric_tensor<dim>(
                      &particle_properties[particle_data_positions.interface_stress],
                      &particle_properties[particle_data_positions.interface_stress] + SymmetricTensor<2, dim>::n_independent_components);
                    // The initial state assumes no slip (the slip rate is bounded by a positive value internally)
                    const SymmetricTensor<2, dim> tau_f = calculate_deviatoric_stress(epsilon, tau_f_old, eta, G);

                    const double g = phase_field_handler.energetic_degradation(pf, volume_fractions);
                    const SymmetricTensor<2, dim> tau_trial = g * tau_b + (1. - g) * tau_f;

                    const Tensor<1, dim> n(ArrayView<double>({&particle_properties[particle_data_positions.normal_direction], dim}));
                    const Tensor<1, dim> s(ArrayView<double>({&particle_properties[particle_data_positions.slip_direction], dim}));
                    const SymmetricTensor<2, dim> S = symmetrize(outer_product(n, s));

                    const double theta_old   = particle_properties[particle_data_positions.slip_state];
                    const double theta_trial = rsf_rheology.slip_state(0, theta_old);
                    const double mu_trial    = calculate_friction_coefficient(0, theta_trial, volume_fractions);

                    if (std::abs(tau_trial * S) - mu_trial * p_bar > 0)
                      {
                        // The material point is in slip mode: perform return-mapping
                        SymmetricTensor<2, dim> tau_f = tau_trial;

                        // Use the slip rate from the previous time step as an initial guess
                        double V     = particle_properties[particle_data_positions.slip_rate];
                        double theta = rsf_rheology.slip_state(V, theta_old);
                        double mu    = calculate_friction_coefficient(V, theta, volume_fractions);

                        const double F_init = std::abs(tau_f * S) - p_bar * mu - eta_d * V;
                        if (F_init > mu * p_bar * 1.e-8)
                          {
                            const unsigned int n_iter_max = 30;
                            const double tol = F_init * 1.e-8;

                            const Tensor<1, dim> &grad_pf = particle_solution_gradients[phase_field_component_index];

                            const double eta_ve = calculate_viscoelastic_viscosity(eta, G);
                            const double g = phase_field_handler.energetic_degradation(pf, volume_fractions);
                            const double gamma = phase_field_handler.crack_surface_density(pf, grad_pf);

                            double F = F_init;
                            unsigned int n_iter = 0;
                            while (F < tol)
                              {
                                const double dmu_over_dV = calculate_friction_coefficient_derivative(V, theta, volume_fractions);
                                const double dF_over_dV = -(p_bar * dmu_over_dV + eta_d);
                                AssertThrow(dF_over_dV != 0, ExcInternalError());

                                V -= F / dF_over_dV;
                                theta = rsf_rheology.slip_state(V, theta_old);
                                mu    = calculate_friction_coefficient(V, theta, volume_fractions);
                                tau_f = (tau_trial - (2.0 * (1.0 - g) * gamma * V * eta_ve) * S) * S;
                                F     = std::abs(tau_f * S) - mu * p_bar - eta_d * V;

                                ++n_iter;
                                if (n_iter > n_iter_max)
                                  break;
                              }

                            AssertThrow(F / F_init < tol,
                                        ExcMessage("The local return-mapping for the phase field RSF model did not converge."));
                          }
                      }
                  }
              }
          }
    }



    template <int dim>
    void PhaseFieldRSF<dim>::update_history_states()
    {

    }



    template <int dim>
    bool PhaseFieldRSF<dim>::is_compressible() const
    {
      return equation_of_state.is_compressible();
    }


    
    template <int dim>
    std::vector<double>
    PhaseFieldRSF<dim>::get_critical_energy_release_rates() const
    {
      return critical_energy_release_rates;
    }



    template <int dim>
    std::vector<double>
    PhaseFieldRSF<dim>::get_threshold_crack_driving_forces() const
    {
      const unsigned int n_comp = elastic_shear_moduli.size();
      std::vector<double> threshold_crack_driving_forces(n_comp);
      for (unsigned int j = 0; j < n_comp; ++j)
        threshold_crack_driving_forces[j] = cohesions[j] * cohesions[j] / (2.0 * elastic_shear_moduli[j]);

      return threshold_crack_driving_forces;
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
    calculate_viscoelastic_viscosity(const double creep_viscosity,
                                     const double shear_modulus) const
    {
      const double time_step = (this->get_timestep_number() > 0 ? this->get_timestep() : initial_time_step);
      return (1.0 - std::exp(-time_step * shear_modulus / creep_viscosity)) * creep_viscosity;
    }



    template <int dim>
    SymmetricTensor<2, dim>
    PhaseFieldRSF<dim>::
    calculate_deviatoric_stress(const SymmetricTensor<2, dim> &strain_rate,
                                const SymmetricTensor<2, dim> &old_stress,
                                const double                   creep_viscosity,
                                const double                   shear_modulus) const
    {
      const double time_step = (this->get_timestep_number() > 0 ? this->get_timestep() : initial_time_step);
      const double fraction = std::exp(-time_step * shear_modulus / creep_viscosity);
      return (2.0 * creep_viscosity * (1.0 - fraction)) * strain_rate + fraction * old_stress;
    }



    template <int dim>
    double
    PhaseFieldRSF<dim>::
    calculate_friction_coefficient(const double V,
                                   const double theta,
                                   const std::vector<double> &volume_fractions) const
    {
      double mu = 0;
      for (unsigned int j = 0; j < volume_fractions.size(); ++j)
        if (volume_fractions[j] > 0)
          mu += volume_fractions[j] * rsf_rheology.friction_coefficient(j, V, theta, true);

      return mu;
    }



    template <int dim>
    double
    PhaseFieldRSF<dim>::
    calculate_friction_coefficient_derivative(const double V,
                                              const double theta,
                                              const std::vector<double> &volume_fractions) const
    {
      if (V < std::numeric_limits<double>::epsilon())
        return 0.0;

      const double dV = V * 1.e-7;
      const double V_diff = V + dV;

      double mu = 0, mu_diff = 0;
      for (unsigned int j = 0; j < volume_fractions.size(); ++j)
        if (volume_fractions[j] > 0)
          {
            mu      += rsf_rheology.friction_coefficient(j, V, theta, true);
            mu_diff += rsf_rheology.friction_coefficient(j, V_diff, theta, true);
          }

      return (mu_diff - mu) / dV;
    }



    template <int dim>
    double
    PhaseFieldRSF<dim>::
    calculate_friction_strength(const Point<dim>          &position,
                                const double               slip_rate,
                                const double               slip_state,
                                const std::vector<double> &volume_fractions) const
    {
      const double mu = calculate_friction_coefficient(slip_rate, slip_state, volume_fractions);
      const double p_bar = this->get_adiabatic_conditions().pressure(position);
      const double eta_d = MaterialUtilities::average_value(volume_fractions, radiation_damping_coefficients, MaterialUtilities::arithmetic);
      return mu * p_bar + eta_d * slip_rate;
    }



    template <int dim>
    const Rheology::RateStateFriction<dim> &
    PhaseFieldRSF<dim>::get_rate_state_friction_model() const
    {
      return rsf_rheology;
    }



    template <int dim>
    bool
    PhaseFieldRSF<dim>::is_fractured(const double phase_field) const
    {
      return phase_field > phase_field_kinetic_threshold;
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

          prm.declare_entry("Phase field kinetic threshold", "1.e-4",
                            Patterns::Double(0, 1),
                            "The kinetic threshold of phase field, above which the material is judged "
                            "to be damaged, and the fault normal vector stored in particles will be "
                            "initialized.");

          prm.declare_entry("Phase field geometric threshold", "0.1",
                            Patterns::Double(0, 1),
                            "The geometric threshold of phase field, above which the fault normal "
                            "vector is considered to be ``fixed''. If the phase field is between the "
                            "kinetic threshold and the geometric threshold, then the fault normal is "
                            "determined by both the stress state and the normal vector stored in "
                            "particles.");

          prm.declare_entry("Initial time step", "1.",
                            Patterns::Double(0),
                            "The initial time step size. It is used for evolving the stress at the "
                            "zeroth time step. Units: years if the 'Use years instead of seconds' "
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

          prm.declare_entry("Cohesions", "1.e7",
                            Patterns::List(Patterns::Double(0)),
                            "List of cohesions, $C$, for background material and compositional fields, "
                            "for a total of N+1 values, where N is the number of all compositional fields or only "
                            "those corresponding to chemical compositions. Units: \\si{\\pascal}.");

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

          phase_field_kinetic_threshold   = prm.get_double("Phase field kinetic threshold");
          phase_field_geometric_threshold = prm.get_double("Phase field geometric threshold");
          AssertThrow(phase_field_kinetic_threshold <= phase_field_geometric_threshold,
                      ExcMessage("The phase field geometric threshold must be greater than or equal to "
                                 "the phase field kinetic threshold."));

          initial_time_step = prm.get_double("Initial time step");

          viscosity_averaging = MaterialUtilities::parse_compositional_averaging_operation("Viscosity averaging scheme", prm);

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

          options.property_name = "Critical energy release rates";
          critical_energy_release_rates = Utilities::MapParsing::parse_map_to_double_array(prm.get("Critical energy release rates"), options);

          options.property_name = "Radiation damping coefficients";
          radiation_damping_coefficients = Utilities::MapParsing::parse_map_to_double_array(prm.get("Radiation damping coefficients"), options);
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
