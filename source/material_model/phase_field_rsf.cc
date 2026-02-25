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
#include <aspect/particle/interpolator/interface.h>
#include <aspect/particle/particle_domain.h>
#include <aspect/solution_evaluator.h>
#include <aspect/newton.h>

#include <random>

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

      // Initialize the particle data positions
      const auto &particle_data_info = this->get_phase_field_handler().get_associated_particle_manager().get_property_manager().get_data_info();

      particle_data_positions.crack_driving_force = particle_data_info.get_position_by_field_name("crack_driving_force");
      particle_data_positions.slip_rate           = particle_data_info.get_position_by_field_name("slip_rate");
      particle_data_positions.slip_state          = particle_data_info.get_position_by_field_name("slip_state");
      particle_data_positions.normal              = particle_data_info.get_position_by_field_name("normal");
      particle_data_positions.slip_direction      = particle_data_info.get_position_by_field_name("slip_direction");
      particle_data_positions.stress              = particle_data_info.get_position_by_field_name("stress");

      particle_data_positions.chemical_fields.clear();
      for (const unsigned int index : this->introspection().chemical_composition_field_indices())
        particle_data_positions.chemical_fields.push_back(
          particle_data_info.get_position_by_field_name(this->get_parameters().mapped_particle_properties.find(index)->second.first));

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
    PhaseFieldRSF<dim>::
    evaluate(const MaterialModel::MaterialModelInputs<dim> &in,
             MaterialModel::MaterialModelOutputs<dim> &out) const
    {
      const unsigned int n_compositions = this->introspection().n_chemical_composition_fields() + 1;
      EquationOfStateOutputs<dim> eos_outputs(n_compositions);

      const std::shared_ptr<MaterialModel::ImplicitConstitutiveOutputs<dim>> implicit_constitutive_outputs
        = out.template get_additional_output_object<MaterialModel::ImplicitConstitutiveOutputs<dim>>();

      // In principle, the values and gradients of the phase field should be calculated
      // with the particle-domain-based shape functions, which are used in the assembly
      // of the phase field system. However, those shape functions are not cached ---
      // we only have their particle-domain-wise averages, i.e. the CPDI weighting functions.
      // Since the particle-domain-based shape functions can be regardes as approximations
      // of the Q1 shape functions, we use the Q1 function values and gradients here.
      const std::shared_ptr<const MaterialModel::PhaseFieldInputs<dim>> phase_field_inputs 
        = in.template get_additional_input_object<MaterialModel::PhaseFieldInputs<dim>>();

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

          
        }
    }



    template <int dim>
    void PhaseFieldRSF<dim>::perform_return_mapping()
    {
      const Introspection<dim> &introspection = this->introspection();
      const PhaseFieldHandler<dim> &phase_field_handler = this->get_phase_field_handler();
      Particles::ParticleHandler<dim> &particle_handler = particle_manager->get_particle_handler();

      const unsigned int phase_field_component_index 
        = introspection.component_indices.compositional_fields[introspection.compositional_index_for_name("phase_field")];

      // Stuff for evaluating the values and gradients of FE variables at particle locations
      std::unique_ptr<SolutionEvaluator<dim>> evaluator =
        construct_solution_evaluator(*this, update_values | update_gradients);

      std::vector<EvaluationFlags::EvaluationFlags> evaluation_flags(introspection.n_components, EvaluationFlags::nothing);
      evaluation_flags[introspection.component_indices.temperature] = EvaluationFlags::values;
      evaluation_flags[phase_field_component_index] = EvaluationFlags::values | EvaluationFlags::gradients;

      std::vector<Point<dim>> reference_locations;

      std::vector<double> clp_dof_values(this->get_fe().dofs_per_cell);
      std::vector<double> particle_clp_values(introspection.n_components);
      std::vector<Tensor<1, dim>> particle_clp_gradients(introspection.n_components);

      // Stuff for evaluating the strain rates at quadrature points
      FEValues<dim> fe_values(this->get_mapping(), 
                              this->get_fe(),
                              introspection.quadratures.velocities,
                              update_gradients | update_quadrature_points);

      std::vector<SymmetricTensor<2, dim>> qp_strain_rates(fe_values.n_quadrature_points);

      // Vectors storing the chemical composition fields and the volume fractions
      std::vector<double> chemical_field_values(introspection.n_chemical_composition_fields());
      std::vector<double> volume_fractions;

      // Perform return-mapping at locally-owned and ghost particles
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (!cell->is_artificial())
          {
            const unsigned int n_particles_in_cell = particle_handler.n_particles_in_cell(cell);
            AssertThrow(n_particles_in_cell > 0, ExcInternalError());

            // Collect the particle locations
            reference_locations.resize(n_particles_in_cell);
            const auto particles_in_cell = particle_handler.particles_in_cell(cell);
            auto particle = particles_in_cell.begin();
            for (unsigned int p = 0; p < n_particles_in_cell; ++p, ++particle)
              reference_locations[p] = particle->get_reference_location();

            // Collect the values of the current linearization point restricted to the current cell's DoFs
            cell->get_dof_values(this->get_current_linearization_point(),
                                 clp_dof_values.begin(), clp_dof_values.end());

            // Update the evaluator and evaluate the temperature values at the particle locations
            evaluator->reinit(cell, {reference_locations.data(), reference_locations.size()});
            evaluator->evaluate({clp_dof_values.data(), clp_dof_values.size()}, evaluation_flags);

            // Evaluate the strain rates at quadrature points
            fe_values.reinit(cell);
            fe_values[introspection.extractors.velocities].get_function_symmetric_gradients(this->get_current_linearization_point(), qp_strain_rates);

            particle = particles_in_cell.begin();
            for (unsigned int p = 0; p < n_particles_in_cell; ++p, ++particle)
              {
                const ArrayView<double> particle_properties = particle->get_properties();

                // Check if the material point is fractured
                evaluator->get_solution(p, {&particle_clp_values[0], particle_clp_values.size()}, evaluation_flags);
                const double pf = particle_clp_values[phase_field_component_index];

                if (pf > phase_field_kinetic_threshold)
                  {
                    // Compute the volume fractions
                    for (unsigned int j = 0; j < chemical_field_values.size(); ++j)
                      chemical_field_values[j] = particle_properties[particle_data_positions.chemical_fields[j]];
                    volume_fractions = MaterialUtilities::compute_composition_fractions(chemical_field_values);

                    const double p_bar = this->get_adiabatic_conditions().pressure(particle->get_location());
                    const double eta_d = MaterialUtilities::average_value(volume_fractions,
                                                                          radiation_damping_coefficients,
                                                                          MaterialUtilities::arithmetic);

                    // Pick the strain rate at the nearest quadrature point
                    unsigned int nearest_quarature_point = 0;
                    double distance = particle->get_location().distance(fe_values.quadrature_point(0));
                    for (unsigned int q = 1; q < fe_values.n_quadrature_points; ++q)
                      {
                        const double new_distance = particle->get_location().distance(fe_values.quadrature_point(q));
                        if (new_distance < distance)
                          {
                            nearest_quarature_point = q;
                            distance = new_distance;
                          }
                      }
                    const SymmetricTensor<2, dim> epsilon = qp_strain_rates[nearest_quarature_point];

                    // Compute the trial state (predictor)
                    const double T = particle_clp_values[introspection.component_indices.temperature];
                    const double eta = calculate_creep_viscosity(T, volume_fractions);
                    const double G   = MaterialUtilities::average_value(volume_fractions, elastic_shear_moduli, viscosity_averaging);
                    const SymmetricTensor<2, dim> tau_old = Utilities::Tensors::to_symmetric_tensor<dim>(
                      &particle_properties[particle_data_positions.stress],
                      &particle_properties[particle_data_positions.stress] + SymmetricTensor<2, dim>::n_independent_components);
                    const SymmetricTensor<2, dim> tau_trial = calculate_bulk_stress(eta, G, epsilon, tau_old);

                    const SymmetricTensor<2, dim> S = Utilities::Tensors::to_symmetric_tensor<dim>(
                      &particle_properties[particle_data_positions.slip_direction],
                      &particle_properties[particle_data_positions.slip_direction] + SymmetricTensor<2, dim>::n_independent_components);

                    // The initial state assumes no slip (the slip rate is bounded by a positive value internally)
                    const double theta_old   = particle_properties[particle_data_positions.slip_state];
                    const double theta_trial = rsf_rheology.slip_state(0, theta_old);
                    const double mu_trial    = calculate_friction_coefficient(0, theta_trial, volume_fractions);

                    if (std::abs(tau_trial * S) - mu_trial * p_bar > 0)
                      {
                        // The material point is in slip mode: perform return-mapping
                        SymmetricTensor<2, dim> tau_f = tau_trial;

                        // Use the slip rate from the previous time step as initial guess
                        double V     = particle_properties[particle_data_positions.slip_rate];
                        double theta = rsf_rheology.slip_state(V, theta_old);
                        double mu    = calculate_friction_coefficient(V, theta, volume_fractions);

                        const double F_init = std::abs(tau_f * S) - p_bar * mu - eta_d * V;
                        if (F_init > mu * p_bar * 1.e-8)
                          {
                            const unsigned int n_iter_max = 30;
                            const double tol = F_init * 1.e-8;

                            evaluator->get_gradients(p, {&particle_clp_gradients[0], particle_clp_gradients.size()}, evaluation_flags);
                            const Tensor<1, dim> &grad_pf = particle_clp_gradients[phase_field_component_index];

                            const double eta_ve = calculate_viscoelastic_viscosity(eta, G);
                            const double g = phase_field_handler.energetic_degradation(pf, volume_fractions);
                            const double gamma = phase_field_handler.crack_surface_density(pf, grad_pf);

                            double F = F_init;
                            unsigned int n_iter = 0;
                            while (F < tol)
                              {
                                const double dmu_dV = calculate_friction_coefficient_derivative(V, theta, volume_fractions);
                                const double dF_dV = -(p_bar * dmu_dV + eta_d);
                                AssertThrow(dF_dV != 0, ExcInternalError());

                                V -= F / dF_dV;
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
    SymmetricTensor<2, dim>
    PhaseFieldRSF<dim>::
    calculate_bulk_stress(const double                   temperature,
                          const SymmetricTensor<2, dim> &strain_rate,
                          const SymmetricTensor<2, dim> &stress_old,
                          const std::vector<double>     &volume_fractions) const
    {
      const double eta = calculate_creep_viscosity(temperature, volume_fractions);
      const double G   = MaterialUtilities::average_value(volume_fractions, elastic_shear_moduli, viscosity_averaging);
      const double dt  = (this->get_timestep_number() > 0 ? this->get_timestep() : initial_time_step);
      const double ratio = std::exp(-dt * G / eta);
      return (2.0 * eta * (1.0 - ratio)) * strain_rate + ratio * stress_old;
    }



    template <int dim>
    const Rheology::RateStateFriction<dim> &
    PhaseFieldRSF<dim>::get_rate_state_friction_model() const
    {
      return rsf_rheology;
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
