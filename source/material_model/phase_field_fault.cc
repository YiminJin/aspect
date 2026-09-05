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

#include <aspect/material_model/phase_field_fault.h>
#include <aspect/material_model/utilities.h>
#include <aspect/phase_field.h>
#include <aspect/particle/manager.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/reconstructed_fault/utilities.h>
#include <aspect/newton.h>
#include <aspect/simulator.h>
#include <aspect/postprocess/visualization.h>
#include <aspect/postprocess/particles.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_cartesian.h>
#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/numerics/vector_tools_evaluate.h>

#include <numeric>

namespace aspect
{
  // -----------------------------------------------------------------------------
  // File-local helper types
  // -----------------------------------------------------------------------------

  namespace
  {
    template <int dim>
    bool
    cohesive_history_is_initialized(
      const std::vector<ReconstructedFault<dim>> &faults,
      const unsigned int cohesive_position,
      const unsigned int normalization_position)
    {
      bool any_initialized = false;
      bool any_uninitialized = false;
      for (const ReconstructedFault<dim> &fault : faults)
        for (unsigned int vertex = 0; vertex < fault.n_vertices(); ++vertex)
          {
            const bool cohesive_is_initialized =
              fault.property_value_is_initialized(vertex, cohesive_position);
            const bool normalization_is_initialized =
              fault.property_value_is_initialized(vertex, normalization_position);
            AssertThrow(cohesive_is_initialized == normalization_is_initialized,
                        ExcMessage("A reconstructed fault contains partially initialized "
                                   "cohesive history."));

            if (cohesive_is_initialized)
              {
                const ArrayView<const double> properties = fault.get_properties(vertex);
                AssertThrow(std::isfinite(properties[cohesive_position])
                            && std::isfinite(properties[normalization_position])
                            && properties[cohesive_position] >= 0.0
                            && properties[normalization_position] > 0.0,
                            ExcMessage("Stored cohesive history is physically inadmissible."));
                any_initialized = true;
              }
            else
              any_uninitialized = true;
          }

      AssertThrow(!(any_initialized && any_uninitialized),
                  ExcMessage("A reconstructed-fault collection contains a mixture of initialized "
                             "and uninitialized cohesive history."));
      return any_initialized;
    }


    struct NormalizationSideState
    {
      double panel_start = 0.0;
      double panel_width = 0.0;
      double integral = 0.0;
      double window_span = 0.0;
      double window_integral = 0.0;
      unsigned int successive_small_windows = 0;
      unsigned int refinement_depth = 0;
      unsigned int accepted_extensions = 0;
      bool boundary_search = false;
      bool boundary_final_panel = false;
      double boundary_low = 0.0;
      double boundary_high = 0.0;
      unsigned int boundary_bisections = 0;
      bool complete = false;
    };


    struct NormalizationEvaluationRequest
    {
      unsigned int profile;
      unsigned int side;
      bool boundary_probe;
      unsigned int first_point;
      std::vector<double> zeta;
    };


    template <int dim, typename Profile>
    Point<dim>
    normalization_profile_point(const Profile &profile,
                                const unsigned int side,
                                const double zeta)
    {
      return profile.origin
             + (side == 0 ? 1.0 : -1.0) * zeta * profile.normal;
    }


    template <int dim>
    std::map<types::particle_index, std::vector<double>>
    interpolate_surface_chemical_compositions(
      ReconstructedFaultManager<dim> &fault_manager,
      const std::vector<unsigned int> &property_indices)
    {
      std::map<types::particle_index, std::vector<double>> compositions;
      for (unsigned int c = 0; c < property_indices.size(); ++c)
        {
          const std::map<types::particle_index, std::vector<double>> values =
            fault_manager.interpolate_property_at_particle_projections(
              property_indices[c]);

          if (c == 0)
            for (const auto &particle : values)
              compositions.emplace(
                particle.first,
                std::vector<double>(property_indices.size(),
                                    numbers::signaling_nan<double>()));
          else
            AssertDimension(values.size(), compositions.size());

          for (const auto &particle : values)
            {
              AssertDimension(particle.second.size(), 1);
              const auto composition = compositions.find(particle.first);
              Assert(composition != compositions.end(), ExcInternalError());
              composition->second[c] = particle.second[0];
            }
        }
      return compositions;
    }


    template <int dim>
    double
    interpolate_fault_scalar(const ReconstructedFault<dim> &fault,
                             const unsigned int segment,
                             const double xi,
                             const unsigned int position,
                             const std::string &property_name)
    {
      AssertThrow(fault.property_value_is_initialized(segment, position)
                  && fault.property_value_is_initialized(segment+1, position),
                  ExcMessage("Reconstructed-fault property <" + property_name
                             + "> is uninitialized at a constitutive evaluation point."));
      return (1.0-xi) * fault.get_properties(segment)[position]
             + xi * fault.get_properties(segment+1)[position];
    }
  }

  namespace MaterialModel
  {
    // -----------------------------------------------------------------------------
    // Maxwell constitutive law
    // -----------------------------------------------------------------------------

    template <int dim>
    typename PhaseFieldFault<dim>::MaxwellCoefficients
    PhaseFieldFault<dim>::
    compute_maxwell_coefficients(const double viscosity,
                                 const double shear_modulus,
                                 const double time_step)
    {
      const double exponent = -time_step * shear_modulus / viscosity;
      const double beta = std::exp(exponent);
      const double kappa = -viscosity * std::expm1(exponent);
      return {beta, kappa};
    }



    template <int dim>
    SymmetricTensor<2,dim>
    PhaseFieldFault<dim>::
    compute_maxwell_stress(
      const MaxwellCoefficients &coefficients,
      const SymmetricTensor<2,dim> &effective_bulk_strain_rate,
      const SymmetricTensor<2,dim> &old_stress)
    {
      return 2.0 * coefficients.kappa * effective_bulk_strain_rate
             + coefficients.beta * old_stress;
    }



    // -----------------------------------------------------------------------------
    // Material-model interface
    // -----------------------------------------------------------------------------

    template <int dim>
    void
    PhaseFieldFault<dim>::
    evaluate(const MaterialModel::MaterialModelInputs<dim> &in,
             MaterialModel::MaterialModelOutputs<dim> &out) const
    {
      EquationOfStateOutputs<dim> eos_outputs(
        this->introspection().n_chemical_composition_fields() + 1);

      for (unsigned int i = 0; i < in.n_evaluation_points(); ++i)
        {
          const std::vector<double> volume_fractions =
            MaterialUtilities::compute_only_composition_fractions(
              in.composition[i],
              this->introspection().chemical_composition_field_indices());

          // Fill in the equation-of-state outputs
          equation_of_state.evaluate(in, i, eos_outputs);

          out.densities[i] = MaterialUtilities::average_value(
            volume_fractions, eos_outputs.densities, MaterialUtilities::arithmetic);
          out.thermal_expansion_coefficients[i] = MaterialUtilities::average_value(
            volume_fractions, eos_outputs.thermal_expansion_coefficients,
            MaterialUtilities::arithmetic);
          out.specific_heat[i] = MaterialUtilities::average_value(
            volume_fractions, eos_outputs.specific_heat_capacities,
            MaterialUtilities::arithmetic);
          out.thermal_conductivities[i] = MaterialUtilities::average_value(
            volume_fractions, thermal_conductivities, MaterialUtilities::arithmetic);
          out.compressibilities[i] = MaterialUtilities::average_value(
            volume_fractions, eos_outputs.compressibilities,
            MaterialUtilities::arithmetic);
          out.entropy_derivative_pressure[i] = MaterialUtilities::average_value(
            volume_fractions, eos_outputs.entropy_derivative_pressure,
            MaterialUtilities::arithmetic);
          out.entropy_derivative_temperature[i] = MaterialUtilities::average_value(
            volume_fractions, eos_outputs.entropy_derivative_temperature,
            MaterialUtilities::arithmetic);

          if (in.requests_property(MaterialProperties::viscosity))
            {
              // Set the output viscosity to be the viscoelastic viscosity (It will not be used in the assemblers,
              // but might be requested by some other functions, like Simulator::compute_pressure_scaling_factor()).
              const double G = MaterialUtilities::average_value(
                volume_fractions, elastic_shear_moduli, viscosity_averaging);
              const double eta = compute_creep_viscosity(volume_fractions, in.temperature[i]);
              const double time_step = (this->get_timestep_number() > 0
                                        ? this->get_timestep()
                                        : initial_time_step);
              const MaxwellCoefficients coefficients =
                compute_maxwell_coefficients(eta, G, time_step);
              out.viscosities[i] = coefficients.kappa;
            }
        }
    }



    template <int dim>
    bool PhaseFieldFault<dim>::is_compressible() const
    {
      return equation_of_state.is_compressible();
    }



    template <int dim>
    std::vector<double>
    PhaseFieldFault<dim>::get_critical_crack_driving_forces() const
    {
      const unsigned int n_comp = elastic_shear_moduli.size();
      std::vector<double> critical_crack_driving_forces(n_comp);
      for (unsigned int j = 0; j < n_comp; ++j)
        critical_crack_driving_forces[j] = cohesions[j] * cohesions[j] / (2.0 * elastic_shear_moduli[j]);

      return critical_crack_driving_forces;
    }


    
    template <int dim>
    std::vector<double>
    PhaseFieldFault<dim>::get_critical_energy_release_rates() const
    {
      return critical_energy_release_rates;
    }



    template <int dim>
    double
    PhaseFieldFault<dim>::get_phase_field_activation_threshold() const
    {
      return phase_field_activation_threshold;
    }



    template <int dim>
    double
    PhaseFieldFault<dim>::get_phase_field_upper_admissibility_threshold() const
    {
      return 0.99;
    }



    template <int dim>
    void
    PhaseFieldFault<dim>::initialize()
    {
      if (!this->get_parameters().reconstruct_faults)
        return;

      const std::vector<unsigned int> &chemical_field_indices =
        this->introspection().chemical_composition_field_indices();
      const std::vector<std::string> &chemical_field_names =
        this->introspection().chemical_composition_field_names();
      AssertDimension(chemical_field_names.size(), chemical_field_indices.size());
      for (unsigned int c = 0; c < chemical_field_indices.size(); ++c)
        {
          const unsigned int field_index = chemical_field_indices[c];
          AssertThrow(this->get_parameters().compositional_field_methods[field_index]
                      == Parameters<dim>::AdvectionFieldMethod::particles,
                      ExcMessage("Distributed I_h evaluation requires every chemical "
                                 "composition field to be advected by particles."));
          AssertThrow(this->get_parameters().mapped_particle_properties.find(field_index)
                      != this->get_parameters().mapped_particle_properties.end(),
                      ExcMessage("Distributed I_h evaluation requires every chemical "
                                 "composition field to be mapped to a particle property."));
        }

      ReconstructedFaultManager<dim> &fault_manager =
        this->get_reconstructed_fault_manager();
      if (fault_friction.has_state_variable())
        fault_property_indices.state = fault_manager.register_property(
          "phase field fault state", 1);
      fault_property_indices.cohesive_traction = fault_manager.register_property(
        "phase field fault cohesive traction", 1);
      fault_property_indices.previous_normalization_integral =
        fault_manager.register_property("phase field fault previous I h", 1);

      fault_property_indices.chemical_compositions.clear();
      fault_property_indices.chemical_compositions.reserve(
        chemical_field_indices.size());
      for (unsigned int c = 0; c < chemical_field_indices.size(); ++c)
        fault_property_indices.chemical_compositions.push_back(
          fault_manager.register_property(
            "phase field fault chemical composition " + chemical_field_names[c], 1));
    }


    template <int dim>
    typename PhaseFieldFault<dim>::ReconstructedFaultPointResponse
    PhaseFieldFault<dim>::evaluate_reconstructed_fault_point(
      const ReconstructedFaultPointInputs &inputs) const
    {
      AssertThrow(dim == 2, ExcNotImplemented());
      AssertThrow(std::isfinite(inputs.slip_rate)
                  && inputs.slip_rate >= fault_friction.get_minimum_slip_rate(),
                  ExcMessage("A reconstructed-fault constitutive evaluation requires "
                             "V >= V_min."));

      const ReconstructedFaultManager<dim> &fault_manager =
        this->get_reconstructed_fault_manager();
      const ReconstructedFault<dim> &fault =
        fault_manager.get_fault(inputs.fault_index);
      AssertIndexRange(inputs.segment_index, fault.n_cells());
      Assert(inputs.xi >= 0.0 && inputs.xi <= 1.0, ExcInternalError());
      const LocalizationResponse localization =
        evaluate_reconstructed_fault_localization(
          inputs.fault_index, inputs.segment_index, inputs.xi,
          inputs.phase_field, inputs.previous_phase_field,
          "surface residual");

      const double eta = compute_creep_viscosity(inputs.bulk_material_fractions,
                                                 inputs.temperature);
      const double G = MaterialUtilities::average_value(
        inputs.bulk_material_fractions, elastic_shear_moduli, viscosity_averaging);
      const double time_step = (this->get_timestep_number() > 0
                                ? this->get_timestep()
                                : initial_time_step);
      const MaxwellCoefficients coefficients =
        compute_maxwell_coefficients(eta, G, time_step);
      const CohesiveResponse cohesive = compute_cohesive_response(
        coefficients, localization.current_I_h, localization.previous_I_h,
        localization.previous_cohesive_traction, inputs.slip_rate,
        localization.current_h, localization.previous_h);

      const SymmetricTensor<2,dim> trial_stress = compute_maxwell_stress(
        coefficients, inputs.strain_rate, inputs.old_maxwell_stress);
      const SymmetricTensor<2,dim> stress_without_current_slip =
        trial_stress
        - 2.0 * coefficients.kappa * cohesive.history_correction
          * inputs.slip_tensor;
      const SymmetricTensor<2,dim> stress =
        stress_without_current_slip
        - 2.0 * coefficients.kappa * cohesive.localization_factor
          * inputs.slip_rate * inputs.slip_tensor;

      const double theta = fault_friction.has_state_variable()
                           ? interpolate_fault_scalar(
                               fault,
                               inputs.segment_index, inputs.xi,
                               fault_manager.get_property_information()[
                                 fault_property_indices.state].position,
                               "phase field fault state")
                           : 0.0;
      const double mu = fault_friction.has_state_variable()
                        ? fault_friction.friction_coefficient(
                            localization.surface_material_fractions,
                            inputs.slip_rate, theta)
                        : fault_friction.friction_coefficient(
                            localization.surface_material_fractions,
                            inputs.slip_rate);
      const double dmu_dV = fault_friction.has_state_variable()
                            ? fault_friction.friction_coefficient_derivative_wrt_slip_rate(
                                localization.surface_material_fractions,
                                inputs.slip_rate, theta)
                            : fault_friction.friction_coefficient_derivative_wrt_slip_rate(
                                localization.surface_material_fractions,
                                inputs.slip_rate);
      const double sigma_n = use_adiabatic_pressure_in_fault_friction
                             ? this->get_adiabatic_conditions().pressure(inputs.position)
                             : inputs.dynamic_pressure
                               - stress * inputs.normal_tensor;
      const double damping = MaterialUtilities::average_value(
        localization.surface_material_fractions, radiation_damping_coefficients,
        MaterialUtilities::arithmetic);

      ReconstructedFaultPointResponse response;
      response.residual_density = stress * inputs.slip_tensor
                                  - cohesive.cohesive_traction
                                  - mu * sigma_n
                                  - damping * inputs.slip_rate;
      response.minus_derivative_wrt_slip_rate =
        2.0 * coefficients.kappa * cohesive.localization_factor
        * (inputs.slip_tensor * inputs.slip_tensor)
        + coefficients.kappa/localization.current_I_h
        + sigma_n*dmu_dV
        + damping;
      response.kappa = coefficients.kappa;
      response.localization_factor = cohesive.localization_factor;
      response.friction_coefficient = mu;
      response.uses_adiabatic_friction_pressure =
        use_adiabatic_pressure_in_fault_friction;
      return response;
    }


    template <int dim>
    typename PhaseFieldFault<dim>::ReconstructedFaultBulkPointResponse
    PhaseFieldFault<dim>::evaluate_reconstructed_fault_bulk_point(
      const ReconstructedFaultBulkPointInputs &inputs) const
    {
      const LocalizationResponse localization =
        evaluate_reconstructed_fault_localization(
          inputs.fault_index, inputs.segment_index, inputs.xi,
          inputs.phase_field, inputs.previous_phase_field,
          "bulk residual");

      const double eta = compute_creep_viscosity(inputs.bulk_material_fractions,
                                                 inputs.temperature);
      const double G = MaterialUtilities::average_value(
        inputs.bulk_material_fractions, elastic_shear_moduli, viscosity_averaging);
      const double time_step = (this->get_timestep_number() > 0
                                ? this->get_timestep()
                                : initial_time_step);
      const MaxwellCoefficients coefficients =
        compute_maxwell_coefficients(eta, G, time_step);
      const CohesiveResponse cohesive = compute_cohesive_response(
        coefficients, localization.current_I_h, localization.previous_I_h,
        localization.previous_cohesive_traction, 0.0,
        localization.current_h, localization.previous_h);

      ReconstructedFaultBulkPointResponse response;
      response.kappa = coefficients.kappa;
      response.localization_factor = cohesive.localization_factor;
      response.history_correction = cohesive.history_correction;
      return response;
    }


    template <int dim>
    double
    PhaseFieldFault<dim>::minimum_fault_slip_rate() const
    {
      return fault_friction.get_minimum_slip_rate();
    }


    template <int dim>
    void
    PhaseFieldFault<dim>::validate_reconstructed_fault_constitutive_state() const
    {
      const ReconstructedFaultManager<dim> &fault_manager =
        this->get_reconstructed_fault_manager();
      const auto &faults = fault_manager.get_faults();
      AssertThrow(!faults.empty(),
                  ExcMessage("Reconstructed-fault constitutive evaluation requires "
                             "fault geometry."));
      AssertThrow(!use_adiabatic_pressure_in_fault_friction
                  || this->get_adiabatic_conditions().is_initialized(),
                  ExcMessage("Adiabatic fault-friction pressure requires initialized "
                             "adiabatic conditions."));
      AssertThrow(fault_property_indices.cohesive_traction
                    != numbers::invalid_unsigned_int
                  && fault_property_indices.previous_normalization_integral
                    != numbers::invalid_unsigned_int,
                  ExcMessage("Reconstructed-fault cohesive properties have not been "
                             "registered."));
      const unsigned int cohesive_position =
        fault_manager.get_property_information()[
          fault_property_indices.cohesive_traction].position;
      const unsigned int normalization_position =
        fault_manager.get_property_information()[
          fault_property_indices.previous_normalization_integral].position;
      AssertThrow(cohesive_history_is_initialized(
                    faults, cohesive_position, normalization_position),
                  ExcMessage("Reconstructed-fault cohesive history has not been initialized."));

      AssertThrow(current_normalization_integrals.size() == faults.size(),
                  ExcMessage("Current reconstructed-fault I_h has not been computed."));
      for (unsigned int fault_index = 0;
           fault_index < faults.size(); ++fault_index)
        {
          const ReconstructedFault<dim> &fault = faults[fault_index];
          AssertThrow(current_normalization_integrals[fault_index].size()
                      == fault.n_vertices(),
                      ExcMessage("Current reconstructed-fault I_h has the wrong number "
                                 "of vertices for fault "
                                 + Utilities::int_to_string(fault_index) + "."));
          for (unsigned int vertex = 0; vertex < fault.n_vertices(); ++vertex)
            AssertThrow(std::isfinite(
                          current_normalization_integrals[fault_index][vertex])
                        && current_normalization_integrals[fault_index][vertex] > 0.0,
                        ExcMessage("Current reconstructed-fault I_h must be finite and "
                                   "positive at fault "
                                   + Utilities::int_to_string(fault_index)
                                   + " vertex " + Utilities::int_to_string(vertex) + "."));
        }

      if (fault_friction.has_state_variable())
        {
          AssertThrow(fault_property_indices.state != numbers::invalid_unsigned_int,
                      ExcMessage("The reconstructed-fault rate-and-state property has "
                                 "not been registered."));
          const unsigned int state_position =
            fault_manager.get_property_information()[fault_property_indices.state].position;
          for (unsigned int fault_index = 0; fault_index < faults.size(); ++fault_index)
            for (unsigned int vertex = 0;
                 vertex < faults[fault_index].n_vertices(); ++vertex)
              AssertThrow(faults[fault_index].property_value_is_initialized(
                            vertex, state_position)
                          && std::isfinite(
                            faults[fault_index].get_properties(vertex)[state_position])
                          && faults[fault_index].get_properties(vertex)[state_position] > 0.0,
                          ExcMessage("Rate-and-state fault friction requires a positive "
                                     "initialized Theta at fault "
                                     + Utilities::int_to_string(fault_index)
                                     + " vertex " + Utilities::int_to_string(vertex) + "."));
        }
    }


    // -----------------------------------------------------------------------------
    // Cohesive constitutive law
    // -----------------------------------------------------------------------------

    template <int dim>
    typename PhaseFieldFault<dim>::LocalizationResponse
    PhaseFieldFault<dim>::evaluate_reconstructed_fault_localization(
      const unsigned int fault_index,
      const unsigned int segment_index,
      const double xi,
      const double phase_field,
      const double previous_phase_field,
      const std::string &context) const
    {
      const ReconstructedFaultManager<dim> &fault_manager =
        this->get_reconstructed_fault_manager();
      const ReconstructedFault<dim> &fault = fault_manager.get_fault(fault_index);
      AssertIndexRange(segment_index, fault.n_cells());
      Assert(xi >= 0.0 && xi <= 1.0, ExcInternalError());
      AssertDimension(current_normalization_integrals.size(),
                      fault_manager.get_faults().size());

      std::vector<double> surface_compositions(
        fault_property_indices.chemical_compositions.size());
      for (unsigned int c = 0; c < surface_compositions.size(); ++c)
        {
          const unsigned int property =
            fault_property_indices.chemical_compositions[c];
          const unsigned int position =
            fault_manager.get_property_information()[property].position;
          surface_compositions[c] = interpolate_fault_scalar(
            fault, segment_index, xi, position,
            fault_manager.get_property_information()[property].name);
        }

      LocalizationResponse response;
      response.surface_material_fractions =
        MaterialUtilities::compute_composition_fractions(surface_compositions);
      const unsigned int previous_I_h_position =
        fault_manager.get_property_information()[
          fault_property_indices.previous_normalization_integral].position;
      const unsigned int cohesive_position =
        fault_manager.get_property_information()[
          fault_property_indices.cohesive_traction].position;
      response.current_I_h =
        (1.0-xi) * current_normalization_integrals[fault_index][segment_index]
        + xi * current_normalization_integrals[fault_index][segment_index+1];
      response.previous_I_h = interpolate_fault_scalar(
        fault, segment_index, xi, previous_I_h_position,
        "phase field fault previous I h");
      response.previous_cohesive_traction = interpolate_fault_scalar(
        fault, segment_index, xi, cohesive_position,
        "phase field fault cohesive traction");

      const double current_phi = normalization_effective_phase_field(
        phase_field, context);
      const double previous_phi = normalization_effective_phase_field(
        previous_phase_field, context + " history");
      const PhaseFieldHandler<dim> &phase_field_handler =
        this->get_phase_field_handler();
      const double current_degradation =
        phase_field_handler.energetic_degradation(
          response.surface_material_fractions, current_phi);
      const double previous_degradation =
        phase_field_handler.energetic_degradation(
          response.surface_material_fractions, previous_phi);
      response.current_h = normalization_integrand(
        current_phi, current_degradation, context);
      response.previous_h = normalization_integrand(
        previous_phi, previous_degradation, context + " history");
      return response;
    }


    template <int dim>
    typename PhaseFieldFault<dim>::CohesiveResponse
    PhaseFieldFault<dim>::compute_cohesive_response(
      const MaxwellCoefficients &coefficients,
      const double current_I_h,
      const double previous_I_h,
      const double previous_cohesive_traction,
      const double slip_rate,
      const double current_h,
      const double previous_h)
    {
      AssertThrow(coefficients.kappa > 0.0,
                  ExcMessage("The cohesive effective viscosity kappa must be positive."));
      AssertThrow(std::isfinite(current_I_h) && current_I_h > 0.0,
                  ExcMessage("The current cohesive normalization integral must be finite and positive."));
      AssertThrow(std::isfinite(previous_I_h) && previous_I_h > 0.0,
                  ExcMessage("The previous cohesive normalization integral must be finite and positive."));
      AssertThrow(std::isfinite(previous_cohesive_traction)
                  && previous_cohesive_traction >= 0.0,
                  ExcMessage("The previous cohesive traction must be finite and nonnegative."));
      AssertThrow(std::isfinite(slip_rate) && slip_rate >= 0.0,
                  ExcMessage("The cohesive slip rate must be finite and nonnegative."));
      AssertThrow(std::isfinite(current_h) && current_h >= 0.0
                  && std::isfinite(previous_h) && previous_h >= 0.0,
                  ExcMessage("The current and previous cohesive degradation functions h "
                             "must be finite and nonnegative."));

      CohesiveResponse response;
      response.cohesive_traction =
        (coefficients.kappa * slip_rate
         + coefficients.beta * previous_I_h * previous_cohesive_traction)
        / current_I_h;
      response.localization_factor = current_h/current_I_h;
      response.history_correction =
        coefficients.beta * previous_cohesive_traction/coefficients.kappa
        * (current_h * previous_I_h/current_I_h - previous_h);
      response.crack_strain_rate =
        response.localization_factor * slip_rate + response.history_correction;

      AssertThrow(std::isfinite(response.cohesive_traction)
                  && std::isfinite(response.localization_factor)
                  && std::isfinite(response.history_correction)
                  && std::isfinite(response.crack_strain_rate),
                  ExcMessage("The common cohesive law produced a non-finite response."));
      return response;
    }


    // -----------------------------------------------------------------------------
    // Initial cohesive-state initialization
    // -----------------------------------------------------------------------------


    template <int dim>
    void
    PhaseFieldFault<dim>::commit_cohesive_state(
      const std::vector<std::vector<double>> &cohesive_tractions)
    {
      ReconstructedFaultManager<dim> &fault_manager =
        this->get_reconstructed_fault_manager();
      const auto &faults = fault_manager.get_faults();
      const unsigned int cohesive_position =
        fault_manager.get_property_information()[
          fault_property_indices.cohesive_traction].position;
      const unsigned int normalization_position =
        fault_manager.get_property_information()[
          fault_property_indices.previous_normalization_integral].position;

      for (unsigned int fault = 0; fault < faults.size(); ++fault)
        {
          for (unsigned int vertex = 0; vertex < faults[fault].n_vertices(); ++vertex)
            {
              AssertThrow(std::isfinite(cohesive_tractions[fault][vertex])
                          && cohesive_tractions[fault][vertex] >= 0.0,
                          ExcMessage("Committed cohesive traction must be finite and nonnegative."));
              AssertThrow(std::isfinite(current_normalization_integrals[fault][vertex])
                          && current_normalization_integrals[fault][vertex] > 0.0,
                          ExcMessage("Committed previous I_h must be finite and positive."));
            }
        }

      // Validation above makes the following update atomic with respect to
      // constitutive input errors.
      for (unsigned int fault = 0; fault < faults.size(); ++fault)
        for (unsigned int vertex = 0; vertex < faults[fault].n_vertices(); ++vertex)
          {
            ArrayView<double> properties =
              fault_manager.get_fault(fault).get_properties(vertex);
            properties[cohesive_position] = cohesive_tractions[fault][vertex];
            properties[normalization_position] =
              current_normalization_integrals[fault][vertex];
          }
    }



    template <int dim>
    void
    PhaseFieldFault<dim>::initialize_cohesive_state_from_initial_fields()
    {
      AssertThrow(dim == 2, ExcNotImplemented());
      ReconstructedFaultManager<dim> &fault_manager =
        this->get_reconstructed_fault_manager();
      const auto &faults = fault_manager.get_faults();
      AssertThrow(!faults.empty(),
                  ExcMessage("Initial cohesive state requires reconstructed fault geometry."));
      const unsigned int cohesive_position =
        fault_manager.get_property_information()[
          fault_property_indices.cohesive_traction].position;
      const unsigned int normalization_position =
        fault_manager.get_property_information()[
          fault_property_indices.previous_normalization_integral].position;

      if (cohesive_history_is_initialized(faults,
                                          cohesive_position,
                                          normalization_position))
        return;

      compute_normalization_integrals();
      const auto projection = fault_manager.project_particle_scalar(
        evaluate_initial_cohesive_particle_values());
      initial_cohesive_projection_diagnostics = projection.diagnostics;
      commit_cohesive_state(projection.nodal_values);

      for (unsigned int fault = 0; fault < projection.diagnostics.size(); ++fault)
        {
          const auto &diagnostic = projection.diagnostics[fault];
          this->get_pcout()
            << "   Initial cohesive q profile variation for fault " << fault
            << ": weighted RMS=" << diagnostic.weighted_rms_residual
            << " Pa, maximum=" << diagnostic.maximum_absolute_residual
            << " Pa, normalized RMS="
            << diagnostic.normalized_weighted_rms_residual
            << ", normalized maximum="
            << diagnostic.normalized_maximum_absolute_residual << std::endl;
        }
    }



    template <int dim>
    std::map<types::particle_index, double>
    PhaseFieldFault<dim>::evaluate_initial_cohesive_particle_values()
    {
      const PhaseFieldHandler<dim> &phase_field_handler =
        this->get_phase_field_handler();
      const Particle::Manager<dim> &particle_manager =
        phase_field_handler.get_associated_particle_manager();
      const auto &particle_handler = particle_manager.get_particle_handler();
      const auto &particle_data = particle_manager.get_property_manager().get_data_info();
      AssertThrow(particle_data.fieldname_exists("crack_driving_force"),
                  ExcMessage("Initial cohesive state requires the particle property "
                             "'crack_driving_force'."));
      const unsigned int H_position =
        particle_data.get_position_by_field_name("crack_driving_force");

      std::vector<Point<dim>> particle_positions;
      particle_positions.reserve(particle_handler.n_locally_owned_particles());
      for (const auto &particle : particle_handler)
        particle_positions.push_back(particle.get_location());

      Utilities::MPI::RemotePointEvaluation<dim> point_cache;
      point_cache.reinit(phase_field_handler.get_grid_cache(), particle_positions);
      const unsigned int phase_field_component =
        this->introspection().variable("phase_field").first_component_index;
      const std::vector<double> phase_field_values =
        VectorTools::point_values<1>(point_cache,
                                     this->get_dof_handler(),
                                     this->get_solution(),
                                     VectorTools::EvaluationFlags::avg,
                                     phase_field_component);
      double local_minimum_phi = std::numeric_limits<double>::max();
      double local_maximum_phi = -std::numeric_limits<double>::max();
      bool local_nonfinite_phi = false;
      for (const double phi : phase_field_values)
        {
          local_nonfinite_phi = local_nonfinite_phi || !std::isfinite(phi);
          if (std::isfinite(phi))
            {
              local_minimum_phi = std::min(local_minimum_phi, phi);
              local_maximum_phi = std::max(local_maximum_phi, phi);
            }
        }
      const unsigned int any_nonfinite_phi = Utilities::MPI::max(
        local_nonfinite_phi ? 1u : 0u, this->get_mpi_communicator());
      AssertThrow(any_nonfinite_phi == 0,
                  ExcMessage("Initial cohesive state encountered a non-finite phase field."));
      const double minimum_phi = Utilities::MPI::min(
        local_minimum_phi, this->get_mpi_communicator());
      const double maximum_phi = Utilities::MPI::max(
        local_maximum_phi, this->get_mpi_communicator());
      validate_normalization_phase_field_minimum(
        minimum_phi, "initial cohesive q projection");
      AssertThrow(maximum_phi <= 1.0,
                  ExcMessage("Initial cohesive state violates the physical upper phase-field "
                             "bound: maximum phi_h=" + Utilities::to_string(maximum_phi)
                             + ". The upper phase-field bound is not clipped."));

      const std::vector<unsigned int> chemical_field_indices =
        this->introspection().chemical_composition_field_indices();
      AssertDimension(fault_property_indices.chemical_compositions.size(),
                      chemical_field_indices.size());
      const std::map<types::particle_index, std::vector<double>> surface_compositions =
        interpolate_surface_chemical_compositions(
          this->get_reconstructed_fault_manager(),
          fault_property_indices.chemical_compositions);

      std::map<types::particle_index, double> particle_q;
      std::string local_error;
      unsigned int particle_index = 0;
      for (const auto &particle : particle_handler)
        {
          const auto surface_composition = surface_compositions.find(particle.get_id());
          if (!chemical_field_indices.empty()
              && surface_composition == surface_compositions.end())
            {
              ++particle_index;
              continue;
            }

          const std::vector<double> chemical_compositions =
            chemical_field_indices.empty()
            ? std::vector<double>()
            : surface_composition->second;
          const std::vector<double> material_fractions =
            MaterialUtilities::compute_composition_fractions(chemical_compositions);
          const double G = MaterialUtilities::average_value(
            material_fractions, elastic_shear_moduli, viscosity_averaging);
          const double H = particle.get_properties()[H_position];
          const double phi = std::max(phase_field_values[particle_index++], 0.0);

          double q = 0.0;
          if (!std::isfinite(H) || H < 0.0)
            {
              if (local_error.empty())
                local_error = "Initial cohesive q has inadmissible H at particle "
                              + Utilities::int_to_string(particle.get_id()) + ".";
            }
          else
            {
              const double degradation = phase_field_handler.energetic_degradation(
                material_fractions, phi);
              q = degradation * std::sqrt(2.0*G*H);
              if (!std::isfinite(q) && local_error.empty())
                local_error = "Initial cohesive q is non-finite at particle "
                              + Utilities::int_to_string(particle.get_id()) + ".";
            }
          particle_q.emplace(particle.get_id(), q);
        }

      // All ranks must report input errors before the projection collective.
      const unsigned int rank =
        Utilities::MPI::this_mpi_process(this->get_mpi_communicator());
      const unsigned int n_processes =
        Utilities::MPI::n_mpi_processes(this->get_mpi_communicator());
      const unsigned int error_rank = Utilities::MPI::min(
        local_error.empty() ? n_processes : rank, this->get_mpi_communicator());
      const std::string error = error_rank < n_processes
                                ? Utilities::MPI::broadcast(
                                    this->get_mpi_communicator(), local_error, error_rank)
                                : std::string();
      AssertThrow(error_rank == n_processes, ExcMessage(error));
      return particle_q;
    }


    // -----------------------------------------------------------------------------
    // Normalization-integral evaluation
    // -----------------------------------------------------------------------------

    template <int dim>
    void
    PhaseFieldFault<dim>::compute_normalization_integrals()
    {
      ReconstructedFaultManager<dim> &fault_manager =
        this->get_reconstructed_fault_manager();
      const std::vector<ReconstructedFault<dim>> &faults = fault_manager.get_faults();
      current_normalization_integrals.clear();
      current_normalization_integrals.resize(faults.size());
      current_minimum_raw_normalization_phase_field = numbers::signaling_nan<double>();
      if (faults.empty())
        return;

      const PhaseFieldHandler<dim> &phase_field_handler =
        this->get_phase_field_handler();

      project_surface_chemical_compositions();

      const std::vector<NormalizationProfile> profiles =
        build_owned_normalization_profiles();

      const unsigned int mpi_rank =
        Utilities::MPI::this_mpi_process(this->get_mpi_communicator());
      const unsigned int n_mpi_processes =
        Utilities::MPI::n_mpi_processes(this->get_mpi_communicator());

      const double length_scale = phase_field_handler.get_length_scale();
      double local_minimum_raw_phase_field = std::numeric_limits<double>::max();
      std::string local_minimum_raw_phase_field_context;

      const auto evaluate_points =
        [this](const std::vector<Point<dim>> &points)
        {
          return evaluate_normalization_points(points);
        };

      const auto integrand =
        [&](const NormalizationProfile &profile,
            const unsigned int side,
            const double zeta,
            const Point<dim> &point,
            const NormalizationPointSample &sample)
        {
          const auto projection = fault_manager.project_to_normal_profiles(point);
          AssertThrow(!projection.active
                      || projection.fault_index == profile.fault_index,
                      ExcMessage("Unsupported reconstructed-fault overlap during I_h evaluation: "
                                 "profile " + Utilities::int_to_string(profile.id)
                                 + " encountered fault "
                                 + Utilities::int_to_string(projection.fault_index)
                                 + " before its local tail terminated."));

          const std::string context =
            "fault " + Utilities::int_to_string(profile.fault_index)
            + ", segment " + Utilities::int_to_string(profile.segment_index)
            + ", profile " + Utilities::int_to_string(profile.id)
            + ", side " + (side == 0 ? std::string("+n") : std::string("-n"))
            + ", zeta=" + Utilities::to_string(zeta)
            + ", point=(" + Utilities::to_string(point[0])
            + "," + Utilities::to_string(point[1]) + ")";
          if (sample.phase_field < local_minimum_raw_phase_field)
            {
              local_minimum_raw_phase_field = sample.phase_field;
              local_minimum_raw_phase_field_context = context;
            }
          const double effective_phase_field =
            this->normalization_effective_phase_field(sample.phase_field, context);
          const double degradation = phase_field_handler.energetic_degradation(
            profile.material_fractions, effective_phase_field);
          return this->normalization_integrand(effective_phase_field, degradation, context);
        };

      const std::vector<double> profile_integrals =
        integrate_normalization_profiles(profiles,
                                         length_scale,
                                         normalization_quadrature_tolerance,
                                         normalization_tail_tolerance,
                                         this->get_mpi_communicator(),
                                         evaluate_points,
                                         integrand);

      current_minimum_raw_normalization_phase_field = Utilities::MPI::min(
        local_minimum_raw_phase_field, this->get_mpi_communicator());
      const unsigned int minimum_rank = Utilities::MPI::min(
        local_minimum_raw_phase_field
          == current_minimum_raw_normalization_phase_field
        ? mpi_rank
        : n_mpi_processes,
        this->get_mpi_communicator());
      const std::string minimum_context = Utilities::MPI::broadcast(
        this->get_mpi_communicator(), local_minimum_raw_phase_field_context,
        minimum_rank);
      validate_normalization_phase_field_minimum(
        current_minimum_raw_normalization_phase_field, minimum_context);

      project_normalization_integrals_to_fault(profiles, profile_integrals);
    }


    // -----------------------------------------------------------------------------
    // Normalization phase-field utilities
    // -----------------------------------------------------------------------------


    template <int dim>
    double
    PhaseFieldFault<dim>::normalization_effective_phase_field(
      const double raw_phase_field,
      const std::string &context)
    {
      AssertThrow(std::isfinite(raw_phase_field) && raw_phase_field <= 1.0,
                  ExcMessage("Internal phase-field invariant violation during I_h evaluation: "
                             "the raw physical phase field must be finite and no greater than "
                             "one, but phi_h=" + Utilities::to_string(raw_phase_field)
                             + " at " + context + ". The upper phase-field bound is not clipped."));
      return std::max(raw_phase_field, 0.0);
    }



    template <int dim>
    void
    PhaseFieldFault<dim>::validate_normalization_phase_field_minimum(
      const double minimum_raw_phase_field,
      const std::string &context)
    {
      AssertThrow(minimum_raw_phase_field
                  >= -normalization_phase_field_undershoot_tolerance,
                  ExcMessage("I_h phase-field undershoot exceeds the internal empirical "
                             "error-detection threshold: "
                             "minimum raw phi_h="
                             + Utilities::to_string(minimum_raw_phase_field)
                             + ", threshold="
                             + Utilities::to_string(
                                 normalization_phase_field_undershoot_tolerance)
                             + " at " + context
                             + ". Bounded negative samples are evaluated with "
                               "phi_eff=max(phi_h,0); the activation threshold is not used. "
                               "This guard is not a physical parameter, solver tolerance, "
                               "or convergence-control parameter."));
    }



    template <int dim>
    double
    PhaseFieldFault<dim>::normalization_integrand(
      const double phase_field,
      const double degradation,
      const std::string &context)
    {
      AssertThrow(std::isfinite(degradation) && degradation > 0.0,
                  ExcMessage("I_h singularity at " + context + ": phi="
                             + Utilities::to_string(phase_field) + ", g="
                             + Utilities::to_string(degradation)
                             + ". I_h requires a finite, strictly positive "
                               "degradation function."));
      const double value = 1.0 / degradation - 1.0;
      AssertThrow(std::isfinite(value),
                  ExcMessage("I_h singularity at " + context + ": phi="
                             + Utilities::to_string(phase_field) + ", g="
                             + Utilities::to_string(degradation)
                             + ". The value 1/g-1 is non-finite."));
      return value;
    }


    // -----------------------------------------------------------------------------
    // Adaptive normalization-profile integration
    // -----------------------------------------------------------------------------


    template <int dim>
    std::vector<double>
    PhaseFieldFault<dim>::integrate_normalization_profiles(
      const std::vector<NormalizationProfile> &profiles,
      const double length_scale,
      const double quadrature_tolerance,
      const double tail_tolerance,
      const MPI_Comm communicator,
      const NormalizationPointEvaluator &evaluate_points,
      const NormalizationIntegrandEvaluator &integrand)
    {
      // Each rank advances only its owned profile sides, but all ranks enter
      // the same batched point-evaluation collectives until every side has
      // satisfied either the domain-boundary or two-window tail criterion.
      std::vector<std::array<NormalizationSideState,2>> states(profiles.size());
      std::vector<Point<dim>> origin_points;
      origin_points.reserve(profiles.size());
      for (const NormalizationProfile &profile : profiles)
        origin_points.push_back(profile.origin);
      const std::vector<NormalizationPointSample> origin_samples =
        evaluate_points(origin_points);
      for (unsigned int i = 0; i < profiles.size(); ++i)
        {
          AssertThrow(origin_samples[i].found,
                      ExcMessage("The origin of reconstructed-fault I_h profile "
                                 + Utilities::int_to_string(profiles[i].id)
                                 + " was not found in the bulk mesh."));
          const double initial_width =
            0.5 * std::min(length_scale, origin_samples[i].cell_diameter);
          states[i][0].panel_width = initial_width;
          states[i][1].panel_width = initial_width;
        }

      const QGauss<1> quadrature_4(4);
      const QGauss<1> quadrature_8(8);

      while (true)
        {
          unsigned int local_incomplete_sides = 0;
          std::vector<Point<dim>> points;
          std::vector<NormalizationEvaluationRequest> requests;
          for (unsigned int p = 0; p < profiles.size(); ++p)
            for (unsigned int side = 0; side < 2; ++side)
              {
                NormalizationSideState &state = states[p][side];
                if (state.complete)
                  continue;
                ++local_incomplete_sides;

                NormalizationEvaluationRequest request;
                request.profile = p;
                request.side = side;
                request.boundary_probe = state.boundary_search;
                request.first_point = points.size();
                if (state.boundary_search)
                  {
                    const double midpoint =
                      0.5 * (state.boundary_low + state.boundary_high);
                    request.zeta.push_back(midpoint);
                    points.push_back(normalization_profile_point<dim>(
                      profiles[p], side, midpoint));
                  }
                else
                  {
                    for (unsigned int q = 0; q < quadrature_4.size(); ++q)
                      request.zeta.push_back(
                        state.panel_start
                        + state.panel_width * quadrature_4.point(q)[0]);
                    for (unsigned int q = 0; q < quadrature_8.size(); ++q)
                      request.zeta.push_back(
                        state.panel_start
                        + state.panel_width * quadrature_8.point(q)[0]);
                    request.zeta.push_back(state.panel_start + state.panel_width);
                    for (const double zeta : request.zeta)
                      points.push_back(normalization_profile_point<dim>(
                        profiles[p], side, zeta));
                  }
                requests.push_back(std::move(request));
              }

          const unsigned int global_incomplete_sides =
            Utilities::MPI::sum(local_incomplete_sides, communicator);
          if (global_incomplete_sides == 0)
            break;

          const std::vector<NormalizationPointSample> samples =
            evaluate_points(points);
          for (const NormalizationEvaluationRequest &request : requests)
            {
              const NormalizationProfile &profile = profiles[request.profile];
              NormalizationSideState &state = states[request.profile][request.side];
              if (request.boundary_probe)
                {
                  const double midpoint = request.zeta[0];
                  if (samples[request.first_point].found)
                    state.boundary_low = midpoint;
                  else
                    state.boundary_high = midpoint;
                  ++state.boundary_bisections;
                  AssertThrow(state.boundary_bisections <= 64,
                              ExcMessage("I_h domain-boundary bisection exceeded 64 iterations "
                                         "for profile " + Utilities::int_to_string(profile.id) + "."));

                  const double coordinate_scale =
                    std::max(1.0, normalization_profile_point<dim>(
                               profile, request.side, state.boundary_high).norm());
                  if (std::nextafter(state.boundary_low, state.boundary_high)
                      == state.boundary_high
                      || state.boundary_high-state.boundary_low
                         <= 16.0 * std::numeric_limits<double>::epsilon()
                            * coordinate_scale)
                    {
                      state.boundary_search = false;
                      if (state.boundary_low > state.panel_start)
                        {
                          state.panel_width = state.boundary_low-state.panel_start;
                          state.boundary_final_panel = true;
                        }
                      else
                        state.complete = true;
                    }
                  continue;
                }

              double first_missing = std::numeric_limits<double>::max();
              double last_found_before_missing = state.panel_start;
              const std::vector<double> &zeta = request.zeta;
              std::vector<unsigned int> order(request.zeta.size());
              std::iota(order.begin(), order.end(), 0);
              std::sort(order.begin(), order.end(),
                        [&zeta](const unsigned int a, const unsigned int b)
                        { return zeta[a] < zeta[b]; });
              for (const unsigned int i : order)
                if (!samples[request.first_point+i].found)
                  {
                    first_missing = request.zeta[i];
                    break;
                  }
                else
                  last_found_before_missing = request.zeta[i];

              if (first_missing < std::numeric_limits<double>::max())
                {
                  state.boundary_search = true;
                  state.boundary_low = last_found_before_missing;
                  state.boundary_high = first_missing;
                  state.boundary_bisections = 0;
                  continue;
                }

              double integral_4 = 0.0;
              double integral_8 = 0.0;
              double panel_cell_diameter = std::numeric_limits<double>::max();
              for (unsigned int q = 0; q < quadrature_4.size(); ++q)
                {
                  const unsigned int i = request.first_point + q;
                  integral_4 += quadrature_4.weight(q)
                                * integrand(profile, request.side, request.zeta[q],
                                            points[i], samples[i]);
                  panel_cell_diameter =
                    std::min(panel_cell_diameter, samples[i].cell_diameter);
                }
              for (unsigned int q = 0; q < quadrature_8.size(); ++q)
                {
                  const unsigned int local_i = quadrature_4.size() + q;
                  const unsigned int i = request.first_point + local_i;
                  integral_8 += quadrature_8.weight(q)
                                * integrand(profile, request.side,
                                            request.zeta[local_i], points[i], samples[i]);
                  panel_cell_diameter =
                    std::min(panel_cell_diameter, samples[i].cell_diameter);
                }
              integral_4 *= state.panel_width;
              integral_8 *= state.panel_width;

              if (std::abs(integral_8-integral_4)
                  > quadrature_tolerance
                    * std::max(std::abs(integral_8), length_scale))
                {
                  ++state.refinement_depth;
                  AssertThrow(state.refinement_depth <= 64,
                              ExcMessage("I_h panel refinement exceeded depth 64 for profile "
                                         + Utilities::int_to_string(profile.id) + "."));
                  state.panel_width *= 0.5;
                  continue;
                }

              AssertThrow(std::isfinite(integral_8),
                          ExcMessage("I_h panel quadrature produced an unusable integral."));
              state.integral += integral_8;
              state.window_span += state.panel_width;
              state.window_integral += integral_8;
              ++state.accepted_extensions;
              AssertThrow(state.accepted_extensions <= 4096,
                          ExcMessage("I_h tail extension exceeded 4096 accepted panels for profile "
                                     + Utilities::int_to_string(profile.id) + "."));

              if (state.boundary_final_panel)
                state.complete = true;
              else if (state.window_span >= length_scale)
                {
                  if (state.window_integral
                      <= tail_tolerance * std::max(state.integral, length_scale))
                    ++state.successive_small_windows;
                  else
                    state.successive_small_windows = 0;
                  state.window_span = 0.0;
                  state.window_integral = 0.0;
                  if (state.successive_small_windows >= 2)
                    state.complete = true;
                }

              if (!state.complete)
                {
                  state.panel_start += state.panel_width;
                  state.panel_width =
                    std::min({2.0*state.panel_width,
                              0.5*length_scale,
                              0.5*panel_cell_diameter});
                  state.refinement_depth = 0;
                }
            }
        }

      std::vector<double> integrals(profiles.size());
      for (unsigned int p = 0; p < profiles.size(); ++p)
        integrals[p] = states[p][0].integral + states[p][1].integral;
      return integrals;
    }


    // -----------------------------------------------------------------------------
    // Surface composition and normalization-profile construction
    // -----------------------------------------------------------------------------


    template <int dim>
    void
    PhaseFieldFault<dim>::project_surface_chemical_compositions()
    {
      const std::vector<unsigned int> &chemical_field_indices =
        this->introspection().chemical_composition_field_indices();
      AssertDimension(fault_property_indices.chemical_compositions.size(),
                      chemical_field_indices.size());
      if (chemical_field_indices.empty())
        return;

      ReconstructedFaultManager<dim> &fault_manager =
        this->get_reconstructed_fault_manager();

      std::vector<typename ReconstructedFaultManager<dim>::ParticlePropertyProjection>
        projections;
      projections.reserve(chemical_field_indices.size());
      for (unsigned int c = 0; c < chemical_field_indices.size(); ++c)
        {
          const auto &particle_property =
            this->get_parameters().mapped_particle_properties.at(
              chemical_field_indices[c]);
          const auto &fault_property =
            fault_manager.get_property_information()[
              fault_property_indices.chemical_compositions[c]];

          typename ReconstructedFaultManager<dim>::ParticlePropertyProjection projection;
          projection.particle_property_name = particle_property.first;
          projection.first_particle_component = particle_property.second;
          projection.fault_property_name = fault_property.name;
          projection.first_fault_component = 0;
          projection.n_components = 1;

          projections.push_back(std::move(projection));
        }

      fault_manager.project_particle_properties(projections);
    }



    template <int dim>
    std::vector<typename PhaseFieldFault<dim>::NormalizationPointSample>
    PhaseFieldFault<dim>::evaluate_normalization_points(
      const std::vector<Point<dim>> &points) const
    {
      const PhaseFieldHandler<dim> &phase_field_handler =
        this->get_phase_field_handler();
      Utilities::MPI::RemotePointEvaluation<dim> cache;
      cache.reinit(phase_field_handler.get_grid_cache(), points);
      const unsigned int phase_field_component =
        this->introspection().variable("phase_field").first_component_index;
      const std::vector<double> phase_field_values =
        VectorTools::point_values<1>(cache,
                                     this->get_dof_handler(),
                                     this->get_solution(),
                                     VectorTools::EvaluationFlags::avg,
                                     phase_field_component);

      const std::vector<double> cell_diameters =
        cache.template evaluate_and_process<double>(
          [](const ArrayView<double> &values,
             const typename Utilities::MPI::RemotePointEvaluation<dim>::CellData &cell_data)
          {
            for (const unsigned int cell_index : cell_data.cell_indices())
              {
                const double diameter =
                  cell_data.get_active_cell_iterator(cell_index)->diameter();
                ArrayView<double> cell_values =
                  cell_data.get_data_view(cell_index, values);
                std::fill(cell_values.begin(), cell_values.end(), diameter);
              }
          });

      const std::vector<unsigned int> &point_ptrs = cache.get_point_ptrs();
      std::vector<NormalizationPointSample> samples(points.size());
      for (unsigned int point = 0; point < points.size(); ++point)
        if (cache.point_found(point))
          {
            samples[point].found = true;
            samples[point].phase_field = phase_field_values[point];
            samples[point].cell_diameter = std::numeric_limits<double>::max();
            for (unsigned int entry = point_ptrs[point];
                 entry < point_ptrs[point+1]; ++entry)
              samples[point].cell_diameter = std::min(
                samples[point].cell_diameter, cell_diameters[entry]);
          }
      return samples;
    }



    template <int dim>
    std::vector<typename PhaseFieldFault<dim>::NormalizationProfile>
    PhaseFieldFault<dim>::build_owned_normalization_profiles() const
    {
      const ReconstructedFaultManager<dim> &fault_manager =
        this->get_reconstructed_fault_manager();
      const auto &fault_property_info = fault_manager.get_property_information();
      const auto &faults = fault_manager.get_faults();

      const QGauss<1> surface_quadrature(3);

      unsigned int n_profiles = 0;
      for (const ReconstructedFault<dim> &fault : faults)
        n_profiles += fault.n_cells() * surface_quadrature.size();

      const unsigned int rank =
        Utilities::MPI::this_mpi_process(this->get_mpi_communicator());
      const unsigned int n_processes =
        Utilities::MPI::n_mpi_processes(this->get_mpi_communicator());
      const unsigned int first_owned_profile = n_profiles * rank / n_processes;
      const unsigned int end_owned_profile = n_profiles * (rank + 1) / n_processes;

      std::vector<unsigned int> chemical_composition_positions;
      chemical_composition_positions.reserve(
        fault_property_indices.chemical_compositions.size());
      for (const unsigned int property_index :
           fault_property_indices.chemical_compositions)
        {
          AssertDimension(fault_property_info[property_index].n_components, 1);
          chemical_composition_positions.push_back(
            fault_property_info[property_index].position);
        }

      std::vector<NormalizationProfile> profiles;
      profiles.reserve(end_owned_profile - first_owned_profile);
      unsigned int profile_id = 0;
      for (unsigned int fault_index = 0; fault_index < faults.size(); ++fault_index)
        {
          const ReconstructedFault<dim> &fault = faults[fault_index];
          for (unsigned int segment = 0; segment < fault.n_cells(); ++segment)
            {
              const Tensor<1,dim> tangent =
                fault.vertex(segment + 1) - fault.vertex(segment);
              const double segment_length = tangent.norm();
              Tensor<1,dim> normal;
              normal[0] = -tangent[1] / segment_length;
              normal[1] = tangent[0] / segment_length;

              for (unsigned int q = 0; q < surface_quadrature.size(); ++q, ++profile_id)
                if (profile_id >= first_owned_profile && profile_id < end_owned_profile)
                  {
                    NormalizationProfile profile;
                    profile.id = profile_id;
                    profile.fault_index = fault_index;
                    profile.segment_index = segment;
                    profile.xi = surface_quadrature.point(q)[0];
                    profile.surface_weight = surface_quadrature.weight(q) * segment_length;
                    profile.origin = (1.0 - profile.xi) * fault.vertex(segment)
                                     + profile.xi * fault.vertex(segment + 1);
                    profile.normal = normal;

                    std::vector<double> chemical_compositions(
                      chemical_composition_positions.size());
                    for (unsigned int c = 0;
                         c < chemical_composition_positions.size(); ++c)
                      chemical_compositions[c] =
                        (1.0 - profile.xi)
                        * fault.get_properties(segment)[
                          chemical_composition_positions[c]]
                        + profile.xi
                        * fault.get_properties(segment + 1)[
                          chemical_composition_positions[c]];
                    profile.material_fractions =
                      MaterialUtilities::compute_composition_fractions(
                        chemical_compositions);

                    profiles.push_back(std::move(profile));
                  }
            }
        }

      return profiles;
    }


    // -----------------------------------------------------------------------------
    // Projection of normalization integrals
    // -----------------------------------------------------------------------------


    template <int dim>
    void
    PhaseFieldFault<dim>::project_normalization_integrals_to_fault(
      const std::vector<NormalizationProfile> &profiles,
      const std::vector<double> &profile_integrals)
    {
      const auto &faults = this->get_reconstructed_fault_manager().get_faults();
      struct FaultSystem
      {
        std::vector<double> diagonal;
        std::vector<double> off_diagonal;
        std::vector<double> rhs;
      };
      std::vector<FaultSystem> local_systems(faults.size());
      for (unsigned int fault = 0; fault < faults.size(); ++fault)
        {
          local_systems[fault].diagonal.assign(faults[fault].n_vertices(), 0.0);
          local_systems[fault].off_diagonal.assign(faults[fault].n_cells(), 0.0);
          local_systems[fault].rhs.assign(faults[fault].n_vertices(), 0.0);
        }

      for (unsigned int profile_index = 0;
           profile_index < profiles.size(); ++profile_index)
        {
          const NormalizationProfile &profile = profiles[profile_index];
          const double normalization = profile_integrals[profile_index];
          AssertThrow(std::isfinite(normalization) && normalization > 0.0,
                      ExcMessage("I_h profile " + Utilities::int_to_string(profile.id)
                                 + " produced a non-positive or non-finite integral."));
          FaultSystem &system = local_systems[profile.fault_index];
          const double shape[2] = {1.0-profile.xi, profile.xi};
          const unsigned int vertex = profile.segment_index;
          system.diagonal[vertex] += profile.surface_weight * shape[0] * shape[0];
          system.diagonal[vertex+1] += profile.surface_weight * shape[1] * shape[1];
          system.off_diagonal[vertex] += profile.surface_weight * shape[0] * shape[1];
          system.rhs[vertex] += profile.surface_weight * shape[0] * normalization;
          system.rhs[vertex+1] += profile.surface_weight * shape[1] * normalization;
        }

      unsigned int packed_size = 0;
      for (const ReconstructedFault<dim> &fault : faults)
        packed_size += 3 * fault.n_vertices() - 1;
      std::vector<double> local_values(packed_size, 0.0);
      unsigned int position = 0;
      for (const FaultSystem &system : local_systems)
        {
          std::copy(system.diagonal.begin(), system.diagonal.end(),
                    local_values.begin()+position);
          position += system.diagonal.size();
          std::copy(system.off_diagonal.begin(), system.off_diagonal.end(),
                    local_values.begin()+position);
          position += system.off_diagonal.size();
          std::copy(system.rhs.begin(), system.rhs.end(),
                    local_values.begin()+position);
          position += system.rhs.size();
        }
      std::vector<double> global_values(packed_size);
      Utilities::MPI::sum(local_values, this->get_mpi_communicator(), global_values);
      position = 0;
      for (unsigned int fault = 0; fault < faults.size(); ++fault)
        {
          const unsigned int n_vertices = faults[fault].n_vertices();
          std::vector<double> diagonal(global_values.begin()+position,
                                       global_values.begin()+position+n_vertices);
          position += n_vertices;
          std::vector<double> off_diagonal(global_values.begin()+position,
                                           global_values.begin()+position+n_vertices-1);
          position += n_vertices-1;
          std::vector<double> rhs(global_values.begin()+position,
                                  global_values.begin()+position+n_vertices);
          position += n_vertices;
          current_normalization_integrals[fault] =
            ReconstructedFaultUtilities::solve_tridiagonal_system(
              diagonal, off_diagonal, rhs);
        }
    }



    // -----------------------------------------------------------------------------
    // Material parameters and parsing
    // -----------------------------------------------------------------------------

    template <int dim>
    double
    PhaseFieldFault<dim>::
    compute_creep_viscosity(const std::vector<double> &volume_fractions,
                            const double               temperature) const
    {
      const unsigned int n_compositions = volume_fractions.size();

      const double dT_over_Tref =
        (temperature - reference_temperature) / reference_temperature;
      std::vector<double> composition_viscosities(n_compositions);
      for (unsigned int j = 0; j < n_compositions; ++j)
        composition_viscosities[j] = std::max(minimum_viscosity,
                                              std::min(maximum_viscosity,
                                                       reference_viscosities[j]
                                                       * std::exp(-thermal_viscosity_exponents[j]
                                                                  * dT_over_Tref)));

      return MaterialUtilities::average_value(volume_fractions,
                                              composition_viscosities,
                                              viscosity_averaging);
    }

    template <int dim>
    void
    PhaseFieldFault<dim>::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Phase field fault");
        {
          EquationOfState::MulticomponentIncompressible<dim>::declare_parameters(prm);
          Rheology::FaultFriction<dim>::declare_parameters(prm);

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

          prm.declare_entry("Use adiabatic pressure in fault friction", "false",
                            Patterns::Bool(),
                            "Use the adiabatic-model pressure as the complete normal pressure "
                            "in the reconstructed-fault friction term. If false, use the dynamic "
                            "pressure minus the deviatoric normal traction.");

          prm.declare_entry("Evolve phase field", "true",
                            Patterns::Bool(),
                            "Whether to evolve the phase field during the simulation. If set to "
                            "false, then the crack driving force and the direction vectors will be "
                            "frozen after initialization. This is useful when conducting benchmarks "
                            "with pre-existing faults.");

          prm.declare_entry("I h quadrature tolerance", "1e-8",
                            Patterns::Double(0),
                            "Relative tolerance used to compare the four- and eight-point "
                            "normal-profile quadrature rules.");

          prm.declare_entry("I h tail tolerance", "1e-8",
                            Patterns::Double(0),
                            "Relative integral tolerance used to terminate each normal-profile tail.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    template <int dim>
    void
    PhaseFieldFault<dim>::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Phase field fault");
        {
          // Equation of state parameters
          equation_of_state.initialize_simulator(this->get_simulator());
          equation_of_state.parse_parameters(prm);

          // Fault-friction parameters
          fault_friction.initialize_simulator(this->get_simulator());
          fault_friction.parse_parameters(prm);

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
          if (this->convert_output_to_years())
            initial_time_step *= year_in_seconds;

          evolve_phase_field = prm.get_bool("Evolve phase field");
          use_adiabatic_pressure_in_fault_friction =
            prm.get_bool("Use adiabatic pressure in fault friction");
          normalization_quadrature_tolerance =
            prm.get_double("I h quadrature tolerance");
          normalization_tail_tolerance =
            prm.get_double("I h tail tolerance");
          AssertThrow(numbers::is_finite(normalization_quadrature_tolerance)
                      && normalization_quadrature_tolerance > 0.0
                      && numbers::is_finite(normalization_tail_tolerance)
                      && normalization_tail_tolerance > 0.0,
                      ExcMessage("The I_h quadrature and tail tolerances must be positive."));

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

          AssertThrow(numbers::is_finite(minimum_viscosity) && minimum_viscosity > 0.0,
                      ExcMessage("The minimum viscosity of the phase field fault material model "
                                 "must be finite and positive."));
          AssertThrow(numbers::is_finite(maximum_viscosity)
                      && maximum_viscosity >= minimum_viscosity,
                      ExcMessage("The maximum viscosity of the phase field fault material model "
                                 "must be finite and no smaller than the minimum viscosity."));
          AssertThrow(numbers::is_finite(initial_time_step) && initial_time_step > 0.0,
                      ExcMessage("The initial time step of the phase field fault material model "
                                 "must be finite and positive."));
          for (const double viscosity : reference_viscosities)
            AssertThrow(numbers::is_finite(viscosity) && viscosity > 0.0,
                        ExcMessage("Every reference viscosity of the phase field fault material "
                                   "model must be finite and positive."));
          for (const double shear_modulus : elastic_shear_moduli)
            AssertThrow(numbers::is_finite(shear_modulus) && shear_modulus > 0.0,
                        ExcMessage("Every elastic shear modulus of the phase field fault material "
                                   "model must be finite and positive."));

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
    }
  }
}

// -----------------------------------------------------------------------------
// Material-model registration
// -----------------------------------------------------------------------------

namespace aspect
{
namespace MaterialModel
  {
    ASPECT_REGISTER_MATERIAL_MODEL(PhaseFieldFault,
                                   "phase field fault",
                                   "")
  }
}
