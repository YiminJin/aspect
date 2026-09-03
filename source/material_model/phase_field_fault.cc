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
#include <aspect/reconstructed_fault.h>
#include <aspect/newton.h>
#include <aspect/simulator.h>
#include <aspect/postprocess/visualization.h>
#include <aspect/postprocess/particles.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_cartesian.h>
#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/numerics/vector_tools_evaluate.h>

#include <cstdint>
#include <cstring>

namespace aspect
{
  namespace
  {
    /**
     * Compare a double's representation with the generic-property sentinel
     * without executing a floating-point classification instruction. Generic
     * reconstructed-fault properties start as signaling NaNs, which intentionally
     * trip ASPECT's debug floating-point trap if passed to std::isfinite().
     */
    bool
    is_uninitialized_property_sentinel(const double value)
    {
      static_assert(sizeof(double) == sizeof(std::uint64_t),
                    "The cohesive-state sentinel check requires a 64-bit double.");
      static_assert(std::numeric_limits<double>::is_iec559,
                    "The cohesive-state sentinel check requires IEEE-754 doubles.");
      std::uint64_t bits;
      const double sentinel = numbers::signaling_nan<double>();
      std::uint64_t sentinel_bits;
      std::memcpy(&bits, &value, sizeof(bits));
      std::memcpy(&sentinel_bits, &sentinel, sizeof(sentinel_bits));
      return bits == sentinel_bits;
    }
  }

  namespace MaterialModel
  {
    template <int dim>
    typename PhaseFieldFault<dim>::MaxwellCoefficients
    PhaseFieldFault<dim>::
    compute_maxwell_coefficients(const double viscosity,
                                 const double shear_modulus,
                                 const double time_step)
    {
      AssertThrow(numbers::is_finite(viscosity) && viscosity > 0.0,
                  ExcMessage("The Maxwell viscosity must be finite and positive."));
      AssertThrow(numbers::is_finite(shear_modulus) && shear_modulus > 0.0,
                  ExcMessage("The Maxwell shear modulus must be finite and positive."));
      AssertThrow(numbers::is_finite(time_step) && time_step >= 0.0,
                  ExcMessage("The Maxwell time step must be finite and nonnegative."));

      const double exponent = -time_step * shear_modulus / viscosity;
      const double beta = std::exp(exponent);
      const double kappa = -viscosity * std::expm1(exponent);

      AssertThrow(numbers::is_finite(beta) && beta >= 0.0 && beta <= 1.0,
                  ExcMessage("The Maxwell relaxation factor is not finite or lies outside [0,1]."));
      AssertThrow(numbers::is_finite(kappa) && kappa >= 0.0,
                  ExcMessage("The Maxwell effective viscosity is not finite or is negative."));

      return {beta, kappa};
    }



    template <int dim>
    SymmetricTensor<2,dim>
    PhaseFieldFault<dim>::
    compute_maxwell_stress(
      const MaxwellCoefficients &coefficients,
      const SymmetricTensor<2,dim> &effective_bulk_strain_rate,
      const SymmetricTensor<2,dim> &previous_stress)
    {
      Assert(numbers::is_finite(coefficients.beta)
             && numbers::is_finite(coefficients.kappa),
             ExcInternalError());
      return 2.0 * coefficients.kappa * effective_bulk_strain_rate
             + coefficients.beta * previous_stress;
    }



    template <int dim>
    typename PhaseFieldFault<dim>::CohesiveResponse
    PhaseFieldFault<dim>::compute_cohesive_response(
      const MaxwellCoefficients &coefficients,
      const double current_normalization_integral,
      const double previous_normalization_integral,
      const double previous_cohesive_traction,
      const double slip_rate,
      const double current_h,
      const double previous_h)
    {
      AssertThrow(std::isfinite(coefficients.beta)
                  && coefficients.beta >= 0.0 && coefficients.beta <= 1.0,
                  ExcMessage("The cohesive relaxation factor beta must lie in [0,1]."));
      AssertThrow(std::isfinite(coefficients.kappa) && coefficients.kappa > 0.0,
                  ExcMessage("The cohesive effective viscosity kappa must be finite and positive."));
      AssertThrow(std::isfinite(current_normalization_integral)
                  && current_normalization_integral > 0.0,
                  ExcMessage("The current cohesive normalization integral must be finite and positive."));
      AssertThrow(std::isfinite(previous_normalization_integral)
                  && previous_normalization_integral > 0.0,
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
         + coefficients.beta * previous_normalization_integral
           * previous_cohesive_traction)
        / current_normalization_integral;
      response.localization_factor = current_h/current_normalization_integral;
      response.history_correction =
        coefficients.beta * previous_cohesive_traction/coefficients.kappa
        * (current_h * previous_normalization_integral
           / current_normalization_integral - previous_h);
      response.crack_strain_rate =
        response.localization_factor * slip_rate + response.history_correction;

      AssertThrow(std::isfinite(response.cohesive_traction)
                  && response.cohesive_traction >= 0.0
                  && std::isfinite(response.localization_factor)
                  && response.localization_factor >= 0.0
                  && std::isfinite(response.history_correction)
                  && std::isfinite(response.crack_strain_rate),
                  ExcMessage("The common cohesive law produced a non-finite or inadmissible response."));
      return response;
    }



    template <int dim>
    void
    PhaseFieldFault<dim>::
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
              const double eta = calculate_creep_viscosity(volume_fractions, in.temperature[i]);
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

      ReconstructedFaultManager<dim> &fault_manager =
        this->get_reconstructed_fault_manager();
      const unsigned int n_chemical_fields =
        this->introspection().n_chemical_composition_fields();
      if (n_chemical_fields > 0)
        fault_composition_property_index =
          fault_manager.register_property(
            "phase field fault chemical compositions", n_chemical_fields);

      cohesive_traction_property_index = fault_manager.register_property(
        "phase field fault cohesive traction", 1);
      previous_normalization_integral_property_index = fault_manager.register_property(
        "phase field fault previous I h", 1);
    }



    template <int dim>
    void
    PhaseFieldFault<dim>::commit_cohesive_state(
      const std::vector<std::vector<double>> &cohesive_tractions)
    {
      ReconstructedFaultManager<dim> &fault_manager =
        this->get_reconstructed_fault_manager();
      const auto &faults = fault_manager.get_faults();
      AssertDimension(cohesive_tractions.size(), faults.size());
      AssertDimension(current_normalization_integrals.size(), faults.size());
      AssertIndexRange(cohesive_traction_property_index,
                       fault_manager.get_property_information().size());
      AssertIndexRange(previous_normalization_integral_property_index,
                       fault_manager.get_property_information().size());
      const unsigned int cohesive_position =
        fault_manager.get_property_information()[cohesive_traction_property_index].position;
      const unsigned int normalization_position =
        fault_manager.get_property_information()[previous_normalization_integral_property_index].position;

      for (unsigned int fault = 0; fault < faults.size(); ++fault)
        {
          AssertDimension(cohesive_tractions[fault].size(), faults[fault].n_vertices());
          AssertDimension(current_normalization_integrals[fault].size(),
                          faults[fault].n_vertices());
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
      AssertIndexRange(cohesive_traction_property_index,
                       fault_manager.get_property_information().size());
      AssertIndexRange(previous_normalization_integral_property_index,
                       fault_manager.get_property_information().size());
      const unsigned int cohesive_position =
        fault_manager.get_property_information()[cohesive_traction_property_index].position;
      const unsigned int normalization_position =
        fault_manager.get_property_information()[previous_normalization_integral_property_index].position;

      bool any_initialized = false;
      bool any_uninitialized = false;
      for (const auto &fault : faults)
        for (unsigned int vertex = 0; vertex < fault.n_vertices(); ++vertex)
          {
            const ArrayView<const double> properties = fault.get_properties(vertex);
            const bool cohesive_is_uninitialized =
              is_uninitialized_property_sentinel(properties[cohesive_position]);
            const bool normalization_is_uninitialized =
              is_uninitialized_property_sentinel(properties[normalization_position]);
            AssertThrow(cohesive_is_uninitialized == normalization_is_uninitialized,
                        ExcMessage("A reconstructed fault contains partially initialized "
                                   "cohesive history."));
            if (!cohesive_is_uninitialized)
              {
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
      if (any_initialized)
        return;

      compute_normalization_integrals();

      const PhaseFieldHandler<dim> &phase_field_handler =
        this->get_phase_field_handler();
      const Particle::Manager<dim> &particle_manager =
        phase_field_handler.get_associated_particle_manager();
      const auto &particle_handler = particle_manager.get_particle_handler();
      const auto &particle_data = particle_manager.get_property_manager().get_data_info();
      AssertThrow(particle_data.fieldname_exists("crack_driving_force"),
                  ExcMessage("Initial cohesive state requires the particle property "
                             "'crack_driving_force'."));
      const unsigned int crack_driving_force_position =
        particle_data.get_position_by_field_name("crack_driving_force");

      const unsigned int phase_field_component =
        this->introspection().variable("phase_field").first_component_index;
      std::vector<Point<dim>> particle_positions;
      particle_positions.reserve(particle_handler.n_locally_owned_particles());
      for (const auto &particle : particle_handler)
        particle_positions.push_back(particle.get_location());

      Utilities::MPI::RemotePointEvaluation<dim> point_cache;
      point_cache.reinit(phase_field_handler.get_grid_cache(), particle_positions);
      const std::vector<double> phase_field_values =
        VectorTools::point_values<1>(point_cache,
                                     this->get_dof_handler(),
                                     this->get_solution(),
                                     VectorTools::EvaluationFlags::avg,
                                     phase_field_component);
      AssertDimension(phase_field_values.size(), particle_positions.size());

      double local_minimum_phase_field = std::numeric_limits<double>::max();
      double local_maximum_phase_field = -std::numeric_limits<double>::max();
      bool local_nonfinite_phase_field = false;
      for (const double phase_field : phase_field_values)
        {
          local_nonfinite_phase_field = local_nonfinite_phase_field
                                        || !std::isfinite(phase_field);
          if (std::isfinite(phase_field))
            {
              local_minimum_phase_field = std::min(local_minimum_phase_field, phase_field);
              local_maximum_phase_field = std::max(local_maximum_phase_field, phase_field);
            }
        }
      const unsigned int any_nonfinite_phase_field = Utilities::MPI::max(
        local_nonfinite_phase_field ? 1u : 0u, this->get_mpi_communicator());
      AssertThrow(any_nonfinite_phase_field == 0,
                  ExcMessage("Initial cohesive state encountered a non-finite phase field."));
      const double minimum_phase_field = Utilities::MPI::min(
        local_minimum_phase_field, this->get_mpi_communicator());
      const double maximum_phase_field = Utilities::MPI::max(
        local_maximum_phase_field, this->get_mpi_communicator());
      validate_normalization_phase_field_minimum(
        minimum_phase_field, "initial cohesive q projection");
      AssertThrow(maximum_phase_field <= 1.0,
                  ExcMessage("Initial cohesive state violates the physical upper phase-field "
                             "bound: maximum phi_h="
                             + Utilities::to_string(maximum_phase_field)
                             + ". The upper phase-field bound is not clipped."));

      const std::vector<unsigned int> chemical_field_indices =
        this->introspection().chemical_composition_field_indices();
      std::vector<unsigned int> chemical_property_positions;
      chemical_property_positions.reserve(chemical_field_indices.size());
      for (const unsigned int field_index : chemical_field_indices)
        {
          AssertThrow(this->get_parameters().compositional_field_methods[field_index]
                      == Parameters<dim>::AdvectionFieldMethod::particles,
                      ExcMessage("Initial cohesive state requires every chemical composition "
                                 "field to be advected by particles."));
          const auto mapped_property =
            this->get_parameters().mapped_particle_properties.find(field_index);
          AssertThrow(mapped_property
                      != this->get_parameters().mapped_particle_properties.end(),
                      ExcMessage("Initial cohesive state requires every chemical composition "
                                 "field to be mapped to a particle property."));
          AssertThrow(particle_data.fieldname_exists(mapped_property->second.first),
                      ExcMessage("A mapped chemical particle property is not registered."));
          const unsigned int n_components =
            particle_data.get_components_by_field_name(mapped_property->second.first);
          AssertThrow(mapped_property->second.second < n_components,
                      ExcMessage("A mapped chemical particle-property component is out of range."));
          chemical_property_positions.push_back(
            particle_data.get_position_by_field_name(mapped_property->second.first)
            + mapped_property->second.second);
        }

      std::map<types::particle_index, double> particle_q_values;
      std::string local_error;
      unsigned int particle_index = 0;
      for (const auto &particle : particle_handler)
        {
          const ArrayView<const double> properties = particle.get_properties();
          std::vector<double> chemical_compositions(chemical_property_positions.size());
          for (unsigned int c = 0; c < chemical_property_positions.size(); ++c)
            chemical_compositions[c] = properties[chemical_property_positions[c]];
          const std::vector<double> volume_fractions =
            MaterialUtilities::compute_composition_fractions(chemical_compositions);
          const double shear_modulus = MaterialUtilities::average_value(
            volume_fractions, elastic_shear_moduli, viscosity_averaging);
          const double H = properties[crack_driving_force_position];
          const double phi = std::max(phase_field_values[particle_index++], 0.0);
          double q = 0.0;
          if (!std::isfinite(shear_modulus) || shear_modulus <= 0.0
              || !std::isfinite(H) || H < 0.0)
            {
              if (local_error.empty())
                local_error = "Initial cohesive q has inadmissible G or H at particle "
                              + Utilities::int_to_string(particle.get_id()) + ".";
            }
          else
            {
              const double degradation = phase_field_handler.energetic_degradation(
                volume_fractions, phi);
              if (!std::isfinite(degradation) || degradation < 0.0)
                {
                  if (local_error.empty())
                    local_error = "Initial cohesive q has an inadmissible degradation "
                                  "factor at particle "
                                  + Utilities::int_to_string(particle.get_id()) + ".";
                }
              else
                {
                  q = degradation * std::sqrt(2.0*shear_modulus*H);
                  if ((!std::isfinite(q) || q < 0.0) && local_error.empty())
                    local_error = "Initial cohesive q is inadmissible at particle "
                                  + Utilities::int_to_string(particle.get_id()) + ".";
                }
            }
          particle_q_values.emplace(particle.get_id(), q);
        }
      AssertDimension(particle_index, phase_field_values.size());

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

      const auto projection = fault_manager.project_particle_scalar(particle_q_values);
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
      AssertThrow(std::isfinite(minimum_raw_phase_field)
                  && minimum_raw_phase_field
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
      AssertThrow(std::isfinite(phase_field)
                  && phase_field >= 0.0
                  && phase_field <= 1.0,
                  ExcMessage("Internal phase-field invariant violation during I_h evaluation: "
                             "the effective phase field must be finite and in [0,1], but phi_eff="
                             + Utilities::to_string(phase_field) + " at " + context + "."));
      AssertThrow(std::isfinite(degradation) && degradation > 0.0,
                  ExcMessage("I_h singularity at " + context + ": phi="
                             + Utilities::to_string(phase_field) + ", g="
                             + Utilities::to_string(degradation)
                             + ". I_h requires a finite, strictly positive "
                               "degradation function."));
      Assert(degradation <= 1.0 + 64.0 * std::numeric_limits<double>::epsilon(),
             ExcInternalError());
      const double value = 1.0 / degradation - 1.0;
      AssertThrow(std::isfinite(value) && value >= 0.0,
                  ExcMessage("I_h singularity at " + context + ": phi="
                             + Utilities::to_string(phase_field) + ", g="
                             + Utilities::to_string(degradation)
                             + ". The value 1/g-1 is negative or non-finite."));
      return value;
    }



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
      AssertThrow(dim == 2, ExcNotImplemented());
      AssertThrow(std::isfinite(length_scale) && length_scale > 0.0,
                  ExcMessage("I_h evaluation requires a positive finite phase-field length scale."));
      AssertThrow(std::isfinite(quadrature_tolerance) && quadrature_tolerance > 0.0
                  && std::isfinite(tail_tolerance) && tail_tolerance > 0.0,
                  ExcMessage("The I_h quadrature and tail tolerances must be positive and finite."));

      struct SideState
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

      std::vector<std::array<SideState,2>> states(profiles.size());
      std::vector<Point<dim>> origin_points;
      origin_points.reserve(profiles.size());
      for (const NormalizationProfile &profile : profiles)
        origin_points.push_back(profile.origin);
      const std::vector<NormalizationPointSample> origin_samples =
        evaluate_points(origin_points);
      AssertDimension(origin_samples.size(), profiles.size());
      for (unsigned int i = 0; i < profiles.size(); ++i)
        {
          AssertThrow(origin_samples[i].found,
                      ExcMessage("The origin of reconstructed-fault I_h profile "
                                 + Utilities::int_to_string(profiles[i].id)
                                 + " was not found in the bulk mesh."));
          AssertThrow(std::isfinite(origin_samples[i].cell_diameter)
                      && origin_samples[i].cell_diameter > 0.0,
                      ExcMessage("I_h profile origin has an invalid bulk-cell diameter."));
          const double initial_width =
            0.5 * std::min(length_scale, origin_samples[i].cell_diameter);
          states[i][0].panel_width = initial_width;
          states[i][1].panel_width = initial_width;
        }

      const QGauss<1> quadrature_4(4);
      const QGauss<1> quadrature_8(8);
      struct Request
      {
        unsigned int profile;
        unsigned int side;
        bool boundary_probe;
        unsigned int first_point;
        std::vector<double> zeta;
      };

      const auto point_at = [](const NormalizationProfile &profile,
                               const unsigned int side,
                               const double zeta)
      {
        return profile.origin
               + (side == 0 ? 1.0 : -1.0) * zeta * profile.normal;
      };

      while (true)
        {
          unsigned int local_incomplete_sides = 0;
          std::vector<Point<dim>> points;
          std::vector<Request> requests;
          for (unsigned int p = 0; p < profiles.size(); ++p)
            for (unsigned int side = 0; side < 2; ++side)
              {
                SideState &state = states[p][side];
                if (state.complete)
                  continue;
                ++local_incomplete_sides;

                Request request;
                request.profile = p;
                request.side = side;
                request.boundary_probe = state.boundary_search;
                request.first_point = points.size();
                if (state.boundary_search)
                  {
                    const double midpoint =
                      0.5 * (state.boundary_low + state.boundary_high);
                    request.zeta.push_back(midpoint);
                    points.push_back(point_at(profiles[p], side, midpoint));
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
                      points.push_back(point_at(profiles[p], side, zeta));
                  }
                requests.push_back(std::move(request));
              }

          const unsigned int global_incomplete_sides =
            Utilities::MPI::sum(local_incomplete_sides, communicator);
          if (global_incomplete_sides == 0)
            break;

          const std::vector<NormalizationPointSample> samples =
            evaluate_points(points);
          AssertDimension(samples.size(), points.size());
          for (const Request &request : requests)
            {
              const NormalizationProfile &profile = profiles[request.profile];
              SideState &state = states[request.profile][request.side];
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
                    std::max(1.0, point_at(profile, request.side,
                                           state.boundary_high).norm());
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
              std::vector<unsigned int> order(request.zeta.size());
              std::iota(order.begin(), order.end(), 0);
              std::sort(order.begin(), order.end(),
                        [&](const unsigned int a, const unsigned int b)
                        { return request.zeta[a] < request.zeta[b]; });
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

              AssertThrow(std::isfinite(integral_8) && integral_8 >= 0.0,
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



    template <int dim>
    void
    PhaseFieldFault<dim>::compute_normalization_integrals()
    {
      AssertThrow(dim == 2, ExcNotImplemented());

      ReconstructedFaultManager<dim> &fault_manager =
        this->get_reconstructed_fault_manager();
      const std::vector<ReconstructedFault<dim>> &faults = fault_manager.get_faults();
      current_normalization_integrals.clear();
      current_normalization_integrals.resize(faults.size());
      current_minimum_raw_normalization_phase_field =
        numbers::signaling_nan<double>();
      if (faults.empty())
        return;

      const std::vector<unsigned int> chemical_field_indices =
        this->introspection().chemical_composition_field_indices();
      unsigned int fault_composition_position = numbers::invalid_unsigned_int;
      if (!chemical_field_indices.empty())
        {
          AssertIndexRange(fault_composition_property_index,
                           fault_manager.get_property_information().size());
          const auto &property_information =
            fault_manager.get_property_information()[fault_composition_property_index];
          AssertDimension(property_information.n_components,
                          chemical_field_indices.size());
          fault_composition_position = property_information.position;

          std::vector<typename ReconstructedFaultManager<dim>::ParticlePropertyProjection>
            projections;
          projections.reserve(chemical_field_indices.size());
          for (unsigned int c = 0; c < chemical_field_indices.size(); ++c)
            {
              const unsigned int field_index = chemical_field_indices[c];
              AssertThrow(this->get_parameters().compositional_field_methods[field_index]
                          == Parameters<dim>::AdvectionFieldMethod::particles,
                          ExcMessage("Distributed I_h evaluation requires every chemical "
                                     "composition field to be advected by particles."));
              const auto mapped_property =
                this->get_parameters().mapped_particle_properties.find(field_index);
              AssertThrow(mapped_property
                          != this->get_parameters().mapped_particle_properties.end(),
                          ExcMessage("Distributed I_h evaluation requires every chemical "
                                     "composition field to be mapped to a particle property."));

              typename ReconstructedFaultManager<dim>::ParticlePropertyProjection projection;
              projection.particle_property_name = mapped_property->second.first;
              projection.first_particle_component = mapped_property->second.second;
              projection.fault_property_name = property_information.name;
              projection.first_fault_component = c;
              projection.n_components = 1;
              projections.push_back(projection);
            }
          fault_manager.project_particle_properties(projections);
        }

      const PhaseFieldHandler<dim> &phase_field_handler =
        this->get_phase_field_handler();
      const unsigned int phase_field_component =
        this->introspection().variable("phase_field").first_component_index;

      const auto evaluate_points =
        [&](const std::vector<Point<dim>> &points)
        {
          Utilities::MPI::RemotePointEvaluation<dim> cache;
          cache.reinit(phase_field_handler.get_grid_cache(), points);
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
          AssertDimension(phase_field_values.size(), points.size());
          AssertDimension(point_ptrs.size(), points.size() + 1);
          for (unsigned int i = 0; i < points.size(); ++i)
            if (cache.point_found(i))
              {
                samples[i].found = true;
                samples[i].phase_field = phase_field_values[i];
                samples[i].cell_diameter = std::numeric_limits<double>::max();
                for (unsigned int j = point_ptrs[i]; j < point_ptrs[i+1]; ++j)
                  samples[i].cell_diameter =
                    std::min(samples[i].cell_diameter, cell_diameters[j]);
                Assert(samples[i].cell_diameter < std::numeric_limits<double>::max(),
                       ExcInternalError());
              }
          return samples;
        };

      const QGauss<1> surface_quadrature(3);
      unsigned int n_profiles = 0;
      for (const ReconstructedFault<dim> &fault : faults)
        n_profiles += fault.n_cells() * surface_quadrature.size();

      const unsigned int mpi_rank =
        Utilities::MPI::this_mpi_process(this->get_mpi_communicator());
      const unsigned int n_mpi_processes =
        Utilities::MPI::n_mpi_processes(this->get_mpi_communicator());
      const unsigned int first_owned_profile = n_profiles * mpi_rank / n_mpi_processes;
      const unsigned int end_owned_profile = n_profiles * (mpi_rank + 1) / n_mpi_processes;

      std::vector<NormalizationProfile> profiles;
      profiles.reserve(end_owned_profile - first_owned_profile);
      unsigned int profile_id = 0;
      for (unsigned int fault_index = 0; fault_index < faults.size(); ++fault_index)
        {
          const ReconstructedFault<dim> &fault = faults[fault_index];
          for (unsigned int segment = 0; segment < fault.n_cells(); ++segment)
            {
              const Tensor<1,dim> tangent = fault.vertex(segment+1) - fault.vertex(segment);
              const double segment_length = tangent.norm();
              AssertThrow(std::isfinite(segment_length) && segment_length > 0.0,
                          ExcMessage("I_h evaluation encountered a degenerate fault segment."));
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
                    profile.origin = (1.0-profile.xi) * fault.vertex(segment)
                                     + profile.xi * fault.vertex(segment+1);
                    profile.normal = normal;

                    std::vector<double> chemical_compositions(chemical_field_indices.size());
                    for (unsigned int c = 0; c < chemical_field_indices.size(); ++c)
                      chemical_compositions[c] =
                        (1.0-profile.xi)
                        * fault.get_properties(segment)[fault_composition_position+c]
                        + profile.xi
                        * fault.get_properties(segment+1)[fault_composition_position+c];
                    profile.material_fractions =
                      MaterialUtilities::compute_composition_fractions(chemical_compositions);

                    double fraction_sum = 0.0;
                    for (const double fraction : profile.material_fractions)
                      {
                        AssertThrow(std::isfinite(fraction) && fraction >= 0.0,
                                    ExcMessage("I_h evaluation produced an invalid fault-surface "
                                               "material fraction."));
                        fraction_sum += fraction;
                      }
                    AssertThrow(std::abs(fraction_sum-1.0) <= 1.e-12,
                                ExcMessage("I_h fault-surface material fractions do not sum to one."));
                    profiles.push_back(std::move(profile));
                  }
            }
        }
      AssertDimension(profile_id, n_profiles);

      const double length_scale = phase_field_handler.get_length_scale();
      double local_minimum_raw_phase_field = std::numeric_limits<double>::max();
      std::string local_minimum_raw_phase_field_context;
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
      AssertIndexRange(minimum_rank, n_mpi_processes);
      const std::string minimum_context = Utilities::MPI::broadcast(
        this->get_mpi_communicator(), local_minimum_raw_phase_field_context,
        minimum_rank);
      validate_normalization_phase_field_minimum(
        current_minimum_raw_normalization_phase_field, minimum_context);

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
      for (unsigned int p = 0; p < profiles.size(); ++p)
        {
          const NormalizationProfile &profile = profiles[p];
          const double normalization = profile_integrals[p];
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
      AssertDimension(position, packed_size);
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
          for (const double normalization : current_normalization_integrals[fault])
            AssertThrow(std::isfinite(normalization),
                        ExcMessage("The consistent Q1 fault projection produced a non-finite "
                                   "nodal I_h coefficient."));
        }
      AssertDimension(position, packed_size);
    }



    template <int dim>
    double
    PhaseFieldFault<dim>::
    calculate_creep_viscosity(const std::vector<double> &volume_fractions,
                              const double               temperature) const
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
          normalization_quadrature_tolerance = prm.get_double("I h quadrature tolerance");
          normalization_tail_tolerance = prm.get_double("I h tail tolerance");
          AssertThrow(normalization_quadrature_tolerance > 0.0
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

// explicit instantiation
namespace aspect
{
namespace MaterialModel
  {
    ASPECT_REGISTER_MATERIAL_MODEL(PhaseFieldFault,
                                   "phase field fault",
                                   "")
  }
}
