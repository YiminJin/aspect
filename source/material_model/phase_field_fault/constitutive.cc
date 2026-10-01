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
#include <aspect/geometry_model/box.h>
#include <aspect/plugins.h>
#include <boost/math/tools/roots.hpp>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_cartesian.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/numerics/vector_tools_evaluate.h>

#include <numeric>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <set>

namespace aspect
{
  namespace
  {
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
    template <int dim>
    typename PhaseFieldFault<dim>::MaxwellCoefficients
    PhaseFieldFault<dim>::
    compute_maxwell_coefficients(const double viscosity,
                                 const double shear_modulus,
                                 const double time_step)
    {
      const double exponent = -time_step * shear_modulus / viscosity;
      const double beta = std::exp(exponent);
      const double eta_ve = -viscosity * std::expm1(exponent);
      return {beta, eta_ve};
    }

    template <int dim>
    SymmetricTensor<2,dim>
    PhaseFieldFault<dim>::
    compute_maxwell_stress(
      const MaxwellCoefficients &coefficients,
      const SymmetricTensor<2,dim> &effective_bulk_strain_rate,
      const SymmetricTensor<2,dim> &old_stress)
    {
      return 2.0 * coefficients.eta_ve * effective_bulk_strain_rate
             + coefficients.beta * old_stress;
    }

    template <int dim>
    SymmetricTensor<2,dim>
    PhaseFieldFault<dim>::evaluate_frozen_maxwell_stress(
      const double temperature,
      const std::vector<double> &composition,
      const SymmetricTensor<2,dim> &old_stress) const
    {
      const auto fractions = MaterialUtilities::compute_only_composition_fractions(
        composition, this->introspection().chemical_composition_field_indices());
      const double eta = compute_creep_viscosity(fractions, temperature);
      const double G = MaterialUtilities::average_value(
        fractions, elastic_shear_moduli, viscosity_averaging);
      const double dt = this->get_timestep_number() > 0
                        ? this->get_timestep() : initial_time_step;
      return compute_maxwell_coefficients(eta, G, dt).beta * old_stress;
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
      const MaxwellCoefficients bulk_coefficients =
        compute_maxwell_coefficients(eta, G, time_step);
      const double surface_eta = compute_creep_viscosity(
        localization.surface_material_fractions,
        localization.surface_temperature);
      const double surface_G = MaterialUtilities::average_value(
        localization.surface_material_fractions, elastic_shear_moduli,
        viscosity_averaging);
      const MaxwellCoefficients surface_coefficients =
        compute_maxwell_coefficients(surface_eta, surface_G, time_step);
      const CohesiveResponse cohesive = compute_cohesive_response(
        surface_coefficients, localization.current_I_h, localization.previous_I_h,
        localization.previous_cohesive_traction, inputs.slip_rate,
        localization.current_h, localization.previous_h, mature_frictional_fault);

      const SymmetricTensor<2,dim> trial_stress = compute_maxwell_stress(
        bulk_coefficients, inputs.strain_rate, inputs.old_maxwell_stress);
      const SymmetricTensor<2,dim> stress_without_current_slip =
        trial_stress
        - 2.0 * bulk_coefficients.eta_ve * cohesive.history_correction
          * inputs.slip_tensor;
      const SymmetricTensor<2,dim> stress =
        stress_without_current_slip
        - 2.0 * bulk_coefficients.eta_ve * cohesive.localization_factor
          * inputs.slip_rate * inputs.slip_tensor;

      // Mechanics uses the committed nodal state, interpolated in the same
      // continuous Q1 space. Aging is applied once after the accepted solve.
      const double theta = fault_friction.has_state_variable()
                           ? interpolate_fault_scalar(
                               fault, inputs.segment_index, inputs.xi,
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
      const auto background = reconstructed_fault_background_tractions(
        inputs.fault_index, inputs.segment_index, inputs.xi);
      const double background_shear = background.first, background_normal = background.second;
      // The bulk unknowns/history may represent stress changes. Background
      // tractions are fixed surface data, not another bulk Maxwell load.
      const double sigma_n = background_normal + (use_adiabatic_pressure_in_fault_friction
                             ? this->get_adiabatic_conditions().pressure(inputs.position)
                             : inputs.dynamic_pressure
                               - stress * inputs.normal_tensor);
      const double damping = MaterialUtilities::average_value(
        localization.surface_material_fractions, radiation_damping_coefficients,
        MaterialUtilities::arithmetic);

      const double surface_resistance = cohesive.cohesive_traction;
      const double cohesive_tangent = mature_frictional_fault ? 0.0
                                     : surface_coefficients.eta_ve/localization.current_I_h;
      ReconstructedFaultPointResponse response;
      response.stress = stress;
      if (inputs.capture_stress_components)
        {
          response.stress_time_step = time_step;
          response.stress_beta = bulk_coefficients.beta;
          // Report the terms from this incoming-history evaluation. Do not
          // subtract two large stresses to infer a tiny current-step increment.
          response.stress_components[0] = bulk_coefficients.beta * inputs.old_maxwell_stress;
          response.stress_components[1] = 2.0 * bulk_coefficients.eta_ve * inputs.strain_rate;
          response.stress_components[2] =
            -2.0 * bulk_coefficients.eta_ve * cohesive.history_correction * inputs.slip_tensor
            -2.0 * bulk_coefficients.eta_ve * cohesive.localization_factor
              * inputs.slip_rate * inputs.slip_tensor;
        }
      response.normalization_integral = localization.current_I_h;
      response.shear_traction = background_shear + stress * inputs.slip_tensor;
      response.normal_traction = sigma_n;
      response.background_normal_traction = background_normal;
      response.cohesive_traction = surface_resistance;
      response.friction_traction = mu * sigma_n;
      response.damping_traction = damping * inputs.slip_rate;
      response.residual_density = response.shear_traction
                                  - surface_resistance
                                  - mu * sigma_n
                                  - damping * inputs.slip_rate;
      response.minus_derivative_wrt_slip_rate =
        2.0 * bulk_coefficients.eta_ve * cohesive.localization_factor
        * (inputs.slip_tensor * inputs.slip_tensor)
        + cohesive_tangent
        + sigma_n*dmu_dV
        + damping;
      response.eta_ve = bulk_coefficients.eta_ve;
      response.localization_factor = cohesive.localization_factor;
      response.friction_coefficient = mu;
      response.friction_derivative_wrt_slip_rate = dmu_dV;
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
      const MaxwellCoefficients bulk_coefficients =
        compute_maxwell_coefficients(eta, G, time_step);
      const double surface_eta = compute_creep_viscosity(
        localization.surface_material_fractions,
        localization.surface_temperature);
      const double surface_G = MaterialUtilities::average_value(
        localization.surface_material_fractions, elastic_shear_moduli,
        viscosity_averaging);
      const MaxwellCoefficients surface_coefficients =
        compute_maxwell_coefficients(surface_eta, surface_G, time_step);
      const CohesiveResponse cohesive = compute_cohesive_response(
        surface_coefficients, localization.current_I_h, localization.previous_I_h,
        localization.previous_cohesive_traction, 0.0,
        localization.current_h, localization.previous_h, mature_frictional_fault);

      ReconstructedFaultBulkPointResponse response;
      response.eta_ve = bulk_coefficients.eta_ve;
      response.localization_factor = cohesive.localization_factor;
      response.history_correction = cohesive.history_correction;
      return response;
    }

    template <int dim>
    double
    PhaseFieldFault<dim>::compute_crack_driving_force_candidate(
      const double time_step,
      const MaxwellCoefficients &surface_coefficients,
      const double current_degradation,
      const double previous_h,
      const double current_cohesive_traction,
      const double previous_cohesive_traction)
    {
      AssertThrow(time_step > 0.0 && surface_coefficients.eta_ve > 0.0,
                  ExcMessage("The cohesive-work update requires positive dt and eta_ve."));
      AssertThrow(std::isfinite(current_degradation)
                  && current_degradation > 0.0
                  && current_degradation <= 1.0,
                  ExcMessage("The cohesive-work update requires 0 < g <= 1."));

      if (current_degradation == 1.0)
        {
          AssertThrow(previous_h == 0.0,
                      ExcMessage("An exactly intact current fault point with nonzero "
                                 "previous h is inadmissible healing."));
          return time_step * current_cohesive_traction
                 * current_cohesive_traction
                 / (2.0*surface_coefficients.eta_ve);
        }

      // This factorization is algebraically the finite-step cohesive-work
      // expression and avoids squaring two nearly equal large terms directly.
      const double a = current_cohesive_traction/current_degradation;
      const double b = surface_coefficients.beta * previous_h
                       * previous_cohesive_traction
                       / (1.0-current_degradation);
      const double candidate = time_step*(a-b)*(a+b)
                               / (2.0*surface_coefficients.eta_ve);
      AssertThrow(std::isfinite(candidate),
                  ExcMessage("The finite-step cohesive-work update is non-finite."));
      return candidate;
    }

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
      AssertDimension(current_fault_surface_temperatures.size(),
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
      AssertDimension(current_fault_surface_temperatures[fault_index].size(),
                      fault.n_vertices());
      response.surface_temperature =
        (1.0-xi) * current_fault_surface_temperatures[fault_index][segment_index]
        + xi * current_fault_surface_temperatures[fault_index][segment_index+1];
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
      // The initial mechanical solve has no earlier physical profile. Use
      // the converged phi_0 with its initialized I_h snapshot, not the zero
      // old_solution placeholder. This changes evaluation, not retained history.
      const double previous_phi = this->get_timestep_number() == 0
                                  ? current_phi
                                  : normalization_effective_phase_field(
                                      previous_phase_field, context + " history");
      const PhaseFieldHandler<dim> &phase_field_handler =
        this->get_phase_field_handler();
      const double current_degradation =
        phase_field_handler.energetic_degradation(
          response.surface_material_fractions, current_phi);
      response.current_degradation = current_degradation;
      response.current_h = normalization_integrand(
        current_phi, current_degradation, context);
      // An unchanged local phase sample uses exactly the same material mixture.
      // Reuse h, but do not infer this from a frozen Eulerian profile alone:
      // advected parents can sample different coordinates at the next timestep.
      response.previous_h = current_phi == previous_phi
        ? response.current_h
        : normalization_integrand(previous_phi,
            phase_field_handler.energetic_degradation(
              response.surface_material_fractions, previous_phi), context + " history");
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
      const double previous_h,
      const bool mature)
    {
      AssertThrow(coefficients.eta_ve > 0.0,
                  ExcMessage("The cohesive viscoelastic viscosity eta_ve must be positive."));
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
        mature ? 0.0 : (coefficients.eta_ve * slip_rate
         + coefficients.beta * previous_I_h * previous_cohesive_traction)
        / current_I_h;
      response.localization_factor = current_h/current_I_h;
      if (mature)
        AssertThrow(previous_cohesive_traction == 0.0
                    && current_h == previous_h && current_I_h == previous_I_h,
                    ExcMessage("Mature friction requires zero cohesive history and a fixed profile."));
      response.history_correction = current_h == previous_h && current_I_h == previous_I_h
        ? 0.0
        : coefficients.beta * previous_cohesive_traction/coefficients.eta_ve
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


    // Registration remains in phase_field_fault.cc; instantiate only moved members.
#define INSTANTIATE(dim) \
    template PhaseFieldFault<dim>::MaxwellCoefficients PhaseFieldFault<dim>::compute_maxwell_coefficients(const double, const double, const double); \
    template SymmetricTensor<2,dim> PhaseFieldFault<dim>::compute_maxwell_stress(const MaxwellCoefficients &, const SymmetricTensor<2,dim> &, const SymmetricTensor<2,dim> &); \
    template SymmetricTensor<2,dim> PhaseFieldFault<dim>::evaluate_frozen_maxwell_stress(const double, const std::vector<double> &, const SymmetricTensor<2,dim> &) const; \
    template PhaseFieldFault<dim>::ReconstructedFaultPointResponse PhaseFieldFault<dim>::evaluate_reconstructed_fault_point(const ReconstructedFaultPointInputs &) const; \
    template PhaseFieldFault<dim>::ReconstructedFaultBulkPointResponse PhaseFieldFault<dim>::evaluate_reconstructed_fault_bulk_point(const ReconstructedFaultBulkPointInputs &) const; \
    template double PhaseFieldFault<dim>::compute_crack_driving_force_candidate(const double, const MaxwellCoefficients &, const double, const double, const double, const double); \
    template PhaseFieldFault<dim>::LocalizationResponse PhaseFieldFault<dim>::evaluate_reconstructed_fault_localization(const unsigned int, const unsigned int, const double, const double, const double, const std::string &) const; \
    template PhaseFieldFault<dim>::CohesiveResponse PhaseFieldFault<dim>::compute_cohesive_response(const MaxwellCoefficients &, const double, const double, const double, const double, const double, const double, const bool); \
    template double PhaseFieldFault<dim>::compute_creep_viscosity(const std::vector<double> &, const double) const;

    ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
  }
}
