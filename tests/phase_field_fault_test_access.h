/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#ifndef _aspect_tests_phase_field_fault_test_access_h
#define _aspect_tests_phase_field_fault_test_access_h

#include <aspect/material_model/phase_field_fault.h>

namespace aspect
{
  namespace MaterialModel
  {
    namespace internal
    {
      /** Narrow test seam for private PhaseFieldFault Stage B-G operations. */
      template <int dim>
      class PhaseFieldFaultTestAccess
      {
        public:
          using PointSample = typename PhaseFieldFault<dim>::NormalizationPointSample;
          using NormalizationPointLookupCache =
            typename PhaseFieldFault<dim>::NormalizationPointLookupCache;
          using CohesiveResponse = typename PhaseFieldFault<dim>::CohesiveResponse;

          static const std::vector<std::vector<double>> &
          compute_normalization_integrals(PhaseFieldFault<dim> &model)
          {
            model.compute_normalization_integrals();
            return model.current_normalization_integrals;
          }

          static auto normalization_cache_status(const PhaseFieldFault<dim> &model)
          {
            const auto &cache = model.normalization_value_cache;
            return std::make_tuple(cache.valid, cache.hits, cache.integrations,
                                   cache.last_requested_points);
          }

          static void invalidate_normalization_cache(PhaseFieldFault<dim> &model)
          {
            model.invalidate_normalization_cache();
          }

          static double
          current_minimum_raw_normalization_phase_field(
            const PhaseFieldFault<dim> &model)
          {
            return model.current_minimum_raw_normalization_phase_field;
          }

          static const std::vector<std::vector<double>> &
          current_normalization_integrals(const PhaseFieldFault<dim> &model)
          {
            return model.current_normalization_integrals;
          }

          static void restore_diagnostic_normalization(
            PhaseFieldFault<dim> &model, const std::vector<std::vector<double>> &values)
          {
            model.current_normalization_integrals=values;
          }

          static double
          normalization_effective_phase_field(
            const double raw_phase_field,
            const std::string &context = "test profile")
          {
            return PhaseFieldFault<dim>::normalization_effective_phase_field(
              raw_phase_field, context);
          }

          static void
          validate_normalization_phase_field_minimum(
            const double minimum_raw_phase_field,
            const std::string &context = "test profile")
          {
            PhaseFieldFault<dim>::validate_normalization_phase_field_minimum(
              minimum_raw_phase_field, context);
          }

          static constexpr double
          normalization_phase_field_undershoot_tolerance()
          {
            return PhaseFieldFault<dim>::normalization_phase_field_undershoot_tolerance;
          }

          static double
          normalization_integrand(const double phase_field,
                                  const double degradation,
                                  const std::string &context = "test profile")
          {
            return PhaseFieldFault<dim>::normalization_integrand(
              phase_field, degradation, context);
          }

          static std::vector<double>
          integrate_normalization_profiles(
            const std::vector<Point<dim>> &origins,
            const std::vector<Tensor<1,dim>> &normals,
            const double length_scale,
            const double quadrature_tolerance,
            const double tail_tolerance,
            const MPI_Comm communicator,
            const typename PhaseFieldFault<dim>::NormalizationPointEvaluator &evaluate_points,
            const std::function<double(double)> &degradation)
          {
            AssertDimension(origins.size(), normals.size());
            std::vector<typename PhaseFieldFault<dim>::NormalizationProfile> profiles(origins.size());
            for (unsigned int i = 0; i < profiles.size(); ++i)
              {
                profiles[i].id = i;
                profiles[i].fault_index = 0;
                profiles[i].segment_index = 0;
                profiles[i].origin = origins[i];
                profiles[i].normal = normals[i];
              }

            const auto integrand =
              [&degradation](const typename PhaseFieldFault<dim>::NormalizationProfile &profile,
                             const unsigned int side,
                             const double zeta,
                             const Point<dim> &,
                             const typename PhaseFieldFault<dim>::NormalizationPointSample &sample)
              {
                const std::string context =
                  "test profile " + Utilities::int_to_string(profile.id)
                  + ", side " + Utilities::int_to_string(side)
                  + ", zeta=" + Utilities::to_string(zeta);
                const double phi = PhaseFieldFault<dim>::normalization_effective_phase_field(
                  sample.phase_field, context);
                return PhaseFieldFault<dim>::normalization_integrand(
                  phi, degradation(phi), context);
              };

            return PhaseFieldFault<dim>::integrate_normalization_profiles(
              profiles, length_scale, quadrature_tolerance, tail_tolerance,
              communicator, evaluate_points, integrand);
          }

          static CohesiveResponse
          compute_cohesive_response(const double beta,
                                    const double kappa,
                                    const double current_normalization_integral,
                                    const double previous_normalization_integral,
                                    const double previous_cohesive_traction,
                                    const double slip_rate,
                                    const double current_h,
                                    const double previous_h,
                                    const bool mature = false)
          {
            return PhaseFieldFault<dim>::compute_cohesive_response(
              {beta, kappa}, current_normalization_integral,
              previous_normalization_integral, previous_cohesive_traction,
              slip_rate, current_h, previous_h, mature);
          }

          static double
          compute_crack_driving_force_candidate(
            const double time_step,
            const double beta,
            const double kappa,
            const double current_degradation,
            const double previous_h,
            const double current_cohesive_traction,
            const double previous_cohesive_traction)
          {
            return PhaseFieldFault<dim>::compute_crack_driving_force_candidate(
              time_step, {beta, kappa}, current_degradation, previous_h,
              current_cohesive_traction, previous_cohesive_traction);
          }

          static void
          initialize_cohesive_state_from_initial_fields(PhaseFieldFault<dim> &model)
          {
            model.initialize_cohesive_state_from_initial_fields();
            model.compute_fault_surface_temperatures();
          }

          static void
          commit_cohesive_state(
            PhaseFieldFault<dim> &model,
            const std::vector<std::vector<double>> &cohesive_tractions)
          {
            model.validate_cohesive_state_commit(cohesive_tractions);
            model.commit_cohesive_state(cohesive_tractions);
          }

          static void
          scale_current_normalization_integrals(PhaseFieldFault<dim> &model,
                                                const double factor)
          {
            for (auto &fault_values : model.current_normalization_integrals)
              for (double &value : fault_values)
                value *= factor;
          }

          static const std::vector<typename ReconstructedFaultManager<dim>::
                                   ParticleScalarProjectionDiagnostics> &
          initial_cohesive_projection_diagnostics(const PhaseFieldFault<dim> &model)
          {
            return model.initial_cohesive_projection_diagnostics;
          }

          static const Rheology::FaultFriction<dim> &
          fault_friction(const PhaseFieldFault<dim> &model)
          {
            return model.fault_friction;
          }

          // Diagnostic use only: reuse the existing prescribed-normal-traction
          // constitutive/Jacobian path with a spatial reference-pressure plugin.
          static void prescribed_friction_normal(PhaseFieldFault<dim> &model, bool enabled)
          {
            model.use_adiabatic_pressure_in_fault_friction = enabled;
          }

          static std::vector<double>
          surface_material_fractions_at_vertex(
            const PhaseFieldFault<dim> &model,
            const ReconstructedFaultManager<dim> &fault_manager,
            const unsigned int fault_index,
            const unsigned int vertex)
          {
            const ReconstructedFault<dim> &fault =
              fault_manager.get_fault(fault_index);
            std::vector<double> compositions(
              model.fault_property_indices.chemical_compositions.size());
            for (unsigned int c = 0; c < compositions.size(); ++c)
              {
                const auto &property = fault_manager.get_property_information()[
                  model.fault_property_indices.chemical_compositions[c]];
                compositions[c] = fault.get_properties(vertex)[property.position];
              }
            return MaterialUtilities::compute_composition_fractions(compositions);
          }

          static std::pair<std::vector<double>, double>
          surface_material_state_at_projection(
            const PhaseFieldFault<dim> &model,
            const unsigned int fault_index,
            const unsigned int segment_index,
            const double xi)
          {
            const typename PhaseFieldFault<dim>::LocalizationResponse state =
              model.evaluate_reconstructed_fault_localization(
                fault_index, segment_index, xi, 0.5, 0.4,
                "transverse-temperature test");
            return {state.surface_material_fractions, state.surface_temperature};
          }

      };
    }
  }
}

#endif
