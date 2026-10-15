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

#include <aspect/material_model/phase_field_fault/fe_strain_internal.h>

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

          /** Value-only FE request must not allocate derivative data or consume mechanics. */
          static std::vector<double> fe_normalization_only(
            const PhaseFieldFault<dim> &model,const LinearAlgebra::BlockVector &trial)
          {
            const auto result=model.evaluate_fe_normalization(trial,false);
            AssertThrow(result.admissible && result.derivative.m()==0
                        && result.derivative_scale.m()==0 && result.integrals.empty()
                        && result.action.empty(),ExcMessage("Normalization-only request prepared derivatives."));
            return result.values;
          }

          static auto normalization_cache_status(const PhaseFieldFault<dim> &model)
          {
            const auto &cache = model.normalization_value_cache;
            return std::make_tuple(cache.valid, cache.hits, cache.integrations,
                                   cache.last_requested_points);
          }

          static auto normalization_profiles(const PhaseFieldFault<dim> &model, const bool all = false)
          {
            return model.build_owned_normalization_profiles(all);
          }

          static const auto &normalization_cells(const PhaseFieldFault<dim> &model)
          {
            return model.normalization_cell_cache;
          }

          static const auto &normalization_continuations(const PhaseFieldFault<dim> &model)
          {
            return model.boundary_continuations;
          }

          static const std::vector<double> &fe_slip_memory(const PhaseFieldFault<dim> &model)
          {
            return model.fe_strain_state->Z;
          }

          static auto fe_material_coefficients(const PhaseFieldFault<dim> &model)
          {
            return std::make_pair(model.fe_strain_state->G,model.fe_strain_state->eta);
          }

          /** Restore a controlled published history path, including derived caches. */
          static std::function<void()> save_fe_history(PhaseFieldFault<dim> &model)
          {
            const auto saved=std::make_shared<typename PhaseFieldFault<dim>::FEStrainState>(*model.fe_strain_state);
            return [&model,saved]()
            {
              model.fe_strain_state=std::make_shared<typename PhaseFieldFault<dim>::FEStrainState>(*saved);
            };
          }

          /** Reproject independent samples with either representation, restoring state. */
          static std::vector<double> probe_normalization_projection(
            PhaseFieldFault<dim> &model, const std::vector<double> &integrals,
            const bool lumped)
          {
            const auto saved = model.current_normalization_integrals;
            const bool saved_representation = model.lump_normalization_projection;
            try
              {
                model.lump_normalization_projection = lumped;
                model.project_normalization_integrals_to_fault(model.build_owned_normalization_profiles(), integrals);
                const auto result = model.current_normalization_integrals.at(0);
                model.current_normalization_integrals = saved;
                model.lump_normalization_projection = saved_representation;
                return result;
              }
            catch (...)
              {
                model.current_normalization_integrals = saved;
                model.lump_normalization_projection = saved_representation;
                throw;
              }
          }

          static std::vector<double> probe_normalization_projection(
            PhaseFieldFault<dim> &model, const std::vector<double> &integrals)
          {
            return probe_normalization_projection(model, integrals, model.lump_normalization_projection);
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
                                    const double eta_ve,
                                    const double current_normalization_integral,
                                    const double previous_normalization_integral,
                                    const double previous_cohesive_traction,
                                    const double slip_rate,
                                    const double current_h,
                                    const double previous_h,
                                    const bool mature = false)
          {
            return PhaseFieldFault<dim>::compute_cohesive_response(
            {beta, eta_ve}, current_normalization_integral,
            previous_normalization_integral, previous_cohesive_traction,
            slip_rate, current_h, previous_h, mature);
          }

          static double
          compute_crack_driving_force_candidate(
            const double time_step,
            const double beta,
            const double eta_ve,
            const double current_degradation,
            const double previous_h,
            const double current_cohesive_traction,
            const double previous_cohesive_traction)
          {
            return PhaseFieldFault<dim>::compute_crack_driving_force_candidate(
                     time_step, {beta, eta_ve}, current_degradation, previous_h,
                     current_cohesive_traction, previous_cohesive_traction);
          }

          /** Native coefficients for isolated prescribed-traction path tests. */
          static double recoverable_phase_driving_force(const double G, const double g, const double q)
          {
            return PhaseFieldFault<dim>::compute_recoverable_phase_driving_force(G,g,q);
          }

          static bool uses_surface_history(const PhaseFieldFault<dim> &m)
          {
            return m.surface_cohesive_history;
          }
          static std::function<void()> diagnostic_fe_reference(PhaseFieldFault<dim> &m,
                                                               const LinearAlgebra::BlockVector &reference,const std::vector<double> &Z,const std::vector<double> &I)
          {
            const auto saved=m.fe_strain_state;
            m.set_fe_diagnostic_reference(reference,Z,I);
            return [&m,saved]()
            {
              m.fe_strain_state=saved;
            };
          }

          /** Stage-only native FE representation probe; publishes no history. */
          static auto prepared_fe_state(const PhaseFieldFault<dim> &model,
                                        const LinearAlgebra::BlockVector &bulk)
          {
            std::vector<std::vector<double>> rates(1,
                                                    model.get_reconstructed_fault_manager().get_slip_rate(0));
            const auto q=model.compute_surface_cohesive_candidates(model.get_timestep(),rates);
            return model.prepare_fe_strain_commit(bulk,q);
          }

          template<class Update>
          static auto prepared_fe_stress(const PhaseFieldFault<dim> &model,
                                         const Update &update,const CellId &cell,
                                         const Point<dim> &point,
                                         double *evaluation_scale = nullptr)
          {
            return model.candidate_fe_strain_stress(update,cell,point,evaluation_scale);
          }
          static bool evolves_history(const PhaseFieldFault<dim> &m)
          {
            return m.evolve_phase_field;
          }
          static double committed_q(const PhaseFieldFault<dim> &m,unsigned int f,unsigned int s,double xi)
          {
            return m.evaluate_reconstructed_fault_localization(f,s,xi,0.,0.,"test").previous_cohesive_traction;
          }
          static auto surface_candidates(const PhaseFieldFault<dim> &m, double dt,
                                         const std::vector<std::vector<double>> &v)
          {
            return m.compute_surface_cohesive_candidates(dt,v);
          }
          static void set_uniform_viscosity(PhaseFieldFault<dim> &m,double eta)
          {
            std::fill(m.reference_viscosities.begin(),m.reference_viscosities.end(),eta);
            m.maximum_viscosity=std::max(m.maximum_viscosity,eta);
          }
          static auto viscosity_state(const PhaseFieldFault<dim> &m)
          {
            return std::make_pair(m.reference_viscosities,m.maximum_viscosity);
          }
          static void restore_viscosity(PhaseFieldFault<dim> &m,
                                        const std::pair<std::vector<double>,double> &v)
          {
            m.reference_viscosities=v.first;
            m.maximum_viscosity=v.second;
          }

          static bool uses_recoverable_history(const PhaseFieldFault<dim> &model)
          {
            return model.recoverable_phase_history;
          }

          static double surface_shear_modulus(const PhaseFieldFault<dim> &model,
                                              const unsigned int fault, const unsigned int segment,
                                              const double xi)
          {
            const auto state=surface_material_state_at_projection(model,fault,segment,xi);
            return MaterialUtilities::average_value(state.first,model.elastic_shear_moduli,
                                                    model.viscosity_averaging);
          }

          static std::pair<double,double>
          maxwell_coefficients(const double viscosity, const double modulus,
                               const double dt)
          {
            const auto c = PhaseFieldFault<dim>::compute_maxwell_coefficients(
                             viscosity, modulus, dt);
            return {c.beta, c.eta_ve};
          }

          /** Exercise the production constitutive update in prescribed-flow tests. */
          static SymmetricTensor<2,dim>
          maxwell_stress(const double viscosity, const double modulus, const double dt,
                         const SymmetricTensor<2,dim> &strain_rate,
                         const SymmetricTensor<2,dim> &old_stress)
          {
            return PhaseFieldFault<dim>::compute_maxwell_stress(
                     PhaseFieldFault<dim>::compute_maxwell_coefficients(viscosity, modulus, dt),
                     strain_rate, old_stress);
          }

          /** Read native surface coefficients without changing any history. */
          static std::pair<double,double>
          surface_maxwell_coefficients(const PhaseFieldFault<dim> &model,
                                       const unsigned int fault,
                                       const unsigned int segment,
                                       const double xi, const double dt)
          {
            const auto state = surface_material_state_at_projection(model, fault, segment, xi);
            const auto coefficients = model.compute_maxwell_coefficients(
                                        model.compute_creep_viscosity(state.first, state.second),
                                        MaterialUtilities::average_value(state.first, model.elastic_shear_moduli,
                                                                         model.viscosity_averaging), dt);
            return {coefficients.beta, coefficients.eta_ve};
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
