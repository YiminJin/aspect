/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include <aspect/material_model/phase_field_fault.h>
#include <aspect/material_model/utilities.h>
#include <aspect/particle/manager.h>
#include <aspect/postprocess/interface.h>
#include <aspect/simulator_access.h>

#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/numerics/vector_tools_evaluate.h>

namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    class VerifyPhaseFieldFaultIh : public Interface<dim>,
      public SimulatorAccess<dim>
    {
      public:
        std::pair<std::string,std::string>
        execute(TableHandler &) override
        {
          const auto *const_model = dynamic_cast<const MaterialModel::PhaseFieldFault<dim> *>(
            &this->get_material_model());
          AssertThrow(const_model != nullptr, ExcInternalError());
          auto &model = const_cast<MaterialModel::PhaseFieldFault<dim> &>(*const_model);
          MaterialModel::internal::PhaseFieldFaultTestAccess<dim>
            ::initialize_cohesive_state_from_initial_fields(model);
          const auto &normalizations =
            MaterialModel::internal::PhaseFieldFaultTestAccess<dim>
              ::compute_normalization_integrals(model);
          const double minimum_raw_phase_field =
            MaterialModel::internal::PhaseFieldFaultTestAccess<dim>
              ::current_minimum_raw_normalization_phase_field(model);
          AssertThrow(std::isfinite(minimum_raw_phase_field)
                      && minimum_raw_phase_field >= -MaterialModel::internal::
                           PhaseFieldFaultTestAccess<dim>
                             ::normalization_phase_field_undershoot_tolerance(),
                      ExcMessage("Stage C lifecycle fixture violated the bounded-undershoot invariant."));

          unsigned int n_values = 0;
          double minimum = std::numeric_limits<double>::max();
          double maximum = 0.0;
          double sum = 0.0;
          for (const auto &fault_values : normalizations)
            for (const double value : fault_values)
              {
                AssertThrow(std::isfinite(value) && value > 0.0,
                            ExcMessage("Stage C produced an unusable nodal I_h."));
                ++n_values;
                minimum = std::min(minimum, value);
                maximum = std::max(maximum, value);
                sum += value;
              }
          AssertThrow(n_values > 0, ExcMessage("Stage C produced no nodal I_h values."));

          auto &fault_manager = this->get_reconstructed_fault_manager();
          const unsigned int cohesive_property = fault_manager.get_property_index(
            "phase field fault cohesive traction");
          const unsigned int previous_I_h_property = fault_manager.get_property_index(
            "phase field fault previous I h");
          const unsigned int cohesive_position =
            fault_manager.get_property_information()[cohesive_property].position;
          const unsigned int previous_I_h_position =
            fault_manager.get_property_information()[previous_I_h_property].position;
          for (unsigned int fault = 0; fault < fault_manager.get_faults().size(); ++fault)
            for (unsigned int vertex = 0;
                 vertex < fault_manager.get_fault(fault).n_vertices(); ++vertex)
              {
                const ArrayView<const double> properties =
                  fault_manager.get_fault(fault).get_properties(vertex);
                AssertThrow(std::isfinite(properties[cohesive_position])
                            && properties[cohesive_position] >= 0.0,
                            ExcMessage("Stage D produced an invalid initial cohesive traction."));
                AssertThrow(properties[previous_I_h_position]
                            == normalizations[fault][vertex],
                            ExcMessage("Stage D did not commit the initial I_h consistently."));
              }
          const auto &cohesive_diagnostics =
            MaterialModel::internal::PhaseFieldFaultTestAccess<dim>
              ::initial_cohesive_projection_diagnostics(model);
          AssertDimension(cohesive_diagnostics.size(), fault_manager.get_faults().size());
          for (const auto &diagnostic : cohesive_diagnostics)
            AssertThrow(std::isfinite(diagnostic.weighted_rms_residual)
                        && diagnostic.weighted_rms_residual >= 0.0
                        && std::isfinite(diagnostic.maximum_absolute_residual)
                        && diagnostic.maximum_absolute_residual >= 0.0
                        && std::isfinite(diagnostic.normalized_weighted_rms_residual)
                        && diagnostic.normalized_weighted_rms_residual >= 0.0
                        && std::isfinite(diagnostic.normalized_maximum_absolute_residual)
                        && diagnostic.normalized_maximum_absolute_residual >= 0.0,
                        ExcMessage("Stage D produced invalid cohesive-profile diagnostics."));

          // The fixture composition varies only normal to the horizontal fault.
          // Verify that the committed initial traction uses the projected surface
          // mixture, and that the particle-local mixture would give a measurably
          // different result.
          const unsigned int composition_property = fault_manager.get_property_index(
            "phase field fault chemical compositions");
          const auto surface_compositions =
            fault_manager.interpolate_property_at_particle_projections(
              composition_property);
          const auto &particle_manager =
            this->get_phase_field_handler().get_associated_particle_manager();
          const auto &particle_handler = particle_manager.get_particle_handler();
          const auto &particle_data =
            particle_manager.get_property_manager().get_data_info();
          const unsigned int H_position =
            particle_data.get_position_by_field_name("crack_driving_force");
          const unsigned int chemical_field =
            this->introspection().chemical_composition_field_indices().at(0);
          const auto mapped_property =
            this->get_parameters().mapped_particle_properties.find(chemical_field);
          AssertThrow(mapped_property
                      != this->get_parameters().mapped_particle_properties.end(),
                      ExcInternalError());
          const unsigned int local_composition_position =
            particle_data.get_position_by_field_name(mapped_property->second.first)
            + mapped_property->second.second;

          std::vector<Point<dim>> particle_positions;
          particle_positions.reserve(particle_handler.n_locally_owned_particles());
          for (const auto &particle : particle_handler)
            particle_positions.push_back(particle.get_location());
          Utilities::MPI::RemotePointEvaluation<dim> point_cache;
          point_cache.reinit(this->get_phase_field_handler().get_grid_cache(),
                             particle_positions);
          const std::vector<double> phase_field_values =
            VectorTools::point_values<1>(
              point_cache,
              this->get_dof_handler(),
              this->get_solution(),
              VectorTools::EvaluationFlags::avg,
              this->introspection().variable("phase_field").first_component_index);

          constexpr double background_G = 1.e10;
          constexpr double rock_G = 4.e10;
          std::map<types::particle_index, double> surface_q;
          std::map<types::particle_index, double> particle_local_q;
          double maximum_particle_q = 0.0;
          double maximum_particle_path_difference = 0.0;
          double minimum_local_composition = 1.0;
          double maximum_local_composition = 0.0;
          double maximum_surface_local_composition_difference = 0.0;
          unsigned int particle_index = 0;
          for (const auto &particle : particle_handler)
            {
              const auto surface = surface_compositions.find(particle.get_id());
              if (surface != surface_compositions.end())
                {
                  AssertDimension(surface->second.size(), 1);
                  const double local_composition =
                    particle.get_properties()[local_composition_position];
                  const double surface_composition = surface->second[0];
                  minimum_local_composition =
                    std::min(minimum_local_composition, local_composition);
                  maximum_local_composition =
                    std::max(maximum_local_composition, local_composition);
                  maximum_surface_local_composition_difference = std::max(
                    maximum_surface_local_composition_difference,
                    std::abs(surface_composition-local_composition));

                  const std::vector<double> surface_fractions =
                    MaterialModel::MaterialUtilities::compute_composition_fractions(
                      {surface_composition});
                  const std::vector<double> local_fractions =
                    MaterialModel::MaterialUtilities::compute_composition_fractions(
                      {local_composition});
                  const double surface_G =
                    surface_fractions[0]*background_G + surface_fractions[1]*rock_G;
                  const double local_G =
                    local_fractions[0]*background_G + local_fractions[1]*rock_G;
                  const double phi = MaterialModel::internal::PhaseFieldFaultTestAccess<dim>
                                     ::normalization_effective_phase_field(
                                       phase_field_values[particle_index],
                                       "Stage D surface-mixture regression");
                  const double H = particle.get_properties()[H_position];
                  const double surface_g =
                    this->get_phase_field_handler().energetic_degradation(
                      surface_fractions, phi);
                  const double local_g =
                    this->get_phase_field_handler().energetic_degradation(
                      local_fractions, phi);
                  const double surface_q_value =
                    surface_g*std::sqrt(2.0*surface_G*H);
                  const double local_q_value =
                    local_g*std::sqrt(2.0*local_G*H);
                  surface_q.emplace(particle.get_id(), surface_q_value);
                  particle_local_q.emplace(particle.get_id(), local_q_value);
                  maximum_particle_q = std::max(maximum_particle_q,
                                                std::abs(surface_q_value));
                  maximum_particle_path_difference = std::max(
                    maximum_particle_path_difference,
                    std::abs(surface_q_value-local_q_value));
                }
              ++particle_index;
            }
          AssertDimension(particle_index, phase_field_values.size());
          minimum_local_composition = Utilities::MPI::min(
            minimum_local_composition, this->get_mpi_communicator());
          maximum_local_composition = Utilities::MPI::max(
            maximum_local_composition, this->get_mpi_communicator());
          maximum_surface_local_composition_difference = Utilities::MPI::max(
            maximum_surface_local_composition_difference,
            this->get_mpi_communicator());
          maximum_particle_q = Utilities::MPI::max(
            maximum_particle_q, this->get_mpi_communicator());
          maximum_particle_path_difference = Utilities::MPI::max(
            maximum_particle_path_difference, this->get_mpi_communicator());
          AssertThrow(maximum_local_composition-minimum_local_composition > 0.2,
                      ExcMessage("The Stage D regression fixture does not contain a sufficient "
                                 "normal particle-composition gradient."));
          AssertThrow(maximum_surface_local_composition_difference > 0.1,
                      ExcMessage("The Stage D regression fixture does not distinguish the "
                                 "surface and particle-local compositions."));
          AssertThrow(maximum_particle_path_difference > 1.e-3*maximum_particle_q,
                      ExcMessage("The Stage D regression fixture does not distinguish the "
                                 "surface-mixture and particle-local initial-q values: maximum "
                                 "particle difference="
                                 + Utilities::to_string(maximum_particle_path_difference)
                                 + ", maximum surface-mixture q="
                                 + Utilities::to_string(maximum_particle_q) + "."));

          const auto expected_surface_projection =
            fault_manager.project_particle_scalar(surface_q);
          const auto old_particle_local_projection =
            fault_manager.project_particle_scalar(particle_local_q);
          double maximum_expected_traction = 0.0;
          double maximum_old_path_difference = 0.0;
          for (unsigned int fault = 0; fault < fault_manager.get_faults().size(); ++fault)
            for (unsigned int vertex = 0;
                 vertex < fault_manager.get_fault(fault).n_vertices(); ++vertex)
              {
                const double committed =
                  fault_manager.get_fault(fault).get_properties(vertex)[cohesive_position];
                const double expected = expected_surface_projection.nodal_values[fault][vertex];
                AssertThrow(std::abs(committed-expected)
                            <= 1.e-12*std::max(1.0, std::abs(expected)),
                            ExcMessage("Initial cohesive traction does not use the projected "
                                       "surface material mixture."));
                maximum_expected_traction = std::max(maximum_expected_traction,
                                                     std::abs(expected));
                maximum_old_path_difference = std::max(
                  maximum_old_path_difference,
                  std::abs(expected-old_particle_local_projection.nodal_values[fault][vertex]));
              }
          AssertThrow(maximum_old_path_difference > 1.e-4*maximum_expected_traction,
                      ExcMessage("The Stage D regression fixture does not distinguish the "
                                 "projected surface-mixture and particle-local-composition "
                                 "paths: maximum nodal difference="
                                 + Utilities::to_string(maximum_old_path_difference)
                                 + ", maximum expected traction="
                                 + Utilities::to_string(maximum_expected_traction) + "."));

          AssertThrow(std::isfinite(minimum) && std::isfinite(maximum)
                      && std::isfinite(sum), ExcInternalError());
          return {"Distributed I_h:", "verified"};
        }
    };

    ASPECT_REGISTER_POSTPROCESSOR(VerifyPhaseFieldFaultIh,
                                  "verify phase field fault I h",
                                  "Run a lifecycle smoke test of the private Stage C distributed "
                                  "I_h evaluator, Stage D cohesive initialization, and the "
                                  "profile-uniform surface-mixture invariant for initial q. "
                                  "Fault reconstruction accuracy is not tested.")
  }
}
