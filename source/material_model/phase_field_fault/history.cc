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
  // -----------------------------------------------------------------------------
  // Shared implementation helper and file-local helper types
  // -----------------------------------------------------------------------------

  // Used by both history preparation and normalization. Keep one definition;
  // normalization.cc declares it privately without adding a public header API.
  namespace internal
  {
    void
    throw_if_history_error(const std::string &local_error, const MPI_Comm communicator)
    {
      const unsigned int n_ranks = Utilities::MPI::n_mpi_processes(communicator);
      const unsigned int rank = Utilities::MPI::this_mpi_process(communicator);
      const unsigned int error_rank = Utilities::MPI::min(
        local_error.empty() ? n_ranks : rank, communicator);
      if (error_rank < n_ranks)
        {
          const std::string error = Utilities::MPI::broadcast(communicator, local_error, error_rank);
          AssertThrow(false, ExcMessage(error));
        }
    }


  }

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


    template <int dim>
    bool
    fault_scalar_property_is_initialized(
      const std::vector<ReconstructedFault<dim>> &faults,
      const unsigned int position,
      const std::string &name)
    {
      bool any_initialized = false;
      bool any_uninitialized = false;
      for (const ReconstructedFault<dim> &fault : faults)
        for (unsigned int vertex = 0; vertex < fault.n_vertices(); ++vertex)
          if (fault.property_value_is_initialized(vertex, position))
            {
              const double value = fault.get_properties(vertex)[position];
              AssertThrow(std::isfinite(value) && value > 0.0,
                          ExcMessage("Stored reconstructed-fault property <" + name
                                     + "> must be finite and positive."));
              any_initialized = true;
            }
          else
            any_uninitialized = true;

      AssertThrow(!(any_initialized && any_uninitialized),
                  ExcMessage("Reconstructed-fault property <" + name
                             + "> is only partially initialized."));
      return any_initialized;
    }


    template <int dim>
    void
    validate_initial_fault_state_mapping(
      const Introspection<dim> &introspection,
      const Parameters<dim> &parameters)
    {
      const auto &generic_fields = introspection.get_indices_for_fields_of_type(
        CompositionalFieldDescription::generic);
      unsigned int matching_fields = 0;
      for (const unsigned int field : generic_fields)
        {
          const auto mapping = parameters.mapped_particle_properties.find(field);
          if (mapping != parameters.mapped_particle_properties.end()
              && mapping->second.first == "phase field fault state")
            {
              AssertThrow(mapping->second.second == 0,
                          ExcMessage("The generic compositional field supplying "
                                     "initial fault state Theta must map to component "
                                     "zero of particle property 'phase field fault state'."));
              AssertThrow(parameters.compositional_field_methods[field]
                          == Parameters<dim>::AdvectionFieldMethod::particles,
                          ExcMessage("The generic compositional field supplying initial "
                                     "fault state Theta must be advected by particles."));
              ++matching_fields;
            }
        }
      AssertThrow(matching_fields == 1,
                  ExcMessage("Rate-and-state reconstructed-fault friction requires "
                             "exactly one generic particle-advected compositional "
                             "field mapped to particle property 'phase field fault "
                             "state', component zero."));
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


  }

  namespace MaterialModel
  {
    using aspect::internal::throw_if_history_error;
    template <int dim>
    void
    PhaseFieldFault<dim>::initialize()
    {
      if (!this->get_parameters().reconstruct_faults)
        return;
      normalization_cell_cache.change_connection = this->get_triangulation().signals.any_change.connect(
        [this]() { normalization_cell_cache.mesh_changed = true; });
      normalization_cell_cache.create_connection = this->get_triangulation().signals.create.connect(
        [this]() { normalization_cell_cache.mesh_changed = true; });
      performance_timer = std::make_unique<TimerOutput>(
        std::cout,
        std::getenv("ASPECT_FAULT_PERFORMANCE") && this->get_pcout().is_active()
        ? TimerOutput::summary : TimerOutput::never, TimerOutput::wall_times);
      this->get_signals().post_resume_load_user_data.connect(
        [this](parallel::distributed::Triangulation<dim> &)
        {
          invalidate_normalization_cache();
          restore_frozen_normalization_after_restart = mature_frictional_fault && !evolve_phase_field;
          AssertThrow(this->get_reconstructed_fault_manager().has_property(
                        "mature fault reference geometry") == mature_frictional_fault,
                      ExcMessage("Cannot change cohesive/mature fault mode on restart."));
        });

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
        {
          fault_property_indices.state = fault_manager.register_property(
            "phase field fault state", 1);
          if (this->get_parameters().nonlinear_solver
              == Parameters<dim>::NonlinearSolver::single_Advection_iterated_Newton_Stokes)
            validate_initial_fault_state_mapping(
              this->introspection(), this->get_parameters());
        }
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
      if (mature_frictional_fault)
        fault_manager.register_property("mature fault reference geometry", dim);
    }

    template <int dim>
    void
    PhaseFieldFault<dim>::prepare_reconstructed_fault_mechanical_solve()
    {
      TimerOutput::Scope coarse_timer(this->get_computing_timer(), "Fault: Property preparation");
      TimerOutput::Scope timer(*performance_timer, "Fault: Property preparation");
      AssertThrow(dim == 2, ExcNotImplemented());
      ReconstructedFaultManager<dim> &fault_manager =
        this->get_reconstructed_fault_manager();
      const auto &faults = fault_manager.get_faults();
      AssertThrow(!faults.empty(),
                  ExcMessage("A reconstructed-fault mechanical solve requires fault geometry."));

      const bool fresh_timestep_zero =
        this->get_timestep_number() == 0
        && !this->get_parameters().resume_computation;

      // This marker is both checkpoint mode provenance and the fixed geometry
      // contract. Only fresh initialization may populate it.
      if (mature_frictional_fault)
        {
          const auto position = fault_manager.get_property_information()[
            fault_manager.get_property_index("mature fault reference geometry")].position;
          for (unsigned int f=0; f<faults.size(); ++f)
            for (unsigned int v=0; v<faults[f].n_vertices(); ++v)
              for (unsigned int d=0; d<dim; ++d)
                {
                  if (fresh_timestep_zero && !faults[f].property_value_is_initialized(v,position+d))
                    fault_manager.get_fault(f).get_properties(v)[position+d]=faults[f].vertex(v)[d];
                  AssertThrow(faults[f].get_properties(v)[position+d]==faults[f].vertex(v)[d],
                              ExcMessage("Mature frictional faults require fixed reconstructed geometry."));
                }
        }

      // Validate/reuse the transient normalization profile for this solve. Missing
      // cohesive history may be constructed only for a genuinely fresh model.
      const unsigned int cohesive_position =
        fault_manager.get_property_information()[
          fault_property_indices.cohesive_traction].position;
      const unsigned int normalization_position =
        fault_manager.get_property_information()[
          fault_property_indices.previous_normalization_integral].position;
      const bool cohesive_is_initialized = cohesive_history_is_initialized(
        faults, cohesive_position, normalization_position);
      AssertThrow(cohesive_is_initialized || fresh_timestep_zero,
                  ExcMessage("Restarted or later-time reconstructed-fault mechanics "
                             "requires complete committed cohesive history."));
      if (!cohesive_is_initialized)
        initialize_cohesive_state_from_initial_fields();
      else
        compute_normalization_integrals();
      compute_fault_surface_temperatures();

      // Rate-and-state friction needs one complete positive Theta field. On a
      // fresh model it is projected from the user-mapped particle field;
      // restart and later-time paths must retain the committed history.
      if (fault_friction.has_state_variable())
        {
          const unsigned int state_position =
            fault_manager.get_property_information()[fault_property_indices.state].position;
          const bool state_is_initialized = fault_scalar_property_is_initialized(
            faults, state_position, "phase field fault state");
          AssertThrow(state_is_initialized || fresh_timestep_zero,
                      ExcMessage("Restarted or later-time rate-and-state fault mechanics "
                                 "requires complete committed Theta history."));
          if (!state_is_initialized)
            {
              validate_initial_fault_state_mapping(
                this->introspection(), this->get_parameters());
              fault_manager.project_particle_properties(
              {
                {"phase field fault state", 0,
                 "phase field fault state", 0, 1}
              });
            }
          fault_scalar_property_is_initialized(
            faults, state_position, "phase field fault state");
        }

      // V is the only Stage-I nonlinear constitutive variable. Initialize its
      // committed/current state at the admissible lower bound only at fresh t=0.
      if (!fault_manager.slip_rates_are_initialized())
        {
          AssertThrow(fresh_timestep_zero,
                      ExcMessage("Restarted or later-time reconstructed-fault mechanics "
                                 "requires committed slip-rate state."));
          for (unsigned int fault = 0; fault < faults.size(); ++fault)
            fault_manager.initialize_slip_rate(
              fault,
              std::vector<double>(faults[fault].n_vertices(),
                                  minimum_fault_slip_rate()));
        }

      // Newton may start only after all pointwise constitutive inputs form one
      // complete frozen state.
      validate_reconstructed_fault_constitutive_state();
    }

    template <int dim>
    void
    PhaseFieldFault<dim>::compute_fault_surface_temperatures()
    {
      const auto &faults = this->get_reconstructed_fault_manager().get_faults();
      std::vector<Point<dim>> points;
      for (const auto &fault : faults)
        for (unsigned int vertex = 0; vertex < fault.n_vertices(); ++vertex)
          points.push_back(fault.vertex(vertex));

      Utilities::MPI::RemotePointEvaluation<dim> point_cache;
      point_cache.reinit(this->get_phase_field_handler().get_grid_cache(), points);
      const std::vector<double> temperatures = VectorTools::point_values<1>(
        point_cache,
        this->get_dof_handler(),
        this->get_solution(),
        VectorTools::EvaluationFlags::avg,
        this->introspection().component_indices.temperature);

      current_fault_surface_temperatures.clear();
      current_fault_surface_temperatures.resize(faults.size());
      unsigned int point = 0;
      for (unsigned int fault = 0; fault < faults.size(); ++fault)
        {
          current_fault_surface_temperatures[fault].resize(
            faults[fault].n_vertices());
          for (double &temperature : current_fault_surface_temperatures[fault])
            {
              temperature = temperatures[point++];
              AssertThrow(std::isfinite(temperature),
                          ExcMessage("Fault-surface temperature is non-finite."));
            }
        }
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
      AssertThrow(current_fault_surface_temperatures.size() == faults.size(),
                  ExcMessage("Current reconstructed-fault surface temperature has not "
                             "been computed."));
      for (unsigned int fault_index = 0;
           fault_index < faults.size(); ++fault_index)
        {
          const ReconstructedFault<dim> &fault = faults[fault_index];
          AssertThrow(current_normalization_integrals[fault_index].size()
                      == fault.n_vertices(),
                      ExcMessage("Current reconstructed-fault I_h has the wrong number "
                                 "of vertices for fault "
                                 + Utilities::int_to_string(fault_index) + "."));
          AssertThrow(current_fault_surface_temperatures[fault_index].size()
                      == fault.n_vertices(),
                      ExcMessage("Current reconstructed-fault surface temperature has "
                                 "the wrong number of vertices for fault "
                                 + Utilities::int_to_string(fault_index) + "."));
          for (unsigned int vertex = 0; vertex < fault.n_vertices(); ++vertex)
            {
              AssertThrow(std::isfinite(
                            current_normalization_integrals[fault_index][vertex])
                          && current_normalization_integrals[fault_index][vertex] > 0.0,
                          ExcMessage("Current reconstructed-fault I_h must be finite and "
                                     "positive at fault "
                                     + Utilities::int_to_string(fault_index)
                                     + " vertex " + Utilities::int_to_string(vertex) + "."));
              AssertThrow(std::isfinite(
                            current_fault_surface_temperatures[fault_index][vertex]),
                          ExcMessage("Current reconstructed-fault surface temperature "
                                     "is non-finite."));
            }
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

    // Per-call scratch storage, never a second owner of committed history.
    // Keep the sampling cache and audit streams alive through publication.
    template <int dim>
    struct PhaseFieldFault<dim>::HistorySamples
    {
      std::vector<Point<dim>> points;
      Utilities::MPI::RemotePointEvaluation<dim> point_cache;
      std::vector<Tensor<2,dim>> velocity_gradients;
      std::vector<double> temperatures;
      std::vector<double> phase_fields;
      std::vector<double> previous_phase_fields;
    };

    template <int dim>
    struct PhaseFieldFault<dim>::HistoryCandidates
    {
      struct ParticleCandidate
      {
        SymmetricTensor<2,dim> stress;
        double crack_driving_force;
      };
      std::map<types::particle_index, double> cohesive_samples;
      std::map<types::particle_index, ParticleCandidate> particle_candidates;
      std::ofstream source_history_audit;
      std::set<std::string> trace_cells;
      std::ofstream stress_cycle_audit;
      typename ReconstructedFaultManager<dim>::ParticleScalarProjectionResult cohesive_projection;
      std::vector<std::vector<double>> state_candidates;
    };

    template <int dim>
    void
    PhaseFieldFault<dim>::commit_reconstructed_fault_mechanical_history(
      const LinearAlgebra::BlockVector &accepted_bulk_state)
    {
      TimerOutput::Scope timer(*performance_timer, "Fault: History commit");
      validate_reconstructed_fault_constitutive_state();

      // Timestep zero supplies the initial kinematic solution only. Its
      // constitutive histories were initialized explicitly during preparation.
      if (this->get_timestep_number() == 0)
        return;

      const double time_step = this->get_timestep();
      AssertThrow(std::isfinite(time_step) && time_step > 0.0,
                  ExcMessage("Later-time reconstructed-fault history requires a "
                             "positive timestep."));

      Particle::Manager<dim> &particle_manager =
        this->get_phase_field_handler().get_associated_particle_manager();
      const auto &property_manager = particle_manager.get_property_manager();
      const auto &particle_data = property_manager.get_data_info();
      AssertThrow(property_manager.plugin_name_exists("maxwell stress")
                  && particle_data.fieldname_exists("crack_driving_force"),
                  ExcMessage("Stage-J history commit requires particle properties "
                             "'maxwell stress' and 'crack_driving_force'."));
      const unsigned int stress_position =
        particle_data.get_position_by_plugin_index(
          property_manager.get_plugin_index_by_name("maxwell stress"));
      const unsigned int H_position =
        particle_data.get_position_by_field_name("crack_driving_force");

      std::vector<unsigned int> chemical_positions;
      for (const unsigned int field :
           this->introspection().chemical_composition_field_indices())
        {
          const auto property =
            this->get_parameters().mapped_particle_properties.find(field);
          AssertThrow(property !=
                      this->get_parameters().mapped_particle_properties.end(),
                      ExcMessage("Stage-J history commit requires mapped particle "
                                 "chemical compositions."));
          chemical_positions.push_back(
            particle_data.get_position_by_field_name(property->second.first)
            + property->second.second);
        }

      HistorySamples samples;
      sample_accepted_history(accepted_bulk_state, samples);

      HistoryCandidates candidates;
      compute_history_candidates(samples, time_step, stress_position, H_position,
                                 chemical_positions, candidates);
      validate_history_candidates(candidates);
      publish_history_candidates(candidates, stress_position, H_position);
    }

    template <int dim>
    void
    PhaseFieldFault<dim>::sample_accepted_history(
      const LinearAlgebra::BlockVector &accepted_bulk_state,
      HistorySamples &samples)
    {
      ReconstructedFaultManager<dim> &fault_manager =
        this->get_reconstructed_fault_manager();
      const auto &associations =
        fault_manager.get_locally_owned_particle_fault_associations();
      auto &points = samples.points;
      points.reserve(associations.size());
      for (const auto &association : associations)
        points.push_back(association.position);

      // Bulk samples use the accepted solution, whereas the surface state is
      // interpolated from the frozen reconstructed-fault Q1 fields.
      auto &point_cache = samples.point_cache;
      point_cache.reinit(this->get_phase_field_handler().get_grid_cache(), points);
      samples.velocity_gradients = VectorTools::point_gradients<dim>(
        point_cache, this->get_dof_handler(), accepted_bulk_state,
        VectorTools::EvaluationFlags::avg,
        this->introspection().component_indices.velocities[0]);
      samples.temperatures = VectorTools::point_values<1>(
        point_cache, this->get_dof_handler(), accepted_bulk_state,
        VectorTools::EvaluationFlags::avg,
        this->introspection().component_indices.temperature);
      const unsigned int phase_field_component =
        this->introspection().variable("phase_field").first_component_index;
      samples.phase_fields = VectorTools::point_values<1>(
        point_cache, this->get_dof_handler(), accepted_bulk_state,
        VectorTools::EvaluationFlags::avg, phase_field_component);
      samples.previous_phase_fields = VectorTools::point_values<1>(
        point_cache, this->get_dof_handler(), this->get_old_solution(),
        VectorTools::EvaluationFlags::avg, phase_field_component);
    }

    template <int dim>
    void
    PhaseFieldFault<dim>::compute_history_candidates(
      const HistorySamples &samples,
      const double time_step,
      const unsigned int stress_position,
      const unsigned int H_position,
      const std::vector<unsigned int> &chemical_positions,
      HistoryCandidates &candidates)
    {
      ReconstructedFaultManager<dim> &fault_manager =
        this->get_reconstructed_fault_manager();
      const auto &faults = fault_manager.get_faults();
      auto &particle_handler = this->get_phase_field_handler()
                               .get_associated_particle_manager().get_particle_handler();
      const auto &associations =
        fault_manager.get_locally_owned_particle_fault_associations();
      using ParticleCandidate = typename HistoryCandidates::ParticleCandidate;
      const auto &points = samples.points;
      const auto &velocity_gradients = samples.velocity_gradients;
      const auto &temperatures = samples.temperatures;
      const auto &phase_fields = samples.phase_fields;
      const auto &previous_phase_fields = samples.previous_phase_fields;
      auto &cohesive_samples = candidates.cohesive_samples;
      auto &particle_candidates = candidates.particle_candidates;
      auto &source_history_audit = candidates.source_history_audit;
      auto &trace_cells = candidates.trace_cells;
      auto &stress_cycle_audit = candidates.stress_cycle_audit;
      auto &cohesive_projection = candidates.cohesive_projection;
      auto &state_candidates = candidates.state_candidates;

      if (std::getenv("ASPECT_STRESS_CYCLE_TRACE"))
        {
          const auto rank=std::to_string(Utilities::MPI::this_mpi_process(this->get_mpi_communicator()));
          std::ifstream cells(this->get_output_directory()+"stress_trace_cells_rank"+rank+".txt");
          for (std::string id; cells>>id;) trace_cells.insert(id);
          stress_cycle_audit.open(this->get_output_directory()+"stress_update_"
                                 +std::to_string(this->get_timestep_number())+"_rank"+rank+".csv");
          stress_cycle_audit<<std::setprecision(17)
            <<"step,time_s,dt,particle_index,particle_id,cell,x,y,ref_x,ref_y,sample_x,sample_y,beta,kappa,grad_xx,grad_xy,grad_yx,grad_yy,old_xx,old_yy,old_xy,eps_xx,eps_yy,eps_xy,crack_xx,crack_yy,crack_xy,new_xx,new_yy,new_xy\n";
        }
      if (std::getenv("ASPECT_FAULT_SOURCE_HISTORY_DIAGNOSTIC"))
        {
          source_history_audit.open(this->get_output_directory()+"continued_source_history_"
            +std::to_string(this->get_timestep_number())+"_rank"
            +std::to_string(Utilities::MPI::this_mpi_process(this->get_mpi_communicator()))+".csv");
          source_history_audit<<std::setprecision(17)
            <<"id,x,y,phi,chi,V,kappa,beta,eps_xx,eps_yy,eps_xy,crack_xx,crack_yy,crack_xy,old_xx,old_yy,old_xy,new_xx,new_yy,new_xy\n";
        }

      // First evaluate the accepted cohesive traction samples. Projection is
      // completed before any particle history candidate uses the new traction.
      unsigned int particle_index = 0;
      // A particle-local numerical failure must reach every rank before the
      // next projection collective. No persistent histories have been written.
      std::string local_error;
      try
        {
          for (const auto &particle : particle_handler)
            {
              const auto &association = associations[particle_index];
              Assert(particle.get_id() == association.particle_id, ExcInternalError());
              if (association.active)
                {
                  const LocalizationResponse localization =
                    evaluate_reconstructed_fault_localization(
                      association.fault_index, association.segment_index,
                      association.xi, phase_fields[particle_index],
                      previous_phase_fields[particle_index], "history commit");
                  const MaxwellCoefficients surface_coefficients =
                    compute_maxwell_coefficients(
                      compute_creep_viscosity(
                        localization.surface_material_fractions,
                        localization.surface_temperature),
                      MaterialUtilities::average_value(
                        localization.surface_material_fractions,
                        elastic_shear_moduli, viscosity_averaging),
                      time_step);
                  const double slip_rate = fault_manager.interpolate_slip_rate(
                    association.fault_index, association.segment_index,
                    association.xi);
                  cohesive_samples.emplace(
                    particle.get_id(),
                    compute_cohesive_response(
                      surface_coefficients, localization.current_I_h,
                      localization.previous_I_h,
                      localization.previous_cohesive_traction, slip_rate,
                      localization.current_h,
                      localization.previous_h, mature_frictional_fault).cohesive_traction);
                }
              ++particle_index;
            }
        }
      catch (const std::exception &exception)
        {
          local_error = exception.what();
        }
      throw_if_history_error(local_error, this->get_mpi_communicator());

      cohesive_projection =
        fault_manager.project_particle_scalar(cohesive_samples);

      try
        {
          if (fault_friction.has_state_variable())
            {
              state_candidates.resize(faults.size());
              const unsigned int state_position =
                fault_manager.get_property_information()[
                  fault_property_indices.state].position;
              for (unsigned int fault = 0; fault < faults.size(); ++fault)
                {
                  state_candidates[fault].resize(faults[fault].n_vertices());
                  for (unsigned int vertex = 0;
                       vertex < faults[fault].n_vertices(); ++vertex)
                    {
                      // Construct the surface mixture at the same Q1 vertex used
                      // by friction. Dc is global in the current aging law.
                      std::vector<double> surface_compositions(
                        fault_property_indices.chemical_compositions.size());
                      for (unsigned int c = 0; c < surface_compositions.size(); ++c)
                        {
                          const auto &property = fault_manager.get_property_information()[
                            fault_property_indices.chemical_compositions[c]];
                          surface_compositions[c] =
                            faults[fault].get_properties(vertex)[property.position];
                        }
                      const std::vector<double> surface_fractions =
                        MaterialUtilities::compute_composition_fractions(
                          surface_compositions);
                      state_candidates[fault][vertex] = fault_friction.update_state(
                        surface_fractions,
                        fault_manager.get_slip_rate(fault)[vertex],
                        faults[fault].get_properties(vertex)[state_position],
                        time_step);
                      AssertThrow(std::isfinite(state_candidates[fault][vertex])
                                  && state_candidates[fault][vertex] > 0.0,
                                  ExcMessage("The accepted rate-and-state history is "
                                             "inadmissible."));
                    }
                }
            }

          // Re-evaluate the accepted local strain decomposition with projected
          // Q1 traction for H. Bulk and surface coefficients deliberately use
          // different temperatures and material mixtures.
          particle_index = 0;
          for (const auto &particle : particle_handler)
            {
              auto association = associations[particle_index];
              if (!association.active)
                {
                  // Bulk source continuation changes Maxwell strain subtraction,
                  // never particle ownership or the surface projection above.
                  const auto source = fault_manager.project_to_bulk_source(association.position, true);
                  if (source.active)
                    {
                      AssertThrow(mature_frictional_fault && !evolve_phase_field,
                                  ExcMessage("Continued bottom source requires a frozen mature fault."));
                      association.active = true;
                      association.fault_index = source.fault_index;
                      association.segment_index = source.segment_index;
                      association.xi = source.xi;
                    }
                }
              const ArrayView<const double> properties = particle.get_properties();
              SymmetricTensor<2,dim> old_stress;
              for (unsigned int component = 0;
                   component < SymmetricTensor<2,dim>::n_independent_components;
                   ++component)
                old_stress[SymmetricTensor<2,dim>::unrolled_to_component_indices(
                  component)] = properties[stress_position+component];
              if (benchmark_retained_stress)
                old_stress = benchmark_retained_stress(
                  particle.get_surrounding_cell()->id(), particle.get_location());

              std::vector<double> bulk_compositions(chemical_positions.size());
              for (unsigned int c = 0; c < chemical_positions.size(); ++c)
                bulk_compositions[c] = properties[chemical_positions[c]];
              const std::vector<double> bulk_fractions =
                MaterialUtilities::compute_composition_fractions(bulk_compositions);
              const MaxwellCoefficients bulk_coefficients =
                compute_maxwell_coefficients(
                  compute_creep_viscosity(bulk_fractions,
                                          temperatures[particle_index]),
                  MaterialUtilities::average_value(
                    bulk_fractions, elastic_shear_moduli, viscosity_averaging),
                  time_step);

              SymmetricTensor<2,dim> effective_strain_rate =
                symmetrize(velocity_gradients[particle_index]);
              SymmetricTensor<2,dim> continued_crack;
              double continued_chi=0., continued_V=0.;
              double new_H = properties[H_position];
              if (association.active)
                {
                  const ReconstructedFault<dim> &fault =
                    faults[association.fault_index];
                  Tensor<1,dim> tangent =
                    fault.vertex(association.segment_index+1)
                    - fault.vertex(association.segment_index);
                  tangent /= tangent.norm();
                  Tensor<1,dim> normal;
                  normal[0] = -tangent[1];
                  normal[1] = tangent[0];
                  const SymmetricTensor<2,dim> slip_tensor =
                    this->get_reconstructed_fault_manager().get_shear_sense(association.fault_index)
                    * symmetrize(outer_product(tangent, normal));

                  const LocalizationResponse localization =
                    evaluate_reconstructed_fault_localization(
                      association.fault_index, association.segment_index,
                      association.xi, phase_fields[particle_index],
                      previous_phase_fields[particle_index], "history commit");
                  const MaxwellCoefficients surface_coefficients =
                    compute_maxwell_coefficients(
                      compute_creep_viscosity(
                        localization.surface_material_fractions,
                        localization.surface_temperature),
                      MaterialUtilities::average_value(
                        localization.surface_material_fractions,
                        elastic_shear_moduli, viscosity_averaging),
                      time_step);
                  const double slip_rate = fault_manager.interpolate_slip_rate(
                    association.fault_index, association.segment_index,
                    association.xi);
                  const CohesiveResponse cohesive = compute_cohesive_response(
                    surface_coefficients, localization.current_I_h,
                    localization.previous_I_h,
                    localization.previous_cohesive_traction, slip_rate,
                    localization.current_h, localization.previous_h, mature_frictional_fault);
                  effective_strain_rate -= cohesive.crack_strain_rate*slip_tensor;
                  if (!associations[particle_index].active)
                    {
                      continued_crack=cohesive.crack_strain_rate*slip_tensor;
                      continued_chi=cohesive.localization_factor;
                      continued_V=slip_rate;
                    }

                  AssertThrow(std::isfinite(new_H) && new_H >= 0.0,
                              ExcMessage("Stored crack-driving history is inadmissible."));
                  // This parameter freezes the driving history, not the phase-field
                  // solve. Maxwell and surface histories still follow mechanics.
                  if (evolve_phase_field)
                    {
                      const double xi = association.xi;
                      const double projected_traction =
                        (1.0-xi) * cohesive_projection.nodal_values[
                          association.fault_index][association.segment_index]
                        + xi * cohesive_projection.nodal_values[
                          association.fault_index][association.segment_index+1];
                      const double H_candidate =
                        compute_crack_driving_force_candidate(
                          time_step, surface_coefficients,
                          localization.current_degradation,
                          localization.previous_h, projected_traction,
                          localization.previous_cohesive_traction);
                      new_H = std::max(new_H, H_candidate);
                    }
                }

              ParticleCandidate candidate;
              candidate.stress = compute_maxwell_stress(
                bulk_coefficients, effective_strain_rate, old_stress);
              candidate.crack_driving_force = new_H;
              AssertThrow(std::isfinite(candidate.stress.norm())
                          && std::isfinite(candidate.crack_driving_force)
                          && candidate.crack_driving_force >= 0.0,
                          ExcMessage("Stage-J particle history candidate is inadmissible."));
              particle_candidates.emplace(particle.get_id(), candidate);
              // Capture the inputs and candidate actually used before publication.
              // In particular, old_stress is the parent value, not reconstructed FE history.
              if (stress_cycle_audit.is_open() && trace_cells.count(particle.get_surrounding_cell()->id().to_string()))
                {
                  const auto x=particle.get_location(), r=particle.get_reference_location();
                  const auto &gradient=velocity_gradients[particle_index];
                  stress_cycle_audit<<this->get_timestep_number()<<','<<this->get_time()<<','<<time_step<<','
                    <<particle_index<<','<<particle.get_id()<<','<<particle.get_surrounding_cell()->id()<<','
                    <<x[0]<<','<<x[1]<<','<<r[0]<<','<<r[1]<<','<<points[particle_index][0]<<','<<points[particle_index][1]
                    <<','<<bulk_coefficients.beta<<','<<bulk_coefficients.kappa<<','
                    <<gradient[0][0]<<','<<gradient[0][1]<<','<<gradient[1][0]<<','<<gradient[1][1];
                  for (const auto &tensor:{old_stress,symmetrize(gradient),
                                          symmetrize(gradient)-effective_strain_rate,candidate.stress})
                    stress_cycle_audit<<','<<tensor[0][0]<<','<<tensor[1][1]<<','<<tensor[0][1];
                  stress_cycle_audit<<'\n';
                }
              if (source_history_audit.is_open() && association.active && !associations[particle_index].active)
                {
                  const auto position=particle.get_location();
                  source_history_audit<<particle.get_id()<<','<<position[0]<<','<<position[1]<<','
                    <<phase_fields[particle_index]<<','<<continued_chi<<','<<continued_V<<','
                    <<bulk_coefficients.kappa<<','<<bulk_coefficients.beta;
                  for (const auto &tensor : {symmetrize(velocity_gradients[particle_index]),
                                            continued_crack,old_stress,candidate.stress})
                    source_history_audit<<','<<tensor[0][0]<<','<<tensor[1][1]<<','<<tensor[0][1];
                  source_history_audit<<'\n';
                }
              ++particle_index;
            }
        }
      catch (const std::exception &exception)
        {
          local_error = exception.what();
        }
      throw_if_history_error(local_error, this->get_mpi_communicator());

    }

    template <int dim>
    void
    PhaseFieldFault<dim>::validate_history_candidates(
      const HistoryCandidates &candidates)
    {
      auto &particle_handler = this->get_phase_field_handler()
                               .get_associated_particle_manager().get_particle_handler();
      const auto &particle_candidates = candidates.particle_candidates;
      const auto &cohesive_projection = candidates.cohesive_projection;
      // Finish the existing collective and local checks before publication.
      const unsigned int local_valid =
        particle_candidates.size() == particle_handler.n_locally_owned_particles()
        ? 1u : 0u;
      AssertThrow(Utilities::MPI::min(local_valid, this->get_mpi_communicator()) == 1,
                  ExcMessage("Stage-J history candidates are incomplete."));

      validate_cohesive_state_commit(cohesive_projection.nodal_values);
      for (const auto &particle : particle_handler)
        AssertThrow(particle_candidates.find(particle.get_id())
                    != particle_candidates.end(),
                    ExcMessage("Stage-J history candidates do not match the locally "
                               "owned particle IDs."));

      for (unsigned int fault = 0;
           fault < cohesive_projection.diagnostics.size(); ++fault)
        this->get_pcout()
          << "   Cohesive traction profile variation for fault " << fault
          << ": weighted RMS="
          << cohesive_projection.diagnostics[fault].weighted_rms_residual
          << " Pa, maximum="
          << cohesive_projection.diagnostics[fault].maximum_absolute_residual
          << " Pa" << std::endl;
    }

    template <int dim>
    void
    PhaseFieldFault<dim>::publish_history_candidates(
      const HistoryCandidates &candidates,
      const unsigned int stress_position,
      const unsigned int H_position)
    {
      ReconstructedFaultManager<dim> &fault_manager =
        this->get_reconstructed_fault_manager();
      const auto &faults = fault_manager.get_faults();
      auto &particle_handler = this->get_phase_field_handler()
                               .get_associated_particle_manager().get_particle_handler();
      using ParticleCandidate = typename HistoryCandidates::ParticleCandidate;
      const auto &particle_candidates = candidates.particle_candidates;
      const auto &cohesive_projection = candidates.cohesive_projection;
      const auto &state_candidates = candidates.state_candidates;
      commit_cohesive_state(cohesive_projection.nodal_values);

      if (fault_friction.has_state_variable())
        {
          const unsigned int state_position =
            fault_manager.get_property_information()[
              fault_property_indices.state].position;
          for (unsigned int fault = 0; fault < faults.size(); ++fault)
            for (unsigned int vertex = 0;
                 vertex < faults[fault].n_vertices(); ++vertex)
              fault_manager.get_fault(fault).get_properties(vertex)[state_position]
                = state_candidates[fault][vertex];
        }

      for (auto &particle : particle_handler)
        {
          const ParticleCandidate &candidate =
            particle_candidates.find(particle.get_id())->second;
          ArrayView<double> properties = particle.get_properties();
          for (unsigned int component = 0;
               component < SymmetricTensor<2,dim>::n_independent_components;
               ++component)
            properties[stress_position+component] =
              candidate.stress[
                SymmetricTensor<2,dim>::unrolled_to_component_indices(component)];
          properties[H_position] = candidate.crack_driving_force;
        }
    }

    template <int dim>
    double
    PhaseFieldFault<dim>::compute_reconstructed_fault_time_step(
      const double cfl_number) const
    {
      AssertThrow(std::isfinite(cfl_number) && cfl_number > 0.0,
                  ExcMessage("The reconstructed-fault timestep requires a positive CFL number."));
      const ReconstructedFaultManager<dim> &fault_manager =
        this->get_reconstructed_fault_manager();
      AssertThrow(fault_manager.slip_rates_are_initialized(),
                  ExcMessage("The reconstructed-fault timestep requires committed V."));

      double time_step = std::numeric_limits<double>::max();
      for (unsigned int fault = 0;
           fault < fault_manager.get_faults().size(); ++fault)
        for (unsigned int vertex = 0;
             vertex < fault_manager.get_fault(fault).n_vertices(); ++vertex)
          {
            std::vector<double> surface_compositions(
              fault_property_indices.chemical_compositions.size());
            for (unsigned int c = 0; c < surface_compositions.size(); ++c)
              {
                const auto &property = fault_manager.get_property_information()[
                  fault_property_indices.chemical_compositions[c]];
                surface_compositions[c] = fault_manager.get_fault(fault)
                                          .get_properties(vertex)[property.position];
              }
            const std::vector<double> surface_fractions =
              MaterialUtilities::compute_composition_fractions(
                surface_compositions);
            time_step = std::min(
              time_step,
              fault_friction.compute_time_step(
                surface_fractions,
                fault_manager.get_timestep_committed_slip_rate(fault)[vertex],
                cfl_number, true));
          }
      return time_step;
    }

    template <int dim>
    void
    PhaseFieldFault<dim>::validate_cohesive_state_commit(
      const std::vector<std::vector<double>> &cohesive_tractions) const
    {
      const ReconstructedFaultManager<dim> &fault_manager =
        this->get_reconstructed_fault_manager();
      const auto &faults = fault_manager.get_faults();
      AssertThrow(cohesive_tractions.size() == faults.size(),
                  ExcMessage("Committed cohesive traction has the wrong number of faults."));
      AssertThrow(current_normalization_integrals.size() == faults.size(),
                  ExcMessage("Committed previous I_h has the wrong number of faults."));

      for (unsigned int fault = 0; fault < faults.size(); ++fault)
        {
          AssertThrow(cohesive_tractions[fault].size()
                      == faults[fault].n_vertices(),
                      ExcMessage("Committed cohesive traction has the wrong number "
                                 "of fault vertices."));
          AssertThrow(current_normalization_integrals[fault].size()
                      == faults[fault].n_vertices(),
                      ExcMessage("Committed previous I_h has the wrong number of "
                                 "fault vertices."));
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
    }

    template <int dim>
    void
    PhaseFieldFault<dim>::commit_cohesive_state(
      const std::vector<std::vector<double>> &cohesive_tractions) noexcept
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

      compute_normalization_integrals();
      if (cohesive_history_is_initialized(faults,
                                          cohesive_position,
                                          normalization_position))
        return;
      const auto projection = fault_manager.project_particle_scalar(
        evaluate_initial_cohesive_particle_values());
      initial_cohesive_projection_diagnostics = projection.diagnostics;
      validate_cohesive_state_commit(projection.nodal_values);
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
      if (mature_frictional_fault)
        {
          std::map<types::particle_index,double> zero;
          for (const auto &p : this->get_reconstructed_fault_manager().get_locally_owned_particle_fault_associations())
            if (p.active) zero.emplace(p.particle_id,0.0);
          return zero;
        }
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


    // Registration remains in phase_field_fault.cc; instantiate only moved members.
#define INSTANTIATE(dim) \
    template void PhaseFieldFault<dim>::initialize(); \
    template void PhaseFieldFault<dim>::prepare_reconstructed_fault_mechanical_solve(); \
    template void PhaseFieldFault<dim>::compute_fault_surface_temperatures(); \
    template void PhaseFieldFault<dim>::validate_reconstructed_fault_constitutive_state() const; \
    template void PhaseFieldFault<dim>::commit_reconstructed_fault_mechanical_history(const LinearAlgebra::BlockVector &); \
    template double PhaseFieldFault<dim>::compute_reconstructed_fault_time_step(const double) const; \
    template void PhaseFieldFault<dim>::validate_cohesive_state_commit(const std::vector<std::vector<double>> &) const; \
    template void PhaseFieldFault<dim>::commit_cohesive_state(const std::vector<std::vector<double>> &) noexcept; \
    template void PhaseFieldFault<dim>::initialize_cohesive_state_from_initial_fields(); \
    template std::map<types::particle_index,double> PhaseFieldFault<dim>::evaluate_initial_cohesive_particle_values();

    ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
  }
}
