/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include <aspect/reconstructed_fault/manager.h>
#include <aspect/phase_field.h>
#include <aspect/particle/manager.h>
#include <aspect/particle/particle_domain.h>
#include <aspect/material_model/utilities.h>
#include <aspect/utilities.h>

#include <deal.II/fe/fe_values.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <set>

namespace aspect
{
  namespace
  {


    template <int dim>
    std::vector<Tensor<1,dim>>
    reference_normals(const std::vector<Point<dim>> &points)
    {
      std::vector<Tensor<1,dim>> normals(points.size());
      for (unsigned int i = 0; i < points.size(); ++i)
        {
          Tensor<1,dim> tangent = (i == 0 ? points[1] - points[0]
                                   : (i + 1 == points.size() ? points[i] - points[i-1]
                                      : points[i+1] - points[i-1]));
          const double norm = tangent.norm();
          AssertThrow(std::isfinite(norm) && norm > 0.0,
                      ExcMessage("The prescribed fault geometry does not define a finite "
                                 "reference normal at every structural point."));
          tangent /= norm;
          normals[i][0] = -tangent[1];
          normals[i][1] = tangent[0];
        }
      return normals;
    }


    struct StructuralCoordinates
    {
      double distance;
      double signed_distance;
      unsigned int segment;
      double xi;
    };


    template <int dim>
    StructuralCoordinates
    structural_coordinates(const std::vector<Point<dim>> &points,
                           const Point<dim> &position)
    {
      double min_r_squared = std::numeric_limits<double>::infinity();
      double r = numbers::signaling_nan<double>();
      double eta = numbers::signaling_nan<double>();
      unsigned int closest_segment = numbers::invalid_unsigned_int;
      double segment_coordinate = numbers::signaling_nan<double>();
      for (unsigned int segment = 0; segment + 1 < points.size(); ++segment)
        {
          const Tensor<1,dim> segment_vector = points[segment+1] - points[segment];
          const double segment_length = segment_vector.norm();
          const Tensor<1,dim> tangent = segment_vector / segment_length;
          const double coordinate = std::clamp(((position-points[segment]) * tangent) / segment_length,
                                               0.0, 1.0);
          const Point<dim> closest_point = points[segment] + coordinate * segment_vector;
          const double r_squared = position.distance_square(closest_point);
          if (r_squared < min_r_squared)
            {
              min_r_squared = r_squared;
              r = std::sqrt(r_squared);
              const Tensor<1,dim> normal({-tangent[1], tangent[0]});
              eta = (position - closest_point) * normal;
              closest_segment = segment;
              segment_coordinate = coordinate;
            }
        }
      return {r, eta, closest_segment, segment_coordinate};
    }


    struct InitialFaultSupport
    {
      double reconstruction_radius;
      std::vector<double> prescribed_half_widths;
    };


    template <int dim>
    std::vector<InitialFaultSupport>
    determine_initial_fault_support(
      const PhaseFieldHandler<dim> &phase_field_handler,
      const DoFHandler<dim> &dof_handler,
      const MPI_Comm mpi_communicator,
      const std::vector<PrescribedInitialFault<dim>> &prescribed_faults)
    {
      double local_cell_margin = 0.0;
      for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
          local_cell_margin = std::max(local_cell_margin, cell->diameter());
      const double cell_margin = Utilities::MPI::max(
                                   local_cell_margin, mpi_communicator);

      std::vector<InitialFaultSupport> support(prescribed_faults.size());
      std::map<double, double> profile_support_by_core_value;
      for (unsigned int fault = 0; fault < prescribed_faults.size(); ++fault)
        {
          support[fault].reconstruction_radius = cell_margin;
          support[fault].prescribed_half_widths.resize(
            prescribed_faults[fault].core_phase_field_values.size());

          for (unsigned int vertex = 0;
               vertex < prescribed_faults[fault].core_phase_field_values.size(); ++vertex)
            {
              const double phi_hat =
                prescribed_faults[fault].core_phase_field_values[vertex];
              auto profile_support = profile_support_by_core_value.find(phi_hat);
              if (profile_support == profile_support_by_core_value.end())
                {
                  double half_width = 0.0;
                  for (const auto &profile :
                       phase_field_handler.get_phase_field_profiles(phi_hat))
                    half_width = std::max(
                                   half_width, profile->get_coordinate_values().back());
                  profile_support = profile_support_by_core_value.emplace(
                                      phi_hat, half_width).first;
                }

              support[fault].prescribed_half_widths[vertex] = profile_support->second;
              support[fault].reconstruction_radius = std::max(
                                                       support[fault].reconstruction_radius,
                                                       profile_support->second + cell_margin);
            }
        }
      return support;
    }


    template <int dim>
    struct InitialFaultReconstruction
    {
      std::vector<Point<dim>> vertices;
      std::vector<double> projection_half_widths;
      FaultReconstructionDiagnostics diagnostics;
    };



  }

  template <int dim>
  ReconstructedFaultManager<dim>::ReconstructedFaultManager(const Simulator<dim> &simulator)
  {
    this->initialize_simulator(simulator);
    performance_timer = std::make_unique<TimerOutput>(
      std::cout,
      std::getenv("ASPECT_FAULT_PERFORMANCE") && this->get_pcout().is_active()
      ? TimerOutput::summary : TimerOutput::never, TimerOutput::wall_times);
  }


  template <int dim>
  void
  ReconstructedFaultManager<dim>::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Fault reconstruction");
    {
      prm.declare_entry("Structural point spacing", "1000",
                        Patterns::Double(0.0),
                        "Target arc-length spacing of reconstructed-fault vertices. Units: meter.");
      prm.declare_entry("Fit prescribed geometry to phase field", "true",
                        Patterns::Bool(),
                        "Fit normal offsets to the solved phase ridge. If false, retain the "
                        "resampled prescribed polyline exactly, using the same profile support policy.");
      prm.declare_entry("Boundary completion", "legacy", Patterns::Selection("legacy|automatic prescribed"),
                        "Legacy retains existing boundary behavior. Automatic prescribed classifies every "
                        "fault contact and requires a compatible fully prescribed frozen 2-D phase field, "
                        "supported exterior Q1 data and paired mechanical work. Unsupported contacts fail.");
      prm.declare_entry("Ridge coefficient", "1",
                        Patterns::Double(0.0),
                        "Dimensionless second-difference ridge coefficient applied after "
                        "normalizing the data system by its total phase-field weight.");
      prm.declare_entry("Prescribed faults file", "",
                        Patterns::FileName(),
                        "ASCII file containing prescribed faults. Each vertex line contains "
                        "the spatial coordinates followed by the core phase-field value. "
                        "A line containing only `---' separates connected faults. Lines may "
                        "contain comments beginning with `#'.");
    }
    prm.leave_subsection();
  }


  template <int dim>
  void
  ReconstructedFaultManager<dim>::parse_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Fault reconstruction");
    {
      structural_spacing = prm.get_double("Structural point spacing");
      fit_prescribed_geometry = prm.get_bool("Fit prescribed geometry to phase field");
      automatic_boundary_completion = prm.get("Boundary completion") == "automatic prescribed";
      AssertThrow(!automatic_boundary_completion || (dim==2 && !fit_prescribed_geometry),
                  ExcMessage("Automatic boundary completion requires 2D prescribed fixed geometry "
                             "(Fit prescribed geometry to phase field=false); 3D contacts are unsupported."));
      ridge_coefficient = prm.get_double("Ridge coefficient");
      prescribed_faults_filename = prm.get("Prescribed faults file");
      AssertThrow(std::isfinite(structural_spacing) && structural_spacing > 0.0,
                  ExcMessage("Fault reconstruction structural point spacing must be positive."));
      AssertThrow(std::isfinite(ridge_coefficient) && ridge_coefficient >= 0.0,
                  ExcMessage("Fault reconstruction ridge coefficient must be nonnegative."));
      AssertThrow(!prescribed_faults_filename.empty(),
                  ExcMessage("Fault reconstruction requires `Prescribed faults file'."));
    }
    prm.leave_subsection();

    prescribed_faults_filename =
      Utilities::expand_ASPECT_SOURCE_DIR(prescribed_faults_filename);
    const std::string file_contents = Utilities::read_and_distribute_file_content(
                                        prescribed_faults_filename, this->get_mpi_communicator());
    prescribed_faults = ReconstructedFaultUtilities::parse_prescribed_faults<dim>(
                          file_contents, prescribed_faults_filename);

    this->get_signals().post_refinement_load_user_data.connect(
      [this] (parallel::distributed::Triangulation<dim> &)
    {
      boundary_contacts_valid = false;
      invalidate_stokes_qp_projection_cache();
    });
    this->get_signals().post_resume_load_user_data.connect(
      [this] (parallel::distributed::Triangulation<dim> &)
    {
      boundary_contacts_valid = false;
      invalidate_stokes_qp_projection_cache();
    });
    this->get_signals().post_mesh_deformation.connect(
      [this] (const SimulatorAccess<dim> &)
    {
      boundary_contacts_valid = false;
      invalidate_stokes_qp_projection_cache();
    });

    // Particle managers connect to this signal before the reconstructed-fault
    // manager is parsed. Consequently, their slots generate and initialize
    // particles (and particle domains) before this slot initializes H. The
    // signal is emitted for every initial adaptive-refinement cycle that sets
    // initial conditions, before the corresponding timestep solve.
    this->get_signals().post_set_initial_state.connect(
      [this] (const SimulatorAccess<dim> &)
    {
      initial_reconstruction_complete = false;
      initialize_crack_driving_force(prescribed_faults);
    });
  }


  // -----------------------------------------------------------------------------
  // Fault reconstruction
  // -----------------------------------------------------------------------------

  template <int dim>
  void
  ReconstructedFaultManager<dim>::initialize_crack_driving_force(
    const std::vector<PrescribedInitialFault<dim>> &faults)
  {
    prescribed_faults = faults;
    boundary_contacts_valid = false;
    if (automatic_boundary_completion)
      prepare_boundary_contacts();
    if (faults.empty())
      return;

    AssertThrow(dim == 2,
                ExcMessage("Prescribed initial fault initialization is currently implemented only in 2D."));

    PhaseFieldHandler<dim> &phase_field_handler = this->get_phase_field_handler();

    const auto *phase_field_model =
      dynamic_cast<const MaterialModel::PhaseFieldModel<dim> *>(
        &this->get_material_model());
    AssertThrow(phase_field_model != nullptr,
                ExcMessage("Prescribed initial faults require a phase-field material model."));
    const double activation_threshold =
      phase_field_model->get_phase_field_activation_threshold();
    const double upper_admissibility_threshold =
      phase_field_model->get_phase_field_upper_admissibility_threshold();

    for (unsigned int fault_index = 0; fault_index < faults.size(); ++fault_index)
      {
        const PrescribedInitialFault<dim> &fault = faults[fault_index];
        ReconstructedFaultUtilities::closest_point_distance_and_core_phase_field(fault, Point<dim>());
        for (unsigned int vertex = 0; vertex < fault.core_phase_field_values.size(); ++vertex)
          AssertThrow(fault.core_phase_field_values[vertex] >= activation_threshold
                      && fault.core_phase_field_values[vertex] <= upper_admissibility_threshold,
                      ExcMessage("Core phase-field value " + Utilities::int_to_string(vertex)
                                 + " of prescribed fault " + Utilities::int_to_string(fault_index)
                                 + " is outside the acceptable phase-field range."));
      }

    Particle::Manager<dim> &particle_manager =
      phase_field_handler.get_associated_particle_manager();
    const auto &particle_data_info = particle_manager.get_property_manager().get_data_info();
    const unsigned int crack_driving_force_index =
      particle_data_info.get_position_by_field_name("crack_driving_force");

    std::vector<unsigned int> chemical_property_indices;
    for (const unsigned int field_index :
         this->introspection().chemical_composition_field_indices())
      {
        AssertThrow(this->get_parameters().compositional_field_methods[field_index]
                    == Parameters<dim>::AdvectionFieldMethod::particles,
                    ExcMessage("Prescribed initial faults require every chemical composition field "
                               "to be advected by particles."));
        const auto mapped_property =
          this->get_parameters().mapped_particle_properties.find(field_index);
        AssertThrow(mapped_property
                    != this->get_parameters().mapped_particle_properties.end(),
                    ExcMessage("A chemical composition field is not mapped to a particle property."));
        chemical_property_indices.push_back(
          particle_data_info.get_position_by_field_name(mapped_property->second.first)
          + mapped_property->second.second);
      }

    std::map<double, std::vector<std::unique_ptr<PhaseField::PhaseFieldProfile>>> profile_cache;
    Particles::ParticleHandler<dim> &particle_handler = particle_manager.get_particle_handler();
    std::vector<double> chemical_compositions(chemical_property_indices.size());
    std::map<types::particle_index, double> particle_updates;

    for (auto &particle : particle_handler)
      {
        const ArrayView<double> properties = particle.get_properties();
        for (unsigned int c = 0; c < chemical_property_indices.size(); ++c)
          chemical_compositions[c] = properties[chemical_property_indices[c]];
        const std::vector<double> volume_fractions =
          MaterialModel::MaterialUtilities::compute_composition_fractions(chemical_compositions);

        unsigned int contributing_faults = 0;
        double prescribed_crack_driving_force = numbers::signaling_nan<double>();
        unsigned int first_contributing_fault = numbers::invalid_unsigned_int;
        unsigned int second_contributing_fault = numbers::invalid_unsigned_int;

        for (unsigned int fault_index = 0; fault_index < faults.size(); ++fault_index)
          {
            const auto r_and_phi_hat =
              ReconstructedFaultUtilities::closest_point_distance_and_core_phase_field(faults[fault_index],
                                                                                       particle.get_location());
            const double r = r_and_phi_hat.first;
            const double phi_hat = r_and_phi_hat.second;

            auto profiles = profile_cache.find(phi_hat);
            if (profiles == profile_cache.end())
              profiles = profile_cache.emplace(
                           phi_hat,
                           phase_field_handler.get_phase_field_profiles(phi_hat)).first;

            std::vector<double> material_phase_fields(profiles->second.size());
            for (unsigned int material = 0; material < profiles->second.size(); ++material)
              material_phase_fields[material] = profiles->second[material]->value(r);

            const double phase_field = MaterialModel::MaterialUtilities::average_value(
                                         volume_fractions,
                                         material_phase_fields,
                                         MaterialModel::MaterialUtilities::arithmetic);

            if (phase_field > activation_threshold)
              {
                ++contributing_faults;
                if (contributing_faults == 1)
                  {
                    first_contributing_fault = fault_index;
                    prescribed_crack_driving_force =
                      phase_field_handler.stationary_crack_driving_force(volume_fractions,
                                                                         phase_field,
                                                                         phi_hat);
                  }
                else if (contributing_faults == 2)
                  second_contributing_fault = fault_index;
              }
          }

        AssertThrow(contributing_faults <= 1,
                    ExcMessage("Prescribed initial faults "
                               + Utilities::int_to_string(first_contributing_fault) + " and "
                               + Utilities::int_to_string(second_contributing_fault) + " "
                               "both contribute at particle "
                               + Utilities::int_to_string(particle.get_id()) + " located at "
                               + Utilities::to_string(particle.get_location()[0]) + ", "
                               + Utilities::to_string(particle.get_location()[1]) + "."));

        if (contributing_faults == 1)
          particle_updates.emplace(particle.get_local_index(), prescribed_crack_driving_force);
      }

    for (auto &particle : particle_handler)
      {
        const auto update = particle_updates.find(particle.get_local_index());
        if (update != particle_updates.end())
          particle.get_properties()[crack_driving_force_index] = update->second;
      }

  }


  template <int dim>
  void
  ReconstructedFaultManager<dim>::reconstruct_initial_faults()
  {
    if (initial_reconstruction_complete || prescribed_faults.empty())
      return;

    this->get_pcout() << "   Reconstruction initial faults... " << std::flush;

    AssertThrow(dim == 2, ExcNotImplemented());
    const auto *phase_field_model =
      dynamic_cast<const MaterialModel::PhaseFieldModel<dim> *>(
        &this->get_material_model());
    AssertThrow(phase_field_model != nullptr,
                ExcMessage("Fault reconstruction requires a phase-field material model."));
    const double activation_threshold =
      phase_field_model->get_phase_field_activation_threshold();

    const PhaseFieldHandler<dim> &phase_field_handler =
      this->get_phase_field_handler();

    // Determine mesh-aware reconstruction support before changing any manager
    // state; all faults use these radii when checking tubular-region overlap.
    const std::vector<InitialFaultSupport> fault_support =
      determine_initial_fault_support(phase_field_handler,
                                      this->get_dof_handler(),
                                      this->get_mpi_communicator(),
                                      prescribed_faults);
    std::vector<double> reconstruction_radii(fault_support.size());
    for (unsigned int fault_index = 0;
         fault_index < fault_support.size(); ++fault_index)
      reconstruction_radii[fault_index] =
        fault_support[fault_index].reconstruction_radius;

    // Replacing geometry invalidates every property/slip-rate association and
    // both particle and Stokes-QP caches as one lifecycle transition.
    reconstructed_faults.clear();
    projection_half_widths.clear();
    shear_senses.clear();
    timestep_committed_slip_rates.clear();
    current_newton_slip_rates.clear();
    trial_slip_rates.clear();
    slip_rate_initialized.clear();
    prescribed_slip_rates.clear();
    slip_rate_nonlinear_solve_active = false;
    slip_rate_trial_active = false;
    invalidate_particle_projection_cache();
    invalidate_stokes_qp_projection_cache();
    diagnostics.clear();

    // Fit each prescribed polyline independently using the shared support data,
    // then store its reconstructed geometry and diagnostics in fault order.
    for (unsigned int fault_index = 0;
         fault_index < prescribed_faults.size(); ++fault_index)
      reconstruct_initial_fault(
        fault_index,
        fault_support[fault_index].reconstruction_radius,
        reconstruction_radii,
        fault_support[fault_index].prescribed_half_widths,
        activation_threshold);

    // Mark reconstruction complete only after every fault has been fitted;
    // subsequent projections rebuild against the final metadata generation.
    ++projection_metadata_version;
    invalidate_particle_projection_cache();
    initial_reconstruction_complete = true;

    this->get_pcout() << "done." << std::endl << std::endl;
  }


  template <int dim>
  void
  ReconstructedFaultManager<dim>::reconstruct_initial_fault(
    const unsigned int fault_index,
    const double reconstruction_radius,
    const std::vector<double> &all_reconstruction_radii,
    const std::vector<double> &prescribed_half_widths,
    const double phase_field_activation_threshold)
  {
    const PrescribedInitialFault<dim> &prescribed_fault =
      prescribed_faults[fault_index];

    // Resample the prescribed geometry into structural Q1 nodes and interpolate
    // the projection widths onto that common parameterization.
    std::vector<Point<dim>> reference_points;
    if (fit_prescribed_geometry)
      reference_points = ReconstructedFaultUtilities::resample_reference_fault(
                           prescribed_fault.vertices, structural_spacing);
    else
      // Preserve supplied corners and boundary-condition transition vertices.
      // In fixed-geometry mode resample each input segment, not its whole length.
      for (unsigned int segment=0; segment+1<prescribed_fault.vertices.size(); ++segment)
        {
          const auto points = ReconstructedFaultUtilities::resample_reference_fault<dim>(
            {prescribed_fault.vertices[segment], prescribed_fault.vertices[segment+1]},
            structural_spacing);
          reference_points.insert(reference_points.end(), points.begin()+(segment>0), points.end());
        }
    const std::vector<Tensor<1,dim>> normals = reference_normals(reference_points);
    const unsigned int n_points = reference_points.size();

    InitialFaultReconstruction<dim> result;
    result.projection_half_widths.resize(n_points);
    for (unsigned int vertex = 0; vertex < n_points; ++vertex)
      {
        const StructuralCoordinates coordinates =
          structural_coordinates(prescribed_fault.vertices, reference_points[vertex]);
        result.projection_half_widths[vertex] =
          (1.0-coordinates.xi)
          * prescribed_half_widths[coordinates.segment]
          + coordinates.xi
          * prescribed_half_widths[coordinates.segment+1];
      }

    if (!fit_prescribed_geometry)
      {
        // Boundary-truncated profiles need not define an unbiased ridge fit.
        // Keep prescribed topology/coordinates, without changing admission widths.
        add_reconstructed_fault(reference_points, result.projection_half_widths);
        diagnostics.emplace_back();
        return;
      }

    std::vector<double> local_matrix(n_points*n_points, 0.0);
    std::vector<double> local_rhs(n_points, 0.0);
    std::vector<double> local_support(n_points, 0.0);
    double local_weight = 0.0;

    // Assemble this rank's phase-field-weighted least-squares fit for normal
    // offsets, excluding inactive QPs and rejecting overlapping fault supports.
    const QGauss<dim> quadrature(this->get_fe().degree + 1);
    FEValues<dim> fe_values(this->get_mapping(),
                            this->get_fe(), quadrature,
                            update_values | update_quadrature_points | update_JxW_values);
    const FEValuesExtractors::Scalar phase_field_component(
      this->introspection().variable("phase_field").first_component_index);
    std::vector<double> phase_field_values(quadrature.size());

    for (const auto &cell : this->get_dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          fe_values[phase_field_component].get_function_values(
            this->get_solution(), phase_field_values);
          for (unsigned int q = 0; q < quadrature.size(); ++q)
            {
              AssertThrow(std::isfinite(phase_field_values[q]),
                          ExcMessage("A non-finite Q1 phase-field value was encountered."));
              const double weight = std::max(
                                      phase_field_values[q] - phase_field_activation_threshold, 0.0);
              if (weight == 0.0)
                continue;

              const StructuralCoordinates coordinates = structural_coordinates(
                                                          reference_points, fe_values.quadrature_point(q));
              if (coordinates.distance > reconstruction_radius)
                continue;

              for (unsigned int other_fault = fault_index + 1;
                   other_fault < prescribed_faults.size(); ++other_fault)
                {
                  const double other_distance =
                    ReconstructedFaultUtilities::closest_point_distance_and_core_phase_field(
                      prescribed_faults[other_fault], fe_values.quadrature_point(q)).first;
                  AssertThrow(other_distance > all_reconstruction_radii[other_fault],
                              ExcMessage("The active reconstruction regions of prescribed faults "
                                         + Utilities::int_to_string(fault_index) + " and "
                                         + Utilities::int_to_string(other_fault) + " overlap."));
                }

              const double factor = fe_values.JxW(q) * weight;
              const double shape_values[2] = {1.0-coordinates.xi, coordinates.xi};
              local_weight += factor;
              for (unsigned int a = 0; a < 2; ++a)
                {
                  const unsigned int row = coordinates.segment + a;
                  local_rhs[row] += factor * shape_values[a] * coordinates.signed_distance;
                  local_support[row] += factor * shape_values[a];
                  for (unsigned int b = 0; b < 2; ++b)
                    local_matrix[row*n_points + coordinates.segment+b] +=
                      factor * shape_values[a] * shape_values[b];
                }
            }
        }

    // The fault geometry is replicated, so collectively sum the small dense
    // fit and coverage diagnostics before solving identically on every rank.
    std::vector<double> matrix(local_matrix.size());
    std::vector<double> rhs(local_rhs.size());
    result.diagnostics.structural_support.resize(local_support.size());
    Utilities::MPI::sum(local_matrix, this->get_mpi_communicator(), matrix);
    Utilities::MPI::sum(local_rhs, this->get_mpi_communicator(), rhs);
    Utilities::MPI::sum(local_support, this->get_mpi_communicator(),
                        result.diagnostics.structural_support);
    result.diagnostics.total_weight = Utilities::MPI::sum(
                                        local_weight, this->get_mpi_communicator());
    AssertThrow(std::isfinite(result.diagnostics.total_weight)
                && result.diagnostics.total_weight > 0.0,
                ExcMessage("Fault " + Utilities::int_to_string(fault_index)
                           + " has no phase-field reconstruction weight."));

    for (unsigned int vertex = 0; vertex < n_points; ++vertex)
      AssertThrow(std::isfinite(result.diagnostics.structural_support[vertex])
                  && result.diagnostics.structural_support[vertex] > 0.0,
                  ExcMessage("A structural vertex of fault "
                             + Utilities::int_to_string(fault_index)
                             + " has no phase-field support."));

    // Ridge-regularized offsets move only along prescribed normals. The fitted
    // fault must remain inside the tubular region used to construct the fit.
    result.diagnostics.offsets = ReconstructedFaultUtilities::solve_normal_offsets(
                                   matrix, rhs, result.diagnostics.total_weight, ridge_coefficient);
    result.vertices.resize(n_points);
    for (unsigned int vertex = 0; vertex < n_points; ++vertex)
      {
        AssertThrow(std::abs(result.diagnostics.offsets[vertex])
                    <= reconstruction_radius,
                    ExcMessage("The fitted offset of fault "
                               + Utilities::int_to_string(fault_index)
                               + " leaves its tubular reconstruction region."));
        result.vertices[vertex] = reference_points[vertex]
                                  + result.diagnostics.offsets[vertex] * normals[vertex];
      }
    add_reconstructed_fault(result.vertices, result.projection_half_widths);
    diagnostics.push_back(std::move(result.diagnostics));
  }



  template <int dim>
  void ReconstructedFaultManager<dim>::set_shear_sense(
    const unsigned int fault_index, const int sense)
  {
    AssertThrow(fault_index < reconstructed_faults.size()
                && (sense == -1 || sense == 1),
                ExcMessage("Fault shear sense must be +1 or -1 on an existing fault."));
    const auto existing = shear_senses.find(fault_index);
    if (existing != shear_senses.end())
      {
        AssertThrow(existing->second == sense,
                    ExcMessage("Cannot change an initialized/checkpointed fault shear sense."));
        return;
      }
    AssertThrow(!slip_rate_nonlinear_solve_active
                && stokes_qp_cache_diagnostics.rebuild_count == 0,
                ExcMessage("Configure fault shear sense before mechanical cache construction."));
    shear_senses.emplace(fault_index, sense);
    invalidate_stokes_qp_projection_cache();
  }


  template <int dim>
  int ReconstructedFaultManager<dim>::get_shear_sense(const unsigned int fault_index) const
  {
    AssertIndexRange(fault_index, reconstructed_faults.size());
    const auto entry = shear_senses.find(fault_index);
    return entry == shear_senses.end() ? 1 : entry->second;
  }


  template <int dim>
  unsigned int
  ReconstructedFaultManager<dim>::add_reconstructed_fault(
    const std::vector<Point<dim>> &vertices,
    const std::vector<double> &half_widths)
  {
    Assert(!slip_rate_nonlinear_solve_active && !slip_rate_trial_active,
           ExcMessage("Reconstructed-fault geometry cannot change during a nonlinear solve."));
    AssertThrow(vertices.size() >= 2,
                ExcMessage("A reconstructed fault must contain at least two vertices."));
    AssertThrow(half_widths.size() == vertices.size(),
                ExcMessage("A reconstructed fault requires one projection half width per vertex."));
    AssertThrow(vertices.front() != vertices.back(),
                ExcMessage("Closed-loop reconstructed faults are unsupported."));
    for (const double half_width : half_widths)
      AssertThrow(std::isfinite(half_width) && half_width > 0.0,
                  ExcMessage("Reconstructed-fault projection half widths must be positive and finite."));

    ReconstructedFault<dim> fault(vertices);
    fault.initialize_properties(n_property_components);
    reconstructed_faults.push_back(std::move(fault));
    projection_half_widths.push_back(half_widths);
    timestep_committed_slip_rates.emplace_back();
    current_newton_slip_rates.emplace_back();
    trial_slip_rates.emplace_back();
    slip_rate_initialized.push_back(false);
    prescribed_slip_rates.emplace_back();
    ++projection_metadata_version;
    invalidate_particle_projection_cache();
    invalidate_stokes_qp_projection_cache();
    return reconstructed_faults.size() - 1;
  }


  // -----------------------------------------------------------------------------
  // Property registration
  // -----------------------------------------------------------------------------

  template <int dim>
  unsigned int
  ReconstructedFaultManager<dim>::register_property(
    const std::string &name,
    const unsigned int n_components)
  {
    AssertThrow(reconstructed_faults.empty(),
                ExcMessage("Reconstructed-fault vertex properties must be registered "
                           "before reconstructed geometry exists."));
    AssertThrow(!name.empty(),
                ExcMessage("Reconstructed-fault vertex property names must not be empty."));
    AssertThrow(name != "slip_rate",
                ExcMessage("The reconstructed-fault vertex property name <slip_rate> is reserved "
                           "for the distinguished kinematic field."));
    AssertThrow(n_components > 0,
                ExcMessage("Reconstructed-fault vertex properties must have at least one component."));
    AssertThrow(!has_property(name),
                ExcMessage("A reconstructed-fault vertex property named <" + name
                           + "> is already registered."));

    const unsigned int property_index = property_information.size();
    property_information.push_back({name, n_components, n_property_components});
    property_indices.emplace(name, property_index);
    n_property_components += n_components;
    return property_index;
  }


  template <int dim>
  bool
  ReconstructedFaultManager<dim>::has_property(const std::string &name) const
  {
    return property_indices.find(name) != property_indices.end();
  }


  template <int dim>
  unsigned int
  ReconstructedFaultManager<dim>::get_property_index(const std::string &name) const
  {
    const auto property = property_indices.find(name);
    AssertThrow(property != property_indices.end(),
                ExcMessage("No reconstructed-fault vertex property named <" + name
                           + "> is registered."));
    return property->second;
  }


  template <int dim>
  const std::vector<typename ReconstructedFaultManager<dim>::PropertyInformation> &
  ReconstructedFaultManager<dim>::get_property_information() const
  {
    return property_information;
  }


  // -----------------------------------------------------------------------------
  // Restart reconstruction and serialization support
  // -----------------------------------------------------------------------------

  template <int dim>
  void
  ReconstructedFaultManager<dim>::rebuild_after_deserialization()
  {
    AssertThrow(projection_half_widths.size() == reconstructed_faults.size()
                && timestep_committed_slip_rates.size() == reconstructed_faults.size()
                && slip_rate_initialized.size() == reconstructed_faults.size(),
                ExcMessage("Invalid reconstructed-fault vector layout in checkpoint."));

    property_indices.clear();
    unsigned int expected_position = 0;
    for (unsigned int property = 0; property < property_information.size(); ++property)
      {
        const PropertyInformation &information = property_information[property];
        AssertThrow(!information.name.empty() && information.name != "slip_rate",
                    ExcMessage("Invalid reconstructed-fault property schema in checkpoint."));
        AssertThrow(information.n_components > 0
                    && information.position == expected_position
                    && property_indices.emplace(information.name, property).second,
                    ExcMessage("Invalid reconstructed-fault property schema in checkpoint."));
        expected_position += information.n_components;
      }
    AssertThrow(expected_position == n_property_components,
                ExcMessage("Invalid reconstructed-fault property component count in checkpoint."));

    for (unsigned int fault = 0; fault < reconstructed_faults.size(); ++fault)
      {
        AssertThrow(reconstructed_faults[fault].n_vertices() >= 2
                    && reconstructed_faults[fault].vertex(0)
                    != reconstructed_faults[fault].vertex(
                      reconstructed_faults[fault].n_vertices()-1),
                    ExcMessage("Invalid reconstructed-fault geometry in checkpoint."));
        for (unsigned int vertex = 0;
             vertex < reconstructed_faults[fault].n_vertices(); ++vertex)
          for (unsigned int d = 0; d < dim; ++d)
            AssertThrow(std::isfinite(reconstructed_faults[fault].vertex(vertex)[d]),
                        ExcMessage("Invalid reconstructed-fault vertex in checkpoint."));
        for (unsigned int segment = 0;
             segment < reconstructed_faults[fault].n_cells(); ++segment)
          {
            const double segment_length = reconstructed_faults[fault].vertex(segment).distance(
                                            reconstructed_faults[fault].vertex(segment+1));
            AssertThrow(std::isfinite(segment_length) && segment_length > 0.0,
                        ExcMessage("Degenerate reconstructed-fault segment in checkpoint."));
          }
        AssertThrow(projection_half_widths[fault].size()
                    == reconstructed_faults[fault].n_vertices(),
                    ExcMessage("Invalid reconstructed-fault half-width layout in checkpoint."));
        for (const double half_width : projection_half_widths[fault])
          AssertThrow(std::isfinite(half_width) && half_width > 0.0,
                      ExcMessage("Invalid reconstructed-fault half width in checkpoint."));
        AssertThrow(reconstructed_faults[fault].n_property_components == n_property_components
                    && reconstructed_faults[fault].property_values.size()
                    == reconstructed_faults[fault].n_vertices() * n_property_components,
                    ExcMessage("Invalid reconstructed-fault property layout in checkpoint."));
        for (unsigned int vertex = 0;
             vertex < reconstructed_faults[fault].n_vertices(); ++vertex)
          for (unsigned int component = 0;
               component < n_property_components; ++component)
            if (reconstructed_faults[fault].property_value_is_initialized(vertex,
                                                                          component))
              AssertThrow(std::isfinite(
                            reconstructed_faults[fault].get_properties(vertex)[component]),
                          ExcMessage("Invalid reconstructed-fault property value in checkpoint."));
        if (slip_rate_initialized[fault])
          {
            AssertThrow(timestep_committed_slip_rates[fault].size()
                        == reconstructed_faults[fault].n_vertices(),
                        ExcMessage("Invalid reconstructed-fault slip-rate layout in checkpoint."));
            for (const double value : timestep_committed_slip_rates[fault])
              AssertThrow(std::isfinite(value) && value >= 0.0,
                          ExcMessage("Invalid reconstructed-fault slip rate in checkpoint: "
                                     "values must be finite and nonnegative."));
          }
        else
          AssertThrow(timestep_committed_slip_rates[fault].empty(),
                      ExcMessage("Uninitialized slip-rate checkpoint data must be empty."));
      }

    current_newton_slip_rates = timestep_committed_slip_rates;
    trial_slip_rates.assign(reconstructed_faults.size(), {});
    slip_rate_nonlinear_solve_active = false;
    slip_rate_trial_active = false;
    // Rebuild the transient per-fault layout; callers reapply prescribed rows.
    prescribed_slip_rates.assign(reconstructed_faults.size(), {});
    diagnostics.clear();
    boundary_contacts_valid = false;
    automatic_source_ready = false;
    ++projection_metadata_version;
    invalidate_particle_projection_cache();
    invalidate_stokes_qp_projection_cache();
  }


  template <int dim>
  void
  ReconstructedFaultManager<dim>::enable_top_source_continuation()
  {
    AssertThrow(bottom_source_fault!=numbers::invalid_unsigned_int,
                ExcMessage("Configure the straight through-bottom fault before its top continuation."));
    const auto &fault=reconstructed_faults[bottom_source_fault];
    AssertThrow(std::abs(fault.vertex(fault.n_vertices()-1)[1]-source_box_upper[1])
                <1e-10*(source_box_upper-source_box_lower).norm(),
                ExcMessage("The last fault vertex must cross the physical top."));
    if (!top_source_continuation) invalidate_stokes_qp_projection_cache();
    top_source_continuation=true;
  }


  template <int dim>
  ReconstructedFaultUtilities::NormalProfileProjection
  ReconstructedFaultManager<dim>::project_to_normal_profiles(
    const Point<dim> &position) const
  {
    ReconstructedFaultUtilities::internal::
    validate_normal_profile_projection_position(position);
    return ReconstructedFaultUtilities::internal::project_to_normal_profiles_unchecked(
             reconstructed_faults, projection_half_widths, position);
  }


  // -----------------------------------------------------------------------------
  // Stokes quadrature-point geometry cache
  // -----------------------------------------------------------------------------

  template <int dim>
  void
  ReconstructedFaultManager<dim>::enable_bottom_source_continuation(
    const unsigned int fault_index, const Point<dim> &lower, const Point<dim> &upper)
  {
    AssertThrow(!automatic_boundary_completion,
                ExcMessage("Legacy and automatic boundary source continuation are mutually exclusive."));
    AssertThrow(dim == 2 && fault_index < reconstructed_faults.size(),
                ExcMessage("Bottom source continuation requires a 2-D fault."));
    for (unsigned int d=0; d<dim; ++d)
      AssertThrow(std::isfinite(lower[d]) && std::isfinite(upper[d]) && upper[d]>lower[d],
                  ExcMessage("Invalid physical box for source continuation."));
    const auto &fault=reconstructed_faults[fault_index];
    const double tolerance=1e-10*(upper-lower).norm();
    const Point<dim> origin=fault.vertex(0);
    Tensor<1,dim> tangent=fault.vertex(1)-origin;
    tangent/=tangent.norm();
    AssertThrow(std::abs(origin[1]-lower[1])<tolerance && tangent[1]>0.,
                ExcMessage("The first fault vertex must cross the physical bottom upwards."));
    for (unsigned int v=0; v<fault.n_vertices(); ++v)
      {
        const Tensor<1,dim> offset=fault.vertex(v)-origin;
        AssertThrow((offset-(offset*tangent)*tangent).norm()<tolerance,
                    ExcMessage("Bottom source continuation requires a straight fault."));
      }
    if (bottom_source_fault!=fault_index || source_box_lower!=lower || source_box_upper!=upper)
      invalidate_stokes_qp_projection_cache();
    bottom_source_fault=fault_index;
    source_box_lower=lower;
    source_box_upper=upper;
  }


  template <int dim>
  ReconstructedFaultUtilities::NormalProfileProjection
  ReconstructedFaultManager<dim>::project_to_bulk_source(
    const Point<dim> &position, const bool normal_profiles_checked) const
  {
    ReconstructedFaultUtilities::NormalProfileProjection result;
    if (!normal_profiles_checked)
      result=ReconstructedFaultUtilities::internal::project_to_normal_profiles_unchecked(
        reconstructed_faults, projection_half_widths, position);
    if (automatic_boundary_completion)
      {
        AssertThrow(automatic_source_ready, ExcMessage("Automatic boundary completion must be qualified before assembly."));
        const double tolerance=boundary_geometry_tolerance;
        for (const auto &contact:boundary_contacts)
          {
            const auto offset=position-contact.position;
            const double s=offset*contact.inward_tangent, r=offset*contact.normal;
            const auto &fault=reconstructed_faults[contact.fault_index];
            const unsigned int segment=contact.endpoint==0 ? 0 : fault.n_cells()-1;
            const auto &a=fault.vertex(segment), &b=fault.vertex(segment+1);
            // Bound endpoint subtraction/dot-product roundoff and the angular
            // error from forming the resampled terminal tangent. This is a
            // floating-point handoff band, not a physical extension of support.
            const double endpoint_roundoff=32.*std::numeric_limits<double>::epsilon()
              *(position.norm()+contact.position.norm()
                +offset.norm()*(1.+(a.norm()+b.norm())/a.distance(b)));
            if (contact.influence_length==0. || s>endpoint_roundoff
                || offset*contact.inward_boundary_normal < -tolerance
                || std::abs(r)>contact.transverse_extent) continue;
            // This enclosure contains every nonzero physical Q1 cell in the
            // verified terminal profile. It is not the normal-profile cutoff.
            // Raw and resampled endpoint frames can put the same point on
            // opposite sides of s=0 at roundoff. The strip owns an already
            // admitted point; otherwise the endpoint fills this narrow gap.
            if (result.active && result.fault_index==contact.fault_index
                && s>=-endpoint_roundoff)
              continue;
            AssertThrow(!result.active, ExcMessage("Overlapping physical boundary continuation associations."));
            result.active=true;
            result.fault_index=contact.fault_index;
            result.segment_index=segment;
            result.xi=contact.endpoint==0 ? 0. : 1.;
            result.signed_distance=r;
          }
        return result;
      }
    if (result.active || bottom_source_fault==numbers::invalid_unsigned_int)
      return result;
    for (unsigned int d=0; d<dim; ++d)
      if (position[d]<source_box_lower[d] || position[d]>source_box_upper[d])
        return result;

    const auto &fault=reconstructed_faults[bottom_source_fault];
    Tensor<1,dim> tangent=fault.vertex(1)-fault.vertex(0), normal;
    tangent/=tangent.norm();
    normal[0]=-tangent[1]; normal[1]=tangent[0];
    const Tensor<1,dim> offset=position-fault.vertex(0);
    const double distance=offset*normal;
    // Only the in-box tangent-extension wedge gets endpoint fields. Do not
    // reapply the segment's normal cutoff here: physical Q1 phase can remain
    // positive beyond it. Zero physical phase produces exactly zero source.
    const bool beyond_top=top_source_continuation
      && (position-fault.vertex(fault.n_vertices()-1))*tangent>0.;
    if (offset*tangent<0. || beyond_top)
      {
        result.active=true;
        result.fault_index=bottom_source_fault;
        result.segment_index=beyond_top ? fault.n_cells()-1 : 0;
        result.xi=beyond_top ? 1. : 0.;
        result.signed_distance=distance;
      }
    return result;
  }

  template <int dim>
  bool
  ReconstructedFaultManager<dim>::stokes_qp_projection_cache_is_valid() const
  {
    if (!stokes_qp_projection_cache_valid
        || cached_stokes_qp_projection_metadata_version
        != projection_metadata_version
        || cached_stokes_qp_fault_geometry_versions.size()
        != reconstructed_faults.size())
      return false;

    for (unsigned int fault = 0; fault < reconstructed_faults.size(); ++fault)
      if (cached_stokes_qp_fault_geometry_versions[fault]
          != reconstructed_faults[fault].geometry_version())
        return false;
    return true;
  }


  template <int dim>
  void
  ReconstructedFaultManager<dim>::rebuild_stokes_qp_projection_cache()
  {
    TimerOutput::Scope timer(this->get_computing_timer(), "Fault: Stokes QP cache build");
    AssertThrow(dim == 2, ExcNotImplemented());
    AssertThrow(!reconstructed_faults.empty(),
                ExcMessage("The Stokes QP projection cache requires reconstructed "
                           "fault geometry."));
    ReconstructedFaultUtilities::internal::
    validate_normal_profile_projection_geometry(reconstructed_faults,
                                                projection_half_widths);

    // Build associations from the exact velocity quadrature used by the Stokes
    // assembler; cell id and QP order are part of the cache identity.
    const Quadrature<dim> &quadrature =
      this->introspection().quadratures.velocities;
    FEValues<dim> fe_values(this->get_mapping(), this->get_fe(), quadrature,
                            update_quadrature_points);
    std::map<CellId, std::vector<StokesQPFaultAssociation>> candidate;
    unsigned int n_active_q_points = 0;

    for (const auto &cell : this->get_dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          std::vector<StokesQPFaultAssociation> entries(quadrature.size());
          for (unsigned int q = 0; q < quadrature.size(); ++q)
            {
              StokesQPFaultAssociation &entry = entries[q];
              entry.position = fe_values.quadrature_point(q);
              const ReconstructedFaultUtilities::NormalProfileProjection projection =
                project_to_bulk_source(entry.position);
              if (!projection.active)
                continue;

              entry.active = true;
              entry.fault_index = projection.fault_index;
              entry.segment_index = projection.segment_index;
              entry.xi = projection.xi;
              entry.shape_1 = entry.xi;
              entry.shape_0 = 1.0-entry.shape_1;
              entry.signed_distance = projection.signed_distance;
              const ReconstructedFault<dim> &fault =
                reconstructed_faults[projection.fault_index];
              entry.tangent = fault.vertex(projection.segment_index+1)
                              - fault.vertex(projection.segment_index);
              entry.tangent /= entry.tangent.norm();
              entry.normal[0] = -entry.tangent[1];
              entry.normal[1] = entry.tangent[0];
              ++n_active_q_points;
            }
          candidate.emplace(cell->id(), std::move(entries));
        }

    // Publish only a complete cache, together with the geometry/quadrature
    // generations that later B and residual actions validate before reuse.
    stokes_qp_projection_cache = std::move(candidate);
    cached_stokes_quadrature_points = quadrature.get_points();
    cached_stokes_quadrature_weights = quadrature.get_weights();
    cached_stokes_qp_fault_geometry_versions.resize(reconstructed_faults.size());
    for (unsigned int fault = 0; fault < reconstructed_faults.size(); ++fault)
      cached_stokes_qp_fault_geometry_versions[fault] =
        reconstructed_faults[fault].geometry_version();
    cached_stokes_qp_projection_metadata_version = projection_metadata_version;
    stokes_qp_cache_diagnostics.n_active_q_points = n_active_q_points;
    ++stokes_qp_cache_diagnostics.rebuild_count;
    stokes_qp_projection_cache_valid = true;
  }


  template <int dim>
  void
  ReconstructedFaultManager<dim>::prepare_stokes_qp_projection_cache()
  {
    if (!stokes_qp_projection_cache_is_valid())
      rebuild_stokes_qp_projection_cache();
  }


  template <int dim>
  const std::vector<typename ReconstructedFaultManager<dim>::
  StokesQPFaultAssociation> &
  ReconstructedFaultManager<dim>::get_stokes_qp_fault_associations(
    const CellId &cell_id,
    const Quadrature<dim> &quadrature,
    const std::vector<Point<dim>> &quadrature_points) const
  {
    (void)quadrature;
    AssertThrow(stokes_qp_projection_cache_is_valid(),
                ExcMessage("The reconstructed-fault Stokes QP projection cache "
                           "is stale. Rebuild it before bulk assembly."));
    Assert(quadrature.get_points() == cached_stokes_quadrature_points,
           ExcMessage("The reconstructed-fault QP cache reference-point order "
                      "does not match the Stokes assembler quadrature."));
    Assert(quadrature.get_weights() == cached_stokes_quadrature_weights,
           ExcMessage("The reconstructed-fault QP cache weights do not match "
                      "the Stokes assembler quadrature."));
    const auto cell = stokes_qp_projection_cache.find(cell_id);
    Assert(cell != stokes_qp_projection_cache.end(), ExcInternalError());
    AssertDimension(cell->second.size(), quadrature.size());
    AssertDimension(quadrature_points.size(), cell->second.size());
    for (unsigned int q = 0; q < quadrature_points.size(); ++q)
      Assert(cell->second[q].position == quadrature_points[q],
             ExcMessage("The reconstructed-fault QP cache order does not match "
                        "the Stokes FEValues quadrature-point order."));
    return cell->second;
  }


  template <int dim>
  void
  ReconstructedFaultManager<dim>::invalidate_stokes_qp_projection_cache()
  {
    stokes_qp_projection_cache_valid = false;
    stokes_qp_projection_cache.clear();
    cached_stokes_qp_fault_geometry_versions.clear();
    cached_stokes_quadrature_points.clear();
    cached_stokes_quadrature_weights.clear();
    stokes_qp_cache_diagnostics.n_active_q_points = 0;
  }


  template <int dim>
  const typename ReconstructedFaultManager<dim>::StokesQPCacheDiagnostics &
  ReconstructedFaultManager<dim>::get_stokes_qp_cache_diagnostics() const
  {
    return stokes_qp_cache_diagnostics;
  }


  // -----------------------------------------------------------------------------
  // Fault access and diagnostics
  // -----------------------------------------------------------------------------

  template <int dim>
  const std::vector<ReconstructedFault<dim>> &
  ReconstructedFaultManager<dim>::get_faults() const
  {
    return reconstructed_faults;
  }


  template <int dim>
  ReconstructedFault<dim> &
  ReconstructedFaultManager<dim>::get_fault(const unsigned int fault_index)
  {
    AssertIndexRange(fault_index, reconstructed_faults.size());
    return reconstructed_faults[fault_index];
  }


  template <int dim>
  const ReconstructedFault<dim> &
  ReconstructedFaultManager<dim>::get_fault(const unsigned int fault_index) const
  {
    AssertIndexRange(fault_index, reconstructed_faults.size());
    return reconstructed_faults[fault_index];
  }


  template <int dim>
  const std::vector<FaultReconstructionDiagnostics> &
  ReconstructedFaultManager<dim>::get_diagnostics() const
  {
    return diagnostics;
  }


}


// -----------------------------------------------------------------------------
// Explicit instantiations

namespace aspect
{
#define INSTANTIATE(dim) template class ReconstructedFaultManager<dim>;

  ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
}
