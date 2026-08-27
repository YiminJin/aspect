/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include <aspect/reconstructed_fault.h>
#include <aspect/phase_field.h>
#include <aspect/particle/manager.h>
#include <aspect/material_model/utilities.h>
#include <aspect/utilities.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <tuple>

namespace aspect
{
  namespace
  {
    template <int dim>
    std::vector<Point<dim>>
    resample_polyline(const std::vector<Point<dim>> &vertices,
                      const double spacing)
    {
      AssertThrow(dim == 2, ExcNotImplemented());
      AssertThrow(vertices.size() >= 2, ExcMessage("A reference fault requires at least two vertices."));
      AssertThrow(std::isfinite(spacing) && spacing > 0.0,
                  ExcMessage("The fault structural point spacing must be positive and finite."));

      std::vector<double> cumulative_length(vertices.size(), 0.0);
      for (unsigned int i = 1; i < vertices.size(); ++i)
        {
          const double segment_length = vertices[i].distance(vertices[i-1]);
          AssertThrow(std::isfinite(segment_length) && segment_length > 0.0,
                      ExcMessage("The reference fault contains a degenerate segment."));
          cumulative_length[i] = cumulative_length[i-1] + segment_length;
        }

      const double length = cumulative_length.back();
      const unsigned int n_segments = std::max(1U, static_cast<unsigned int>(std::ceil(length / spacing)));
      const double structural_spacing = length / n_segments;
      std::vector<Point<dim>> points(n_segments + 1);
      unsigned int input_segment = 0;
      for (unsigned int i = 0; i <= n_segments; ++i)
        {
          const double s = (i == n_segments ? length : i * structural_spacing);
          while (input_segment + 1 < cumulative_length.size() - 1
                 && s > cumulative_length[input_segment+1])
            ++input_segment;
          const double local_coordinate =
            (s - cumulative_length[input_segment])
            / (cumulative_length[input_segment+1] - cumulative_length[input_segment]);
          points[i] = vertices[input_segment]
                      + local_coordinate * (vertices[input_segment+1] - vertices[input_segment]);
        }
      return points;
    }


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
                      ExcMessage("The resampled reference fault has an invalid tangent."));
          tangent /= norm;
          normals[i][0] = -tangent[1];
          normals[i][1] = tangent[0];
        }
      return normals;
    }


    template <int dim>
    std::tuple<double,double,unsigned int,double>
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
  }


  namespace ReconstructedFaultUtilities
  {
    template <int dim>
    std::vector<Point<dim>>
    resample_reference_fault(const std::vector<Point<dim>> &vertices,
                             const double structural_spacing)
    {
      return resample_polyline(vertices, structural_spacing);
    }


    std::vector<double>
    solve_normal_offsets(const std::vector<double> &matrix_values,
                         const std::vector<double> &rhs_values,
                         const double total_weight,
                         const double ridge_coefficient)
    {
      const unsigned int n_points = rhs_values.size();
      AssertDimension(matrix_values.size(), n_points*n_points);
      AssertThrow(std::isfinite(total_weight) && total_weight > 0.0,
                  ExcMessage("The total phase-field reconstruction weight must be positive."));
      AssertThrow(std::isfinite(ridge_coefficient) && ridge_coefficient >= 0.0,
                  ExcMessage("The fault reconstruction ridge coefficient must be nonnegative."));

      FullMatrix<double> system(n_points, n_points);
      Vector<double> rhs(n_points), offsets(n_points);
      for (unsigned int i = 0; i < n_points; ++i)
        {
          rhs[i] = rhs_values[i] / total_weight;
          for (unsigned int j = 0; j < n_points; ++j)
            system(i,j) = matrix_values[i*n_points+j] / total_weight;
        }
      for (unsigned int j = 1; j + 1 < n_points; ++j)
        for (unsigned int a = 0; a < 3; ++a)
          for (unsigned int b = 0; b < 3; ++b)
            {
              const double d[3] = {1.0, -2.0, 1.0};
              system(j-1+a,j-1+b) += ridge_coefficient * d[a] * d[b];
            }

      system.gauss_jordan();
      system.vmult(offsets, rhs);
      return std::vector<double>(offsets.begin(), offsets.end());
    }


    template <int dim>
    std::vector<PrescribedInitialFault<dim>>
    parse_prescribed_faults(const std::string &file_contents,
                            const std::string &filename)
    {
      std::vector<PrescribedInitialFault<dim>> faults;
      PrescribedInitialFault<dim> fault;
      std::istringstream input(file_contents);
      std::string line;
      unsigned int line_number = 0;

      const auto finish_fault = [&]()
      {
        AssertThrow(fault.vertices.size() >= 2,
                    ExcMessage("Prescribed-fault file <" + filename
                               + "> contains a fault with fewer than two vertices near line "
                               + Utilities::int_to_string(line_number) + "."));
        faults.push_back(std::move(fault));
        fault = PrescribedInitialFault<dim>();
      };

      while (std::getline(input, line))
        {
          ++line_number;
          const std::size_t comment_position = line.find('#');
          if (comment_position != std::string::npos)
            line.erase(comment_position);

          std::istringstream line_stream(line);
          std::string first_entry;
          if (!(line_stream >> first_entry))
            continue;

          if (first_entry == "---")
            {
              std::string extra_entry;
              AssertThrow(!(line_stream >> extra_entry),
                          ExcMessage("Fault separator on line "
                                     + Utilities::int_to_string(line_number)
                                     + " of <" + filename + "> must contain only `---'."));
              AssertThrow(!fault.vertices.empty(),
                          ExcMessage("Empty fault before separator on line "
                                     + Utilities::int_to_string(line_number)
                                     + " of <" + filename + ">."));
              finish_fault();
              continue;
            }

          Point<dim> point;
          try
            {
              point[0] = Utilities::string_to_double(first_entry);
            }
          catch (const std::exception &)
            {
              AssertThrow(false,
                          ExcMessage("Could not read the first coordinate on line "
                                     + Utilities::int_to_string(line_number)
                                     + " of prescribed-fault file <" + filename + ">."));
            }
          for (unsigned int d = 1; d < dim; ++d)
            AssertThrow(line_stream >> point[d],
                        ExcMessage("Line " + Utilities::int_to_string(line_number)
                                   + " of prescribed-fault file <" + filename
                                   + "> contains fewer than "
                                   + Utilities::int_to_string(dim) + " coordinates."));
          double phi_hat;
          AssertThrow(line_stream >> phi_hat,
                      ExcMessage("Line " + Utilities::int_to_string(line_number)
                                 + " of prescribed-fault file <" + filename
                                 + "> is missing the core phase-field value."));
          std::string extra_entry;
          AssertThrow(!(line_stream >> extra_entry),
                      ExcMessage("Line " + Utilities::int_to_string(line_number)
                                 + " of prescribed-fault file <" + filename
                                 + "> contains too many entries."));
          for (unsigned int d = 0; d < dim; ++d)
            AssertThrow(std::isfinite(point[d]),
                        ExcMessage("A coordinate on line "
                                   + Utilities::int_to_string(line_number)
                                   + " of prescribed-fault file <" + filename
                                   + "> is not finite."));
          AssertThrow(std::isfinite(phi_hat),
                      ExcMessage("The core phase-field value on line "
                                 + Utilities::int_to_string(line_number)
                                 + " of prescribed-fault file <" + filename
                                 + "> is not finite."));
          fault.vertices.push_back(point);
          fault.core_phase_field_values.push_back(phi_hat);
        }

      if (!fault.vertices.empty())
        finish_fault();
      AssertThrow(!faults.empty(),
                  ExcMessage("Prescribed-fault file <" + filename
                             + "> does not contain any faults."));
      return faults;
    }


    template <int dim>
    std::pair<double, double>
    closest_point_distance_and_core_phase_field(const PrescribedInitialFault<dim> &fault,
                                                 const Point<dim> &position)
    {
      AssertThrow(dim == 2,
                  ExcMessage("Prescribed initial fault geometry is currently implemented only in 2D."));
      AssertThrow(fault.vertices.size() >= 2,
                  ExcMessage("A prescribed initial fault requires at least two vertices."));
      AssertThrow(fault.core_phase_field_values.size() == fault.vertices.size(),
                  ExcMessage("A prescribed initial fault requires one core phase-field value per vertex."));

      double min_r_squared = std::numeric_limits<double>::infinity();
      double phi_hat = numbers::signaling_nan<double>();

      for (unsigned int segment = 0; segment < fault.vertices.size() - 1; ++segment)
        {
          const Tensor<1,dim> segment_vector =
            fault.vertices[segment+1] - fault.vertices[segment];
          const double squared_segment_length = segment_vector.norm_square();
          AssertThrow(std::isfinite(squared_segment_length) && squared_segment_length > 0.0,
                      ExcMessage("Prescribed initial fault segment "
                                 + Utilities::int_to_string(segment) + " is degenerate."));

          const double segment_coordinate =
            std::clamp(((position - fault.vertices[segment]) * segment_vector)
                       / squared_segment_length,
                       0.0,
                       1.0);
          const Point<dim> closest_point = fault.vertices[segment]
                                           + segment_coordinate * segment_vector;
          const double r_squared = position.distance_square(closest_point);

          if (r_squared < min_r_squared)
            {
              min_r_squared = r_squared;
              phi_hat =
                (1.0 - segment_coordinate) * fault.core_phase_field_values[segment]
                + segment_coordinate * fault.core_phase_field_values[segment+1];
            }
        }

      return {std::sqrt(min_r_squared), phi_hat};
    }


    template <int dim>
    void
    initialize_crack_driving_force(
      PhaseFieldHandler<dim> &phase_field_handler,
      const std::vector<PrescribedInitialFault<dim>> &faults)
    {
      if (faults.empty())
        return;

      AssertThrow(dim == 2,
                  ExcMessage("Prescribed initial fault initialization is currently implemented only in 2D."));

      const auto *phase_field_model =
        dynamic_cast<const MaterialModel::PhaseFieldModel<dim> *>(
          &phase_field_handler.get_material_model());
      AssertThrow(phase_field_model != nullptr,
                  ExcMessage("Prescribed initial faults require a phase-field material model."));
      const std::pair<double, double> phase_field_range = phase_field_model->get_phase_field_range();

      for (unsigned int fault_index = 0; fault_index < faults.size(); ++fault_index)
        {
          const PrescribedInitialFault<dim> &fault = faults[fault_index];
          closest_point_distance_and_core_phase_field(fault, Point<dim>());
          for (unsigned int vertex = 0; vertex < fault.core_phase_field_values.size(); ++vertex)
            AssertThrow(fault.core_phase_field_values[vertex] >= phase_field_range.first
                        && fault.core_phase_field_values[vertex] <= phase_field_range.second,
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
           phase_field_handler.introspection().chemical_composition_field_indices())
        {
          AssertThrow(phase_field_handler.get_parameters().compositional_field_methods[field_index]
                      == Parameters<dim>::AdvectionFieldMethod::particles,
                      ExcMessage("Prescribed initial faults require every chemical composition field "
                                 "to be advected by particles."));
          const auto mapped_property =
            phase_field_handler.get_parameters().mapped_particle_properties.find(field_index);
          AssertThrow(mapped_property
                      != phase_field_handler.get_parameters().mapped_particle_properties.end(),
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
                closest_point_distance_and_core_phase_field(faults[fault_index],
                                                             particle.get_location());
              const double r = r_and_phi_hat.first;
              const double phi_hat = r_and_phi_hat.second;

              auto profiles = profile_cache.find(phi_hat);
              if (profiles == profile_cache.end())
                profiles = profile_cache.emplace(
                  phi_hat,
                  phase_field_handler.get_phase_field_profiles(phi_hat)).first;

              AssertThrow(profiles->second.size() == volume_fractions.size(),
                          ExcMessage("The number of material-specific phase-field profiles does not "
                                     "match the number of particle composition fractions."));
              std::vector<double> material_phase_fields(profiles->second.size());
              for (unsigned int material = 0; material < profiles->second.size(); ++material)
                material_phase_fields[material] = profiles->second[material]->value(r);

              const double phase_field = MaterialModel::MaterialUtilities::average_value(
                volume_fractions,
                material_phase_fields,
                MaterialModel::MaterialUtilities::arithmetic);

              if (phase_field > phase_field_range.first)
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
  }


  template <int dim>
  ReconstructedFaultManager<dim>::ReconstructedFaultManager(const Simulator<dim> &simulator)
  {
    this->initialize_simulator(simulator);
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

    // Particle managers connect to this signal before the reconstructed-fault
    // manager is parsed. Consequently, their slots generate and initialize
    // particles (and particle domains) before this slot initializes H. The
    // signal is emitted for every initial adaptive-refinement cycle that sets
    // initial conditions, before the corresponding timestep solve.
    this->get_signals().post_set_initial_state.connect(
      [this] (const SimulatorAccess<dim> &)
    {
      initial_reconstruction_complete = false;
      ReconstructedFaultUtilities::initialize_crack_driving_force(
        this->get_phase_field_handler(), prescribed_faults);
    });
  }


  template <int dim>
  void
  ReconstructedFaultManager<dim>::initialize_crack_driving_force(
    PhaseFieldHandler<dim> &phase_field_handler,
    const std::vector<PrescribedInitialFault<dim>> &faults)
  {
    prescribed_faults = faults;
    ReconstructedFaultUtilities::initialize_crack_driving_force(phase_field_handler, faults);
  }


  template <int dim>
  void
  ReconstructedFaultManager<dim>::reconstruct_initial_faults(
    PhaseFieldHandler<dim> &phase_field_handler)
  {
    if (initial_reconstruction_complete || prescribed_faults.empty())
      return;

    this->get_pcout() << "   Reconstruction initial faults... " << std::flush;

    AssertThrow(dim == 2, ExcNotImplemented());
    const auto *phase_field_model = dynamic_cast<const MaterialModel::PhaseFieldModel<dim> *>(
      &phase_field_handler.get_material_model());
    AssertThrow(phase_field_model != nullptr,
                ExcMessage("Fault reconstruction requires a phase-field material model."));
    const double phi_min = phase_field_model->get_phase_field_range().first;

    double local_cell_margin = 0.0;
    for (const auto &cell : phase_field_handler.get_dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        local_cell_margin = std::max(local_cell_margin, cell->diameter());
    const double cell_margin = Utilities::MPI::max(local_cell_margin,
                                                    phase_field_handler.get_mpi_communicator());

    reconstructed_faults.clear();
    diagnostics.clear();
    std::vector<double> reconstruction_radii(prescribed_faults.size(), cell_margin);
    for (unsigned int fault_index = 0; fault_index < prescribed_faults.size(); ++fault_index)
      for (const double phi_hat : prescribed_faults[fault_index].core_phase_field_values)
        for (const auto &profile : phase_field_handler.get_phase_field_profiles(phi_hat))
          reconstruction_radii[fault_index] =
            std::max(reconstruction_radii[fault_index],
                     profile->get_coordinate_values().back() + cell_margin);

    for (unsigned int fault_index = 0; fault_index < prescribed_faults.size(); ++fault_index)
      {
        const auto &prescribed_fault = prescribed_faults[fault_index];
        const std::vector<Point<dim>> reference_points =
          ReconstructedFaultUtilities::resample_reference_fault(
            prescribed_fault.vertices, structural_spacing);
        const std::vector<Tensor<1,dim>> normals = reference_normals(reference_points);
        const unsigned int n_points = reference_points.size();

        const double reconstruction_radius = reconstruction_radii[fault_index];

        std::vector<double> local_matrix(n_points*n_points, 0.0);
        std::vector<double> local_rhs(n_points, 0.0);
        std::vector<double> local_support(n_points, 0.0);
        double local_weight = 0.0;

        const QGauss<dim> quadrature(phase_field_handler.get_fe().degree + 1);
        FEValues<dim> fe_values(phase_field_handler.get_mapping(),
                                phase_field_handler.get_fe(), quadrature,
                                update_values | update_quadrature_points | update_JxW_values);
        const FEValuesExtractors::Scalar phase_field_component(
          phase_field_handler.introspection().variable("phase_field").first_component_index);
        std::vector<double> phase_field_values(quadrature.size());

        for (const auto &cell : phase_field_handler.get_dof_handler().active_cell_iterators())
          if (cell->is_locally_owned())
            {
              fe_values.reinit(cell);
              fe_values[phase_field_component].get_function_values(phase_field_handler.get_solution(),
                                                                    phase_field_values);
              for (unsigned int q = 0; q < quadrature.size(); ++q)
                {
                  AssertThrow(std::isfinite(phase_field_values[q]),
                              ExcMessage("A non-finite Q1 phase-field value was encountered."));
                  const double weight = std::max(phase_field_values[q] - phi_min, 0.0);
                  if (weight == 0.0)
                    continue;

                  const auto [r, eta, segment, coordinate] =
                    structural_coordinates(reference_points, fe_values.quadrature_point(q));
                  if (r > reconstruction_radius)
                    continue;

                  for (unsigned int other_fault = fault_index + 1;
                       other_fault < prescribed_faults.size(); ++other_fault)
                    {
                      const double other_r =
                        ReconstructedFaultUtilities::closest_point_distance_and_core_phase_field(
                          prescribed_faults[other_fault], fe_values.quadrature_point(q)).first;
                      AssertThrow(other_r > reconstruction_radii[other_fault],
                                  ExcMessage("The active reconstruction regions of prescribed faults "
                                             + Utilities::int_to_string(fault_index) + " and "
                                             + Utilities::int_to_string(other_fault) + " overlap."));
                    }

                  const double factor = fe_values.JxW(q) * weight;
                  const double shape_values[2] = {1.0-coordinate, coordinate};
                  local_weight += factor;
                  for (unsigned int a = 0; a < 2; ++a)
                    {
                      const unsigned int ia = segment + a;
                      local_rhs[ia] += factor * shape_values[a] * eta;
                      local_support[ia] += factor * shape_values[a];
                      for (unsigned int b = 0; b < 2; ++b)
                        local_matrix[ia*n_points + segment+b] +=
                          factor * shape_values[a] * shape_values[b];
                    }
                }
            }

        std::vector<double> matrix_values(local_matrix.size());
        std::vector<double> rhs_values(local_rhs.size());
        std::vector<double> support(local_support.size());
        Utilities::MPI::sum(local_matrix, phase_field_handler.get_mpi_communicator(), matrix_values);
        Utilities::MPI::sum(local_rhs, phase_field_handler.get_mpi_communicator(), rhs_values);
        Utilities::MPI::sum(local_support, phase_field_handler.get_mpi_communicator(), support);
        const double total_weight = Utilities::MPI::sum(
          local_weight, phase_field_handler.get_mpi_communicator());
        AssertThrow(std::isfinite(total_weight) && total_weight > 0.0,
                    ExcMessage("Fault " + Utilities::int_to_string(fault_index)
                               + " has no phase-field reconstruction weight."));

        for (unsigned int i = 0; i < n_points; ++i)
          {
            AssertThrow(std::isfinite(support[i]) && support[i] > 0.0,
                        ExcMessage("A structural vertex of fault "
                                   + Utilities::int_to_string(fault_index)
                                   + " has no phase-field support."));
          }
        const std::vector<double> offsets =
          ReconstructedFaultUtilities::solve_normal_offsets(matrix_values,
                                                              rhs_values,
                                                              total_weight,
                                                              ridge_coefficient);
        std::vector<Point<dim>> fitted_points(n_points);
        FaultReconstructionDiagnostics fault_diagnostics;
        fault_diagnostics.total_weight = total_weight;
        fault_diagnostics.structural_support = support;
        fault_diagnostics.offsets.resize(n_points);
        for (unsigned int i = 0; i < n_points; ++i)
          {
            AssertThrow(std::isfinite(offsets[i]),
                        ExcMessage("The reconstructed fault offset is non-finite."));
            AssertThrow(std::abs(offsets[i]) <= reconstruction_radius,
                        ExcMessage("The fitted offset of fault "
                                   + Utilities::int_to_string(fault_index)
                                   + " leaves its tubular reconstruction region."));
            fitted_points[i] = reference_points[i] + offsets[i] * normals[i];
            fault_diagnostics.offsets[i] = offsets[i];
          }
        reconstructed_faults.emplace_back(fitted_points);
        diagnostics.push_back(std::move(fault_diagnostics));
      }

    initial_reconstruction_complete = true;

    this->get_pcout() << "done." << std::endl << std::endl;
  }


  template <int dim>
  const std::vector<ReconstructedFault<dim>> &
  ReconstructedFaultManager<dim>::get_faults() const
  {
    return reconstructed_faults;
  }


  template <int dim>
  const std::vector<FaultReconstructionDiagnostics> &
  ReconstructedFaultManager<dim>::get_diagnostics() const
  {
    return diagnostics;
  }


  template <int dim>
  ReconstructedFault<dim>::ReconstructedFault(const std::vector<Point<dim>> &initial_vertices)
    : vertices(initial_vertices)
  {}


  template <int dim>
  bool
  ReconstructedFault<dim>::empty() const
  {
    return vertices.empty();
  }


  template <int dim>
  unsigned int
  ReconstructedFault<dim>::n_vertices() const
  {
    return vertices.size();
  }


  template <int dim>
  unsigned int
  ReconstructedFault<dim>::n_cells() const
  {
    return vertices.empty() ? 0 : vertices.size() - 1;
  }


  template <int dim>
  const Point<dim> &
  ReconstructedFault<dim>::vertex(const unsigned int index) const
  {
    AssertIndexRange(index, vertices.size());
    return vertices[index];
  }


  template <int dim>
  const std::vector<Point<dim>> &
  ReconstructedFault<dim>::get_vertices() const
  {
    return vertices;
  }


  template <int dim>
  void
  ReconstructedFault<dim>::append_vertex(const Point<dim> &new_vertex)
  {
    vertices.push_back(new_vertex);
    ++current_geometry_version;
  }


  template <int dim>
  void
  ReconstructedFault<dim>::append_vertices(const std::vector<Point<dim>> &new_vertices)
  {
    if (new_vertices.empty())
      return;

    vertices.insert(vertices.end(), new_vertices.begin(), new_vertices.end());
    ++current_geometry_version;
  }


  template <int dim>
  std::uint64_t
  ReconstructedFault<dim>::geometry_version() const
  {
    return current_geometry_version;
  }

}


// explicit instantiations
namespace aspect
{
#define INSTANTIATE(dim) \
  namespace ReconstructedFaultUtilities \
  { \
    template std::pair<double, double> \
    closest_point_distance_and_core_phase_field( \
      const PrescribedInitialFault<dim> &, const Point<dim> &); \
    template std::vector<Point<dim>> \
    resample_reference_fault( \
      const std::vector<Point<dim>> &, const double); \
    template std::vector<PrescribedInitialFault<dim>> \
    parse_prescribed_faults( \
      const std::string &, const std::string &); \
    template void \
    initialize_crack_driving_force( \
      PhaseFieldHandler<dim> &, const std::vector<PrescribedInitialFault<dim>> &); \
  } \
  template class ReconstructedFault<dim>; \
  template class ReconstructedFaultManager<dim>;

  ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
}
