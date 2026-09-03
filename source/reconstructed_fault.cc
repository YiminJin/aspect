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
#include <aspect/particle/particle_domain.h>
#include <aspect/material_model/utilities.h>
#include <aspect/utilities.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
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


    std::pair<std::vector<double>, std::vector<double>>
    factor_tridiagonal(const std::vector<double> &diagonal,
                       const std::vector<double> &off_diagonal)
    {
      AssertThrow(!diagonal.empty(), ExcMessage("A projection system must not be empty."));
      AssertDimension(off_diagonal.size() + 1, diagonal.size());
      const double scale = *std::max_element(diagonal.begin(), diagonal.end());
      AssertThrow(std::isfinite(scale) && scale > 0.0,
                  ExcMessage("The particle-to-fault projection matrix has no positive diagonal."));
      const double tolerance = std::numeric_limits<double>::epsilon()
                               * std::max(1.0, static_cast<double>(diagonal.size())) * scale;

      std::vector<double> factor_diagonal(diagonal.size());
      std::vector<double> factor_lower(off_diagonal.size());
      factor_diagonal[0] = diagonal[0];
      AssertThrow(std::isfinite(factor_diagonal[0]) && factor_diagonal[0] > tolerance,
                  ExcMessage("The particle-to-fault projection matrix is singular at its first vertex."));
      for (unsigned int i = 1; i < diagonal.size(); ++i)
        {
          AssertThrow(std::isfinite(off_diagonal[i-1]),
                      ExcMessage("The particle-to-fault projection matrix is non-finite."));
          factor_lower[i-1] = off_diagonal[i-1] / factor_diagonal[i-1];
          factor_diagonal[i] = diagonal[i] - factor_lower[i-1] * off_diagonal[i-1];
          AssertThrow(std::isfinite(factor_diagonal[i]) && factor_diagonal[i] > tolerance,
                      ExcMessage("The particle-to-fault projection matrix is singular at vertex "
                                 + Utilities::int_to_string(i) + "."));
        }
      return {factor_diagonal, factor_lower};
    }


    std::vector<double>
    solve_tridiagonal_factors(const std::vector<double> &factor_diagonal,
                              const std::vector<double> &factor_lower,
                              const ArrayView<const double> &rhs)
    {
      AssertDimension(rhs.size(), factor_diagonal.size());
      AssertDimension(factor_lower.size() + 1, factor_diagonal.size());
      std::vector<double> solution(rhs.begin(), rhs.end());
      for (unsigned int i = 1; i < solution.size(); ++i)
        solution[i] -= factor_lower[i-1] * solution[i-1];
      for (unsigned int i = 0; i < solution.size(); ++i)
        solution[i] /= factor_diagonal[i];
      for (unsigned int i = solution.size() - 1; i > 0; --i)
        solution[i-1] -= factor_lower[i-1] * solution[i];
      return solution;
    }
  }


  namespace ReconstructedFaultUtilities
  {
    template <int dim>
    NormalProfileProjection
    project_to_normal_profiles(
      const std::vector<ReconstructedFault<dim>> &faults,
      const std::vector<std::vector<double>> &half_widths,
      const Point<dim> &position)
    {
      AssertThrow(dim == 2, ExcNotImplemented());
      AssertDimension(half_widths.size(), faults.size());
      NormalProfileProjection result;
      unsigned int admitted_faults = 0;

      for (unsigned int fault_index = 0; fault_index < faults.size(); ++fault_index)
        {
          const ReconstructedFault<dim> &fault = faults[fault_index];
          AssertThrow(fault.n_vertices() >= 2,
                      ExcMessage("Particle projection requires faults with at least two vertices."));
          AssertThrow(fault.vertex(0) != fault.vertex(fault.n_vertices()-1),
                      ExcMessage("Closed-loop faults are unsupported by particle projection."));
          AssertDimension(half_widths[fault_index].size(), fault.n_vertices());

          bool admitted_to_fault = false;
          double smallest_distance = std::numeric_limits<double>::infinity();
          NormalProfileProjection candidate;
          for (unsigned int segment = 0; segment < fault.n_cells(); ++segment)
            {
              const Tensor<1,dim> segment_vector = fault.vertex(segment+1) - fault.vertex(segment);
              const double length_squared = segment_vector.norm_square();
              AssertThrow(std::isfinite(length_squared) && length_squared > 0.0,
                          ExcMessage("Particle projection encountered a degenerate fault segment."));
              const double xi = ((position - fault.vertex(segment)) * segment_vector) / length_squared;
              if (xi < 0.0 || xi > 1.0)
                continue;

              const double width = (1.0-xi) * half_widths[fault_index][segment]
                                   + xi * half_widths[fault_index][segment+1];
              AssertThrow(std::isfinite(width) && width > 0.0,
                          ExcMessage("Particle projection requires positive finite influence half-widths."));
              Tensor<1,dim> tangent = segment_vector / std::sqrt(length_squared);
              const Tensor<1,dim> normal({-tangent[1], tangent[0]});
              const Point<dim> projected_point = fault.vertex(segment) + xi * segment_vector;
              const double signed_distance = (position - projected_point) * normal;
              const double distance = std::abs(signed_distance);
              if (distance <= width && distance < smallest_distance)
                {
                  admitted_to_fault = true;
                  smallest_distance = distance;
                  candidate = {true, fault_index, segment, xi, signed_distance};
                }
            }

          if (admitted_to_fault)
            {
              ++admitted_faults;
              result = candidate;
            }
        }

      AssertThrow(admitted_faults <= 1,
                  ExcMessage("A particle lies in the influence regions of multiple reconstructed faults. "
                             "Overlapping fault influence regions are unsupported."));
      return result;
    }


    std::vector<double>
    solve_tridiagonal_system(const std::vector<double> &diagonal,
                             const std::vector<double> &off_diagonal,
                             const std::vector<double> &rhs)
    {
      const auto factors = factor_tridiagonal(diagonal, off_diagonal);
      return solve_tridiagonal_factors(factors.first, factors.second, make_array_view(rhs));
    }


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
      const double activation_threshold =
        phase_field_model->get_phase_field_activation_threshold();
      const double upper_admissibility_threshold =
        phase_field_model->get_phase_field_upper_admissibility_threshold();

      for (unsigned int fault_index = 0; fault_index < faults.size(); ++fault_index)
        {
          const PrescribedInitialFault<dim> &fault = faults[fault_index];
          closest_point_distance_and_core_phase_field(fault, Point<dim>());
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
    const double phi_min = phase_field_model->get_phase_field_activation_threshold();

    double local_cell_margin = 0.0;
    for (const auto &cell : phase_field_handler.get_dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        local_cell_margin = std::max(local_cell_margin, cell->diameter());
    const double cell_margin = Utilities::MPI::max(local_cell_margin,
                                                    phase_field_handler.get_mpi_communicator());

    reconstructed_faults.clear();
    projection_half_widths.clear();
    timestep_committed_slip_rates.clear();
    current_newton_slip_rates.clear();
    trial_slip_rates.clear();
    slip_rate_initialized.clear();
    slip_rate_nonlinear_solve_active = false;
    slip_rate_trial_active = false;
    invalidate_particle_projection_cache();
    diagnostics.clear();
    std::vector<double> reconstruction_radii(prescribed_faults.size(), cell_margin);
    std::vector<std::vector<double>> prescribed_half_widths(prescribed_faults.size());
    std::map<double, double> profile_support_by_core_value;
    for (unsigned int fault_index = 0; fault_index < prescribed_faults.size(); ++fault_index)
      {
        prescribed_half_widths[fault_index].resize(
          prescribed_faults[fault_index].core_phase_field_values.size());
        for (unsigned int vertex = 0;
             vertex < prescribed_faults[fault_index].core_phase_field_values.size(); ++vertex)
          {
            const double phi_hat =
              prescribed_faults[fault_index].core_phase_field_values[vertex];
            auto support = profile_support_by_core_value.find(phi_hat);
            if (support == profile_support_by_core_value.end())
              {
                double half_width = 0.0;
                for (const auto &profile : phase_field_handler.get_phase_field_profiles(phi_hat))
                  half_width = std::max(half_width,
                                        profile->get_coordinate_values().back());
                AssertThrow(std::isfinite(half_width) && half_width > 0.0,
                            ExcMessage("Stationary profiles must provide a positive finite support."));
                support = profile_support_by_core_value.emplace(phi_hat, half_width).first;
              }
            prescribed_half_widths[fault_index][vertex] = support->second;
            reconstruction_radii[fault_index] =
              std::max(reconstruction_radii[fault_index], support->second + cell_margin);
          }
      }

    for (unsigned int fault_index = 0; fault_index < prescribed_faults.size(); ++fault_index)
      {
        const auto &prescribed_fault = prescribed_faults[fault_index];
        const std::vector<Point<dim>> reference_points =
          ReconstructedFaultUtilities::resample_reference_fault(
            prescribed_fault.vertices, structural_spacing);
        const std::vector<Tensor<1,dim>> normals = reference_normals(reference_points);
        const unsigned int n_points = reference_points.size();
        std::vector<double> reference_half_widths(n_points);
        for (unsigned int vertex = 0; vertex < n_points; ++vertex)
          {
            const auto [distance, signed_distance, segment, xi] =
              structural_coordinates(prescribed_fault.vertices, reference_points[vertex]);
            (void) distance;
            (void) signed_distance;
            reference_half_widths[vertex] =
              (1.0-xi) * prescribed_half_widths[fault_index][segment]
              + xi * prescribed_half_widths[fault_index][segment+1];
          }

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
        add_reconstructed_fault(fitted_points, reference_half_widths);
        diagnostics.push_back(std::move(fault_diagnostics));
      }

    ++projection_metadata_version;
    invalidate_particle_projection_cache();
    initial_reconstruction_complete = true;

    this->get_pcout() << "done." << std::endl << std::endl;
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
    Assert(half_widths.size() == vertices.size(), ExcInternalError());
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
    ++projection_metadata_version;
    invalidate_particle_projection_cache();
    return reconstructed_faults.size() - 1;
  }


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


  template <int dim>
  bool
  ReconstructedFaultManager<dim>::slip_rates_are_initialized() const
  {
    return !reconstructed_faults.empty()
           && slip_rate_initialized.size() == reconstructed_faults.size()
           && std::all_of(slip_rate_initialized.begin(), slip_rate_initialized.end(),
                          [](const bool initialized) { return initialized; });
  }


  template <int dim>
  void
  ReconstructedFaultManager<dim>::initialize_slip_rate(
    const unsigned int fault_index,
    const std::vector<double> &values)
  {
    AssertIndexRange(fault_index, reconstructed_faults.size());
    Assert(!slip_rate_nonlinear_solve_active && !slip_rate_trial_active,
           ExcMessage("Slip rates cannot be initialized during a nonlinear solve."));
    Assert(slip_rate_initialized.size() == reconstructed_faults.size(), ExcInternalError());
    Assert(!slip_rate_initialized[fault_index], ExcInternalError());
    Assert(values.size() == reconstructed_faults[fault_index].n_vertices(),
           ExcInternalError());
    for (const double value : values)
      AssertThrow(std::isfinite(value) && value >= 0.0,
                  ExcMessage("Reconstructed-fault slip rates must be finite and nonnegative."));

    timestep_committed_slip_rates[fault_index] = values;
    current_newton_slip_rates[fault_index] = values;
    slip_rate_initialized[fault_index] = true;
  }


  template <int dim>
  const std::vector<double> &
  ReconstructedFaultManager<dim>::get_slip_rate(const unsigned int fault_index) const
  {
    AssertIndexRange(fault_index, reconstructed_faults.size());
    Assert(slip_rate_initialized.size() == reconstructed_faults.size()
           && slip_rate_initialized[fault_index],
           ExcMessage("The reconstructed-fault slip rate has not been initialized."));
    const std::vector<double> &values = slip_rate_trial_active
                                        ? trial_slip_rates[fault_index]
                                        : current_newton_slip_rates[fault_index];
    AssertDimension(values.size(), reconstructed_faults[fault_index].n_vertices());
    return values;
  }


  template <int dim>
  const std::vector<double> &
  ReconstructedFaultManager<dim>::get_timestep_committed_slip_rate(
    const unsigned int fault_index) const
  {
    AssertIndexRange(fault_index, reconstructed_faults.size());
    Assert(slip_rate_initialized.size() == reconstructed_faults.size()
           && slip_rate_initialized[fault_index],
           ExcMessage("The reconstructed-fault slip rate has not been initialized."));
    AssertDimension(timestep_committed_slip_rates[fault_index].size(),
                    reconstructed_faults[fault_index].n_vertices());
    return timestep_committed_slip_rates[fault_index];
  }


  template <int dim>
  double
  ReconstructedFaultManager<dim>::interpolate_slip_rate(
    const unsigned int fault_index,
    const unsigned int segment_index,
    const double xi) const
  {
    AssertIndexRange(fault_index, reconstructed_faults.size());
    AssertIndexRange(segment_index, reconstructed_faults[fault_index].n_cells());
    Assert(std::isfinite(xi) && xi >= 0.0 && xi <= 1.0, ExcInternalError());
    const std::vector<double> &values = get_slip_rate(fault_index);
    return (1.0-xi) * values[segment_index] + xi * values[segment_index+1];
  }


  template <int dim>
  void
  ReconstructedFaultManager<dim>::begin_slip_rate_nonlinear_solve()
  {
    Assert(slip_rates_are_initialized(),
           ExcMessage("A slip-rate nonlinear solve requires initialized slip rates."));
    Assert(!slip_rate_nonlinear_solve_active && !slip_rate_trial_active,
           ExcInternalError());
    current_newton_slip_rates = timestep_committed_slip_rates;
    trial_slip_rates.assign(reconstructed_faults.size(), {});
    slip_rate_nonlinear_solve_active = true;
  }


  template <int dim>
  void
  ReconstructedFaultManager<dim>::commit_slip_rate_nonlinear_solve()
  {
    Assert(slip_rate_nonlinear_solve_active && !slip_rate_trial_active,
           ExcInternalError());
    timestep_committed_slip_rates = current_newton_slip_rates;
    slip_rate_nonlinear_solve_active = false;
  }


  template <int dim>
  void
  ReconstructedFaultManager<dim>::rollback_slip_rate_nonlinear_solve()
  {
    Assert(slip_rate_nonlinear_solve_active, ExcInternalError());
    current_newton_slip_rates = timestep_committed_slip_rates;
    trial_slip_rates.assign(reconstructed_faults.size(), {});
    slip_rate_trial_active = false;
    slip_rate_nonlinear_solve_active = false;
  }


  template <int dim>
  void
  ReconstructedFaultManager<dim>::begin_slip_rate_trial()
  {
    Assert(slip_rate_nonlinear_solve_active && !slip_rate_trial_active,
           ExcInternalError());
    trial_slip_rates = current_newton_slip_rates;
    slip_rate_trial_active = true;
  }


  template <int dim>
  void
  ReconstructedFaultManager<dim>::set_slip_rate_trial(
    const std::vector<std::vector<double>> &delta_V,
    const double step_length)
  {
    Assert(slip_rate_nonlinear_solve_active && slip_rate_trial_active,
           ExcInternalError());
    AssertThrow(std::isfinite(step_length),
                ExcMessage("The slip-rate trial step length must be finite."));
    Assert(delta_V.size() == reconstructed_faults.size(), ExcInternalError());
    for (unsigned int fault = 0; fault < reconstructed_faults.size(); ++fault)
      {
        Assert(slip_rate_initialized[fault], ExcInternalError());
        Assert(delta_V[fault].size() == reconstructed_faults[fault].n_vertices(),
               ExcInternalError());
        for (const double value : delta_V[fault])
          AssertThrow(std::isfinite(value),
                      ExcMessage("Reconstructed-fault slip-rate updates must be finite."));
      }

    trial_slip_rates = current_newton_slip_rates;
    for (unsigned int fault = 0; fault < trial_slip_rates.size(); ++fault)
      for (unsigned int vertex = 0; vertex < trial_slip_rates[fault].size(); ++vertex)
        {
          const double value = current_newton_slip_rates[fault][vertex]
                               + step_length * delta_V[fault][vertex];
          AssertThrow(std::isfinite(value) && value >= 0.0,
                      ExcMessage("A reconstructed-fault slip-rate trial value must be finite "
                                 "and nonnegative."));
          trial_slip_rates[fault][vertex] = value;
        }
  }


  template <int dim>
  void
  ReconstructedFaultManager<dim>::accept_slip_rate_trial()
  {
    Assert(slip_rate_nonlinear_solve_active && slip_rate_trial_active,
           ExcInternalError());
    current_newton_slip_rates = trial_slip_rates;
    trial_slip_rates.assign(reconstructed_faults.size(), {});
    slip_rate_trial_active = false;
  }


  template <int dim>
  void
  ReconstructedFaultManager<dim>::rollback_slip_rate_trial()
  {
    Assert(slip_rate_nonlinear_solve_active && slip_rate_trial_active,
           ExcInternalError());
    trial_slip_rates.assign(reconstructed_faults.size(), {});
    slip_rate_trial_active = false;
  }


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
    diagnostics.clear();
    ++projection_metadata_version;
    invalidate_particle_projection_cache();
  }


  template <int dim>
  bool
  ReconstructedFaultManager<dim>::particle_projection_cache_is_valid() const
  {
    if (!particle_projection_cache_valid
        || cached_projection_metadata_version != projection_metadata_version
        || cached_fault_geometry_versions.size() != reconstructed_faults.size())
      return false;

    for (unsigned int fault = 0; fault < reconstructed_faults.size(); ++fault)
      if (cached_fault_geometry_versions[fault]
          != reconstructed_faults[fault].geometry_version())
        return false;

    const Particle::Manager<dim> &particle_manager =
      this->get_phase_field_handler().get_associated_particle_manager();
    if (!particle_manager.particle_domains_requested())
      return false;
    const auto &particle_handler = particle_manager.get_particle_handler();
    const auto &particle_domain_handler = particle_manager.get_particle_domain_handler();
    if (particle_handler.n_locally_owned_particles() != particle_projection_cache.size())
      return false;

    unsigned int particle_index = 0;
    for (const auto &particle : particle_handler)
      {
        const double volume = particle_domain_handler
                              .get_particle_domain(particle.get_local_index()).volume();
        const ParticleProjectionCacheEntry &entry = particle_projection_cache[particle_index++];
        if (entry.particle_id != particle.get_id()
            || entry.position != particle.get_location()
            || entry.particle_domain_volume != volume)
          return false;
      }
    return particle_index == particle_projection_cache.size();
  }


  template <int dim>
  void
  ReconstructedFaultManager<dim>::rebuild_particle_projection_cache()
  {
    AssertThrow(dim == 2, ExcNotImplemented());
    AssertThrow(!reconstructed_faults.empty(),
                ExcMessage("Particle projection requires reconstructed fault geometry."));
    AssertDimension(projection_half_widths.size(), reconstructed_faults.size());

    Particle::Manager<dim> &particle_manager =
      this->get_phase_field_handler().get_associated_particle_manager();
    AssertThrow(particle_manager.particle_domains_requested(),
                ExcMessage("Particle-to-fault projection requires particle-domain volumes."));
    const auto &particle_handler = particle_manager.get_particle_handler();
    const auto &particle_domain_handler = particle_manager.get_particle_domain_handler();

    projection_systems.clear();
    projection_systems.resize(reconstructed_faults.size());
    particle_projection_diagnostics.clear();
    particle_projection_diagnostics.resize(reconstructed_faults.size());
    for (unsigned int fault = 0; fault < reconstructed_faults.size(); ++fault)
      {
        AssertThrow(reconstructed_faults[fault].n_vertices() >= 2,
                    ExcMessage("Particle projection requires faults with at least two vertices."));
        AssertDimension(projection_half_widths[fault].size(),
                        reconstructed_faults[fault].n_vertices());
        projection_systems[fault].diagonal.assign(
          reconstructed_faults[fault].n_vertices(), 0.0);
        projection_systems[fault].off_diagonal.assign(
          reconstructed_faults[fault].n_cells(), 0.0);
        particle_projection_diagnostics[fault].weighted_support.assign(
          reconstructed_faults[fault].n_vertices(), 0.0);
      }

    particle_projection_cache.clear();
    particle_projection_cache.reserve(particle_handler.n_locally_owned_particles());
    std::vector<unsigned int> local_contributing_particles(reconstructed_faults.size(), 0);
    for (const auto &particle : particle_handler)
      {
        const double volume = particle_domain_handler
                              .get_particle_domain(particle.get_local_index()).volume();
        AssertThrow(std::isfinite(volume) && volume > 0.0,
                    ExcMessage("Particle-to-fault projection encountered a non-positive "
                               "or non-finite particle-domain volume."));
        for (unsigned int d = 0; d < dim; ++d)
          AssertThrow(std::isfinite(particle.get_location()[d]),
                      ExcMessage("Particle-to-fault projection encountered a non-finite position."));

        const ReconstructedFaultUtilities::NormalProfileProjection projection =
          ReconstructedFaultUtilities::project_to_normal_profiles(
            reconstructed_faults, projection_half_widths, particle.get_location());
        ParticleProjectionCacheEntry entry;
        entry.particle_id = particle.get_id();
        entry.position = particle.get_location();
        entry.particle_domain_volume = volume;
        entry.active = projection.active;
        entry.fault_index = projection.fault_index;
        entry.segment_index = projection.segment_index;
        entry.xi = projection.xi;
        particle_projection_cache.push_back(entry);

        if (entry.active)
          {
            ProjectionSystem &system = projection_systems[entry.fault_index];
            auto &support =
              particle_projection_diagnostics[entry.fault_index].weighted_support;
            const double shape[2] = {1.0-entry.xi, entry.xi};
            const unsigned int first_vertex = entry.segment_index;
            system.diagonal[first_vertex] += volume * shape[0] * shape[0];
            system.diagonal[first_vertex+1] += volume * shape[1] * shape[1];
            system.off_diagonal[first_vertex] += volume * shape[0] * shape[1];
            support[first_vertex] += volume * shape[0];
            support[first_vertex+1] += volume * shape[1];
            ++local_contributing_particles[entry.fault_index];
          }
      }

    unsigned int packed_size = 0;
    for (const ReconstructedFault<dim> &fault : reconstructed_faults)
      packed_size += 3 * fault.n_vertices();
    std::vector<double> local_values(packed_size, 0.0);
    unsigned int position = 0;
    for (unsigned int fault = 0; fault < reconstructed_faults.size(); ++fault)
      {
        const ProjectionSystem &system = projection_systems[fault];
        std::copy(system.diagonal.begin(), system.diagonal.end(),
                  local_values.begin() + position);
        position += system.diagonal.size();
        std::copy(system.off_diagonal.begin(), system.off_diagonal.end(),
                  local_values.begin() + position);
        position += system.off_diagonal.size();
        const auto &support = particle_projection_diagnostics[fault].weighted_support;
        std::copy(support.begin(), support.end(), local_values.begin() + position);
        position += support.size();
        local_values[position++] = local_contributing_particles[fault];
      }
    AssertDimension(position, packed_size);

    std::vector<double> global_values(packed_size);
    Utilities::MPI::sum(local_values, this->get_mpi_communicator(), global_values);
    position = 0;
    for (unsigned int fault = 0; fault < reconstructed_faults.size(); ++fault)
      {
        ProjectionSystem &system = projection_systems[fault];
        std::copy_n(global_values.begin() + position, system.diagonal.size(),
                    system.diagonal.begin());
        position += system.diagonal.size();
        std::copy_n(global_values.begin() + position, system.off_diagonal.size(),
                    system.off_diagonal.begin());
        position += system.off_diagonal.size();
        auto &diagnostic = particle_projection_diagnostics[fault];
        std::copy_n(global_values.begin() + position, diagnostic.weighted_support.size(),
                    diagnostic.weighted_support.begin());
        position += diagnostic.weighted_support.size();
        diagnostic.n_contributing_particles =
          static_cast<unsigned int>(std::llround(global_values[position++]));

        const double support_scale = *std::max_element(diagnostic.weighted_support.begin(),
                                                       diagnostic.weighted_support.end());
        const double support_tolerance = std::numeric_limits<double>::epsilon()
                                         * std::max(1.0, static_cast<double>(system.diagonal.size()))
                                         * support_scale;
        for (unsigned int vertex = 0; vertex < diagnostic.weighted_support.size(); ++vertex)
          AssertThrow(std::isfinite(diagnostic.weighted_support[vertex])
                      && diagnostic.weighted_support[vertex] > support_tolerance,
                      ExcMessage("Fault " + Utilities::int_to_string(fault) + " vertex "
                                 + Utilities::int_to_string(vertex)
                                 + " has insufficient particle projection support."));

        const auto factors = factor_tridiagonal(system.diagonal, system.off_diagonal);
        system.factor_diagonal = factors.first;
        system.factor_lower = factors.second;
      }

    cached_fault_geometry_versions.resize(reconstructed_faults.size());
    for (unsigned int fault = 0; fault < reconstructed_faults.size(); ++fault)
      cached_fault_geometry_versions[fault] = reconstructed_faults[fault].geometry_version();
    cached_projection_metadata_version = projection_metadata_version;
    particle_projection_cache_valid = true;
  }


  template <int dim>
  std::vector<double>
  ReconstructedFaultManager<dim>::solve_projection_system(
    const unsigned int fault_index,
    const ArrayView<const double> &rhs) const
  {
    AssertIndexRange(fault_index, projection_systems.size());
    return solve_tridiagonal_factors(projection_systems[fault_index].factor_diagonal,
                                     projection_systems[fault_index].factor_lower,
                                     rhs);
  }


  template <int dim>
  void
  ReconstructedFaultManager<dim>::project_particle_properties(
    const std::vector<ParticlePropertyProjection> &projections)
  {
    if (projections.empty())
      return;
    AssertThrow(dim == 2, ExcNotImplemented());
    AssertThrow(!reconstructed_faults.empty(),
                ExcMessage("Particle projection requires reconstructed fault geometry."));

    Particle::Manager<dim> &particle_manager =
      this->get_phase_field_handler().get_associated_particle_manager();
    const auto &particle_data = particle_manager.get_property_manager().get_data_info();
    struct ResolvedComponent
    {
      unsigned int particle_position;
      unsigned int fault_position;
      std::string description;
    };
    std::vector<ResolvedComponent> components;
    std::set<unsigned int> destination_components;
    for (const ParticlePropertyProjection &projection : projections)
      {
        AssertThrow(projection.n_components > 0,
                    ExcMessage("A particle property projection must contain at least one component."));
        AssertThrow(particle_data.fieldname_exists(projection.particle_property_name),
                    ExcMessage("No particle property named <" + projection.particle_property_name
                               + "> is registered."));
        const unsigned int particle_components =
          particle_data.get_components_by_field_name(projection.particle_property_name);
        AssertThrow(projection.first_particle_component + projection.n_components
                    <= particle_components,
                    ExcMessage("A particle property projection exceeds the source component range."));
        AssertThrow(has_property(projection.fault_property_name),
                    ExcMessage("No reconstructed-fault property named <"
                               + projection.fault_property_name + "> is registered."));
        const PropertyInformation &fault_property =
          property_information[get_property_index(projection.fault_property_name)];
        AssertThrow(projection.first_fault_component + projection.n_components
                    <= fault_property.n_components,
                    ExcMessage("A particle property projection exceeds the destination component range."));

        const unsigned int particle_position =
          particle_data.get_position_by_field_name(projection.particle_property_name)
          + projection.first_particle_component;
        const unsigned int fault_position = fault_property.position
                                            + projection.first_fault_component;
        for (unsigned int component = 0; component < projection.n_components; ++component)
          {
            AssertThrow(destination_components.insert(fault_position + component).second,
                        ExcMessage("Multiple particle property projections target the same "
                                   "reconstructed-fault property component."));
            components.push_back({particle_position + component,
                                  fault_position + component,
                                  projection.particle_property_name + " component "
                                  + Utilities::int_to_string(projection.first_particle_component
                                                             + component)});
          }
      }

    if (!particle_projection_cache_is_valid())
      rebuild_particle_projection_cache();

    std::vector<unsigned int> fault_vertex_offsets(reconstructed_faults.size() + 1, 0);
    for (unsigned int fault = 0; fault < reconstructed_faults.size(); ++fault)
      fault_vertex_offsets[fault+1] = fault_vertex_offsets[fault]
                                      + reconstructed_faults[fault].n_vertices();
    const unsigned int n_fault_vertices = fault_vertex_offsets.back();
    std::vector<double> local_rhs(components.size() * n_fault_vertices, 0.0);

    const auto &particle_handler = particle_manager.get_particle_handler();
    unsigned int cache_index = 0;
    for (const auto &particle : particle_handler)
      {
        const ParticleProjectionCacheEntry &entry = particle_projection_cache[cache_index++];
        if (!entry.active)
          continue;
        const ArrayView<const double> particle_properties = particle.get_properties();
        const double shape[2] = {1.0-entry.xi, entry.xi};
        const unsigned int first_vertex = fault_vertex_offsets[entry.fault_index]
                                          + entry.segment_index;
        for (unsigned int component = 0; component < components.size(); ++component)
          {
            const double value = particle_properties[components[component].particle_position];
            AssertThrow(std::isfinite(value),
                        ExcMessage("Particle " + Utilities::int_to_string(particle.get_id())
                                   + " has a non-finite value for "
                                   + components[component].description + "."));
            local_rhs[component*n_fault_vertices + first_vertex]
              += entry.particle_domain_volume * shape[0] * value;
            local_rhs[component*n_fault_vertices + first_vertex+1]
              += entry.particle_domain_volume * shape[1] * value;
          }
      }

    std::vector<double> global_rhs(local_rhs.size());
    Utilities::MPI::sum(local_rhs, this->get_mpi_communicator(), global_rhs);
    for (unsigned int component = 0; component < components.size(); ++component)
      for (unsigned int fault = 0; fault < reconstructed_faults.size(); ++fault)
        {
          const unsigned int begin = component*n_fault_vertices + fault_vertex_offsets[fault];
          const ArrayView<const double> rhs = make_array_view(
            global_rhs.cbegin() + begin,
            global_rhs.cbegin() + begin + reconstructed_faults[fault].n_vertices());
          const std::vector<double> projected_values = solve_projection_system(fault, rhs);
          for (unsigned int vertex = 0; vertex < reconstructed_faults[fault].n_vertices(); ++vertex)
            reconstructed_faults[fault].get_properties(vertex)[components[component].fault_position]
              = projected_values[vertex];
        }
  }


  template <int dim>
  std::map<types::particle_index, std::vector<double>>
  ReconstructedFaultManager<dim>::interpolate_property_at_particle_projections(
    const unsigned int property_index)
  {
    AssertThrow(dim == 2, ExcNotImplemented());
    AssertIndexRange(property_index, property_information.size());
    AssertThrow(!reconstructed_faults.empty(),
                ExcMessage("Fault-property interpolation requires reconstructed fault geometry."));

    if (!particle_projection_cache_is_valid())
      rebuild_particle_projection_cache();

    const PropertyInformation &property = property_information[property_index];
    std::map<types::particle_index, std::vector<double>> values;
    for (const ParticleProjectionCacheEntry &entry : particle_projection_cache)
      if (entry.active)
        {
          const ReconstructedFault<dim> &fault = reconstructed_faults[entry.fault_index];
          const ArrayView<const double> first = fault.get_properties(entry.segment_index);
          const ArrayView<const double> second = fault.get_properties(entry.segment_index+1);
          std::vector<double> interpolated(property.n_components);
          for (unsigned int component = 0; component < property.n_components; ++component)
            {
              interpolated[component] =
                (1.0-entry.xi) * first[property.position+component]
                + entry.xi * second[property.position+component];
              AssertThrow(std::isfinite(interpolated[component]),
                          ExcMessage("Fault property <" + property.name
                                     + "> is not finite at the cached projection of particle "
                                     + Utilities::int_to_string(entry.particle_id) + "."));
            }
          AssertThrow(values.emplace(entry.particle_id, std::move(interpolated)).second,
                      ExcMessage("Duplicate locally owned particle ID in the fault-projection cache."));
        }

    return values;
  }


  template <int dim>
  typename ReconstructedFaultManager<dim>::ParticleScalarProjectionResult
  ReconstructedFaultManager<dim>::project_particle_scalar(
    const std::map<types::particle_index, double> &locally_owned_values)
  {
    AssertThrow(dim == 2, ExcNotImplemented());
    AssertThrow(!reconstructed_faults.empty(),
                ExcMessage("Particle projection requires reconstructed fault geometry."));

    if (!particle_projection_cache_is_valid())
      rebuild_particle_projection_cache();

    std::vector<unsigned int> fault_vertex_offsets(reconstructed_faults.size() + 1, 0);
    for (unsigned int fault = 0; fault < reconstructed_faults.size(); ++fault)
      fault_vertex_offsets[fault+1] = fault_vertex_offsets[fault]
                                      + reconstructed_faults[fault].n_vertices();
    const unsigned int n_fault_vertices = fault_vertex_offsets.back();
    std::vector<double> local_rhs(n_fault_vertices, 0.0);

    const auto &particle_handler = this->get_phase_field_handler()
                                   .get_associated_particle_manager()
                                   .get_particle_handler();
    std::string local_error;
    unsigned int cache_index = 0;
    for (const auto &particle : particle_handler)
      {
        const ParticleProjectionCacheEntry &entry = particle_projection_cache[cache_index++];
        if (!entry.active)
          continue;

        const auto value = locally_owned_values.find(particle.get_id());
        if (value == locally_owned_values.end())
          {
            if (local_error.empty())
              local_error = "No scalar projection value was supplied for active particle "
                            + Utilities::int_to_string(particle.get_id()) + ".";
            continue;
          }
        if (!std::isfinite(value->second))
          {
            if (local_error.empty())
              local_error = "A non-finite scalar projection value was supplied for particle "
                            + Utilities::int_to_string(particle.get_id()) + ".";
            continue;
          }

        const double shape[2] = {1.0-entry.xi, entry.xi};
        const unsigned int first_vertex = fault_vertex_offsets[entry.fault_index]
                                          + entry.segment_index;
        local_rhs[first_vertex] += entry.particle_domain_volume * shape[0] * value->second;
        local_rhs[first_vertex+1] += entry.particle_domain_volume * shape[1] * value->second;
      }
    AssertDimension(cache_index, particle_projection_cache.size());

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

    std::vector<double> global_rhs(local_rhs.size());
    Utilities::MPI::sum(local_rhs, this->get_mpi_communicator(), global_rhs);

    ParticleScalarProjectionResult result;
    result.nodal_values.resize(reconstructed_faults.size());
    result.diagnostics.resize(reconstructed_faults.size());
    for (unsigned int fault = 0; fault < reconstructed_faults.size(); ++fault)
      {
        const ArrayView<const double> rhs = make_array_view(
          global_rhs.cbegin() + fault_vertex_offsets[fault],
          global_rhs.cbegin() + fault_vertex_offsets[fault+1]);
        result.nodal_values[fault] = solve_projection_system(fault, rhs);
        for (const double value : result.nodal_values[fault])
          AssertThrow(std::isfinite(value),
                      ExcMessage("Particle scalar projection produced a non-finite nodal value."));
      }

    std::vector<double> local_squared_residual(reconstructed_faults.size(), 0.0);
    std::vector<double> local_weight(reconstructed_faults.size(), 0.0);
    std::vector<double> local_maximum_residual(reconstructed_faults.size(), 0.0);
    std::vector<double> local_maximum_value(reconstructed_faults.size(), 0.0);
    cache_index = 0;
    for (const auto &particle : particle_handler)
      {
        const ParticleProjectionCacheEntry &entry = particle_projection_cache[cache_index++];
        if (!entry.active)
          continue;
        const double value = locally_owned_values.find(particle.get_id())->second;
        const double projected =
          (1.0-entry.xi) * result.nodal_values[entry.fault_index][entry.segment_index]
          + entry.xi * result.nodal_values[entry.fault_index][entry.segment_index+1];
        const double residual = std::abs(value-projected);
        local_squared_residual[entry.fault_index] +=
          entry.particle_domain_volume * residual * residual;
        local_weight[entry.fault_index] += entry.particle_domain_volume;
        local_maximum_residual[entry.fault_index] =
          std::max(local_maximum_residual[entry.fault_index], residual);
        local_maximum_value[entry.fault_index] =
          std::max(local_maximum_value[entry.fault_index], std::abs(value));
      }

    for (unsigned int fault = 0; fault < reconstructed_faults.size(); ++fault)
      {
        const double squared_residual = Utilities::MPI::sum(
          local_squared_residual[fault], this->get_mpi_communicator());
        const double weight = Utilities::MPI::sum(
          local_weight[fault], this->get_mpi_communicator());
        const double maximum_residual = Utilities::MPI::max(
          local_maximum_residual[fault], this->get_mpi_communicator());
        const double maximum_value = Utilities::MPI::max(
          local_maximum_value[fault], this->get_mpi_communicator());
        AssertThrow(std::isfinite(weight) && weight > 0.0,
                    ExcMessage("Particle scalar projection has no positive fault support."));

        ParticleScalarProjectionDiagnostics &diagnostic = result.diagnostics[fault];
        diagnostic.weighted_rms_residual = std::sqrt(squared_residual/weight);
        diagnostic.maximum_absolute_residual = maximum_residual;
        if (maximum_value > 0.0)
          {
            diagnostic.normalized_weighted_rms_residual =
              diagnostic.weighted_rms_residual/maximum_value;
            diagnostic.normalized_maximum_absolute_residual =
              diagnostic.maximum_absolute_residual/maximum_value;
          }
        else
          {
            Assert(diagnostic.weighted_rms_residual == 0.0
                   && diagnostic.maximum_absolute_residual == 0.0,
                   ExcInternalError());
            diagnostic.normalized_weighted_rms_residual = 0.0;
            diagnostic.normalized_maximum_absolute_residual = 0.0;
          }
      }

    return result;
  }


  template <int dim>
  void
  ReconstructedFaultManager<dim>::invalidate_particle_projection_cache()
  {
    particle_projection_cache_valid = false;
    particle_projection_cache.clear();
    projection_systems.clear();
    particle_projection_diagnostics.clear();
    cached_fault_geometry_versions.clear();
  }


  template <int dim>
  const std::vector<typename ReconstructedFaultManager<dim>::ParticleProjectionDiagnostics> &
  ReconstructedFaultManager<dim>::get_particle_projection_diagnostics() const
  {
    return particle_projection_diagnostics;
  }



  template <int dim>
  ReconstructedFaultUtilities::NormalProfileProjection
  ReconstructedFaultManager<dim>::project_to_normal_profiles(
    const Point<dim> &position) const
  {
    return ReconstructedFaultUtilities::project_to_normal_profiles(
      reconstructed_faults, projection_half_widths, position);
  }


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
  ReconstructedFault<dim>::initialize_properties(const unsigned int n_components)
  {
    Assert(property_values.empty(), ExcInternalError());
    n_property_components = n_components;
    property_values.resize(vertices.size() * n_property_components,
                           numbers::signaling_nan<double>());
  }


  template <int dim>
  ArrayView<double>
  ReconstructedFault<dim>::get_properties(const unsigned int vertex_index)
  {
    AssertIndexRange(vertex_index, vertices.size());
    return make_array_view(property_values.begin() + vertex_index * n_property_components,
                           property_values.begin() + (vertex_index + 1) * n_property_components);
  }


  template <int dim>
  ArrayView<const double>
  ReconstructedFault<dim>::get_properties(const unsigned int vertex_index) const
  {
    AssertIndexRange(vertex_index, vertices.size());
    return make_array_view(property_values.cbegin() + vertex_index * n_property_components,
                           property_values.cbegin() + (vertex_index + 1) * n_property_components);
  }


  template <int dim>
  void
  ReconstructedFault<dim>::append_vertex(const Point<dim> &new_vertex)
  {
    vertices.push_back(new_vertex);
    property_values.resize(vertices.size() * n_property_components,
                           numbers::signaling_nan<double>());
    ++current_geometry_version;
  }


  template <int dim>
  void
  ReconstructedFault<dim>::append_vertices(const std::vector<Point<dim>> &new_vertices)
  {
    if (new_vertices.empty())
      return;

    vertices.insert(vertices.end(), new_vertices.begin(), new_vertices.end());
    property_values.resize(vertices.size() * n_property_components,
                           numbers::signaling_nan<double>());
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
    template NormalProfileProjection \
    project_to_normal_profiles( \
      const std::vector<ReconstructedFault<dim>> &, \
      const std::vector<std::vector<double>> &, const Point<dim> &); \
  } \
  template class ReconstructedFault<dim>; \
  template class ReconstructedFaultManager<dim>;

  ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
}
