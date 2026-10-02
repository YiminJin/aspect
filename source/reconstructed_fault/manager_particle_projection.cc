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
#include <aspect/utilities.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <map>
#include <set>

namespace aspect
{
  namespace
  {
    std::pair<std::vector<double>, std::vector<double>>
    factor_tridiagonal(const std::vector<double> &diagonal,
                       const std::vector<double> &off_diagonal)
    {
      AssertThrow(!diagonal.empty(), ExcMessage("A projection system must not be empty."));
      AssertThrow(off_diagonal.size() + 1 == diagonal.size(),
                  ExcMessage("A tridiagonal projection system requires one fewer "
                             "off-diagonal entry than diagonal entries."));
      AssertThrow(std::all_of(diagonal.begin(), diagonal.end(),
                              [](const double value)
      {
        return std::isfinite(value);
      })
      && std::all_of(off_diagonal.begin(), off_diagonal.end(),
                     [](const double value)
      {
        return std::isfinite(value);
      }),
      ExcMessage("The particle-to-fault projection matrix is non-finite."));
      const double scale = *std::max_element(diagonal.begin(), diagonal.end());
      AssertThrow(scale > 0.0,
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
      std::vector<double> solution(rhs.begin(), rhs.end());
      for (unsigned int i = 1; i < solution.size(); ++i)
        solution[i] -= factor_lower[i-1] * solution[i-1];
      for (unsigned int i = 0; i < solution.size(); ++i)
        solution[i] /= factor_diagonal[i];
      for (unsigned int i = solution.size() - 1; i > 0; --i)
        solution[i-1] -= factor_lower[i-1] * solution[i];
      AssertThrow(std::all_of(solution.begin(), solution.end(),
                              [](const double value)
      {
        return std::isfinite(value);
      }),
      ExcMessage("The tridiagonal projection solve produced a non-finite result."));
      return solution;
    }
  }

  // -----------------------------------------------------------------------------
  // Particle-projection cache
  // -----------------------------------------------------------------------------

  template <int dim>
  bool
  ReconstructedFaultManager<dim>::particle_projection_cache_is_valid() const
  {
    TimerOutput::Scope timer(*performance_timer, "Fault: Cache validation");
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
    if (cached_domain_geometry_version != particle_domain_handler.geometry_version())
      return false;
    if (particle_handler.n_locally_owned_particles() != particle_projection_cache.size())
      return false;

    unsigned int particle_index = 0;
    for (const auto &particle : particle_handler)
      {
        const double volume = particle_domain_handler
                              .get_particle_domain(particle.get_local_index()).volume();
        const ParticleFaultAssociation &entry = particle_projection_cache[particle_index++];
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
    TimerOutput::Scope coarse_timer(this->get_computing_timer(), "Fault: Cache build");
    TimerOutput::Scope timer(*performance_timer, "Fault: Cache build total");
    ReconstructedFaultUtilities::DomainQuadratureStatistics quadrature_statistics;
    unsigned long long integration_points = 0;
    Particle::Manager<dim> &particle_manager =
      this->get_phase_field_handler().get_associated_particle_manager();
    AssertThrow(particle_manager.particle_domains_requested(),
                ExcMessage("Particle-to-fault projection requires particle-domain volumes."));
    const auto &particle_handler = particle_manager.get_particle_handler();
    const auto &particle_domain_handler = particle_manager.get_particle_domain_handler();

    // Geometry and influence widths are stable for the lifetime of this cache.
    ReconstructedFaultUtilities::internal::
    validate_normal_profile_projection_geometry(reconstructed_faults,
                                                projection_half_widths);

    // Initialize the local fault-sized Q1 systems and coverage diagnostics.
    projection_systems.clear();
    projection_systems.resize(reconstructed_faults.size());
    particle_projection_diagnostics.clear();
    particle_projection_diagnostics.resize(reconstructed_faults.size());
    for (unsigned int fault = 0; fault < reconstructed_faults.size(); ++fault)
      {
        projection_systems[fault].diagonal.assign(
          reconstructed_faults[fault].n_vertices(), 0.0);
        projection_systems[fault].off_diagonal.assign(
          reconstructed_faults[fault].n_cells(), 0.0);
        particle_projection_diagnostics[fault].weighted_support.assign(
          reconstructed_faults[fault].n_vertices(), 0.0);
      }

    // Associate each locally owned particle once and assemble the local mass
    // matrices with particle-domain volume as the sampling measure.
    particle_projection_cache.clear();
    particle_projection_cache.reserve(particle_handler.n_locally_owned_particles());
    std::vector<unsigned int> local_contributing_particles(reconstructed_faults.size(), 0);
    std::string local_geometry_error;
    try
      {
        for (const auto &particle : particle_handler)
          {
            const double volume = particle_domain_handler
                                  .get_particle_domain(particle.get_local_index()).volume();
            AssertThrow(std::isfinite(volume) && volume > 0.0,
                        ExcMessage("Particle-to-fault projection encountered a non-positive "
                                   "or non-finite particle-domain volume."));
            ReconstructedFaultUtilities::internal::
            validate_normal_profile_projection_position(particle.get_location());

            TimerOutput::Scope association_timer(*performance_timer, "Fault: Parent association");
            const ReconstructedFaultUtilities::NormalProfileProjection projection =
              ReconstructedFaultUtilities::internal::project_to_normal_profiles_unchecked(
                reconstructed_faults, projection_half_widths, particle.get_location());
            association_timer.stop();
            ParticleFaultAssociation entry;
            entry.particle_id = particle.get_id();
            entry.position = particle.get_location();
            entry.particle_domain_volume = volume;
            entry.active = projection.active;
            entry.fault_index = projection.fault_index;
            entry.segment_index = projection.segment_index;
            entry.xi = projection.xi;
            if (entry.active)
              {
                TimerOutput::Scope partition_timer(*performance_timer, "Fault: Domain partition");
                if constexpr (dim == 2)
                  {
                    const auto domain = particle_domain_handler.get_particle_domain(particle.get_local_index());
                    // Periodicity belongs to the bulk domain, not fault topology.
                    // Integrate every physical piece once against the open fault.
                    const auto append = [&](const std::vector<Point<dim>> &polygon)
                    {
                      const auto quadrature = ReconstructedFaultUtilities::domain_quadrature(
                        polygon, reconstructed_faults[entry.fault_index], 3, &quadrature_statistics);
                      entry.quadrature.insert(entry.quadrature.end(), quadrature.begin(), quadrature.end());
                    };
                    if (domain.periodic_fragments().empty())
                      append(domain.vertices());
                    else
                      for (const auto &fragment : domain.periodic_fragments()) append(fragment);
                  }
                else
                  AssertThrow(false, ExcNotImplemented());
                partition_timer.stop();
                integration_points += entry.quadrature.size();
                double integrated_volume = 0.0;
                ProjectionSystem &system = projection_systems[entry.fault_index];
                auto &support =
                  particle_projection_diagnostics[entry.fault_index].weighted_support;
                for (const auto &q : entry.quadrature)
                  {
                    const double shape[2] = {1.0-q.xi, q.xi};
                    const unsigned int first_vertex = q.segment_index;
                    system.diagonal[first_vertex] += q.weight * shape[0] * shape[0];
                    system.diagonal[first_vertex+1] += q.weight * shape[1] * shape[1];
                    system.off_diagonal[first_vertex] += q.weight * shape[0] * shape[1];
                    support[first_vertex] += q.weight * shape[0];
                    support[first_vertex+1] += q.weight * shape[1];
                    integrated_volume += q.weight;
                  }
                AssertThrow(std::abs(integrated_volume-volume) <= 1.e-10*volume,
                            ExcMessage("Surface domain quadrature does not cover the full parent volume."));
                ++local_contributing_particles[entry.fault_index];
              }
            particle_projection_cache.push_back(std::move(entry));
          }
      }
    catch (const std::exception &exception)
      {
        local_geometry_error=exception.what();
      }
    // A local cut/coverage failure must reach every owner before matrix sums.
    const unsigned int rank=Utilities::MPI::this_mpi_process(this->get_mpi_communicator());
    const unsigned int n_ranks=Utilities::MPI::n_mpi_processes(this->get_mpi_communicator());
    const unsigned int error_rank=Utilities::MPI::min(
      local_geometry_error.empty() ? n_ranks : rank,this->get_mpi_communicator());
    const std::string geometry_error=error_rank<n_ranks ? Utilities::MPI::broadcast(
      this->get_mpi_communicator(),local_geometry_error,error_rank) : std::string();
    AssertThrow(error_rank==n_ranks,ExcMessage(geometry_error));

    // The replicated fault ordering gives every rank the same packed layout,
    // so one collective reduction assembles all matrices and diagnostics.
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
    std::vector<double> global_values(packed_size);
    Utilities::MPI::sum(local_values, this->get_mpi_communicator(), global_values);
    // Validate global support and retain one factorization per fault for all
    // subsequent projected quantities.
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

    // Record exactly which geometry and projection metadata this cache uses.
    cached_fault_geometry_versions.resize(reconstructed_faults.size());
    for (unsigned int fault = 0; fault < reconstructed_faults.size(); ++fault)
      cached_fault_geometry_versions[fault] = reconstructed_faults[fault].geometry_version();
    cached_projection_metadata_version = projection_metadata_version;
    cached_domain_geometry_version = particle_domain_handler.geometry_version();
    particle_projection_cache_valid = true;
    if (std::getenv("ASPECT_FAULT_PERFORMANCE") != nullptr)
      {
        const auto mpi = this->get_mpi_communicator();
        const auto sum = [mpi](const unsigned long long value)
        { return Utilities::MPI::sum(value, mpi); };
        const auto points = sum(integration_points);
        const auto straight = sum(quadrature_statistics.straight_calls);
        const auto general = sum(quadrature_statistics.general_calls);
        const auto tests = sum(quadrature_statistics.segment_tests);
        const auto candidates = sum(quadrature_statistics.candidate_segments);
        this->get_pcout() << "Fault quadrature work: points=" << points
                         << ", straight calls=" << straight << ", general calls=" << general
                         << ", segment tests=" << tests << ", candidates=" << candidates
                         << std::endl;
      }
  }


  template <int dim>
  std::vector<unsigned int>
  ReconstructedFaultManager<dim>::fault_vertex_offsets() const
  {
    std::vector<unsigned int> offsets(reconstructed_faults.size() + 1, 0);
    for (unsigned int fault = 0; fault < reconstructed_faults.size(); ++fault)
      offsets[fault+1] = offsets[fault] + reconstructed_faults[fault].n_vertices();
    return offsets;
  }


  template <int dim>
  std::vector<typename ReconstructedFaultManager<dim>::FaultNodalValues>
  ReconstructedFaultManager<dim>::reduce_and_solve_projection_rhs(
    const std::vector<double> &local_rhs,
    const unsigned int n_components) const
  {
    const std::vector<unsigned int> offsets = fault_vertex_offsets();

    std::vector<double> global_rhs(local_rhs.size());
    Utilities::MPI::sum(local_rhs, this->get_mpi_communicator(), global_rhs);

    std::vector<FaultNodalValues> nodal_values(
      n_components,
      FaultNodalValues(reconstructed_faults.size()));
    for (unsigned int component = 0; component < n_components; ++component)
      for (unsigned int fault = 0; fault < reconstructed_faults.size(); ++fault)
        {
          const unsigned int begin = component * offsets.back() + offsets[fault];
          const ArrayView<const double> rhs = make_array_view(
                                                global_rhs.cbegin() + begin,
                                                global_rhs.cbegin() + begin + reconstructed_faults[fault].n_vertices());
          nodal_values[component][fault] =
            solve_tridiagonal_factors(projection_systems[fault].factor_diagonal,
                                      projection_systems[fault].factor_lower,
                                      rhs);
        }
    return nodal_values;
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


  // -----------------------------------------------------------------------------
  // Particle-to-fault projection
  // -----------------------------------------------------------------------------

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
                                                             + component)
                                 });
          }
      }

    if (!particle_projection_cache_is_valid())
      rebuild_particle_projection_cache();

    const std::vector<unsigned int> vertex_offsets = fault_vertex_offsets();
    const unsigned int n_fault_vertices = vertex_offsets.back();
    std::vector<double> local_rhs(components.size() * n_fault_vertices, 0.0);

    const auto &particle_handler = particle_manager.get_particle_handler();
    unsigned int cache_index = 0;
    for (const auto &particle : particle_handler)
      {
        const ParticleFaultAssociation &entry = particle_projection_cache[cache_index++];
        if (!entry.active)
          continue;
        const ArrayView<const double> particle_properties = particle.get_properties();
        for (unsigned int component = 0; component < components.size(); ++component)
          {
            const double value = particle_properties[components[component].particle_position];
            AssertThrow(std::isfinite(value),
                        ExcMessage("Particle " + Utilities::int_to_string(particle.get_id())
                                   + " has a non-finite value for "
                                   + components[component].description + "."));
            for (const auto &q : entry.quadrature)
              {
                const unsigned int first_vertex = vertex_offsets[entry.fault_index]+q.segment_index;
                local_rhs[component*n_fault_vertices + first_vertex]
                += q.weight * (1.0-q.xi) * value;
                local_rhs[component*n_fault_vertices + first_vertex+1]
                += q.weight * q.xi * value;
              }
          }
      }

    const auto nodal_values =
      reduce_and_solve_projection_rhs(local_rhs, components.size());
    for (unsigned int component = 0; component < components.size(); ++component)
      for (unsigned int fault = 0; fault < reconstructed_faults.size(); ++fault)
        for (unsigned int vertex = 0; vertex < reconstructed_faults[fault].n_vertices(); ++vertex)
          reconstructed_faults[fault].get_properties(vertex)[components[component].fault_position]
            = nodal_values[component][fault][vertex];
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
    for (const ParticleFaultAssociation &entry : particle_projection_cache)
      if (entry.active)
        {
          const ReconstructedFault<dim> &fault = reconstructed_faults[entry.fault_index];
          for (unsigned int component = 0; component < property.n_components; ++component)
            {
              const bool first_is_initialized = fault.property_value_is_initialized(
                                                  entry.segment_index, property.position+component);
              const bool second_is_initialized = fault.property_value_is_initialized(
                                                   entry.segment_index+1, property.position+component);
              AssertThrow(first_is_initialized && second_is_initialized,
                          ExcMessage("Fault property <" + property.name
                                     + "> is uninitialized at the cached projection of particle "
                                     + Utilities::int_to_string(entry.particle_id) + "."));
            }

          const ArrayView<const double> first = fault.get_properties(entry.segment_index);
          const ArrayView<const double> second = fault.get_properties(entry.segment_index+1);
          std::vector<double> interpolated(property.n_components);
          for (unsigned int component = 0; component < property.n_components; ++component)
            interpolated[component] =
              (1.0-entry.xi) * first[property.position+component]
              + entry.xi * second[property.position+component];
          values.emplace(entry.particle_id, std::move(interpolated));
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

    const std::vector<unsigned int> vertex_offsets = fault_vertex_offsets();
    const unsigned int n_fault_vertices = vertex_offsets.back();
    std::vector<double> local_rhs(n_fault_vertices, 0.0);

    const auto &particle_handler = this->get_phase_field_handler()
                                   .get_associated_particle_manager()
                                   .get_particle_handler();
    std::string local_error;
    unsigned int cache_index = 0;
    for (const auto &particle : particle_handler)
      {
        const ParticleFaultAssociation &entry = particle_projection_cache[cache_index++];
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

        for (const auto &q : entry.quadrature)
          {
            const unsigned int first_vertex = vertex_offsets[entry.fault_index]+q.segment_index;
            local_rhs[first_vertex] += q.weight * (1.0-q.xi) * value->second;
            local_rhs[first_vertex+1] += q.weight * q.xi * value->second;
          }
      }
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

    ParticleScalarProjectionResult result;
    const auto nodal_values = reduce_and_solve_projection_rhs(local_rhs, 1);
    result.nodal_values = nodal_values[0];
    result.diagnostics.resize(reconstructed_faults.size());

    std::vector<double> local_squared_residual(reconstructed_faults.size(), 0.0);
    std::vector<double> local_weight(reconstructed_faults.size(), 0.0);
    std::vector<double> local_maximum_residual(reconstructed_faults.size(), 0.0);
    std::vector<double> local_maximum_value(reconstructed_faults.size(), 0.0);
    cache_index = 0;
    for (const auto &particle : particle_handler)
      {
        const ParticleFaultAssociation &entry = particle_projection_cache[cache_index++];
        if (!entry.active)
          continue;
        const double value = locally_owned_values.find(particle.get_id())->second;
        for (const auto &q : entry.quadrature)
          {
            const double projected =
              (1.0-q.xi) * result.nodal_values[entry.fault_index][q.segment_index]
              + q.xi * result.nodal_values[entry.fault_index][q.segment_index+1];
            const double residual = std::abs(value-projected);
            local_squared_residual[entry.fault_index] += q.weight * residual * residual;
            local_weight[entry.fault_index] += q.weight;
            local_maximum_residual[entry.fault_index] =
              std::max(local_maximum_residual[entry.fault_index], residual);
          }
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
            diagnostic.normalized_weighted_rms_residual = 0.0;
            diagnostic.normalized_maximum_absolute_residual = 0.0;
          }
      }

    return result;
  }


  template <int dim>
  const std::vector<typename ReconstructedFaultManager<dim>::ParticleFaultAssociation> &
  ReconstructedFaultManager<dim>::get_locally_owned_particle_fault_associations()
  {
    if (!particle_projection_cache_is_valid())
      rebuild_particle_projection_cache();
    return particle_projection_cache;
  }


  template <int dim>
  const std::vector<typename ReconstructedFaultManager<dim>::ParticleProjectionDiagnostics> &
  ReconstructedFaultManager<dim>::get_particle_projection_diagnostics() const
  {
    return particle_projection_diagnostics;
  }




}

// Instantiate only the moved particle-projection definitions. Geometry,
// persistence and Stokes-QP associations remain in manager.cc.
namespace aspect
{
#define INSTANTIATE(dim) \
  template bool ReconstructedFaultManager<dim>::particle_projection_cache_is_valid() const; \
  template void ReconstructedFaultManager<dim>::rebuild_particle_projection_cache(); \
  template std::vector<unsigned int> ReconstructedFaultManager<dim>::fault_vertex_offsets() const; \
  template std::vector<ReconstructedFaultManager<dim>::FaultNodalValues> ReconstructedFaultManager<dim>::reduce_and_solve_projection_rhs(const std::vector<double> &, const unsigned int) const; \
  template void ReconstructedFaultManager<dim>::invalidate_particle_projection_cache(); \
  template void ReconstructedFaultManager<dim>::project_particle_properties(const std::vector<ParticlePropertyProjection> &); \
  template std::map<types::particle_index, std::vector<double>> ReconstructedFaultManager<dim>::interpolate_property_at_particle_projections(const unsigned int); \
  template ReconstructedFaultManager<dim>::ParticleScalarProjectionResult ReconstructedFaultManager<dim>::project_particle_scalar(const std::map<types::particle_index, double> &); \
  template const std::vector<ReconstructedFaultManager<dim>::ParticleFaultAssociation> &ReconstructedFaultManager<dim>::get_locally_owned_particle_fault_associations(); \
  template const std::vector<ReconstructedFaultManager<dim>::ParticleProjectionDiagnostics> &ReconstructedFaultManager<dim>::get_particle_projection_diagnostics() const;

  ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
}
