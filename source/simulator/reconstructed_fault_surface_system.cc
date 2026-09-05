/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include <aspect/simulator/reconstructed_fault_surface_system.h>

#include <aspect/material_model/phase_field_fault.h>
#include <aspect/material_model/utilities.h>
#include <aspect/particle/manager.h>
#include <aspect/phase_field.h>
#include <aspect/reconstructed_fault.h>

#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/numerics/vector_tools_evaluate.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace aspect
{
  namespace
  {
    template <int dim>
    const MaterialModel::PhaseFieldFault<dim> &
    checked_phase_field_fault(const SimulatorAccess<dim> &simulator)
    {
      const auto *phase_field_fault =
        dynamic_cast<const MaterialModel::PhaseFieldFault<dim> *>(
          &simulator.get_material_model());
      AssertThrow(phase_field_fault != nullptr,
                  ExcMessage("The reconstructed-fault surface system requires the "
                             "'Phase field fault' material model."));
      return *phase_field_fault;
    }


    template <int dim>
    struct SurfaceAssembly
    {
      ReconstructedFaultSurfaceResidual residual;
      std::vector<std::vector<double>> diagonal;
      std::vector<std::vector<double>> off_diagonal;
    };


    template <int dim>
    SurfaceAssembly<dim>
    assemble_surface_system(
      const SimulatorAccess<dim> &simulator,
      const MaterialModel::PhaseFieldFault<dim> &phase_field_fault,
      const LinearAlgebra::BlockVector &bulk_state,
      const std::vector<std::vector<double>> &slip_rate,
      const bool assemble_jacobian)
    {
      AssertThrow(dim == 2, ExcNotImplemented());
      phase_field_fault.validate_reconstructed_fault_constitutive_state();

      ReconstructedFaultManager<dim> &fault_manager =
        simulator.get_reconstructed_fault_manager();
      const auto &faults = fault_manager.get_faults();
      AssertThrow(!faults.empty(),
                  ExcMessage("Surface assembly requires reconstructed fault geometry."));
      AssertThrow(slip_rate.size() == faults.size(),
                  ExcMessage("The surface slip-rate vector has the wrong number of faults."));
      for (unsigned int fault = 0; fault < faults.size(); ++fault)
        AssertThrow(slip_rate[fault].size() == faults[fault].n_vertices(),
                    ExcMessage("The surface slip-rate vector has the wrong number of "
                               "vertices for reconstructed fault "
                               + Utilities::int_to_string(fault) + "."));

      const auto &associations =
        fault_manager.get_locally_owned_particle_fault_associations();
      std::vector<Point<dim>> points;
      points.reserve(associations.size());
      for (const auto &association : associations)
        if (association.active)
          points.push_back(association.position);

      Utilities::MPI::RemotePointEvaluation<dim> point_cache;
      point_cache.reinit(simulator.get_phase_field_handler().get_grid_cache(), points);
      const unsigned int velocity_component =
        simulator.introspection().variable("velocity").first_component_index;
      const unsigned int pressure_component =
        simulator.introspection().variable("pressure").first_component_index;
      const unsigned int temperature_component =
        simulator.introspection().variable("temperature").first_component_index;
      const unsigned int phase_field_component =
        simulator.introspection().variable("phase_field").first_component_index;

      const auto velocity_gradients = VectorTools::point_gradients<dim>(
        point_cache, simulator.get_dof_handler(), bulk_state,
        VectorTools::EvaluationFlags::avg, velocity_component);
      const std::vector<double> pressures = VectorTools::point_values<1>(
        point_cache, simulator.get_dof_handler(), bulk_state,
        VectorTools::EvaluationFlags::avg, pressure_component);
      const std::vector<double> temperatures = VectorTools::point_values<1>(
        point_cache, simulator.get_dof_handler(), bulk_state,
        VectorTools::EvaluationFlags::avg, temperature_component);
      const std::vector<double> phase_fields = VectorTools::point_values<1>(
        point_cache, simulator.get_dof_handler(), bulk_state,
        VectorTools::EvaluationFlags::avg, phase_field_component);
      const std::vector<double> previous_phase_fields = VectorTools::point_values<1>(
        point_cache, simulator.get_dof_handler(), simulator.get_old_solution(),
        VectorTools::EvaluationFlags::avg, phase_field_component);

      const Particle::Manager<dim> &particle_manager =
        simulator.get_phase_field_handler().get_associated_particle_manager();
      const auto &particle_handler = particle_manager.get_particle_handler();
      const auto &particle_data = particle_manager.get_property_manager().get_data_info();
      AssertThrow(particle_data.fieldname_exists("maxwell stress"),
                  ExcMessage("Reconstructed-fault surface assembly requires particle "
                             "property 'maxwell stress'."));
      const unsigned int stress_position =
        particle_data.get_position_by_field_name("maxwell stress");

      std::vector<unsigned int> chemical_positions;
      for (const unsigned int field :
           simulator.introspection().chemical_composition_field_indices())
        {
          const auto property = simulator.get_parameters().mapped_particle_properties.find(field);
          AssertThrow(property != simulator.get_parameters().mapped_particle_properties.end(),
                      ExcMessage("Reconstructed-fault surface assembly requires mapped "
                                 "particle chemical compositions."));
          chemical_positions.push_back(
            particle_data.get_position_by_field_name(property->second.first)
            + property->second.second);
        }

      SurfaceAssembly<dim> local;
      local.residual.values.resize(faults.size());
      local.residual.per_fault_weighted_rms.assign(faults.size(), 0.0);
      local.diagonal.resize(faults.size());
      local.off_diagonal.resize(faults.size());
      std::vector<double> local_squared_residual(faults.size(), 0.0);
      std::vector<double> local_weight(faults.size(), 0.0);
      for (unsigned int fault = 0; fault < faults.size(); ++fault)
        {
          local.residual.values[fault].assign(faults[fault].n_vertices(), 0.0);
          local.diagonal[fault].assign(faults[fault].n_vertices(), 0.0);
          local.off_diagonal[fault].assign(faults[fault].n_cells(), 0.0);
        }

      unsigned int association_index = 0;
      unsigned int point_index = 0;
      for (const auto &particle : particle_handler)
        {
          const auto &association = associations[association_index++];
          Assert(particle.get_id() == association.particle_id, ExcInternalError());
          if (!association.active)
            continue;

          const ReconstructedFault<dim> &fault = faults[association.fault_index];
          Tensor<1,dim> tangent = fault.vertex(association.segment_index+1)
                                  - fault.vertex(association.segment_index);
          tangent /= tangent.norm();
          Tensor<1,dim> normal;
          normal[0] = -tangent[1];
          normal[1] = tangent[0];

          typename MaterialModel::PhaseFieldFault<dim>::
            ReconstructedFaultPointInputs inputs;
          inputs.fault_index = association.fault_index;
          inputs.segment_index = association.segment_index;
          inputs.xi = association.xi;
          inputs.position = association.position;
          inputs.slip_rate = (1.0-association.xi)
                             * slip_rate[association.fault_index][association.segment_index]
                             + association.xi
                             * slip_rate[association.fault_index][association.segment_index+1];
          inputs.phase_field = phase_fields[point_index];
          inputs.previous_phase_field = previous_phase_fields[point_index];
          inputs.temperature = temperatures[point_index];
          inputs.dynamic_pressure = pressures[point_index];
          inputs.strain_rate = symmetrize(velocity_gradients[point_index]);
          inputs.slip_tensor = symmetrize(outer_product(tangent, normal));
          inputs.normal_tensor = symmetrize(outer_product(normal, normal));

          const ArrayView<const double> properties = particle.get_properties();
          for (unsigned int component = 0;
               component < SymmetricTensor<2,dim>::n_independent_components;
               ++component)
            inputs.old_maxwell_stress[
              SymmetricTensor<2,dim>::unrolled_to_component_indices(component)] =
                properties[stress_position+component];
          std::vector<double> chemical_compositions(chemical_positions.size());
          for (unsigned int c = 0; c < chemical_positions.size(); ++c)
            chemical_compositions[c] = properties[chemical_positions[c]];
          inputs.bulk_material_fractions =
            MaterialModel::MaterialUtilities::compute_composition_fractions(
              chemical_compositions);

          const auto response =
            phase_field_fault.evaluate_reconstructed_fault_point(inputs);
          AssertThrow(std::isfinite(response.residual_density)
                      && std::isfinite(response.minus_derivative_wrt_slip_rate),
                      ExcMessage("Reconstructed-fault constitutive evaluation produced "
                                 "a non-finite surface coefficient at particle "
                                 + Utilities::int_to_string(particle.get_id()) + "."));
          const double shape[2] = {1.0-association.xi, association.xi};
          const unsigned int vertex = association.segment_index;
          const double weight = association.particle_domain_volume;
          auto &residual = local.residual.values[association.fault_index];
          residual[vertex] += weight*shape[0]*response.residual_density;
          residual[vertex+1] += weight*shape[1]*response.residual_density;
          local_squared_residual[association.fault_index]
            += weight*response.residual_density*response.residual_density;
          local_weight[association.fault_index] += weight;

          if (assemble_jacobian)
            {
              auto &diagonal = local.diagonal[association.fault_index];
              auto &off_diagonal = local.off_diagonal[association.fault_index];
              diagonal[vertex] += weight*shape[0]*shape[0]
                                  * response.minus_derivative_wrt_slip_rate;
              diagonal[vertex+1] += weight*shape[1]*shape[1]
                                    * response.minus_derivative_wrt_slip_rate;
              off_diagonal[vertex] += weight*shape[0]*shape[1]
                                      * response.minus_derivative_wrt_slip_rate;
            }
          ++point_index;
        }
      AssertDimension(association_index, associations.size());
      AssertDimension(point_index, points.size());

      unsigned int packed_size = 2*faults.size();
      for (const auto &fault : faults)
        packed_size += 2*fault.n_vertices() + fault.n_cells();
      std::vector<double> local_values(packed_size, 0.0);
      unsigned int position = 0;
      for (unsigned int fault = 0; fault < faults.size(); ++fault)
        {
          std::copy(local.residual.values[fault].begin(),
                    local.residual.values[fault].end(), local_values.begin()+position);
          position += local.residual.values[fault].size();
          std::copy(local.diagonal[fault].begin(), local.diagonal[fault].end(),
                    local_values.begin()+position);
          position += local.diagonal[fault].size();
          std::copy(local.off_diagonal[fault].begin(), local.off_diagonal[fault].end(),
                    local_values.begin()+position);
          position += local.off_diagonal[fault].size();
          local_values[position++] = local_squared_residual[fault];
          local_values[position++] = local_weight[fault];
        }
      std::vector<double> global_values(packed_size);
      Utilities::MPI::sum(local_values, simulator.get_mpi_communicator(), global_values);

      double total_squared_residual = 0.0;
      double total_weight = 0.0;
      position = 0;
      for (unsigned int fault = 0; fault < faults.size(); ++fault)
        {
          std::copy_n(global_values.begin()+position,
                      local.residual.values[fault].size(),
                      local.residual.values[fault].begin());
          position += local.residual.values[fault].size();
          std::copy_n(global_values.begin()+position, local.diagonal[fault].size(),
                      local.diagonal[fault].begin());
          position += local.diagonal[fault].size();
          std::copy_n(global_values.begin()+position,
                      local.off_diagonal[fault].size(),
                      local.off_diagonal[fault].begin());
          position += local.off_diagonal[fault].size();
          const double squared_residual = global_values[position++];
          const double weight = global_values[position++];
          local.residual.per_fault_weighted_rms[fault] =
            std::sqrt(squared_residual/weight);
          total_squared_residual += squared_residual;
          total_weight += weight;
        }
      local.residual.weighted_rms = std::sqrt(total_squared_residual/total_weight);
      return local;
    }
  }


  template <int dim>
  struct ReconstructedFaultSurfaceSystem<dim>::SurfaceLinearization
  {
    struct FaultFactorization
    {
      SparsityPattern sparsity_pattern;
      SparseMatrix<double> matrix;
#ifdef DEAL_II_WITH_UMFPACK
      std::unique_ptr<SparseDirectUMFPACK> inverse;
#endif
    };

    ReconstructedFaultSurfaceResidual residual;
    std::vector<std::vector<double>> diagonal;
    std::vector<std::vector<double>> off_diagonal;
    std::vector<std::unique_ptr<FaultFactorization>> factorizations;
  };


  template <int dim>
  ReconstructedFaultSurfaceSystem<dim>::ReconstructedFaultSurfaceSystem(
    const Simulator<dim> &simulator)
    :
    SimulatorAccess<dim>(simulator),
    phase_field_fault(checked_phase_field_fault<dim>(*this))
  {}


  template <int dim>
  ReconstructedFaultSurfaceSystem<dim>::~ReconstructedFaultSurfaceSystem() = default;


  template <int dim>
  ReconstructedFaultSurfaceResidual
  ReconstructedFaultSurfaceSystem<dim>::evaluate_surface_residual(
    const LinearAlgebra::BlockVector &bulk_state,
    const FaultVector &slip_rate) const
  {
    return assemble_surface_system(*this, phase_field_fault,
                                   bulk_state, slip_rate, false).residual;
  }


  template <int dim>
  const ReconstructedFaultSurfaceResidual &
  ReconstructedFaultSurfaceSystem<dim>::linearize_surface_system(
    const LinearAlgebra::BlockVector &bulk_state,
    const FaultVector &slip_rate)
  {
    surface_linearization.reset();
#ifndef DEAL_II_WITH_UMFPACK
    AssertThrow(false,
                ExcMessage("The coupled reconstructed-fault solver requires deal.II "
                           "with UMFPACK support."));
#else
    const SurfaceAssembly<dim> assembled =
      assemble_surface_system(*this, phase_field_fault,
                              bulk_state, slip_rate, true);
    auto candidate = std::make_unique<SurfaceLinearization>();
    candidate->residual = assembled.residual;
    candidate->diagonal = assembled.diagonal;
    candidate->off_diagonal = assembled.off_diagonal;
    candidate->factorizations.resize(assembled.diagonal.size());

    for (unsigned int fault = 0; fault < assembled.diagonal.size(); ++fault)
      {
        auto factorization =
          std::make_unique<typename SurfaceLinearization::FaultFactorization>();
        const unsigned int n = assembled.diagonal[fault].size();
        DynamicSparsityPattern dynamic_sparsity(n, n);
        for (unsigned int i = 0; i < n; ++i)
          {
            dynamic_sparsity.add(i, i);
            if (i+1 < n)
              {
                dynamic_sparsity.add(i, i+1);
                dynamic_sparsity.add(i+1, i);
              }
          }
        factorization->sparsity_pattern.copy_from(dynamic_sparsity);
        factorization->matrix.reinit(factorization->sparsity_pattern);
        for (unsigned int i = 0; i < n; ++i)
          {
            factorization->matrix.set(i, i, assembled.diagonal[fault][i]);
            if (i+1 < n)
              {
                factorization->matrix.set(i, i+1, assembled.off_diagonal[fault][i]);
                factorization->matrix.set(i+1, i, assembled.off_diagonal[fault][i]);
              }
          }
        factorization->inverse = std::make_unique<SparseDirectUMFPACK>();
        try
          {
            factorization->inverse->initialize(factorization->matrix);
          }
        catch (const ExceptionBase &exception)
          {
            AssertThrow(false,
                        ExcMessage("Failed to factor reconstructed-fault K_V block "
                                   + Utilities::int_to_string(fault) + ": "
                                   + exception.what()));
          }
        candidate->factorizations[fault] = std::move(factorization);
      }
    surface_linearization = std::move(candidate);
#endif
    return surface_linearization->residual;
  }


  template <int dim>
  void
  ReconstructedFaultSurfaceSystem<dim>::solve_surface_jacobian(
    const FaultVector &rhs,
    FaultVector &solution) const
  {
    AssertThrow(surface_linearization != nullptr,
                ExcMessage("K_V must be assembled before applying its inverse."));
    AssertThrow(rhs.size() == surface_linearization->factorizations.size(),
                ExcMessage("The K_V right-hand side has the wrong number of faults."));
    solution.resize(rhs.size());
    for (unsigned int fault = 0; fault < rhs.size(); ++fault)
      {
        AssertThrow(rhs[fault].size()
                    == surface_linearization->diagonal[fault].size(),
                    ExcMessage("The K_V right-hand side has the wrong number of "
                               "vertices for reconstructed fault "
                               + Utilities::int_to_string(fault) + "."));
        Vector<double> source(rhs[fault].size());
        for (unsigned int i = 0; i < source.size(); ++i)
          source[i] = rhs[fault][i];
        Vector<double> result(source.size());
#ifdef DEAL_II_WITH_UMFPACK
        surface_linearization->factorizations[fault]->inverse->vmult(result, source);
#else
        AssertThrow(false, ExcNotImplemented());
#endif
        solution[fault].assign(result.begin(), result.end());

        double residual_norm = 0.0;
        double matrix_norm = 0.0;
        double solution_norm = 0.0;
        double rhs_norm = 0.0;
        for (unsigned int i = 0; i < result.size(); ++i)
          {
            double value = surface_linearization->diagonal[fault][i]*result[i];
            double row_sum = std::abs(surface_linearization->diagonal[fault][i]);
            if (i > 0)
              {
                value += surface_linearization->off_diagonal[fault][i-1]*result[i-1];
                row_sum += std::abs(surface_linearization->off_diagonal[fault][i-1]);
              }
            if (i+1 < result.size())
              {
                value += surface_linearization->off_diagonal[fault][i]*result[i+1];
                row_sum += std::abs(surface_linearization->off_diagonal[fault][i]);
              }
            residual_norm = std::max(residual_norm, std::abs(value-source[i]));
            matrix_norm = std::max(matrix_norm, row_sum);
            solution_norm = std::max(solution_norm, std::abs(result[i]));
            rhs_norm = std::max(rhs_norm, std::abs(source[i]));
          }
        const double scale = matrix_norm*solution_norm + rhs_norm;
        const double backward_error = residual_norm/std::max(
          scale, std::numeric_limits<double>::min());
        AssertThrow(std::isfinite(backward_error)
                    && backward_error
                       <= 100.0*std::numeric_limits<double>::epsilon()
                          * std::max(1.0, static_cast<double>(result.size())),
                    ExcMessage("The K_V solve for reconstructed fault "
                               + Utilities::int_to_string(fault)
                               + " has excessive scaled backward error "
                               + Utilities::to_string(backward_error)
                               + "; the block is singular or ill-conditioned."));
      }
  }


#define INSTANTIATE(dim) template class ReconstructedFaultSurfaceSystem<dim>;
  ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
}
