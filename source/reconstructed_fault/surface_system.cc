/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include <aspect/reconstructed_fault/surface_system.h>

#include <aspect/material_model/phase_field_fault.h>
#include <aspect/material_model/utilities.h>
#include <aspect/particle/manager.h>
#include <aspect/phase_field.h>
#include <aspect/plugins.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/reconstructed_fault/utilities.h>

#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/numerics/vector_tools_evaluate.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>

namespace aspect
{
  namespace
  {
    template <int dim>
    class RestrictedSurfaceLinearSolve final
      : public ReconstructedFaultSurfaceLinearSolve<dim>
    {
      public:
        RestrictedSurfaceLinearSolve(
          const ReconstructedFaultSurfaceSystem<dim> &owner,
          const unsigned int generation,
          const std::vector<std::vector<double>> &diagonal,
          const std::vector<std::vector<double>> &off_diagonal,
          const ReconstructedFaultActiveSet &active_set)
          : owner(owner), generation(generation), active_set(active_set)
        {
#ifndef DEAL_II_WITH_UMFPACK
          AssertThrow(false, ExcNotImplemented());
#else
          // Materialize K_FF direct-sum I_AA at full fault-vector size. Omitting
          // every free/active edge removes active columns from the free equations.
          factorizations.resize(diagonal.size());
          for (unsigned int fault = 0; fault < diagonal.size(); ++fault)
            {
              AssertDimension(active_set[fault].size(), diagonal[fault].size());
              AssertDimension(off_diagonal[fault].size(),
                              diagonal[fault].empty()
                              ? 0
                              : diagonal[fault].size()-1);
              auto factorization = std::make_unique<FaultFactorization>();
              const unsigned int n = diagonal[fault].size();
              DynamicSparsityPattern dynamic_sparsity(n, n);
              for (unsigned int i = 0; i < n; ++i)
                {
                  dynamic_sparsity.add(i, i);
                  if (i+1 < n && !active_set[fault][i]
                      && !active_set[fault][i+1])
                    {
                      dynamic_sparsity.add(i, i+1);
                      dynamic_sparsity.add(i+1, i);
                    }
                }
              factorization->sparsity_pattern.copy_from(dynamic_sparsity);
              factorization->matrix.reinit(factorization->sparsity_pattern);
              for (unsigned int i = 0; i < n; ++i)
                {
                  factorization->matrix.set(
                    i, i, active_set[fault][i] ? 1.0 : diagonal[fault][i]);
                  if (i+1 < n && !active_set[fault][i]
                      && !active_set[fault][i+1])
                    {
                      factorization->matrix.set(i, i+1, off_diagonal[fault][i]);
                      factorization->matrix.set(i+1, i, off_diagonal[fault][i]);
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
                              ExcMessage("Failed to factor the free K_V block for "
                                         "reconstructed fault "
                                         + Utilities::int_to_string(fault) + ": "
                                         + exception.what()));
                }
              factorizations[fault] = std::move(factorization);
            }
#endif
        }

        void
        solve(const ReconstructedFaultVector &rhs,
              ReconstructedFaultVector &solution) const override
        {
          AssertThrow(generation == owner.get_linearization_generation(),
                      ExcMessage("This restricted reconstructed-fault surface "
                                 "solve has been superseded."));
          AssertThrow(rhs.size() == factorizations.size(),
                      ExcMessage("The restricted K_V right-hand side has the "
                                 "wrong number of faults."));
          solution.resize(rhs.size());
          for (unsigned int fault = 0; fault < rhs.size(); ++fault)
            {
              // Project the right-hand side before solving and overwrite active
              // entries afterward so the semantic inverse returns exact zeros.
              AssertDimension(rhs[fault].size(), active_set[fault].size());
              Vector<double> source(rhs[fault].size());
              for (unsigned int i = 0; i < source.size(); ++i)
                source[i] = active_set[fault][i] ? 0.0 : rhs[fault][i];
              Vector<double> result(source.size());
#ifdef DEAL_II_WITH_UMFPACK
              factorizations[fault]->inverse->vmult(result, source);
#else
              AssertThrow(false, ExcNotImplemented());
#endif
              solution[fault].resize(result.size());
              for (unsigned int i = 0; i < result.size(); ++i)
                solution[fault][i] = active_set[fault][i] ? 0.0 : result[i];

              // A scaled backward error is the numerical admissibility guard
              // for a nonsingular, potentially indefinite principal free block.
              double residual_norm = 0.0;
              double matrix_norm = 0.0;
              double solution_norm = 0.0;
              double rhs_norm = 0.0;
              for (unsigned int i = 0; i < result.size(); ++i)
                {
                  double value = active_set[fault][i]
                                 ? solution[fault][i]
                                 : factorizations[fault]->matrix.el(i, i)
                                   * solution[fault][i];
                  double row_sum = std::abs(
                                     factorizations[fault]->matrix.el(i, i));
                  if (i > 0)
                    {
                      value += factorizations[fault]->matrix.el(i, i-1)
                               * solution[fault][i-1];
                      row_sum += std::abs(
                                   factorizations[fault]->matrix.el(i, i-1));
                    }
                  if (i+1 < result.size())
                    {
                      value += factorizations[fault]->matrix.el(i, i+1)
                               * solution[fault][i+1];
                      row_sum += std::abs(
                                   factorizations[fault]->matrix.el(i, i+1));
                    }
                  residual_norm = std::max(residual_norm,
                                           std::abs(value-source[i]));
                  matrix_norm = std::max(matrix_norm, row_sum);
                  solution_norm = std::max(solution_norm,
                                            std::abs(solution[fault][i]));
                  rhs_norm = std::max(rhs_norm, std::abs(source[i]));
                }
              const double scale = matrix_norm*solution_norm + rhs_norm;
              const double backward_error = residual_norm/std::max(
                                              scale,
                                              std::numeric_limits<double>::min());
              AssertThrow(std::isfinite(backward_error)
                          && backward_error
                          <= 100.0*std::numeric_limits<double>::epsilon()
                          * std::max(1.0, static_cast<double>(result.size())),
                          ExcMessage("The free K_V solve for reconstructed fault "
                                     + Utilities::int_to_string(fault)
                                     + " has excessive scaled backward error "
                                     + Utilities::to_string(backward_error)
                                     + "; the free block is singular or ill-conditioned."));
            }
        }

      private:
        struct FaultFactorization
        {
          SparsityPattern sparsity_pattern;
          SparseMatrix<double> matrix;
#ifdef DEAL_II_WITH_UMFPACK
          std::unique_ptr<SparseDirectUMFPACK> inverse;
#endif
        };

        const ReconstructedFaultSurfaceSystem<dim> &owner;
        const unsigned int generation;
        const ReconstructedFaultActiveSet active_set;
        std::vector<std::unique_ptr<FaultFactorization>> factorizations;
    };
  }


  template <int dim>
  struct ReconstructedFaultSurfaceSystem<dim>::SurfaceAssembly
  {
    struct CouplingPoint
    {
      Point<dim> position;
      unsigned int parent_index;
      unsigned int fault_index;
      unsigned int segment_index;
      double xi;
      double particle_domain_volume;
      double kappa;
      double friction_coefficient;
      SymmetricTensor<2,dim> slip_tensor;
      SymmetricTensor<2,dim> normal_tensor;
      bool uses_adiabatic_friction_pressure;
    };

    ReconstructedFaultSurfaceResidual residual;
    std::vector<std::vector<double>> diagonal;
    std::vector<std::vector<double>> off_diagonal;
    std::vector<std::vector<double>> mass_diagonal;
    std::vector<std::vector<double>> mass_off_diagonal;
    std::vector<CouplingPoint> coupling_points;
  };


  template <int dim>
  typename ReconstructedFaultSurfaceSystem<dim>::SurfaceAssembly
  ReconstructedFaultSurfaceSystem<dim>::assemble_surface_system(
    const LinearAlgebra::BlockVector &bulk_state,
    const FaultVector &slip_rate,
    const bool assemble_jacobian) const
  {
    TimerOutput::Scope total_timer(*performance_timer, "Fault: Surface R/K total");
    AssertThrow(dim == 2, ExcNotImplemented());
    phase_field_fault.validate_reconstructed_fault_constitutive_state();

    ReconstructedFaultManager<dim> &fault_manager =
      this->get_reconstructed_fault_manager();
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

    // Particle/fault associations are locally owned, whereas the bulk FE
    // samples may live remotely. Evaluate all active particle positions in one
    // collective point-evaluation operation and preserve association order.
    const auto &associations =
      fault_manager.get_locally_owned_particle_fault_associations();
    std::vector<Point<dim>> points;
    points.reserve(associations.size());
    for (const auto &association : associations)
      if (association.active)
        points.push_back(association.position);

    Utilities::MPI::RemotePointEvaluation<dim> point_cache;
    TimerOutput::Scope lookup_timer(*performance_timer, "Fault: Parent lookup build");
    point_cache.reinit(grid_cache, points);
    lookup_timer.stop();
    const unsigned int velocity_component =
      this->introspection().component_indices.velocities[0];
    const unsigned int pressure_component =
      this->introspection().component_indices.pressure;
    const unsigned int temperature_component =
      this->introspection().component_indices.temperature;
    const unsigned int phase_field_component =
      this->introspection().variable("phase_field").first_component_index;

    TimerOutput::Scope sample_timer(*performance_timer, "Fault: Parent FE sampling");
    const auto velocity_gradients = VectorTools::point_gradients<dim>(
                                      point_cache, this->get_dof_handler(), bulk_state,
                                      VectorTools::EvaluationFlags::avg, velocity_component);
    const std::vector<double> pressures = VectorTools::point_values<1>(
                                            point_cache, this->get_dof_handler(), bulk_state,
                                            VectorTools::EvaluationFlags::avg, pressure_component);
    const std::vector<double> temperatures = VectorTools::point_values<1>(
                                               point_cache, this->get_dof_handler(), bulk_state,
                                               VectorTools::EvaluationFlags::avg, temperature_component);
    const std::vector<double> phase_fields = VectorTools::point_values<1>(
                                               point_cache, this->get_dof_handler(), bulk_state,
                                               VectorTools::EvaluationFlags::avg, phase_field_component);
    const std::vector<double> previous_phase_fields = VectorTools::point_values<1>(
                                                        point_cache, this->get_dof_handler(), this->get_old_solution(),
                                                        VectorTools::EvaluationFlags::avg, phase_field_component);
    sample_timer.stop();
    TimerOutput::Scope assembly_timer(*performance_timer, "Fault: Surface R/K integrate");

    const Particle::Manager<dim> &particle_manager =
      this->get_phase_field_handler().get_associated_particle_manager();
    const auto &particle_handler = particle_manager.get_particle_handler();
    const auto &property_manager = particle_manager.get_property_manager();
    const auto &particle_data = property_manager.get_data_info();
    AssertThrow(property_manager.plugin_name_exists("maxwell stress"),
                ExcMessage("Reconstructed-fault surface assembly requires particle "
                           "property plugin 'maxwell stress'."));
    const unsigned int stress_position =
      particle_data.get_position_by_plugin_index(
        property_manager.get_plugin_index_by_name("maxwell stress"));

    std::vector<unsigned int> chemical_positions;
    for (const unsigned int field :
         this->introspection().chemical_composition_field_indices())
      {
        const auto property = this->get_parameters().mapped_particle_properties.find(field);
        AssertThrow(property != this->get_parameters().mapped_particle_properties.end(),
                    ExcMessage("Reconstructed-fault surface assembly requires mapped "
                               "particle chemical compositions."));
        chemical_positions.push_back(
          particle_data.get_position_by_field_name(property->second.first)
          + property->second.second);
      }

    // Accumulate each rank's particle-domain quadrature into the replicated Q1
    // surface residual and, when requested, K_V=-dR_Gamma/dV.
    SurfaceAssembly local;
    local.residual.values.resize(faults.size());
    local.residual.shear_traction.resize(faults.size());
    local.residual.cohesive_traction.resize(faults.size());
    local.residual.friction_traction.resize(faults.size());
    local.residual.damping_traction.resize(faults.size());
    local.residual.per_fault_weighted_rms.assign(faults.size(), 0.0);
    local.diagonal.resize(faults.size());
    local.off_diagonal.resize(faults.size());
    local.mass_diagonal.resize(faults.size());
    local.mass_off_diagonal.resize(faults.size());
    std::vector<double> local_squared_residual(faults.size(), 0.0);
    std::vector<double> local_weight(faults.size(), 0.0);
    // Opt-in pilot evidence from the actual pre-publication linearization.
    // Keep rank-local moments separate from the residual and its MPI reduction.
    const bool record_normal_stress = assemble_jacobian
      && std::getenv("ASPECT_FAULT_NORMAL_STRESS_DIAGNOSTIC");
    std::vector<std::vector<std::array<double,10>>> normal_stress_moments;
    if (record_normal_stress)
      {
        AssertThrow(!phase_field_fault.uses_adiabatic_friction_pressure(),
                    ExcMessage("Normal-stress pilot diagnostics require true-pressure friction."));
        normal_stress_moments.resize(faults.size());
        for (unsigned int f=0; f<faults.size(); ++f)
          normal_stress_moments[f].resize(faults[f].n_vertices(),
            {{0,0,0,0, std::numeric_limits<double>::infinity(),
              -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
              -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
              -std::numeric_limits<double>::infinity()}});
      }
    for (unsigned int fault = 0; fault < faults.size(); ++fault)
      {
        local.residual.values[fault].assign(faults[fault].n_vertices(), 0.0);
        local.residual.shear_traction[fault].assign(faults[fault].n_vertices(), 0.0);
        local.residual.cohesive_traction[fault].assign(faults[fault].n_vertices(), 0.0);
        local.residual.friction_traction[fault].assign(faults[fault].n_vertices(), 0.0);
        local.residual.damping_traction[fault].assign(faults[fault].n_vertices(), 0.0);
        local.diagonal[fault].assign(faults[fault].n_vertices(), 0.0);
        local.off_diagonal[fault].assign(faults[fault].n_cells(), 0.0);
        local.mass_diagonal[fault].assign(faults[fault].n_vertices(), 0.0);
        local.mass_off_diagonal[fault].assign(faults[fault].n_cells(), 0.0);
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

        typename MaterialModel::PhaseFieldFault<dim>::
        ReconstructedFaultPointInputs inputs;
        inputs.fault_index = association.fault_index;
        inputs.position = association.position;
        inputs.phase_field = phase_fields[point_index];
        inputs.previous_phase_field = previous_phase_fields[point_index];
        inputs.temperature = temperatures[point_index];
        inputs.dynamic_pressure = pressures[point_index];
        inputs.strain_rate = symmetrize(velocity_gradients[point_index]);

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

        // Parent bulk/history samples stay P0; all surface Q1 inputs and the
        // nonlinear response vary across the domain. K_V and G differentiate
        // exactly these evaluations, including domains crossing fault nodes.
        for (const auto &q : association.quadrature)
          {
            Tensor<1,dim> tangent=fault.vertex(q.segment_index+1)-fault.vertex(q.segment_index);
            tangent/=tangent.norm();
            Tensor<1,dim> normal;
            normal[0]=-tangent[1];
            normal[1]=tangent[0];
            inputs.slip_tensor=symmetrize(outer_product(tangent,normal));
            inputs.normal_tensor=symmetrize(outer_product(normal,normal));
            inputs.segment_index = q.segment_index;
            inputs.xi = q.xi;
            const double left_slip_rate = slip_rate[association.fault_index][q.segment_index];
            inputs.slip_rate = left_slip_rate + q.xi *
              (slip_rate[association.fault_index][q.segment_index+1]-left_slip_rate);
            const auto response =
              phase_field_fault.evaluate_reconstructed_fault_point(inputs);
            AssertThrow(std::isfinite(response.residual_density)
                        && std::isfinite(response.minus_derivative_wrt_slip_rate),
                        ExcMessage("Reconstructed-fault constitutive evaluation produced "
                                   "a non-finite surface coefficient at particle "
                                   + Utilities::int_to_string(particle.get_id()) + "."));
            const double shape[2] = {1.0-q.xi, q.xi};
            const unsigned int vertex = q.segment_index;
            const double weight = q.weight;
            if (record_normal_stress)
              {
                // The response already contains mu*(p-tau:N). Recover its
                // sigma_n without reevaluating Maxwell from committed history.
                AssertThrow(response.friction_coefficient > 0.0,
                            ExcMessage("Normal-stress diagnostic requires positive friction coefficient."));
                const double sigma = response.friction_traction/response.friction_coefficient;
                const double values[3] = {inputs.dynamic_pressure, sigma,
                                         inputs.dynamic_pressure-sigma};
                for (unsigned int i=0; i<2; ++i)
                  if (weight*shape[i] > 0.0)
                    {
                      auto &moments = normal_stress_moments[association.fault_index][vertex+i];
                      moments[0] += weight*shape[i];
                      for (unsigned int c=0; c<3; ++c)
                        {
                          moments[1+c] += weight*shape[i]*values[c];
                          moments[4+2*c] = std::min(moments[4+2*c],values[c]);
                          moments[5+2*c] = std::max(moments[5+2*c],values[c]);
                        }
                    }
              }
            auto &residual = local.residual.values[association.fault_index];
            residual[vertex] += weight*shape[0]*response.residual_density;
            residual[vertex+1] += weight*shape[1]*response.residual_density;
            for (unsigned int i=0; i<2; ++i)
              {
                local.residual.shear_traction[association.fault_index][vertex+i]
                  += weight*shape[i]*response.shear_traction;
                local.residual.cohesive_traction[association.fault_index][vertex+i]
                  += weight*shape[i]*response.cohesive_traction;
                local.residual.friction_traction[association.fault_index][vertex+i]
                  += weight*shape[i]*response.friction_traction;
                local.residual.damping_traction[association.fault_index][vertex+i]
                  += weight*shape[i]*response.damping_traction;
              }
            local_squared_residual[association.fault_index]
            += weight*response.residual_density*response.residual_density;
            local_weight[association.fault_index] += weight;

            auto &mass_diagonal = local.mass_diagonal[association.fault_index];
            auto &mass_off_diagonal =
              local.mass_off_diagonal[association.fault_index];
            mass_diagonal[vertex] += weight*shape[0]*shape[0];
            mass_diagonal[vertex+1] += weight*shape[1]*shape[1];
            mass_off_diagonal[vertex] += weight*shape[0]*shape[1];

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
                local.coupling_points.push_back(
                {
                  association.position,
                  point_index,
                  association.fault_index,
                  q.segment_index,
                  q.xi,
                  q.weight,
                  response.kappa,
                  response.friction_coefficient,
                  inputs.slip_tensor,
                  inputs.normal_tensor,
                  response.uses_adiabatic_friction_pressure
                });
              }
          }
        ++point_index;
      }
    AssertDimension(association_index, associations.size());
    AssertDimension(point_index, points.size());

    // Faults are replicated but particles are distributed. Pack the small
    // surface systems into one collective sum so every rank receives identical
    // residuals, Jacobian coefficients, and diagnostics.
    unsigned int packed_size = 2*faults.size();
    for (const auto &fault : faults)
      packed_size += 8*fault.n_vertices() + 2*fault.n_cells();
    std::vector<double> local_values(packed_size, 0.0);
    unsigned int position = 0;
    for (unsigned int fault = 0; fault < faults.size(); ++fault)
      {
        std::copy(local.residual.values[fault].begin(),
                  local.residual.values[fault].end(), local_values.begin()+position);
        position += local.residual.values[fault].size();
        for (const auto *term : {&local.residual.shear_traction,
                                &local.residual.cohesive_traction,
                                &local.residual.friction_traction,
                                &local.residual.damping_traction})
          {
            std::copy((*term)[fault].begin(), (*term)[fault].end(), local_values.begin()+position);
            position += (*term)[fault].size();
          }
        std::copy(local.diagonal[fault].begin(), local.diagonal[fault].end(),
                  local_values.begin()+position);
        position += local.diagonal[fault].size();
        std::copy(local.off_diagonal[fault].begin(), local.off_diagonal[fault].end(),
                  local_values.begin()+position);
        position += local.off_diagonal[fault].size();
        std::copy(local.mass_diagonal[fault].begin(),
                  local.mass_diagonal[fault].end(), local_values.begin()+position);
        position += local.mass_diagonal[fault].size();
        std::copy(local.mass_off_diagonal[fault].begin(),
                  local.mass_off_diagonal[fault].end(), local_values.begin()+position);
        position += local.mass_off_diagonal[fault].size();
        local_values[position++] = local_squared_residual[fault];
        local_values[position++] = local_weight[fault];
      }
    std::vector<double> global_values(packed_size);
    Utilities::MPI::sum(local_values, this->get_mpi_communicator(), global_values);

    double total_squared_residual = 0.0;
    double total_weight = 0.0;
    position = 0;
    for (unsigned int fault = 0; fault < faults.size(); ++fault)
      {
        std::copy_n(global_values.begin()+position,
                    local.residual.values[fault].size(),
                    local.residual.values[fault].begin());
        position += local.residual.values[fault].size();
        for (auto *term : {&local.residual.shear_traction,
                          &local.residual.cohesive_traction,
                          &local.residual.friction_traction,
                          &local.residual.damping_traction})
          {
            std::copy_n(global_values.begin()+position, (*term)[fault].size(), (*term)[fault].begin());
            position += (*term)[fault].size();
          }
        std::copy_n(global_values.begin()+position, local.diagonal[fault].size(),
                    local.diagonal[fault].begin());
        position += local.diagonal[fault].size();
        std::copy_n(global_values.begin()+position,
                    local.off_diagonal[fault].size(),
                    local.off_diagonal[fault].begin());
        position += local.off_diagonal[fault].size();
        std::copy_n(global_values.begin()+position,
                    local.mass_diagonal[fault].size(),
                    local.mass_diagonal[fault].begin());
        position += local.mass_diagonal[fault].size();
        std::copy_n(global_values.begin()+position,
                    local.mass_off_diagonal[fault].size(),
                    local.mass_off_diagonal[fault].begin());
        position += local.mass_off_diagonal[fault].size();
        const double squared_residual = global_values[position++];
        const double weight = global_values[position++];
        local.residual.per_fault_weighted_rms[fault] =
          (weight > 0.0 ? std::sqrt(squared_residual/weight) : 0.0);
        total_squared_residual += squared_residual;
        total_weight += weight;
      }
    local.residual.weighted_rms =
      (total_weight > 0.0
       ? std::sqrt(total_squared_residual/total_weight)
       : 0.0);
    local.residual.mass_diagonal = local.mass_diagonal;
    local.residual.mass_off_diagonal = local.mass_off_diagonal;
    if (record_normal_stress)
      {
        // Replace only this step/rank's diagnostic on each new linearization.
        // A final file is accepted evidence only if the solve actually converges.
        std::ofstream output(this->get_output_directory()+"constitutive_normal_"
          + std::to_string(this->get_timestep_number())+"_rank"
          + std::to_string(Utilities::MPI::this_mpi_process(this->get_mpi_communicator()))+".csv");
        output.exceptions(std::ios::failbit | std::ios::badbit);
        output << std::setprecision(17)
               << "step,time,fault,node,weight,p_load,sigma_load,tauN_load,p_min,p_max,sigma_min,sigma_max,tauN_min,tauN_max\n";
        for (unsigned int f=0; f<faults.size(); ++f)
          for (unsigned int i=0; i<faults[f].n_vertices(); ++i)
            {
              output << this->get_timestep_number() << ',' << this->get_time() << ',' << f << ',' << i;
              for (const double value : normal_stress_moments[f][i])
                output << ',' << value;
              output << '\n';
            }
      }
    return local;
  }


  template <int dim>
  struct ReconstructedFaultSurfaceSystem<dim>::SurfaceLinearization
  {
    struct CouplingPoint
    {
      Point<dim> position;
      unsigned int parent_index;
      unsigned int fault_index;
      unsigned int segment_index;
      double xi;
      double particle_domain_volume;
      double kappa;
      double friction_coefficient;
      SymmetricTensor<2,dim> slip_tensor;
      SymmetricTensor<2,dim> normal_tensor;
      bool uses_adiabatic_friction_pressure;
    };

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
    std::vector<std::vector<double>> mass_diagonal;
    std::vector<std::vector<double>> mass_off_diagonal;
    std::vector<std::unique_ptr<FaultFactorization>> factorizations;
    std::vector<CouplingPoint> coupling_points;
    std::unique_ptr<Utilities::MPI::RemotePointEvaluation<dim>> point_cache;
  };


  template <int dim>
  ReconstructedFaultSurfaceSystem<dim>::ReconstructedFaultSurfaceSystem(
    const Simulator<dim> &simulator)
    :
    SimulatorAccess<dim>(simulator),
    phase_field_fault(
      Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(
        this->get_material_model())),
    grid_cache(this->get_triangulation(), this->get_mapping())
  {
    performance_timer = std::make_unique<TimerOutput>(
      std::cout,
      std::getenv("ASPECT_FAULT_PERFORMANCE") && this->get_pcout().is_active()
      ? TimerOutput::summary : TimerOutput::never, TimerOutput::wall_times);
  }


  template <int dim>
  ReconstructedFaultSurfaceSystem<dim>::~ReconstructedFaultSurfaceSystem() = default;


  template <int dim>
  ReconstructedFaultSurfaceResidual
  ReconstructedFaultSurfaceSystem<dim>::evaluate_surface_residual(
    const LinearAlgebra::BlockVector &bulk_state,
    const FaultVector &slip_rate) const
  {
    return assemble_surface_system(bulk_state, slip_rate, false).residual;
  }


  template <int dim>
  const ReconstructedFaultSurfaceResidual &
  ReconstructedFaultSurfaceSystem<dim>::linearize_surface_system(
    const LinearAlgebra::BlockVector &bulk_state,
    const FaultVector &slip_rate)
  {
    // Invalidate all previous semantic solves before building any new K_V data.
    TimerOutput::Scope timer(*performance_timer, "Fault: Linearization total");
    surface_linearization.reset();
    ++linearization_generation;
#ifndef DEAL_II_WITH_UMFPACK
    AssertThrow(false,
                ExcMessage("The coupled reconstructed-fault solver requires deal.II "
                           "with UMFPACK support."));
#else
    const SurfaceAssembly assembled =
      assemble_surface_system(bulk_state, slip_rate, true);
    auto candidate = std::make_unique<SurfaceLinearization>();
    candidate->residual = assembled.residual;
    candidate->diagonal = assembled.diagonal;
    candidate->off_diagonal = assembled.off_diagonal;
    candidate->mass_diagonal = assembled.mass_diagonal;
    candidate->mass_off_diagonal = assembled.mass_off_diagonal;
    candidate->coupling_points.reserve(assembled.coupling_points.size());
    for (const auto &point : assembled.coupling_points)
      candidate->coupling_points.push_back(
      {
        point.position,
        point.parent_index,
        point.fault_index,
        point.segment_index,
        point.xi,
        point.particle_domain_volume,
        point.kappa,
        point.friction_coefficient,
        point.slip_tensor,
        point.normal_tensor,
        point.uses_adiabatic_friction_pressure
      });
    candidate->factorizations.resize(assembled.diagonal.size());

    // Each reconstructed fault is an independent replicated Q1 block. Factor
    // the full nonsingular block once for this coupled linearization.
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

    // G reuses the same particle points and constitutive coefficients as K_V;
    // cache their remote bulk-field lookup for the lifetime of this linearization.
    std::vector<Point<dim>> points;
    points.reserve(candidate->coupling_points.empty() ? 0 :
                   candidate->coupling_points.back().parent_index+1);
    for (const auto &point : candidate->coupling_points)
      if (point.parent_index == points.size())
        points.push_back(point.position);
    candidate->point_cache =
      std::make_unique<Utilities::MPI::RemotePointEvaluation<dim>>();
    TimerOutput::Scope lookup_timer(*performance_timer, "Fault: G lookup build");
    candidate->point_cache->reinit(grid_cache, points);
    surface_linearization = std::move(candidate);
#endif
    return surface_linearization->residual;
  }


  template <int dim>
  void
  ReconstructedFaultSurfaceSystem<dim>::solve(
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
        // Solve each replicated fault block independently; no MPI exchange is
        // needed because assembly already made K_V and the right-hand side global.
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

        // Detect singular or ill-conditioned surface blocks by scaled backward
        // error rather than assuming definiteness of K_V.
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


  template <int dim>
  std::unique_ptr<ReconstructedFaultSurfaceLinearSolve<dim>>
  ReconstructedFaultSurfaceSystem<dim>::create_restricted_linear_solve(
    const ReconstructedFaultActiveSet &active_set) const
  {
    AssertThrow(surface_linearization != nullptr,
                ExcMessage("K_V must be assembled before restricting its inverse."));
    AssertThrow(active_set.size() == surface_linearization->diagonal.size(),
                ExcMessage("The reconstructed-fault active set has the wrong "
                           "number of faults."));
    for (unsigned int fault = 0; fault < active_set.size(); ++fault)
      AssertThrow(active_set[fault].size()
                  == surface_linearization->diagonal[fault].size(),
                  ExcMessage("The reconstructed-fault active set has the wrong "
                             "number of vertices for fault "
                             + Utilities::int_to_string(fault) + "."));

    return std::make_unique<RestrictedSurfaceLinearSolve<dim>>(
             *this,
             linearization_generation,
             surface_linearization->diagonal,
             surface_linearization->off_diagonal,
             active_set);
  }


  template <int dim>
  void
  ReconstructedFaultSurfaceSystem<dim>::apply_surface_jacobian(
    const FaultVector &direction,
    FaultVector &result) const
  {
    AssertThrow(surface_linearization != nullptr,
                ExcMessage("K_V must be assembled before applying it."));
    AssertThrow(direction.size() == surface_linearization->diagonal.size(),
                ExcMessage("The K_V direction has the wrong number of faults."));

    result.resize(direction.size());
    for (unsigned int fault = 0; fault < direction.size(); ++fault)
      {
        AssertThrow(direction[fault].size()
                    == surface_linearization->diagonal[fault].size(),
                    ExcMessage("The K_V direction has the wrong number of vertices "
                               "for reconstructed fault "
                               + Utilities::int_to_string(fault) + "."));
        result[fault].resize(direction[fault].size());
        for (unsigned int i = 0; i < direction[fault].size(); ++i)
          {
            double value = surface_linearization->diagonal[fault][i]
                           * direction[fault][i];
            if (i > 0)
              value += surface_linearization->off_diagonal[fault][i-1]
                       * direction[fault][i-1];
            if (i+1 < direction[fault].size())
              value += surface_linearization->off_diagonal[fault][i]
                       * direction[fault][i+1];
            result[fault][i] = value;
          }
      }
  }


  template <int dim>
  double
  ReconstructedFaultSurfaceSystem<dim>::surface_residual_rms(
    const ReconstructedFaultSurfaceResidual &residual,
    const ReconstructedFaultActiveSet &active_set) const
  {
    AssertThrow(surface_linearization != nullptr,
                ExcMessage("The surface system must be linearized before "
                           "measuring its residual."));
    AssertDimension(residual.values.size(),
                    surface_linearization->mass_diagonal.size());
    AssertDimension(active_set.size(), residual.values.size());

    // For weak nodal residual r, sqrt(r^T M^-1 r) is the L2 norm of
    // its consistent-Q1 strong representation. Restrict both r and M to
    // the same free set used by the projected Newton solve.
    double squared_norm = 0.0;
    double measure = 0.0;
    for (unsigned int fault = 0; fault < residual.values.size(); ++fault)
      {
        const auto &mass_diagonal = surface_linearization->mass_diagonal[fault];
        const auto &mass_off_diagonal =
          surface_linearization->mass_off_diagonal[fault];
        AssertDimension(residual.values[fault].size(), mass_diagonal.size());
        AssertDimension(active_set[fault].size(), mass_diagonal.size());

        std::vector<double> restricted_diagonal = mass_diagonal;
        std::vector<double> restricted_off_diagonal = mass_off_diagonal;
        std::vector<double> restricted_rhs = residual.values[fault];
        for (unsigned int vertex = 0; vertex < restricted_rhs.size(); ++vertex)
          if (active_set[fault][vertex])
            {
              restricted_diagonal[vertex] = 1.0;
              restricted_rhs[vertex] = 0.0;
              if (vertex > 0)
                restricted_off_diagonal[vertex-1] = 0.0;
              if (vertex < restricted_off_diagonal.size())
                restricted_off_diagonal[vertex] = 0.0;
            }

        const std::vector<double> strong_residual =
          ReconstructedFaultUtilities::solve_tridiagonal_system(
            restricted_diagonal, restricted_off_diagonal, restricted_rhs);
        for (unsigned int vertex = 0; vertex < restricted_rhs.size(); ++vertex)
          if (!active_set[fault][vertex])
            {
              squared_norm += restricted_rhs[vertex]*strong_residual[vertex];
              measure += mass_diagonal[vertex];
            }
        for (unsigned int segment = 0;
             segment < mass_off_diagonal.size(); ++segment)
          if (!active_set[fault][segment] && !active_set[fault][segment+1])
            measure += 2.0*mass_off_diagonal[segment];
      }

    AssertThrow(std::isfinite(squared_norm) && std::isfinite(measure)
                && squared_norm >= 0.0 && measure >= 0.0,
                ExcMessage("The consistent reconstructed-fault surface residual "
                           "norm is not finite and nonnegative."));
    return measure > 0.0 ? std::sqrt(squared_norm/measure) : 0.0;
  }


  template <int dim>
  void
  ReconstructedFaultSurfaceSystem<dim>::apply_G(
    const LinearAlgebra::BlockVector &physical_bulk_direction,
    FaultVector &result) const
  {
    TimerOutput::Scope timer(*performance_timer, "Fault: G total");
    AssertThrow(surface_linearization != nullptr,
                ExcMessage("The surface system must be linearized before applying G."));
    AssertThrow(physical_bulk_direction.size()
                == this->get_dof_handler().n_dofs(),
                ExcMessage("The reconstructed-fault G direction has the wrong size."));
    const auto &faults = this->get_reconstructed_fault_manager().get_faults();
    result.resize(faults.size());
    unsigned int n_fault_dofs = 0;
    for (unsigned int fault = 0; fault < faults.size(); ++fault)
      {
        result[fault].assign(faults[fault].n_vertices(), 0.0);
        n_fault_dofs += faults[fault].n_vertices();
      }

    // The input is a homogeneous bulk perturbation with pressure already
    // converted from solver scaling to physical units by the condensed system.
    const unsigned int velocity_component =
      this->introspection().component_indices.velocities[0];
    const unsigned int pressure_component =
      this->introspection().component_indices.pressure;
    TimerOutput::Scope sample_timer(*performance_timer, "Fault: G FE sampling");
    const auto velocity_gradients = VectorTools::point_gradients<dim>(
                                      *surface_linearization->point_cache, this->get_dof_handler(),
                                      physical_bulk_direction, VectorTools::EvaluationFlags::avg,
                                      velocity_component);
    const std::vector<double> pressures = VectorTools::point_values<1>(
                                            *surface_linearization->point_cache, this->get_dof_handler(),
                                            physical_bulk_direction, VectorTools::EvaluationFlags::avg,
                                            pressure_component);
    AssertDimension(velocity_gradients.size(), pressures.size());
    sample_timer.stop();
    TimerOutput::Scope action_timer(*performance_timer, "Fault: G integrate/reduce");

    // Apply the pointwise G action with its pressure-mode sign convention. The
    // adiabatic branch has neither the mu*N strain term nor the -mu*delta-p term.
    for (unsigned int p = 0;
         p < surface_linearization->coupling_points.size(); ++p)
      {
        const auto &point = surface_linearization->coupling_points[p];
        const SymmetricTensor<2,dim> strain_rate =
          symmetrize(velocity_gradients[point.parent_index]);
        const SymmetricTensor<2,dim> stress_direction =
          point.uses_adiabatic_friction_pressure
          ? 2.0 * point.kappa * point.slip_tensor
          : 2.0 * point.kappa
          * (point.slip_tensor
             + point.friction_coefficient * point.normal_tensor);
        double value = stress_direction * strain_rate;
        if (!point.uses_adiabatic_friction_pressure)
          value -= point.friction_coefficient * pressures[point.parent_index];

        const double shape[2] = {1.0-point.xi, point.xi};
        result[point.fault_index][point.segment_index] +=
          point.particle_domain_volume * shape[0] * value;
        result[point.fault_index][point.segment_index+1] +=
          point.particle_domain_volume * shape[1] * value;
      }

    // Contributions originate on locally owned particles; sum them so the
    // returned fault vector is replicated identically on all ranks.
    std::vector<double> local_values(n_fault_dofs);
    unsigned int position = 0;
    for (const auto &fault_values : result)
      for (const double value : fault_values)
        local_values[position++] = value;
    std::vector<double> global_values(n_fault_dofs);
    Utilities::MPI::sum(local_values, this->get_mpi_communicator(), global_values);
    position = 0;
    for (auto &fault_values : result)
      for (double &value : fault_values)
        value = global_values[position++];
  }


  template <int dim>
  const ReconstructedFaultSurfaceResidual &
  ReconstructedFaultSurfaceSystem<dim>::get_linearization_residual() const
  {
    AssertThrow(surface_linearization != nullptr,
                ExcMessage("Surface weak diagnostics require a completed linearization."));
    return surface_linearization->residual;
  }


  template <int dim>
  unsigned int
  ReconstructedFaultSurfaceSystem<dim>::get_linearization_generation() const
  {
    return linearization_generation;
  }


#define INSTANTIATE(dim) template class ReconstructedFaultSurfaceSystem<dim>;
  ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
}
