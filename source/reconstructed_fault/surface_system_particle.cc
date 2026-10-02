/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include "surface_system_internal.h"

#include <aspect/material_model/phase_field_fault.h>
#include <aspect/material_model/utilities.h>
#include <aspect/particle/manager.h>
#include <aspect/phase_field.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/reconstructed_fault/utilities.h>

#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/numerics/vector_tools_evaluate.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>

namespace aspect
{
  template <int dim>
  typename ReconstructedFaultSurfaceSystem<dim>::SurfaceAssembly
  ReconstructedFaultSurfaceSystem<dim>::assemble_particle_system(
    const LinearAlgebra::BlockVector &bulk_state,
    const FaultVector &slip_rate,
    const bool assemble_jacobian) const
  {
    AssertThrow(normal_filter_mode=="raw",ExcMessage("Normal filtering requires the bulk work surface measure."));
    AssertThrow(!phase_field_fault.benchmark_retained_stress,
                ExcMessage("Native-history benchmark requires the bulk work surface rule."));
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
    // Disposable history-representation audit. Sample the complete frozen FE
    // tensor at the same parent points; stress components are not Newton unknowns.
    const bool history_audit=std::getenv("ASPECT_FAULT_HISTORY_AUDIT");
    std::vector<std::vector<double>> fe_history(SymmetricTensor<2,dim>::n_independent_components);
    if (history_audit)
      for (const auto &m:this->get_parameters().mapped_particle_properties)
        if (m.second.first=="maxwell stress")
          fe_history[m.second.second]=VectorTools::point_values<1>(point_cache,this->get_dof_handler(),bulk_state,
            VectorTools::EvaluationFlags::avg,this->introspection().component_indices.compositional_fields[m.first]);
    std::vector<std::vector<std::array<double,22>>> history_moments(faults.size());
    if (history_audit && assemble_jacobian)
      for (unsigned int f=0;f<faults.size();++f) history_moments[f].resize(faults[f].n_vertices(),{});
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
    local.residual.normal_traction.resize(faults.size());
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
    const bool record_samples = assemble_jacobian
      && std::getenv("ASPECT_FAULT_STRESS_SAMPLE_DIAGNOSTIC");
    // Keep only 20 low/high samples per support class per rank. All samples
    // still contribute to weak moments, including tensile ones on mixed rows.
    std::array<std::multimap<double,std::string>,3> lowest_samples, highest_samples;
    std::vector<std::vector<std::array<double,10>>> stress_audit_moments;
    std::vector<std::vector<bool>> prescribed;
    if (record_samples)
      {
        prescribed = fault_manager.prescribed_slip_rate_mask();
        stress_audit_moments.resize(faults.size());
        for (unsigned int f=0; f<faults.size(); ++f)
          stress_audit_moments[f].resize(faults[f].n_vertices(), {});
      }
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
        local.residual.normal_traction[fault].assign(faults[fault].n_vertices(), 0.0);
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
        bool parent_touches_free=false, parent_touches_prescribed=false;
        if (record_samples)
          for (const auto &q : association.quadrature)
            for (unsigned int i=0; i<2; ++i)
              if ((i==0 ? 1-q.xi : q.xi)>0)
                {
                  if (prescribed[association.fault_index][q.segment_index+i])
                    parent_touches_prescribed=true;
                  else parent_touches_free=true;
                }
        unsigned int domain_q_index=0;
        for (const auto &q : association.quadrature)
          {
            Tensor<1,dim> tangent=fault.vertex(q.segment_index+1)-fault.vertex(q.segment_index);
            tangent/=tangent.norm();
            Tensor<1,dim> normal;
            normal[0]=-tangent[1];
            normal[1]=tangent[0];
            inputs.slip_tensor=fault_manager.get_shear_sense(association.fault_index)
                               *symmetrize(outer_product(tangent,normal));
            inputs.normal_tensor=symmetrize(outer_product(normal,normal));
            inputs.segment_index = q.segment_index;
            inputs.xi = q.xi;
            const double left_slip_rate = slip_rate[association.fault_index][q.segment_index];
            const double right_slip_rate = slip_rate[association.fault_index][q.segment_index+1];
            // Tip partitions evaluate exact endpoint basis functions. Reading
            // the absolute node avoids losing lower contact by cancellation.
            inputs.slip_rate = ReconstructedFaultUtilities::interpolate_slip_rate(
              left_slip_rate, right_slip_rate, q.xi);
            const auto particle_response =
              phase_field_fault.evaluate_reconstructed_fault_point(inputs);
            const auto &response=particle_response;
            if (history_audit)
              {
                auto fe_inputs=inputs;
                for (unsigned int c=0;c<fe_history.size();++c)
                  fe_inputs.old_maxwell_stress[SymmetricTensor<2,dim>::unrolled_to_component_indices(c)]
                    =fe_history[c][point_index];
                const auto fe_response=phase_field_fault.evaluate_reconstructed_fault_point(fe_inputs);
                if (assemble_jacobian)
                  for (unsigned int i=0;i<2;++i)
                    {
                      const double Ni=i==0 ? 1-q.xi : q.xi;
                      auto &m=history_moments[association.fault_index][q.segment_index+i];
                      m[0]+=q.weight*Ni;
                      m[1]+=q.weight*Ni*inputs.dynamic_pressure;
                      for (unsigned int mode=0;mode<2;++mode)
                        {
                          const auto &r=mode==0 ? particle_response : fe_response;
                          const unsigned int k=2+10*mode;
                          const double values[]={r.shear_traction,r.normal_traction,r.cohesive_traction,
                            r.friction_traction,r.damping_traction,r.residual_density};
                          for (unsigned int j=0;j<6;++j) m[k+j]+=q.weight*Ni*values[j];
                          m[k+6]+=q.weight*Ni*Ni*r.minus_derivative_wrt_slip_rate;
                          m[k+7]+=q.weight*Ni*(1-Ni)*r.minus_derivative_wrt_slip_rate;
                          if (r.normal_traction<0.)
                            {m[k+8]+=q.weight*Ni; m[k+9]+=q.weight*Ni*r.friction_traction;}
                        }
                    }
              }
            AssertThrow(std::isfinite(response.residual_density)
                        && std::isfinite(response.minus_derivative_wrt_slip_rate),
                        ExcMessage("Reconstructed-fault constitutive evaluation produced "
                                   "a non-finite surface coefficient at particle "
                                   + Utilities::int_to_string(particle.get_id()) + "."));
            const double shape[2] = {1.0-q.xi, q.xi};
            const unsigned int vertex = q.segment_index;
            const double weight = q.weight;
            local.residual.minimum_normal_traction =
              std::min(local.residual.minimum_normal_traction, response.normal_traction);
            local.residual.maximum_normal_traction =
              std::max(local.residual.maximum_normal_traction, response.normal_traction);
            if (record_samples)
              {
                const auto f=association.fault_index;
                const double sigma=response.normal_traction;
                const double tau_n=inputs.dynamic_pressure
                                   +response.background_normal_traction-sigma;
                double free_shape=0.;
                for (unsigned int i=0; i<2; ++i)
                  {
                    if (!prescribed[f][vertex+i]) free_shape+=shape[i];
                    auto &m=stress_audit_moments[f][vertex+i];
                    const double w=weight*shape[i];
                    m[0]+=w;
                    m[1]+=w*inputs.dynamic_pressure;
                    m[2]+=w*tau_n;
                    m[3]+=w*response.background_normal_traction;
                    m[4]+=w*sigma;
                    m[5]+=w*response.friction_traction;
                    m[6]+=w*std::abs(response.friction_traction);
                    if (sigma<0.)
                      {
                        m[7]+=w;
                        m[8]+=w*response.friction_traction;
                        m[9]+=w*std::abs(response.friction_traction);
                      }
                  }
                const unsigned int support=free_shape==0. ? 1 : free_shape==1. ? 0 : 2;
                auto &low=lowest_samples[support];
                auto &high=highest_samples[support];
                if (low.size()<20 || high.size()<20
                    || sigma<low.rbegin()->first || sigma>high.begin()->first)
                  {
                    const Point<dim> surface=(1-q.xi)*fault.vertex(vertex)
                                             +q.xi*fault.vertex(vertex+1);
                    std::ostringstream row;
                    row<<std::setprecision(17)<<association.fault_index<<','<<particle.get_id()<<','
                       <<particle.get_surrounding_cell()->id().to_string()<<','<<domain_q_index<<','
                       <<vertex<<','<<q.xi<<','<<inputs.position[0]<<','<<inputs.position[1]<<','
                       <<surface[0]<<','<<surface[1]<<','<<(inputs.position-surface)*normal<<','
                       <<weight<<','<<association.particle_domain_volume<<','<<free_shape<<','
                       <<parent_touches_free<<','<<parent_touches_prescribed<<','<<inputs.slip_rate<<','
                       <<inputs.dynamic_pressure<<','<<tau_n<<','<<response.background_normal_traction<<','
                       <<sigma<<','<<response.shear_traction<<','<<response.friction_coefficient<<','
                       <<response.friction_traction<<','<<inputs.phase_field<<','<<response.localization_factor;
                    low.emplace(sigma,row.str());
                    high.emplace(sigma,row.str());
                    if (low.size()>20) low.erase(std::prev(low.end()));
                    if (high.size()>20) high.erase(high.begin());
                  }
              }
            ++domain_q_index;
            if (record_normal_stress)
              {
                // The response already contains mu*(p-tau:N). Recover its
                // sigma_n without reevaluating Maxwell from committed history.
                AssertThrow(response.friction_coefficient > 0.0,
                            ExcMessage("Normal-stress diagnostic requires positive friction coefficient."));
                const double sigma = response.normal_traction;
                const double values[3] = {inputs.dynamic_pressure, sigma,
                                         inputs.dynamic_pressure
                                         -(sigma-response.background_normal_traction)};
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
                local.residual.normal_traction[association.fault_index][vertex+i]
                  += weight*shape[i]*response.normal_traction;
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
                  response.eta_ve,
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
      packed_size += 9*fault.n_vertices() + 2*fault.n_cells();
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
                                &local.residual.damping_traction,
                                &local.residual.normal_traction})
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
                          &local.residual.damping_traction,
                          &local.residual.normal_traction})
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
    local.residual.minimum_normal_traction = Utilities::MPI::min(
      local.residual.minimum_normal_traction, this->get_mpi_communicator());
    local.residual.maximum_normal_traction = Utilities::MPI::max(
      local.residual.maximum_normal_traction, this->get_mpi_communicator());
    if (record_samples)
      {
        // Rank-local files are replaced on each linearization, never on an
        // unrelated trial evaluation. Use only after final convergence and
        // compare the saved nodal V against the accepted state's rate vector.
        const auto rank=Utilities::MPI::this_mpi_process(this->get_mpi_communicator());
        const auto tag=std::to_string(this->get_timestep_number())+"_rank"+std::to_string(rank)+".csv";
        std::ofstream samples(this->get_output_directory()+"stress_samples_"+tag);
        samples.exceptions(std::ios::failbit | std::ios::badbit);
        samples<<"step,time,rank,selection,support,fault,particle,cell,domain_q,segment,xi,parent_x,parent_y,surface_x,surface_y,signed_normal_distance,weight,domain_volume,free_shape,parent_touches_free,parent_touches_prescribed,V,delta_p,delta_tau_N,sigma_bg,sigma_n,q,mu,mu_sigma,phi,chi\n";
        for (unsigned int c=0; c<3; ++c)
          for (unsigned int side=0; side<2; ++side)
            for (const auto &entry : (side==0 ? lowest_samples[c] : highest_samples[c]))
              samples<<std::setprecision(17)<<this->get_timestep_number()<<','<<this->get_time()<<','
                     <<rank<<','<<(side==0 ? "minimum" : "maximum")<<','<<c<<','<<entry.second<<'\n';
        std::ofstream moments(this->get_output_directory()+"stress_weak_moments_"+tag);
        moments.exceptions(std::ios::failbit | std::ios::badbit);
        moments<<"step,time,rank,fault,node,x,y,prescribed,V,weight,p_load,tauN_load,bg_load,sigma_load,friction_load,abs_friction_load,tensile_weight,tensile_friction_load,tensile_abs_friction_load\n";
        for (unsigned int f=0; f<faults.size(); ++f)
          for (unsigned int i=0; i<faults[f].n_vertices(); ++i)
            {
              moments<<std::setprecision(17)<<this->get_timestep_number()<<','<<this->get_time()<<','<<rank<<','
                     <<f<<','<<i<<','<<faults[f].vertex(i)[0]<<','<<faults[f].vertex(i)[1]<<','
                     <<prescribed[f][i]<<','<<slip_rate[f][i];
              for (double value:stress_audit_moments[f][i]) moments<<','<<value;
              moments<<'\n';
            }
      }
    if (history_audit && assemble_jacobian)
      {
        std::ofstream out(this->get_output_directory()+"history_surface_rank"+
                          std::to_string(Utilities::MPI::this_mpi_process(this->get_mpi_communicator()))+".csv");
        out.exceptions(std::ios::failbit|std::ios::badbit);
        out<<"fault,node,x,y,V,weight,p,particle_q,particle_sigma,particle_C,particle_friction,particle_damping,particle_R,particle_Kdiag,particle_Koff_sum,particle_tensile_weight,particle_tensile_friction,fe_q,fe_sigma,fe_C,fe_friction,fe_damping,fe_R,fe_Kdiag,fe_Koff_sum,fe_tensile_weight,fe_tensile_friction\n";
        for (unsigned int f=0;f<faults.size();++f)
          for (unsigned int v=0;v<faults[f].n_vertices();++v)
            {
              out<<std::setprecision(17)<<f<<','<<v<<','<<faults[f].vertex(v)[0]<<','<<faults[f].vertex(v)[1]<<','<<slip_rate[f][v];
              for (double x:history_moments[f][v]) out<<','<<x;
              out<<'\n';
            }
      }
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


#define INSTANTIATE(dim) \
  template ReconstructedFaultSurfaceSystem<dim>::SurfaceAssembly \
  ReconstructedFaultSurfaceSystem<dim>::assemble_particle_system(const LinearAlgebra::BlockVector &, const FaultVector &, const bool) const;

  ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
}
