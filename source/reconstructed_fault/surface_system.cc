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
#include <aspect/reconstructed_fault/linear_performance.h>
#include <aspect/reconstructed_fault/sparse_coupling.h>
#include "surface_direct_internal.h"

#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/fe/fe_values.h>
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
#include <map>
#include <sstream>
#include <set>

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
          const ReconstructedFaultVector &diagonal,
          const ReconstructedFaultVector &off_diagonal,
          const ReconstructedFaultVector &lower_diagonal,
          const ReconstructedFaultActiveSet &active_set)
          : owner(owner), generation(generation)
        {
          for (unsigned int fault=0; fault<diagonal.size(); ++fault)
            {
              internal::FaultLinearSection timing(internal::FaultLinearTiming::factor);
              factorizations.push_back(std::make_unique<internal::FaultSurfaceDirect>(
                diagonal[fault],off_diagonal[fault],active_set[fault],fault,
                internal::FaultSurfaceDirect::use_pivoting(),
                lower_diagonal.empty() ? std::vector<double>() : lower_diagonal[fault]));
            }
        }

        void solve(const ReconstructedFaultVector &rhs,
                   ReconstructedFaultVector &solution) const override
        {
          AssertThrow(generation == owner.get_linearization_generation(),
                      ExcMessage("This restricted reconstructed-fault surface solve has been superseded."));
          AssertThrow(rhs.size() == factorizations.size(),
                      ExcMessage("The restricted K_V RHS has the wrong number of faults."));
          solution.resize(rhs.size());
          for (unsigned int fault=0; fault<rhs.size(); ++fault)
            {
              internal::FaultLinearSection timing(internal::FaultLinearTiming::inverse);
              factorizations[fault]->solve(rhs[fault],solution[fault]);
            }
        }

      private:
        const ReconstructedFaultSurfaceSystem<dim> &owner;
        const unsigned int generation;
        std::vector<std::unique_ptr<internal::FaultSurfaceDirect>> factorizations;
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
    std::vector<std::vector<double>> lower_diagonal;
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
    if (bulk_work_measure)
      return assemble_bulk_work_system(bulk_state, slip_rate, assemble_jacobian);
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
    const bool use_fe_history=std::getenv("ASPECT_FAULT_HISTORY_FE");
    AssertThrow(!use_fe_history || (history_audit && std::getenv("ASPECT_FAULT_NONCOMMITTING_DIAGNOSTIC")),
                ExcMessage("Alternative surface history is restricted to noncommitting diagnostics."));
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
    // Complete, opt-in domain samples for a bounded frozen-state audit. These
    // are separate files; ordinary graphical and extrema outputs are unchanged.
    std::ofstream theta_samples;
    std::set<unsigned int> theta_segments;
    if (const char *prefix=assemble_jacobian ? std::getenv("ASPECT_FAULT_THETA_QP_EXPORT") : nullptr)
      {
        AssertThrow(std::getenv("ASPECT_FAULT_NONCOMMITTING_DIAGNOSTIC"),ExcMessage("Theta quadrature export requires a disposable solve."));
        const char *selected=std::getenv("ASPECT_FAULT_THETA_AUDIT_SEGMENTS");
        AssertThrow(selected,ExcMessage("Missing theta audit segments."));
        std::istringstream in(selected); unsigned int segment;
        while (in>>segment) theta_segments.insert(segment);
        theta_samples.open(std::string(prefix)+"_rank"+std::to_string(Utilities::MPI::this_mpi_process(this->get_mpi_communicator()))+".csv");
        theta_samples.exceptions(std::ios::failbit|std::ios::badbit);
        theta_samples<<std::setprecision(17)<<"step,time,particle,fault,segment,xi,weight,parent_x,parent_y,V,p,tau_N,sigma_n,q,C,mu,friction,damping,R,minus_dR_dV,phi,chi\n";
      }
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
            inputs.slip_tensor=symmetrize(outer_product(tangent,normal));
            inputs.normal_tensor=symmetrize(outer_product(normal,normal));
            inputs.segment_index = q.segment_index;
            inputs.xi = q.xi;
            const double left_slip_rate = slip_rate[association.fault_index][q.segment_index];
            const double right_slip_rate = slip_rate[association.fault_index][q.segment_index+1];
            // Tip partitions evaluate exact endpoint basis functions. Reading
            // the absolute node avoids losing lower contact by cancellation.
            inputs.slip_rate = q.xi == 0.0 ? left_slip_rate
                              : q.xi == 1.0 ? right_slip_rate
                              : left_slip_rate + q.xi*(right_slip_rate-left_slip_rate);
            const auto particle_response =
              phase_field_fault.evaluate_reconstructed_fault_point(inputs);
            auto response=particle_response;
            if (theta_samples.is_open() && theta_segments.count(q.segment_index))
              theta_samples<<this->get_timestep_number()<<','<<this->get_time()<<','<<particle.get_id()<<','
                <<association.fault_index<<','<<q.segment_index<<','<<q.xi<<','<<q.weight<<','
                <<association.position[0]<<','<<association.position[1]<<','<<inputs.slip_rate<<','
                <<inputs.dynamic_pressure<<','<<inputs.dynamic_pressure+response.background_normal_traction-response.normal_traction<<','
                <<response.normal_traction<<','<<response.shear_traction<<','<<response.cohesive_traction<<','
                <<response.friction_coefficient<<','<<response.friction_traction<<','<<response.damping_traction<<','
                <<response.residual_density<<','<<response.minus_derivative_wrt_slip_rate<<','
                <<inputs.phase_field<<','<<response.localization_factor<<'\n';
            if (history_audit)
              {
                auto fe_inputs=inputs;
                for (unsigned int c=0;c<fe_history.size();++c)
                  fe_inputs.old_maxwell_stress[SymmetricTensor<2,dim>::unrolled_to_component_indices(c)]
                    =fe_history[c][point_index];
                const auto fe_response=phase_field_fault.evaluate_reconstructed_fault_point(fe_inputs);
                if (use_fe_history) response=fe_response;
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

    using FaultFactorization = internal::FaultSurfaceDirect;

    ReconstructedFaultSurfaceResidual residual;
    std::vector<std::vector<double>> diagonal;
    std::vector<std::vector<double>> off_diagonal;
    std::vector<std::vector<double>> lower_diagonal;
    std::vector<std::vector<double>> mass_diagonal;
    std::vector<std::vector<double>> mass_off_diagonal;
    std::vector<std::unique_ptr<FaultFactorization>> factorizations;
    std::vector<CouplingPoint> coupling_points;
    std::unique_ptr<Utilities::MPI::RemotePointEvaluation<dim>> point_cache;
    std::unique_ptr<internal::FaultSparseCoupling> matrix;
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
  void ReconstructedFaultSurfaceSystem<dim>::enable_bulk_work_measure()
  {
    AssertThrow(dim==2 && phase_field_fault.is_mature_frictional_fault()
                && !phase_field_fault.uses_adiabatic_friction_pressure(),
                ExcMessage("Bulk work measure requires a frozen mature 2-D fault with true normal stress."));
    const auto &faults=this->get_reconstructed_fault_manager().get_faults();
    AssertThrow(faults.size()==1,ExcMessage("Bulk work measure currently supports one straight fault."));
    const auto &fault=faults[0];
    auto tangent=fault.vertex(1)-fault.vertex(0);tangent/=tangent.norm();
    const double length=fault.vertex(fault.n_vertices()-1).distance(fault.vertex(0));
    for (unsigned int i=0;i<fault.n_vertices();++i)
      {
        const auto offset=fault.vertex(i)-fault.vertex(0);
        AssertThrow((offset-(offset*tangent)*tangent).norm()<1e-10*length,
                    ExcMessage("Bulk work measure requires a straight fault."));
      }
    if (!bulk_work_measure) surface_linearization.reset();
    bulk_work_measure=true;
  }


  template <int dim>
  typename ReconstructedFaultSurfaceSystem<dim>::SurfaceAssembly
  ReconstructedFaultSurfaceSystem<dim>::assemble_bulk_work_system(
    const LinearAlgebra::BlockVector &bulk_state, const FaultVector &slip_rate,
    const bool assemble_jacobian) const
  {
    TimerOutput::Scope timer(*performance_timer,"Fault: bulk work R/K");
    phase_field_fault.validate_reconstructed_fault_constitutive_state();
    auto &manager=this->get_reconstructed_fault_manager();
    const auto &faults=manager.get_faults();
    AssertDimension(slip_rate.size(),faults.size());
    SurfaceAssembly local;
    const bool candidate_state=std::getenv("ASPECT_FAULT_WITHIN_STEP_STATE");
    if (candidate_state)
      {
        local.lower_diagonal.resize(faults.size());
        for (unsigned int f=0;f<faults.size();++f) local.lower_diagonal[f].assign(faults[f].n_cells(),0.);
      }
    std::ofstream state_export;
    if (assemble_jacobian && (std::getenv("ASPECT_BP3_WITHIN_STEP_DIAGNOSTIC")
                             || std::getenv("ASPECT_BP3_COUPLED_STATE_REPLAY")))
      {
        state_export.open(this->get_output_directory()+"state_qp_rank"+
          std::to_string(Utilities::MPI::this_mpi_process(this->get_mpi_communicator()))+".csv");
        state_export.exceptions(std::ios::failbit|std::ios::badbit);
        state_export<<std::setprecision(17)<<"cell,qp,segment,xi,weight,V,Theta,q,friction,damping,sigma,R,p,Kfixed,Kstate0,Kstate1\n";
      }
    for (auto *v:{&local.residual.values,&local.residual.shear_traction,
                 &local.residual.cohesive_traction,&local.residual.friction_traction,
                 &local.residual.damping_traction,&local.residual.normal_traction,
                 &local.diagonal,&local.mass_diagonal})
      {
        v->resize(faults.size());
        for (unsigned int f=0;f<faults.size();++f) (*v)[f].assign(faults[f].n_vertices(),0.);
      }
    for (auto *v:{&local.off_diagonal,&local.mass_off_diagonal})
      {
        v->resize(faults.size());
        for (unsigned int f=0;f<faults.size();++f) (*v)[f].assign(faults[f].n_cells(),0.);
      }
    std::vector<double> squared(faults.size()),measure(faults.size());
    const auto &intro=this->introspection();
    const auto &quadrature=intro.quadratures.velocities;
    FEValues<dim> fe(this->get_mapping(),this->get_fe(),quadrature,
                     update_values|update_gradients|update_quadrature_points|update_JxW_values);
    const unsigned int nq=quadrature.size();
    std::vector<double> phase(nq),temperature(nq),pressure(nq);
    std::vector<SymmetricTensor<2,dim>> strain(nq);
    std::vector<std::vector<double>> composition(intro.n_compositional_fields,std::vector<double>(nq));
    std::array<unsigned int,SymmetricTensor<2,dim>::n_independent_components> stress_fields;
    stress_fields.fill(numbers::invalid_unsigned_int);
    for (const auto &entry:this->get_parameters().mapped_particle_properties)
      if (entry.second.first=="maxwell stress") stress_fields[entry.second.second]=entry.first;
    manager.prepare_stokes_qp_projection_cache();

    // Each owned physical Stokes QP is visited once. The cached source map
    // already includes both enabled wedges; there is no overlapping tip pass.
    for (const auto &cell:this->get_dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe.reinit(cell);
          const auto &associations=manager.get_stokes_qp_fault_associations(
            cell->id(),quadrature,fe.get_quadrature_points());
          fe[FEValuesExtractors::Scalar(intro.variable("phase_field").first_component_index)].get_function_values(bulk_state,phase);
          fe[intro.extractors.temperature].get_function_values(bulk_state,temperature);
          fe[intro.extractors.pressure].get_function_values(bulk_state,pressure);
          fe[intro.extractors.velocities].get_function_symmetric_gradients(bulk_state,strain);
          for (unsigned int c=0;c<composition.size();++c)
            fe[intro.extractors.compositional_fields[c]].get_function_values(bulk_state,composition[c]);
          for (unsigned int q=0;q<nq;++q)
            if (associations[q].active)
              {
                const auto &a=associations[q];
                AssertDimension(slip_rate[a.fault_index].size(),faults[a.fault_index].n_vertices());
                typename MaterialModel::PhaseFieldFault<dim>::ReconstructedFaultPointInputs input;
                input.fault_index=a.fault_index;input.segment_index=a.segment_index;input.xi=a.xi;
                input.position=fe.quadrature_point(q);
                input.phase_field=phase[q];input.previous_phase_field=phase[q];
                input.temperature=temperature[q];input.dynamic_pressure=pressure[q];input.strain_rate=strain[q];
                input.slip_tensor=symmetrize(outer_product(a.tangent,a.normal));
                input.normal_tensor=symmetrize(outer_product(a.normal,a.normal));
                for (unsigned int c=0;c<stress_fields.size();++c)
                  {
                    AssertIndexRange(stress_fields[c],composition.size());
                    input.old_maxwell_stress[SymmetricTensor<2,dim>::unrolled_to_component_indices(c)]
                      =composition[stress_fields[c]][q];
                  }
                std::vector<double> chemical;
                for (const auto c:intro.chemical_composition_field_indices()) chemical.push_back(composition[c][q]);
                input.bulk_material_fractions=MaterialModel::MaterialUtilities::compute_composition_fractions(chemical);
                const auto &V=slip_rate[a.fault_index];
                input.diagnostic_nodal_rates={{V[a.segment_index],V[a.segment_index+1]}};
                input.slip_rate=a.xi==0. ? V[a.segment_index] : a.xi==1. ? V[a.segment_index+1]
                  : (1-a.xi)*V[a.segment_index]+a.xi*V[a.segment_index+1];
                const auto response=phase_field_fault.evaluate_reconstructed_fault_point(input);
                const double weight=fe.JxW(q)*response.localization_factor;
                if (weight==0.) continue;
                AssertThrow(std::isfinite(weight) && weight>0.
                            && std::isfinite(response.residual_density)
                            && std::isfinite(response.minus_derivative_wrt_slip_rate),
                            ExcMessage("Non-finite bulk work surface response."));

                // One work measure multiplies every traction, K and the mass
                // matrix. Consequently M^{-1}R and the RMS norm remain in Pa.
                const unsigned int f=a.fault_index,j=a.segment_index;
                const double N[2]={1-a.xi,a.xi};
                if (state_export.is_open() && (j==795 || j==796))
                  state_export<<cell->id()<<','<<q<<','<<j<<','<<a.xi<<','<<weight<<','<<input.slip_rate
                    <<','<<response.evaluated_state<<','<<response.shear_traction<<','<<response.friction_traction
                    <<','<<response.damping_traction<<','<<response.normal_traction<<','<<response.residual_density
                    <<','<<pressure[q]<<','<<response.minus_derivative_wrt_slip_rate<<','
                    <<response.diagnostic_state_tangent[0]<<','<<response.diagnostic_state_tangent[1]<<'\n';
                for (unsigned int i=0;i<2;++i)
                  {
                    local.residual.values[f][j+i]+=weight*N[i]*response.residual_density;
                    local.residual.shear_traction[f][j+i]+=weight*N[i]*response.shear_traction;
                    local.residual.friction_traction[f][j+i]+=weight*N[i]*response.friction_traction;
                    local.residual.damping_traction[f][j+i]+=weight*N[i]*response.damping_traction;
                    local.residual.normal_traction[f][j+i]+=weight*N[i]*response.normal_traction;
                    local.mass_diagonal[f][j+i]+=weight*N[i]*N[i];
                    if (assemble_jacobian)
                      local.diagonal[f][j+i]+=weight*N[i]*N[i]*(response.minus_derivative_wrt_slip_rate
                        +response.diagnostic_state_tangent[i]);
                  }
                local.mass_off_diagonal[f][j]+=weight*N[0]*N[1];
                squared[f]+=weight*response.residual_density*response.residual_density;
                measure[f]+=weight;
                local.residual.minimum_normal_traction=std::min(local.residual.minimum_normal_traction,response.normal_traction);
                local.residual.maximum_normal_traction=std::max(local.residual.maximum_normal_traction,response.normal_traction);
                if (assemble_jacobian)
                  {
                    local.off_diagonal[f][j]+=weight*N[0]*N[1]*(response.minus_derivative_wrt_slip_rate
                      +response.diagnostic_state_tangent[1]);
                    if (candidate_state)
                      local.lower_diagonal[f][j]+=weight*N[0]*N[1]*(response.minus_derivative_wrt_slip_rate
                        +response.diagnostic_state_tangent[0]);
                    const unsigned int point_index=local.coupling_points.size();
                    local.coupling_points.push_back({input.position,point_index,f,j,a.xi,weight,response.kappa,
                      response.friction_coefficient,input.slip_tensor,input.normal_tensor,false});
                  }
              }
        }

    // QPs have unique cell ownership. Only fault-sized arrays are reduced;
    // G keeps the same QP coordinates and coefficients on their original ranks.
    const auto comm=this->get_mpi_communicator();
    for (auto *v:{&local.residual.values,&local.residual.shear_traction,
                 &local.residual.cohesive_traction,&local.residual.friction_traction,
                 &local.residual.damping_traction,&local.residual.normal_traction,
                 &local.diagonal,&local.off_diagonal,&local.lower_diagonal,&local.mass_diagonal,&local.mass_off_diagonal})
      for (auto &fault:*v)
        { const auto copy=fault; Utilities::MPI::sum(copy,comm,fault); }
    const auto local_squared=squared,local_measure=measure;
    Utilities::MPI::sum(local_squared,comm,squared);Utilities::MPI::sum(local_measure,comm,measure);
    double total_squared=0.,total_measure=0.;
    local.residual.per_fault_weighted_rms.resize(faults.size());
    for (unsigned int f=0;f<faults.size();++f)
      {
        AssertThrow(measure[f]>0.,ExcMessage("Fault has no positive work measure."));
        local.residual.per_fault_weighted_rms[f]=std::sqrt(squared[f]/measure[f]);
        total_squared+=squared[f];total_measure+=measure[f];
      }
    local.residual.weighted_rms=std::sqrt(total_squared/total_measure);
    local.residual.mass_diagonal=local.mass_diagonal;local.residual.mass_off_diagonal=local.mass_off_diagonal;
    local.residual.minimum_normal_traction=Utilities::MPI::min(local.residual.minimum_normal_traction,comm);
    local.residual.maximum_normal_traction=Utilities::MPI::max(local.residual.maximum_normal_traction,comm);
    return local;
  }


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
    candidate->lower_diagonal = assembled.lower_diagonal;
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
        internal::FaultLinearSection timing(internal::FaultLinearTiming::factor);
        candidate->factorizations[fault] =
          std::make_unique<typename SurfaceLinearization::FaultFactorization>(
            assembled.diagonal[fault], assembled.off_diagonal[fault],
            std::vector<bool>(assembled.diagonal[fault].size(),false), fault,
            internal::FaultSurfaceDirect::use_pivoting(),
            assembled.lower_diagonal.empty() ? std::vector<double>() : assembled.lower_diagonal[fault]);
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
    lookup_timer.stop();
    if (std::getenv("ASPECT_FAULT_EXPLICIT_G"))
      {
        internal::FaultLinearSection timing(internal::FaultLinearTiming::G_setup);
        using Coefficient=std::pair<SymmetricTensor<2,dim>,double>;
        using Row=std::pair<unsigned int,Coefficient>;
        using Rows=std::vector<Row>;
        std::vector<std::map<unsigned int,Coefficient>> parent_coefficients(points.size());
        std::vector<unsigned int> offsets(1,0);
        for (const auto &fault : assembled.diagonal)
          offsets.push_back(offsets.back()+fault.size());
        const auto &point_ptrs=candidate->point_cache->get_point_ptrs();

        // Integrate domain test functions first, holding each parent's bulk
        // sample P0 exactly as in G. Shared-face samples are averaged, not
        // assigned to an arbitrary incident cell: divide before RPE duplicates.
        for (const auto &point : candidate->coupling_points)
          {
            const auto multiplicity=point_ptrs[point.parent_index+1]-point_ptrs[point.parent_index];
            // VectorTools' avg evaluation returns zero for a missing request.
            // Preserve that existing action convention instead of dividing by zero.
            if (multiplicity==0) continue;
            const auto stress=2.*point.kappa*(point.slip_tensor
              +(point.uses_adiabatic_friction_pressure ? 0. : point.friction_coefficient)*point.normal_tensor);
            for (unsigned int end=0; end<2; ++end)
              {
                const double weight=point.particle_domain_volume*(end==0 ? 1.-point.xi : point.xi)/multiplicity;
                auto &c=parent_coefficients[point.parent_index][offsets[point.fault_index]+point.segment_index+end];
                c.first+=weight*stress;
                if (!point.uses_adiabatic_friction_pressure)
                  c.second-=weight*point.friction_coefficient;
              }
          }
        std::vector<Rows> input(points.size());
        for (unsigned int p=0; p<points.size(); ++p)
          for (const auto &row : parent_coefficients[p]) input[p].push_back(row);
        internal::FaultSparseCoupling::Entries entries;
        using Cache=Utilities::MPI::RemotePointEvaluation<dim>;
        candidate->point_cache->template process_and_evaluate<Rows>(input,
          [&](const ArrayView<const Rows> &values, const typename Cache::CellData &cells)
          {
            // RPE routes each coefficient to the original FE-cell owner.
            // Store physical, unconstrained columns; the solver continues to
            // distribute homogeneous constraints and pressure scaling before G.
            const auto &fe=this->get_fe();
            std::vector<types::global_dof_index> indices(fe.dofs_per_cell);
            for (const auto c : cells.cell_indices())
              {
                const auto tria_cell=cells.get_active_cell_iterator(c);
                const typename DoFHandler<dim>::active_cell_iterator cell(
                  &this->get_triangulation(),tria_cell->level(),tria_cell->index(),&this->get_dof_handler());
                const auto unit_points=cells.get_unit_points(c);
                FEValues<dim> fe_values(this->get_mapping(),fe,
                  Quadrature<dim>(std::vector<Point<dim>>(unit_points.begin(),unit_points.end())),
                  update_values | update_gradients);
                fe_values.reinit(cell);
                cell->get_dof_indices(indices);
                const auto data=cells.get_data_view(c,values);
                for (unsigned int p=0; p<data.size(); ++p)
                  for (unsigned int i=0; i<indices.size(); ++i)
                    {
                      const auto component=fe.system_to_component_index(i).first;
                      const bool velocity=this->introspection().component_masks.velocities[component];
                      const bool pressure=component==this->introspection().component_indices.pressure;
                      if (!velocity && !pressure) continue;
                      for (const auto &row : data[p])
                        entries[{row.first,indices[i]}]+=velocity
                          ? row.second.first*fe_values[this->introspection().extractors.velocities].symmetric_gradient(i,p)
                          : row.second.second*fe_values[this->introspection().extractors.pressure].value(i,p);
                    }
              }
          });
        candidate->matrix=std::make_unique<internal::FaultSparseCoupling>();
        candidate->matrix->build(entries);
        this->get_pcout() << "Fault sparse G: rank0 entries=" << candidate->matrix->values.size()
                         << ", bytes=" << candidate->matrix->bytes() << std::endl;
      }
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
        internal::FaultLinearSection timing(internal::FaultLinearTiming::inverse);
        // Replicated K_V/RHS require no communication. The same factor object
        // handles unrestricted and principal-free blocks, including safeguards.
        surface_linearization->factorizations[fault]->solve(rhs[fault],solution[fault]);
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
             surface_linearization->lower_diagonal,
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
              value += (surface_linearization->lower_diagonal.empty()
                        ? surface_linearization->off_diagonal[fault][i-1]
                        : surface_linearization->lower_diagonal[fault][i-1])
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
    AssertThrow(surface_linearization,ExcMessage("G must be linearized before applying it."));
    AssertThrow(physical_bulk_direction.size()==this->get_dof_handler().n_dofs(),
                ExcMessage("The reconstructed-fault G direction has the wrong size."));
    if (!surface_linearization->matrix)
      { apply_G_reference(physical_bulk_direction,result); return; }
    {
      internal::FaultLinearSection timing(internal::FaultLinearTiming::G_sparse);
      unsigned int size=0;
      for (const auto &fault : surface_linearization->diagonal) size+=fault.size();
      std::vector<double> local(size),global(size);
      surface_linearization->matrix->add(physical_bulk_direction,local);
      Utilities::MPI::sum(local,this->get_mpi_communicator(),global);
      result=surface_linearization->diagonal;
      unsigned int i=0;
      for (auto &fault : result) for (auto &value : fault) value=global[i++];
    }
    if (std::getenv("ASPECT_FAULT_COMPARE_COUPLING"))
      {
        FaultVector reference;
        apply_G_reference(physical_bulk_direction,reference);
        double error=0.,scale=0.;
        for (unsigned int f=0; f<result.size(); ++f)
          for (unsigned int i=0; i<result[f].size(); ++i)
            {
              error+=Utilities::fixed_power<2>(result[f][i]-reference[f][i]);
              scale+=Utilities::fixed_power<2>(reference[f][i]);
            }
        auto &diagnostics=internal::FaultLinearTiming::get();
        diagnostics.G_relative_error=std::max(diagnostics.G_relative_error,
                                             scale>0. ? std::sqrt(error/scale) : 0.);
        AssertThrow(std::sqrt(error)<=2.e-11*std::sqrt(scale),
                    ExcMessage("Sparse G disagrees with the independent parent/domain action."));
      }
  }


  template <int dim>
  void
  ReconstructedFaultSurfaceSystem<dim>::apply_G_reference(
    const LinearAlgebra::BlockVector &physical_bulk_direction,
    FaultVector &result) const
  {
    TimerOutput::Scope timer(*performance_timer, "Fault: G total");
    internal::FaultLinearSection linear_timing(internal::FaultLinearTiming::G);
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
