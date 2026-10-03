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

#include "normal_filter_internal.h"

#include <deal.II/fe/fe_values.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

namespace aspect
{
  template <int dim>
  typename ReconstructedFaultSurfaceSystem<dim>::SurfaceAssembly
  ReconstructedFaultSurfaceSystem<dim>::assemble_bulk_work_system(
    const LinearAlgebra::BlockVector &bulk_state, const FaultVector &slip_rate,
    const bool assemble_jacobian) const
  {
    TimerOutput::Scope timer(*performance_timer,"Fault: bulk work R/K");
    phase_field_fault.validate_reconstructed_fault_constitutive_state();
    const bool filtering=normal_filter_mode!="raw";
    AssertThrow(!filtering || !phase_field_fault.uses_adiabatic_friction_pressure(),
                ExcMessage("Experimental normal filtering requires true normal stress."));
    auto &manager=this->get_reconstructed_fault_manager();
    const auto &faults=manager.get_faults();
    AssertDimension(slip_rate.size(),faults.size());
    SurfaceAssembly local;
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
    struct FilterPoint
    {
      unsigned int fault,segment;
      double xi,weight,raw,mu,dmu,residual;
    };
    std::vector<FilterPoint> filter_points;
    FaultVector stiffness=local.mass_diagonal,stiffness_edge=local.mass_off_diagonal;
    // The same incoming-history response as mechanics, not a post-commit update.
    // Rejected trial residuals cannot overwrite this linearization snapshot.
    if (assemble_jacobian && normal_diagnostic_window)
      {
        AssertThrow(!phase_field_fault.uses_adiabatic_friction_pressure(),
                    ExcMessage("Normal split diagnostic requires true normal traction."));
        local.normal_diagnostic = std::make_shared<NormalTractionDiagnostic>();
        auto &d = *local.normal_diagnostic;
        d.step = this->get_timestep_number(); d.time = this->get_time();
        d.pressure_load = d.deviatoric_load = d.background_load = local.residual.values;
        for (auto &load : d.deviatoric_component_loads) load=local.residual.values;
        d.rates = slip_rate;
        d.friction_mass_diagonal=local.mass_diagonal;
        d.friction_mass_off_diagonal=local.mass_off_diagonal;
      }
    const auto &intro=this->introspection();
    const auto &quadrature=intro.quadratures.velocities;
    FEValues<dim> fe(this->get_mapping(),this->get_fe(),quadrature,
                     update_values|update_gradients|update_quadrature_points|update_JxW_values);
    const unsigned int nq=quadrature.size();
    std::vector<double> phase(nq),temperature(nq),pressure(nq);
    std::vector<SymmetricTensor<2,dim>> strain(nq);
    std::vector<Tensor<2,dim>> diagnostic_gradients(nq);
    std::vector<std::vector<double>> composition(intro.n_compositional_fields,std::vector<double>(nq));
    std::array<unsigned int,SymmetricTensor<2,dim>::n_independent_components> stress_fields;
    stress_fields.fill(numbers::invalid_unsigned_int);
    for (const auto &entry:this->get_parameters().mapped_particle_properties)
      if (entry.second.first=="maxwell stress") stress_fields[entry.second.second]=entry.first;
    manager.prepare_stokes_qp_projection_cache();
    std::set<types::particle_index> diagnostic_particle_ids;

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
          // The observer compares the configured particle interpolator with
          // the constrained FE history actually consumed below. These are
          // distinct representations; no particle update occurs here.
          std::vector<SymmetricTensor<2,dim>> particle_interpolated;
          if (local.normal_diagnostic)
            {
              fe[intro.extractors.velocities].get_function_gradients(bulk_state,diagnostic_gradients);
              bool selected=false;
              for (const auto &a:associations)
                if (a.active)
                  selected |= normal_diagnostic_window((1.-a.xi)*faults[a.fault_index].vertex(a.segment_index)
                                                       +a.xi*faults[a.fault_index].vertex(a.segment_index+1));
              // Also cover stencils of cell traces at a line-window edge,
              // even when no ordinary QP falls inside that very short overlap.
              for (const auto &line:normal_diagnostic_lines)
                {
                  const auto ends=cell->bounding_box().get_boundary_points();
                  bool overlaps=true;
                  for (unsigned int c=0;c<dim;++c)
                    overlaps &= std::max(line.first[c],line.second[c])>=ends.first[c]
                                && std::min(line.first[c],line.second[c])<=ends.second[c];
                  selected |= overlaps;
                }
              if (selected)
                {
                  const auto &pm=this->get_particle_manager(0);
                  const auto &handler=pm.get_particle_handler();
                  const auto sp=pm.get_property_manager().get_data_info().get_position_by_field_name("maxwell stress");
                  ComponentMask mask(handler.n_properties_per_particle(),false);
                  for (unsigned int c=0;c<SymmetricTensor<2,dim>::n_independent_components;++c) mask.set(sp+c,true);
                  const auto values=pm.get_interpolator().properties_at_points(handler,fe.get_quadrature_points(),mask,cell);
                  particle_interpolated.resize(nq);
                  for (unsigned int q=0;q<nq;++q)
                    for (unsigned int c=0;c<SymmetricTensor<2,dim>::n_independent_components;++c)
                      particle_interpolated[q][SymmetricTensor<2,dim>::unrolled_to_component_indices(c)]=values[q][sp+c];
                  // Include vertex-neighbour stencils used by distance-weighted
                  // interpolation. Ghost copies are labeled, never counted as
                  // independent owners; each ID appears once per rank snapshot.
                  const auto &vertex_cells=this->get_phase_field_handler().get_grid_cache().get_vertex_to_cell_map();
                  std::set<typename Triangulation<dim>::active_cell_iterator> neighbours;
                  for (const auto v:cell->vertex_indices())
                    neighbours.insert(vertex_cells[cell->vertex_index(v)].begin(),vertex_cells[cell->vertex_index(v)].end());
                  for (const auto &neighbour:neighbours)
                    if (!neighbour->is_artificial())
                      for (const auto &particle:handler.particles_in_cell(neighbour))
                        if (diagnostic_particle_ids.insert(particle.get_id()).second)
                          {
                            SymmetricTensor<2,dim> stress;
                            for (unsigned int c=0;c<SymmetricTensor<2,dim>::n_independent_components;++c)
                              stress[SymmetricTensor<2,dim>::unrolled_to_component_indices(c)]=particle.get_properties()[sp+c];
                            local.normal_diagnostic->particles.push_back({neighbour->id().to_string(),particle.get_id(),neighbour->subdomain_id(),particle.get_location(),stress});
                          }
                }
            }
          // Count unsupported positive-phase samples without inventing a map.
          // Optional native centerline evaluation in the same frozen working
          // state. Axis-aligned cell clipping retains both one-sided cell
          // traces; these points never enter quadrature or mechanical assembly.
          if (local.normal_diagnostic && !normal_diagnostic_lines.empty())
            for (const auto &line:normal_diagnostic_lines)
              {
                const auto box=cell->bounding_box();
                const auto ends=box.get_boundary_points();
                const auto direction=line.second-line.first;
                double enter=0.,leave=1.;
                for (unsigned int c=0;c<dim;++c)
                  if (direction[c]!=0.)
                    {
                      double a=(ends.first[c]-line.first[c])/direction[c];
                      double b=(ends.second[c]-line.first[c])/direction[c];
                      if (a>b) std::swap(a,b);
                      enter=std::max(enter,a);leave=std::min(leave,b);
                    }
                  else if (line.first[c]<ends.first[c] || line.first[c]>ends.second[c]) enter=2.;
                if (enter>leave) continue;
                // Restrict the diagnostic geometry, not the production model.
                for (const auto v:cell->vertex_indices())
                  for (unsigned int c=0;c<dim;++c)
                    AssertThrow(cell->vertex(v)[c]==ends.first[c] || cell->vertex(v)[c]==ends.second[c],
                                ExcMessage("Native diagnostic line requires axis-aligned Box cells."));
                std::vector<double> coordinates={enter,leave};
                const unsigned int panels=static_cast<unsigned int>(std::ceil(direction.norm()/2.));
                for (unsigned int i=static_cast<unsigned int>(std::ceil(enter*panels));i<=panels && i<=leave*panels;++i)
                  coordinates.push_back(double(i)/panels);
                std::sort(coordinates.begin(),coordinates.end());
                coordinates.erase(std::unique(coordinates.begin(),coordinates.end()),coordinates.end());
                std::vector<Point<dim>> reference_points;
                for (const auto t:coordinates)
                  {
                    auto unit=this->get_mapping().transform_real_to_unit_cell(cell,line.first+t*direction);
                    AssertThrow(GeometryInfo<dim>::distance_to_unit_cell(unit)<1e-9,
                                ExcMessage("Clipped diagnostic line is outside its reference cell."));
                    for (unsigned int c=0;c<dim;++c) unit[c]=std::max(0.,std::min(1.,unit[c]));
                    reference_points.push_back(unit);
                  }
                FEValues<dim> line_fe(this->get_mapping(),this->get_fe(),Quadrature<dim>(reference_points),
                                     update_values|update_gradients|update_quadrature_points);
                line_fe.reinit(cell);
                const unsigned int n=reference_points.size();
                std::vector<double> phi(n),temp(n),p(n);
                std::vector<SymmetricTensor<2,dim>> eps(n);
                std::vector<Tensor<2,dim>> gradients(n);
                std::vector<std::vector<double>> fields(intro.n_compositional_fields,std::vector<double>(n));
                line_fe[FEValuesExtractors::Scalar(intro.variable("phase_field").first_component_index)].get_function_values(bulk_state,phi);
                line_fe[intro.extractors.temperature].get_function_values(bulk_state,temp);
                line_fe[intro.extractors.pressure].get_function_values(bulk_state,p);
                line_fe[intro.extractors.velocities].get_function_symmetric_gradients(bulk_state,eps);
                line_fe[intro.extractors.velocities].get_function_gradients(bulk_state,gradients);
                for (unsigned int c=0;c<fields.size();++c)
                  line_fe[intro.extractors.compositional_fields[c]].get_function_values(bulk_state,fields[c]);
                const auto &pm=this->get_particle_manager(0);
                const auto sp=pm.get_property_manager().get_data_info().get_position_by_field_name("maxwell stress");
                ComponentMask mask(pm.get_particle_handler().n_properties_per_particle(),false);
                for (unsigned int c=0;c<SymmetricTensor<2,dim>::n_independent_components;++c) mask.set(sp+c,true);
                const auto particle_values=pm.get_interpolator().properties_at_points(pm.get_particle_handler(),line_fe.get_quadrature_points(),mask,cell);
                for (unsigned int q=0;q<n;++q)
                  {
                    const auto x=line_fe.quadrature_point(q);
                    const auto a=manager.project_to_normal_profiles(x);
                    AssertThrow(a.active,ExcMessage("Native centerline has no fault association."));
                    auto tangent=faults[a.fault_index].vertex(a.segment_index+1)-faults[a.fault_index].vertex(a.segment_index);
                    tangent/=tangent.norm();
                    Tensor<1,dim> normal;normal[0]=-tangent[1];normal[1]=tangent[0];
                    typename MaterialModel::PhaseFieldFault<dim>::ReconstructedFaultPointInputs input;
                    input.position=x; input.fault_index=a.fault_index;input.segment_index=a.segment_index;input.xi=a.xi;
                    input.phase_field=input.previous_phase_field=phi[q];input.temperature=temp[q];
                    input.dynamic_pressure=p[q];input.strain_rate=eps[q];input.capture_stress_components=true;
                    input.slip_tensor=manager.get_shear_sense(a.fault_index)
                                      *symmetrize(outer_product(tangent,normal));
                    input.normal_tensor=symmetrize(outer_product(normal,normal));
                    SymmetricTensor<2,dim> particle_stress;
                    for (unsigned int c=0;c<stress_fields.size();++c)
                      {
                        const auto index=SymmetricTensor<2,dim>::unrolled_to_component_indices(c);
                        input.old_maxwell_stress[index]=fields[stress_fields[c]][q];particle_stress[index]=particle_values[q][sp+c];
                      }
                    if (phase_field_fault.benchmark_retained_stress)
                      input.old_maxwell_stress = phase_field_fault.benchmark_retained_stress(cell->id(),x);
                    std::vector<double> chemical;
                    for (const auto c:intro.chemical_composition_field_indices()) chemical.push_back(fields[c][q]);
                    input.bulk_material_fractions=MaterialModel::MaterialUtilities::compute_composition_fractions(chemical);
                    input.slip_rate=ReconstructedFaultUtilities::interpolate_slip_rate(
                      slip_rate[a.fault_index][a.segment_index],slip_rate[a.fault_index][a.segment_index+1],a.xi);
                    const auto response=phase_field_fault.evaluate_reconstructed_fault_point(input);
                    const auto foot=(1.-a.xi)*faults[a.fault_index].vertex(a.segment_index)+a.xi*faults[a.fault_index].vertex(a.segment_index+1);
                    local.normal_diagnostic->line_samples.push_back({cell->id().to_string(),q,static_cast<unsigned int>(cell->level()),
                      a.fault_index,a.segment_index,x,foot,normal,response.stress,cell->diameter(),a.xi,p[q],
                      -(response.stress*input.normal_tensor),response.background_normal_traction,response.normal_traction,
                      phi[q],response.normalization_integral,response.localization_factor,0.,0.,input.old_maxwell_stress,
                      response.stress_components,reference_points[q],particle_stress,
                      gradients[q],response.stress_time_step,response.stress_beta,response.eta_ve});
                  }
              }
          if (local.normal_diagnostic)
            for (unsigned int q=0;q<nq;++q)
              if (!associations[q].active && phase[q]>0.
                  && normal_diagnostic_window(fe.quadrature_point(q)))
                ++local.normal_diagnostic->unassociated_phase_points;
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
                input.slip_tensor=manager.get_shear_sense(a.fault_index)
                                  *symmetrize(outer_product(a.tangent,a.normal));
                input.normal_tensor=symmetrize(outer_product(a.normal,a.normal));
                for (unsigned int c=0;c<stress_fields.size();++c)
                  {
                    AssertIndexRange(stress_fields[c],composition.size());
                    input.old_maxwell_stress[SymmetricTensor<2,dim>::unrolled_to_component_indices(c)]
                      =composition[stress_fields[c]][q];
                  }
                if (phase_field_fault.benchmark_retained_stress)
                  input.old_maxwell_stress = phase_field_fault.benchmark_retained_stress(
                    cell->id(), fe.quadrature_point(q));
                std::vector<double> chemical;
                for (const auto c:intro.chemical_composition_field_indices()) chemical.push_back(composition[c][q]);
                input.bulk_material_fractions=MaterialModel::MaterialUtilities::compute_composition_fractions(chemical);
                const auto &V=slip_rate[a.fault_index];
                input.slip_rate=ReconstructedFaultUtilities::interpolate_slip_rate(
                  V[a.segment_index],V[a.segment_index+1],a.shape_1);
                input.capture_stress_components = static_cast<bool>(local.normal_diagnostic);
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
                // The same continuous Q1 weights enter source, test and trial.
                const double N[2]={a.shape_0,a.shape_1};
                if (filtering)
                  {
                    filter_points.push_back({f,j,a.shape_1,weight,response.normal_traction,
                      response.friction_coefficient,response.friction_derivative_wrt_slip_rate,response.residual_density});
                  }
                if (filtering || local.normal_diagnostic)
                  internal::add_normal_filter_stiffness(faults[f],input.position,j,
                    a.tangent,weight,stiffness[f],stiffness_edge[f]);
                if (local.normal_diagnostic)
                  {
                    auto &d = *local.normal_diagnostic;
                    for (unsigned int i=0;i<2;++i)
                      d.friction_mass_diagonal[f][j+i]+=weight*N[i]*N[i]*response.friction_coefficient;
                    d.friction_mass_off_diagonal[f][j]+=weight*N[0]*N[1]*response.friction_coefficient;
                    const double deviatoric = -(response.stress * input.normal_tensor);
                    d.integrated_loads[0] += weight*pressure[q];
                    d.integrated_loads[1] += weight*deviatoric;
                    d.integrated_loads[2] += weight*response.background_normal_traction;
                    for (unsigned int c=0;c<3;++c)
                      {
                        const double value=-(response.stress_components[c]*input.normal_tensor);
                        d.integrated_loads[c+3] += weight*value;
                        for (unsigned int i=0;i<2;++i)
                          d.deviatoric_component_loads[c][f][j+i] += weight*N[i]*value;
                      }
                    for (unsigned int i=0;i<2;++i)
                      {
                        d.pressure_load[f][j+i] += weight*N[i]*pressure[q];
                        d.deviatoric_load[f][j+i] += weight*N[i]*deviatoric;
                        d.background_load[f][j+i] += weight*N[i]*response.background_normal_traction;
                      }
                    const Point<dim> foot=(1.-a.xi)*faults[f].vertex(j)+a.xi*faults[f].vertex(j+1);
                    if (normal_diagnostic_window(foot))
                      d.samples.push_back({cell->id().to_string(),q,static_cast<unsigned int>(cell->level()),f,j,
                        input.position, foot,
                        a.normal,response.stress,cell->diameter(),a.xi,pressure[q],deviatoric,
                        response.background_normal_traction,response.normal_traction,
                        phase[q],response.normalization_integral,response.localization_factor,
                        fe.JxW(q),weight,input.old_maxwell_stress,response.stress_components,
                        quadrature.point(q),particle_interpolated[q],diagnostic_gradients[q],
                        response.stress_time_step,response.stress_beta,response.eta_ve,
                        response.friction_coefficient,response.normal_traction});
                  }
                for (unsigned int i=0;i<2;++i)
                  {
                    local.residual.values[f][j+i]+=weight*N[i]*response.residual_density;
                    local.residual.shear_traction[f][j+i]+=weight*N[i]*response.shear_traction;
                    local.residual.friction_traction[f][j+i]+=weight*N[i]*response.friction_traction;
                    local.residual.damping_traction[f][j+i]+=weight*N[i]*response.damping_traction;
                    local.residual.normal_traction[f][j+i]+=weight*N[i]*response.normal_traction;
                    local.mass_diagonal[f][j+i]+=weight*N[i]*N[i];
                    if (assemble_jacobian)
                      local.diagonal[f][j+i]+=weight*N[i]*N[i]*response.minus_derivative_wrt_slip_rate;
                  }
                local.mass_off_diagonal[f][j]+=weight*N[0]*N[1];
                squared[f]+=weight*response.residual_density*response.residual_density;
                measure[f]+=weight;
                local.residual.minimum_normal_traction=std::min(local.residual.minimum_normal_traction,response.normal_traction);
                local.residual.maximum_normal_traction=std::max(local.residual.maximum_normal_traction,response.normal_traction);
                if (assemble_jacobian)
                  {
                    local.off_diagonal[f][j]+=weight*N[0]*N[1]*response.minus_derivative_wrt_slip_rate;
                    const unsigned int point_index=local.coupling_points.size();
                    local.coupling_points.push_back({input.position,point_index,f,j,a.shape_1,weight,response.eta_ve,
                      response.friction_coefficient,input.slip_tensor,input.normal_tensor,
                      response.uses_adiabatic_friction_pressure});
                  }
              }
        }

    // QPs have unique cell ownership. Only fault-sized arrays are reduced;
    // G keeps the same QP coordinates and coefficients on their original ranks.
    const auto comm=this->get_mpi_communicator();
    if (local.normal_diagnostic)
      for (auto &value : local.normal_diagnostic->integrated_loads)
        value=Utilities::MPI::sum(value,comm);
    if (local.normal_diagnostic)
      for (auto *v : {&local.normal_diagnostic->pressure_load,
                      &local.normal_diagnostic->deviatoric_load,
                      &local.normal_diagnostic->background_load,
                      &local.normal_diagnostic->deviatoric_component_loads[0],
                      &local.normal_diagnostic->deviatoric_component_loads[1],
                      &local.normal_diagnostic->deviatoric_component_loads[2]})
        for (auto &fault : *v)
          { const auto copy=fault; Utilities::MPI::sum(copy,comm,fault); }
    if (filtering || local.normal_diagnostic)
      for (auto *v:{&stiffness,&stiffness_edge})
        for (auto &fault:*v) {const auto copy=fault;Utilities::MPI::sum(copy,comm,fault);}
    if (local.normal_diagnostic)
      {
        local.normal_diagnostic->filter_stiffness_diagonal=stiffness;
        local.normal_diagnostic->filter_stiffness_off_diagonal=stiffness_edge;
        for (auto *v:{&local.normal_diagnostic->friction_mass_diagonal,&local.normal_diagnostic->friction_mass_off_diagonal})
          for (auto &fault:*v) {const auto copy=fault;Utilities::MPI::sum(copy,comm,fault);}
      }
    for (auto *v:{&local.residual.values,&local.residual.shear_traction,
                 &local.residual.cohesive_traction,&local.residual.friction_traction,
                 &local.residual.damping_traction,&local.residual.normal_traction,
                 &local.diagonal,&local.off_diagonal,&local.mass_diagonal,&local.mass_off_diagonal})
      for (auto &fault:*v)
        { const auto copy=fault; Utilities::MPI::sum(copy,comm,fault); }
    if (filtering)
      {
        TimerOutput::Scope filter_timer(*performance_timer,"Fault: normal filter");
        normal_filter_cache.resize(faults.size());
        local.residual.raw_normal_traction=local.residual.normal_traction;
        local.residual.normal_filter_coefficients.resize(faults.size());
        for (unsigned int f=0;f<faults.size();++f)
          {
            // Compare assembled operators, not timestep numbers. Geometry,
            // phase, material and continuation changes affect these moments.
            // Trial values never enter this cache. Keep each linearization's
            // shared factor alive even when a later trial rebuilds the cache.
            if (!normal_filter_cache[f] || !normal_filter_cache[f]->matches(local.mass_diagonal[f],
                local.mass_off_diagonal[f],stiffness[f],stiffness_edge[f],normal_filter_length))
              normal_filter_cache[f]=std::make_shared<internal::FaultNormalFilter>(local.mass_diagonal[f],
                  local.mass_off_diagonal[f],stiffness[f],stiffness_edge[f],normal_filter_length,f);
            local.residual.normal_filter_coefficients[f]=normal_filter_cache[f]->solve(local.residual.raw_normal_traction[f]);
          }
        local.normal_filters=normal_filter_cache;
        if (local.normal_diagnostic)
          for (auto *samples:{&local.normal_diagnostic->samples,&local.normal_diagnostic->line_samples})
            for (auto &sample:*samples)
              {
                const auto &z=local.residual.normal_filter_coefficients[sample.fault];
                sample.friction_normal=(1.-sample.xi)*z[sample.segment]+sample.xi*z[sample.segment+1];
              }
        FaultVector correction=local.residual.values,normal=correction,diagonal=correction,edge=local.off_diagonal;
        for (auto *v:{&correction,&normal,&diagonal,&edge}) for (auto &f:*v) std::fill(f.begin(),f.end(),0.);
        local.residual.minimum_raw_normal_traction=local.residual.minimum_normal_traction;
        local.residual.maximum_raw_normal_traction=local.residual.maximum_normal_traction;
        local.residual.minimum_normal_traction=std::numeric_limits<double>::infinity();
        local.residual.maximum_normal_traction=-std::numeric_limits<double>::infinity();
        std::fill(squared.begin(),squared.end(),0.);
        for (const auto &p:filter_points)
          {
            const double N[2]={1.-p.xi,p.xi};const auto &z=local.residual.normal_filter_coefficients[p.fault];
            const double filtered=N[0]*z[p.segment]+N[1]*z[p.segment+1];
            const double difference=filtered-p.raw,friction=p.mu*difference;
            for (unsigned int i=0;i<2;++i)
              {
                correction[p.fault][p.segment+i]+=p.weight*N[i]*friction;
                normal[p.fault][p.segment+i]+=p.weight*N[i]*difference;
                diagonal[p.fault][p.segment+i]+=p.weight*N[i]*N[i]*p.dmu*difference;
              }
            edge[p.fault][p.segment]+=p.weight*N[0]*N[1]*p.dmu*difference;
            squared[p.fault]+=p.weight*(p.residual-friction)*(p.residual-friction);
            local.residual.minimum_normal_traction=std::min(local.residual.minimum_normal_traction,filtered);
            local.residual.maximum_normal_traction=std::max(local.residual.maximum_normal_traction,filtered);
          }
        for (auto *v:{&correction,&normal,&diagonal,&edge})
          for (auto &f:*v) {const auto copy=f;Utilities::MPI::sum(copy,comm,f);}
        for (unsigned int f=0;f<faults.size();++f)
          {
            for (unsigned int i=0;i<faults[f].n_vertices();++i)
              {
                local.residual.values[f][i]-=correction[f][i];
                local.residual.friction_traction[f][i]+=correction[f][i];
                local.residual.normal_traction[f][i]+=normal[f][i];
                if (assemble_jacobian) local.diagonal[f][i]+=diagonal[f][i];
              }
            if (assemble_jacobian)
              for (unsigned int i=0;i<edge[f].size();++i) local.off_diagonal[f][i]+=edge[f][i];
          }
        local.residual.minimum_raw_normal_traction=Utilities::MPI::min(local.residual.minimum_raw_normal_traction,comm);
        local.residual.maximum_raw_normal_traction=Utilities::MPI::max(local.residual.maximum_raw_normal_traction,comm);
      }
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


#define INSTANTIATE(dim) \
  template ReconstructedFaultSurfaceSystem<dim>::SurfaceAssembly \
  ReconstructedFaultSurfaceSystem<dim>::assemble_bulk_work_system(const LinearAlgebra::BlockVector &, const FaultVector &, const bool) const;

  ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
}
