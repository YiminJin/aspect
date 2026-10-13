/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include <aspect/reconstructed_fault/surface_system_internal.h>

#include <aspect/material_model/phase_field_fault.h>
#include <aspect/material_model/utilities.h>
#include <aspect/particle/manager.h>
#include <aspect/phase_field.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/reconstructed_fault/utilities.h>

#include <aspect/reconstructed_fault/normal_filter_internal.h>
#include <aspect/reconstructed_fault/surface_direct_internal.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_values.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <fstream>
#include <iomanip>

namespace aspect
{
  template <int dim>
  std::vector<SymmetricTensor<2,dim>> ReconstructedFaultSurfaceSystem<dim>::sample_normal_diagnostic_particles(
    const typename DoFHandler<dim>::active_cell_iterator &cell,const std::vector<Point<dim>> &points,
    NormalTractionDiagnostic &diagnostic,std::set<types::particle_index> &diagnostic_particle_ids) const
  {
    const unsigned int nq=points.size();
    std::vector<SymmetricTensor<2,dim>> particle_interpolated;
    const auto &pm=this->get_particle_manager(0);
    const auto &handler=pm.get_particle_handler();
    const auto sp=pm.get_property_manager().get_data_info().get_position_by_field_name("maxwell stress");
    ComponentMask mask(handler.n_properties_per_particle(),false);
    for (unsigned int c=0; c<SymmetricTensor<2,dim>::n_independent_components; ++c)
      mask.set(sp+c,true);
    const auto values=pm.get_interpolator().properties_at_points(handler,points,mask,cell);
    particle_interpolated.resize(nq);
    for (unsigned int q=0; q<nq; ++q)
      for (unsigned int c=0; c<SymmetricTensor<2,dim>::n_independent_components; ++c)
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
              for (unsigned int c=0; c<SymmetricTensor<2,dim>::n_independent_components; ++c)
                stress[SymmetricTensor<2,dim>::unrolled_to_component_indices(c)]=particle.get_properties()[sp+c];
              diagnostic.particles.push_back({neighbour->id().to_string(),particle.get_id(),neighbour->subdomain_id(),particle.get_location(),stress});
            }
    return particle_interpolated;
  }

  template <int dim>
  void ReconstructedFaultSurfaceSystem<dim>::sample_normal_diagnostic_lines(
    const typename DoFHandler<dim>::active_cell_iterator &cell,const LinearAlgebra::BlockVector &bulk_state,
    const FaultVector &slip_rate,const std::array<unsigned int,SymmetricTensor<2,dim>::n_independent_components> &stress_fields,
    NormalTractionDiagnostic &diagnostic) const
  {
    const auto &intro=this->introspection();
    auto &manager=this->get_reconstructed_fault_manager();
    const auto &faults=manager.get_faults();
    for (const auto &line:normal_diagnostic_lines)
      {
        const auto box=cell->bounding_box();
        const auto ends=box.get_boundary_points();
        const auto direction=line.second-line.first;
        double enter=0.,leave=1.;
        for (unsigned int c=0; c<dim; ++c)
          if (direction[c]!=0.)
            {
              double a=(ends.first[c]-line.first[c])/direction[c];
              double b=(ends.second[c]-line.first[c])/direction[c];
              if (a>b)
                std::swap(a,b);
              enter=std::max(enter,a);
              leave=std::min(leave,b);
            }
          else if (line.first[c]<ends.first[c] || line.first[c]>ends.second[c])
            enter=2.;
        if (enter>leave)
          continue;
        // Restrict the diagnostic geometry, not the production model.
        for (const auto v:cell->vertex_indices())
          for (unsigned int c=0; c<dim; ++c)
            AssertThrow(cell->vertex(v)[c]==ends.first[c] || cell->vertex(v)[c]==ends.second[c],
                        ExcMessage("Native diagnostic line requires axis-aligned Box cells."));
        std::vector<double> coordinates= {enter,leave};
        const unsigned int panels=static_cast<unsigned int>(std::ceil(direction.norm()/2.));
        for (unsigned int i=static_cast<unsigned int>(std::ceil(enter*panels)); i<=panels && i<=leave*panels; ++i)
          coordinates.push_back(double(i)/panels);
        std::sort(coordinates.begin(),coordinates.end());
        coordinates.erase(std::unique(coordinates.begin(),coordinates.end()),coordinates.end());
        std::vector<Point<dim>> reference_points;
        for (const auto t:coordinates)
          {
            auto unit=this->get_mapping().transform_real_to_unit_cell(cell,line.first+t*direction);
            AssertThrow(GeometryInfo<dim>::distance_to_unit_cell(unit)<1e-9,
                        ExcMessage("Clipped diagnostic line is outside its reference cell."));
            for (unsigned int c=0; c<dim; ++c)
              unit[c]=std::max(0.,std::min(1.,unit[c]));
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
        for (unsigned int c=0; c<fields.size(); ++c)
          line_fe[intro.extractors.compositional_fields[c]].get_function_values(bulk_state,fields[c]);
        const auto &pm=this->get_particle_manager(0);
        const auto sp=pm.get_property_manager().get_data_info().get_position_by_field_name("maxwell stress");
        ComponentMask mask(pm.get_particle_handler().n_properties_per_particle(),false);
        for (unsigned int c=0; c<SymmetricTensor<2,dim>::n_independent_components; ++c)
          mask.set(sp+c,true);
        const auto particle_values=pm.get_interpolator().properties_at_points(pm.get_particle_handler(),line_fe.get_quadrature_points(),mask,cell);
        for (unsigned int q=0; q<n; ++q)
          {
            const auto x=line_fe.quadrature_point(q);
            const auto a=manager.project_to_normal_profiles(x);
            AssertThrow(a.active,ExcMessage("Native centerline has no fault association."));
            auto tangent=faults[a.fault_index].vertex(a.segment_index+1)-faults[a.fault_index].vertex(a.segment_index);
            tangent/=tangent.norm();
            Tensor<1,dim> normal;
            normal[0]=-tangent[1];
            normal[1]=tangent[0];
            typename MaterialModel::PhaseFieldFault<dim>::ReconstructedFaultPointInputs input;
            input.position=x;
            input.fault_index=a.fault_index;
            input.segment_index=a.segment_index;
            input.xi=a.xi;
            input.phase_field=input.previous_phase_field=phi[q];
            input.temperature=temp[q];
            input.dynamic_pressure=p[q];
            input.strain_rate=eps[q];
            input.capture_stress_components=true;
            input.slip_tensor=manager.get_shear_sense(a.fault_index)
                              *symmetrize(outer_product(tangent,normal));
            input.normal_tensor=symmetrize(outer_product(normal,normal));
            SymmetricTensor<2,dim> particle_stress;
            for (unsigned int c=0; c<stress_fields.size(); ++c)
              {
                const auto index=SymmetricTensor<2,dim>::unrolled_to_component_indices(c);
                input.old_maxwell_stress[index]=fields[stress_fields[c]][q];
                particle_stress[index]=particle_values[q][sp+c];
              }
            input.old_maxwell_stress = phase_field_fault.retained_history_stress(cell->id(),x,input.old_maxwell_stress);
            std::vector<double> chemical;
            for (const auto c:intro.chemical_composition_field_indices())
              chemical.push_back(fields[c][q]);
            input.bulk_material_fractions=MaterialModel::MaterialUtilities::compute_composition_fractions(chemical);
            input.slip_rate=ReconstructedFaultUtilities::interpolate_slip_rate(
                              slip_rate[a.fault_index][a.segment_index],slip_rate[a.fault_index][a.segment_index+1],a.xi);
            const auto response=phase_field_fault.evaluate_reconstructed_fault_point(input);
            const auto foot=(1.-a.xi)*faults[a.fault_index].vertex(a.segment_index)+a.xi*faults[a.fault_index].vertex(a.segment_index+1);
            diagnostic.line_samples.push_back({cell->id().to_string(),q,static_cast<unsigned int>(cell->level()),
                                               a.fault_index,a.segment_index,x,foot,normal,response.stress,cell->diameter(),a.xi,p[q],
                                               -(response.stress*input.normal_tensor),response.background_normal_traction,response.normal_traction,
                                               phi[q],response.normalization_integral,response.localization_factor,0.,0.,input.old_maxwell_stress,
                                               response.stress_components,reference_points[q],particle_stress,
                                               gradients[q],response.stress_time_step,response.stress_beta,response.eta_ve
                                              });
          }
      }
  }

  template <int dim>
  void ReconstructedFaultSurfaceSystem<dim>::write_particle_diagnostics(
    const FaultVector &slip_rate,const ParticleDiagnosticData &diagnostic) const
  {
    const auto &faults=this->get_reconstructed_fault_manager().get_faults();
    const bool record_samples=diagnostic.write_samples;
    const bool record_normal_stress=diagnostic.write_normal;
    const auto &lowest_samples=diagnostic.lowest_samples;
    const auto &highest_samples=diagnostic.highest_samples;
    const auto &prescribed=diagnostic.prescribed;
    const auto &stress_audit_moments=diagnostic.stress_moments;
    const auto &normal_stress_moments=diagnostic.normal_moments;
    const auto &history_moments=diagnostic.history_moments;
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
              for (double value:stress_audit_moments[f][i])
                moments<<','<<value;
              moments<<'\n';
            }
      }
    if (diagnostic.write_history)
      {
        std::ofstream out(this->get_output_directory()+"history_surface_rank"+
                          std::to_string(Utilities::MPI::this_mpi_process(this->get_mpi_communicator()))+".csv");
        out.exceptions(std::ios::failbit|std::ios::badbit);
        out<<"fault,node,x,y,V,weight,p,particle_q,particle_sigma,particle_C,particle_friction,particle_damping,particle_R,particle_Kdiag,particle_Koff_sum,particle_tensile_weight,particle_tensile_friction,fe_q,fe_sigma,fe_C,fe_friction,fe_damping,fe_R,fe_Kdiag,fe_Koff_sum,fe_tensile_weight,fe_tensile_friction\n";
        for (unsigned int f=0; f<faults.size(); ++f)
          for (unsigned int v=0; v<faults[f].n_vertices(); ++v)
            {
              out<<std::setprecision(17)<<f<<','<<v<<','<<faults[f].vertex(v)[0]<<','<<faults[f].vertex(v)[1]<<','<<slip_rate[f][v];
              for (double x:history_moments[f][v])
                out<<','<<x;
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
  }

#define INSTANTIATE(dim) \
  template std::vector<SymmetricTensor<2,dim>> ReconstructedFaultSurfaceSystem<dim>::sample_normal_diagnostic_particles(const DoFHandler<dim>::active_cell_iterator &,const std::vector<Point<dim>> &,NormalTractionDiagnostic &,std::set<types::particle_index> &) const; \
  template void ReconstructedFaultSurfaceSystem<dim>::sample_normal_diagnostic_lines(const DoFHandler<dim>::active_cell_iterator &,const LinearAlgebra::BlockVector &,const FaultVector &,const std::array<unsigned int,SymmetricTensor<2,dim>::n_independent_components> &,NormalTractionDiagnostic &) const; \
  template void ReconstructedFaultSurfaceSystem<dim>::write_particle_diagnostics(const FaultVector &,const ParticleDiagnosticData &) const;
  ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
}
