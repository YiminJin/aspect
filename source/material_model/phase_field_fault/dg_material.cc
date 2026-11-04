/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/material_model/phase_field_fault/fe_strain_internal.h>
#include <aspect/material_model/phase_field_fault/dg_history_internal.h>
#include <aspect/reconstructed_fault/objective_stress.h>
#include <aspect/phase_field.h>
#include <cstdlib>
#include <iomanip>
#include <aspect/particle/manager.h>
#include <deal.II/numerics/vector_tools_evaluate.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

namespace aspect { namespace MaterialModel
{
  template<int dim> double PhaseFieldFault<dim>::material_viscosity(
    const Point<dim> &x,const double reference) const
  {
    if(!dg_material_transport)return reference;
    const double multiplier=material_viscosity_multiplier.value(x);
    AssertThrow(std::isfinite(multiplier) && multiplier>0.,
                ExcMessage("Material viscosity multiplier must be finite and positive."));
    return reference*multiplier;
  }

  template<int dim> void PhaseFieldFault<dim>::finish_dg_material_step(const bool accept) noexcept
  {
    dg_material_reaction.reset();material_reaction_solving=false;
    if(!dg_material_trial)return;
    auto &trial=*dg_material_trial;
    if(!accept && trial.bound)
      {
        this->get_reconstructed_fault_manager().swap_prescribed_geometry(*trial.manager);
        fe_strain_state=trial.incoming;
        invalidate_normalization_cache();
        current_normalization_integrals.swap(trial.normalization);
        current_fault_surface_temperatures.swap(trial.temperatures);
        for(auto &p:this->get_phase_field_handler().get_associated_particle_manager().get_particle_handler())
          std::copy(trial.particle_properties.at(p.get_id()).begin(),
                    trial.particle_properties.at(p.get_id()).end(),p.get_properties().begin());
        fe_strain_state->stress_samples.clear();
        fe_strain_state->mechanical_samples.reset();
      }
    dg_material_trial.reset();
  }

  template<int dim> double PhaseFieldFault<dim>::material_surface_viscosity(
    unsigned int f,unsigned int j,double xi,double reference) const
  {
    if(!dg_material_transport)return reference;
    const auto &fault=this->get_reconstructed_fault_manager().get_fault(f);
    return material_viscosity(fault.vertex(j)+xi*(fault.vertex(j+1)-fault.vertex(j)),reference);
  }

  template<int dim> void PhaseFieldFault<dim>::begin_dg_material_step(
    LinearAlgebra::BlockVector &bulk,LinearAlgebra::BlockVector &linear)
  {
    if(!dg_material_transport)return;
    AssertThrow(!(material_bounded_profile || material_trace_inflow)
                || this->get_reconstructed_fault_manager().uses_full_normal_cross_section(),
                ExcMessage("The selected material profile/inflow extension requires Association support = full normal cross-section. "
                           "Stationary seed support can clip the independently supplied/transported Q1 profile while normalization includes its tails."));
    if(this->get_timestep_number()==0)return;
    AssertThrow(!dg_material_trial && mature_frictional_fault && !evolve_phase_field,
                ExcMessage("Material DG requires one growth-disabled accepted transaction."));
    if(restore_frozen_normalization_after_restart)
      {
        // Validate the loaded accepted geometry/profile before constructing
        // its first moving trial. The stationary restart guard must never see
        // the already transported profile as if it were checkpoint data.
        invalidate_normalization_cache();compute_normalization_integrals();
      }
    validate_fe_strain_restrictions();
    const double dt=this->get_timestep();
    AssertThrow(dt>0. && std::isfinite(dt),ExcMessage("Invalid material step interval."));
    material_viscosity_multiplier.set_time(this->get_time());
    auto transaction=std::make_shared<DGMaterialTrial>();
    transaction->incoming=fe_strain_state;
    transaction->step=this->get_timestep_number();transaction->dt=dt;
    transaction->normalization=current_normalization_integrals;
    transaction->temperatures=current_fault_surface_temperatures;
    for(const auto &p:this->get_phase_field_handler().get_associated_particle_manager().get_particle_handler())
      transaction->particle_properties.emplace(p.get_id(),std::vector<double>(p.get_properties().begin(),p.get_properties().end()));
    const auto &old=*transaction->incoming;
    auto &manager=this->get_reconstructed_fault_manager();
    const auto &space=*old.dg_history->space;
    const auto comm=this->get_mpi_communicator();
    const auto phase_block=this->introspection().variable("phase_field").block_index;
    const auto velocity_component=this->introspection().component_indices.velocities[0];
    const QGauss<dim> rule(std::max(8u,fe_bulk_normal_quadrature_points));
    const auto project=[&](const DGMaterialField &field)
      { return project_material_tensor(*old.dg_history,field); };

    const auto value=[&](const DGHistory &h,const auto &ids,const Point<dim> &unit)
      {
        SymmetricTensor<2,dim> result;
        for(unsigned c=0;c<h.trial_stress.size();++c)
          for(unsigned i=0;i<9;++i)
            result[SymmetricTensor<2,dim>::unrolled_to_component_indices(c)]+=h.trial_stress[c][ids[i]]*space.element.shape_value(i,unit);
        return result;
      };
    // Evolve the whole R+B combination. Its subtraction uses the transported
    // profile/normalization below, so no geometry/product-rule term is dropped.
    std::shared_ptr<DGHistory> U;
    if(!material_direct_stress)
    {
    U=project([&](const auto &,const auto &,const auto &x,const double phi,const auto &)
      {return evaluate_mature_dg_memory(x,phi,old.Z,old.I);});
    for(unsigned c=0;c<U->trial_stress.size();++c)
      {
        LinearAlgebra::Vector owned(space.dofs.locally_owned_dofs(),comm);
        owned=U->trial_stress[c];
        for(const auto i:owned.locally_owned_elements())owned[i]+=old.dg_history->residual->trial_stress[c][i];
        owned.compress(VectorOperation::insert);U->trial_stress[c]=owned;
      }
    }
    auto T=std::make_shared<DGHistory>(*old.dg_history);T->residual.reset();
    // Optional diagnostic ledger, not constitutive state or checkpoint data.
    const bool energy_ledger=material_direct_stress && std::getenv("ASPECT_TRUE_STRESS_ENERGY");
    const auto tensor_energy=[&](const DGHistory &history)
      {
        double energy=0.;
        FEValues<dim> fe(this->get_mapping(),space.element,rule,update_JxW_values);
        for(const auto &entry:old.cells)
          {
            fe.reinit(space.cells.at(entry.first));
            std::vector<types::global_dof_index> ids(9);space.cells.at(entry.first)->get_dof_indices(ids);
            for(unsigned q=0;q<rule.size();++q)
              {const auto a=value(history,ids,rule.point(q));
               energy+=fe.JxW(q)*(scalar_product(a,a)+trace(a)*trace(a))/(4*old.G);}
          }
        return Utilities::MPI::sum(energy,comm);
      };
    const double energy_before=energy_ledger ? tensor_energy(*T) : 0.;

    const auto transport=make_dg_transport(this->get_old_solution(),std::max(8u,fe_bulk_normal_quadrature_points),false);
    const unsigned substeps=std::max(1u,static_cast<unsigned>(std::ceil(dt*transport->inverse_advection_time/.1)));
    AssertThrow(substeps<=4096,ExcMessage("Material DG advection requires a smaller physical timestep (CFL substep budget)."));
    const DGField zero=[](const Point<dim> &,double){return SymmetricTensor<2,dim>();};
    const auto point_less=[](const Point<dim> &a,const Point<dim> &b)
    {
      for(unsigned d=0;d<dim;++d)
        {if(a[d]<b[d])return true;if(b[d]<a[d])return false;}
      return false;
    };
    std::map<Point<dim>,SymmetricTensor<2,dim>,decltype(point_less)>
      trial_inflow(point_less),residual_inflow(point_less);
    if(material_trace_inflow)
      {
        std::map<types::global_dof_index,CellId> owner;
        for(const auto &cell:transport->cells)owner.emplace(cell.indices.front(),cell.id);
        for(const auto &load:transport->inflow)
          {
            const auto cell=space.cells.at(owner.at(load.indices.front()));
            const auto unit=this->get_mapping().transform_real_to_unit_cell(cell,load.point);
            trial_inflow.emplace(load.point,value(*T,load.indices,unit));
            if(U)residual_inflow.emplace(load.point,value(*U,load.indices,unit));
          }
      }
    const DGField T_inflow=[&](const Point<dim> &x,double)
      {return material_trace_inflow ? trial_inflow.at(x) : SymmetricTensor<2,dim>();};
    const DGField U_inflow=[&](const Point<dim> &x,double)
      {return material_trace_inflow ? residual_inflow.at(x) : SymmetricTensor<2,dim>();};
    {
    TimerOutput::Scope timer(this->get_computing_timer(),"Material tensor advection");
    for(unsigned k=0;k<substeps;++k)
      {
        const double start=this->get_time()-dt+dt*k/substeps,sub_dt=dt/substeps;
        T=advance_dg_transport(*transport,*T,T_inflow,zero,start,sub_dt);
        if(U)U=advance_dg_transport(*transport,*U,U_inflow,zero,start,sub_dt);
      }
    }
    const double energy_advected=energy_ledger ? tensor_energy(*T) : 0.;
    // Label-following vertices retain all surface histories; V is not their
    // velocity. The same accepted FE field supplies bulk, curve and profile.
    auto vertices=std::vector<std::vector<Point<dim>>>(1,old.vertices);
    auto widths=manager.get_projection_half_widths();
    Utilities::MPI::RemotePointEvaluation<dim> lookup;
    auto &cache=this->get_phase_field_handler().get_grid_cache();
    lookup.reinit(cache,old.vertices);
    AssertThrow(lookup.all_points_found(),ExcMessage("Material curve leaves available native FE coverage."));
    const auto velocity=VectorTools::point_values<dim>(lookup,this->get_dof_handler(),this->get_old_solution(),VectorTools::EvaluationFlags::avg,velocity_component);
    const auto curve_gradients=VectorTools::point_gradients<dim>(lookup,this->get_dof_handler(),this->get_old_solution(),VectorTools::EvaluationFlags::avg,velocity_component);
    for(unsigned i=0;i<old.vertices.size();++i)vertices[0][i]+=dt*velocity[i];
    for(unsigned i=0;i<old.vertices.size();++i)
      {
        // A normal support is a geometric covector, not a stress-spin vector.
        // The physical profile itself is pulled back below; this only moves
        // the finite search support under the same first-order material map.
        const unsigned left=i==0 ? 0 : i-1,right=std::min<unsigned>(i+1,old.vertices.size()-1);
        auto tangent=old.vertices[right]-old.vertices[left];tangent/=tangent.norm();
        Tensor<1,dim> normal;normal[0]=-tangent[1];normal[1]=tangent[0];
        Tensor<2,dim> flow;for(unsigned d=0;d<dim;++d)flow[d][d]=1.;flow+=dt*curve_gradients[i];
        AssertThrow(determinant(flow)>0.,ExcMessage("Material support map reversed orientation; reduce the physical timestep."));
        widths[0][i]/=(transpose(invert(flow))*normal).norm();
      }
    transaction->manager=manager.stage_material_motion(vertices,widths);
    const auto phase_owned=transport_material_profile();
    auto next=std::make_shared<FEStrainState>(old);
    next->displacement_history=phase_owned;
    next->vertices=vertices[0];next->geometry_version=transaction->manager->get_fault(0).geometry_version();
    next->geometry_publications=old.geometry_publications+1;
    next->stress_samples.clear();next->mechanical_samples.reset();next->particle_samples.reset();
    next->objective_anchor.reset();next->objective_parent.reset();
    dg_material_trial=transaction;
    manager.swap_prescribed_geometry(*transaction->manager);transaction->bound=true;
    fe_strain_state=next;
    bulk.block(phase_block)=phase_owned.block(phase_block);linear.block(phase_block)=phase_owned.block(phase_block);
    try
      {
        invalidate_normalization_cache();compute_normalization_integrals();
        next->I=current_normalization_integrals.at(0);
        const auto ip=manager.get_property_information()[fault_property_indices.previous_normalization_integral].position;
        for(unsigned i=0;i<next->I.size();++i)manager.get_fault(0).get_properties(i)[ip]=next->I[i];
        compute_fault_surface_temperatures();
        auto anchor=project([&](const auto &ids,const auto &u,const auto &,double,const auto &gradient)
          {
            if constexpr(dim==2)return ReconstructedFaultUtilities::rotate_plane_strain_stress(value(*T,ids,u),gradient,dt);
            else return SymmetricTensor<2,dim>();
          });
        if(!material_direct_stress)anchor->residual=project([&](const auto &ids,const auto &u,const auto &x,double p,const auto &gradient)
          {
            if constexpr(dim==2)return ReconstructedFaultUtilities::rotate_plane_strain_stress(value(*U,ids,u),gradient,dt)
                                     -evaluate_mature_dg_memory(x,p,next->Z,next->I);
            else return SymmetricTensor<2,dim>();
          });
        if(energy_ledger)
          this->get_pcout()<<std::setprecision(17)<<"   Direct transport energy: step="<<this->get_timestep_number()
                           <<" dt="<<dt<<" accepted="<<energy_before<<" advected="<<energy_advected
                           <<" rotated_projected="<<tensor_energy(*anchor)<<std::endl;
        next->objective_anchor=anchor;next->objective_parent=next->dg_history;
        next->objective_step=this->get_timestep_number();next->objective_dt=dt;
        old.dg_history->space->point_samples.clear();
        old.dg_history->space->point_geometry_generation=numbers::invalid_unsigned_int;
        this->get_pcout()<<"   Material DG staged complete curve/profile/tensor/surface state; advection substeps="<<substeps<<std::endl;
      }
    catch(...){finish_dg_material_step(false);throw;}
  }

#define INSTANTIATE(dim) \
  template double PhaseFieldFault<dim>::material_viscosity(const Point<dim> &,double) const; \
  template double PhaseFieldFault<dim>::material_surface_viscosity(unsigned int,unsigned int,double,double) const; \
  template void PhaseFieldFault<dim>::begin_dg_material_step(LinearAlgebra::BlockVector &,LinearAlgebra::BlockVector &); \
  template void PhaseFieldFault<dim>::finish_dg_material_step(bool) noexcept;
  ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
}}
