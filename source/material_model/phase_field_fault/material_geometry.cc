/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/material_model/phase_field_fault/material_history_internal.h>
#include <aspect/reconstructed_fault/objective_stress.h>
#include <aspect/phase_field.h>
#include <aspect/particle/manager.h>
#include <deal.II/numerics/vector_tools_evaluate.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <cstdlib>
#include <iomanip>

namespace aspect
{
  namespace MaterialModel
  {
    template<int dim> void PhaseFieldFault<dim>::stage_material_geometry(
      const MaterialHistoryInputs &input,LinearAlgebra::BlockVector &bulk,LinearAlgebra::BlockVector &linear)
    {
      const auto &old=input.accepted;
      auto transaction=dg_material_trial;
      auto &manager=this->get_reconstructed_fault_manager();
      const auto phase_block=this->introspection().variable("phase_field").block_index;
      const auto velocity_component=this->introspection().component_indices.velocities[0];
      // Label-following vertices retain all surface histories; V is not their
      // velocity. The same accepted FE field supplies bulk, curve and profile.
      Utilities::MPI::RemotePointEvaluation<dim> lookup;
      auto &cache=this->get_phase_field_handler().get_grid_cache();
      lookup.reinit(cache,old.vertices);
      AssertThrow(lookup.all_points_found(),ExcMessage("Material curve leaves available native FE coverage."));
      const auto velocity=VectorTools::point_values<dim>(lookup,this->get_dof_handler(),input.accepted_velocity,VectorTools::EvaluationFlags::avg,velocity_component);
      const auto curve_gradients=VectorTools::point_gradients<dim>(lookup,this->get_dof_handler(),input.accepted_velocity,VectorTools::EvaluationFlags::avg,velocity_component);
      transaction->manager=manager.stage_material_motion(input.dt, {velocity}, {curve_gradients});
      const auto phase_owned=transport_material_profile(input);
      auto next=std::make_shared<FEStrainState>(old);
      next->displacement_history=phase_owned;
      next->vertices=transaction->manager->get_fault(0).get_vertices();
      next->geometry_version=transaction->manager->get_fault(0).geometry_version();
      next->geometry_publications=old.geometry_publications+1;
      next->stress_samples.clear();
      next->mechanical_samples.reset();
      next->particle_samples.reset();
      next->objective_anchor.reset();
      next->objective_parent.reset();
      transaction->candidate_history=next;
      manager.swap_prescribed_geometry(*transaction->manager);
      transaction->bound=true;
      fe_strain_state=next;
      bulk.block(phase_block)=phase_owned.block(phase_block);
      linear.block(phase_block)=phase_owned.block(phase_block);
      invalidate_normalization_cache();
      compute_normalization_integrals();
      next->I=current_normalization_integrals.at(0);
      const auto ip=manager.get_property_information()[fault_property_indices.previous_normalization_integral].position;
      for (unsigned i=0; i<next->I.size(); ++i)manager.get_fault(0).get_properties(i)[ip]=next->I[i];
      compute_fault_surface_temperatures();
    }
#define INSTANTIATE(dim) \
  template void PhaseFieldFault<dim>::stage_material_geometry(const MaterialHistoryInputs &,LinearAlgebra::BlockVector &,LinearAlgebra::BlockVector &);
    ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
  }
}
