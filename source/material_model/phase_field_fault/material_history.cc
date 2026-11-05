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
    template <int dim>
    double
    PhaseFieldFault<dim>::material_viscosity(
      const Point<dim> &x, const double reference) const
    {
      if (!dg_material_transport)
        return reference;
      const double multiplier = material_viscosity_multiplier.value(x);
      AssertThrow(std::isfinite(multiplier) && multiplier > 0.,
                  ExcMessage("Material viscosity multiplier must be finite and positive."));
      return reference * multiplier;
    }

    template <int dim>
    void
    PhaseFieldFault<dim>::finish_dg_material_step(const bool accept) noexcept
    {
      // The common endpoint owner has already validated/published on success.
      // On failure, restore every bound geometry/history view before dropping
      // the transaction. Clearing scratch alone would leave candidate mechanics.
      dg_material_reaction.reset();
      material_reaction_solving = false;
      if (!dg_material_trial)
        return;
      auto &trial = *dg_material_trial;
      if (!accept && trial.bound)
        {
          this->get_reconstructed_fault_manager().swap_prescribed_geometry(*trial.manager);
          fe_strain_state = trial.accepted_history;
          invalidate_normalization_cache();
          current_normalization_integrals.swap(trial.normalization);
          current_fault_surface_temperatures.swap(trial.temperatures);
          for (auto &p : this->get_phase_field_handler().get_associated_particle_manager().get_particle_handler())
            std::copy(trial.particle_properties.at(p.get_id()).begin(),
            trial.particle_properties.at(p.get_id()).end(), p.get_properties().begin());
          fe_strain_state->observations.stress.clear();
          fe_strain_state->observations.mechanics.reset();
        }
      dg_material_trial.reset();
    }

    template <int dim>
    double
    PhaseFieldFault<dim>::material_surface_viscosity(
      unsigned int f, unsigned int j, double xi, double reference) const
    {
      if (!dg_material_transport)
        return reference;
      const auto &fault = this->get_reconstructed_fault_manager().get_fault(f);
      return material_viscosity(fault.vertex(j) + xi * (fault.vertex(j + 1) - fault.vertex(j)), reference);
    }

    template <int dim>
    typename PhaseFieldFault<dim>::MaterialHistoryInputs
    PhaseFieldFault<dim>::prepare_material_history(
      const LinearAlgebra::BlockVector &accepted_velocity, const double time, const double dt)
    {
      AssertThrow(!(material_bounded_profile || material_trace_inflow)
                  || this->get_reconstructed_fault_manager().uses_full_normal_cross_section(),
                  ExcMessage("The selected material profile/inflow extension requires Association support = full normal cross-section. "
                             "Stationary seed support can clip the independently supplied/transported Q1 profile while normalization includes its tails."));
      AssertThrow(!dg_material_trial && mature_frictional_fault && !evolve_phase_field,
                  ExcMessage("Material DG requires one growth-disabled accepted transaction."));
      if (restore_frozen_normalization_after_restart)
        {
          // Validate the loaded accepted geometry/profile before constructing
          // its first moving trial. The stationary restart guard must never see
          // the already transported profile as if it were checkpoint data.
          invalidate_normalization_cache();
          compute_normalization_integrals();
        }
      validate_fe_strain_restrictions();
      AssertThrow(dt > 0. && std::isfinite(dt), ExcMessage("Invalid material step interval."));
      material_viscosity_multiplier.set_time(time);
      // Capture the accepted parent before transport or geometry binding. These
      // references and backups are the sole rollback source for this attempt.
      auto transaction = std::make_shared<DGMaterialTrial>();
      transaction->accepted_history = fe_strain_state;
      transaction->step = this->get_timestep_number();
      transaction->dt = dt;
      transaction->normalization = current_normalization_integrals;
      transaction->temperatures = current_fault_surface_temperatures;
      for (const auto &p : this->get_phase_field_handler().get_associated_particle_manager().get_particle_handler())
        transaction->particle_properties.emplace(p.get_id(), std::vector<double>(p.get_properties().begin(), p.get_properties().end()));
      dg_material_trial = transaction;
      return {*transaction->accepted_history, accepted_velocity, time, dt, transaction->step};
    }

    template <int dim>
    std::shared_ptr<typename PhaseFieldFault<dim>::DGTransport>
    PhaseFieldFault<dim>::make_dg_transport(const LinearAlgebra::BlockVector &velocity,
                                            const unsigned int order, const bool relax) const
    {
      validate_fe_strain_restrictions();
      AssertThrow(jaumann_dg_stress || (dg_tensor_history && !consistent_endpoint_phase),
                  ExcMessage("Tensor transport requires objective history or the retained PF4 control."));
      return DGTransport::assemble(*this, fe_strain_state->bulk_history.tensor->primary, fe_strain_state->cells,
                                   velocity, order, relax ? fe_strain_state->G / fe_strain_state->eta : 0., jaumann_dg_stress);
    }
    template <int dim>
    std::shared_ptr<typename PhaseFieldFault<dim>::DGHistory>
    PhaseFieldFault<dim>::project_dg_transport_field(const DGTransport &op, const DGField &field,
                                                     double time, unsigned int order) const
    {
      auto result = std::make_shared<DGHistory>();
      result->primary = *DGTransport::project(*this, op, field, time, order);
      return result;
    }
    template <int dim>
    std::shared_ptr<typename PhaseFieldFault<dim>::DGHistory>
    PhaseFieldFault<dim>::advance_dg_transport(const DGTransport &op, const DGHistory &history,
                                               const DGField &inflow, const DGField &source, double time, double dt) const
    {
      AssertThrow(!history.residual, ExcMessage("Transport one independent tensor at a time."));
      auto result = std::make_shared<DGHistory>();
      result->representation = history.representation;
      result->primary = *DGTransport::advance(*this, op, history.primary, inflow, source, time, dt);
      return result;
    }
#define INSTANTIATE(dim) \
  template double PhaseFieldFault<dim>::material_viscosity(const Point<dim> &,double) const; \
  template double PhaseFieldFault<dim>::material_surface_viscosity(unsigned int,unsigned int,double,double) const; \
  template PhaseFieldFault<dim>::MaterialHistoryInputs PhaseFieldFault<dim>::prepare_material_history(const LinearAlgebra::BlockVector &,double,double); \
  template void PhaseFieldFault<dim>::finish_dg_material_step(bool) noexcept; \
  template std::shared_ptr<PhaseFieldFault<dim>::DGTransport> PhaseFieldFault<dim>::make_dg_transport(const LinearAlgebra::BlockVector &,unsigned int,bool) const; \
  template std::shared_ptr<PhaseFieldFault<dim>::DGHistory> PhaseFieldFault<dim>::project_dg_transport_field(const DGTransport &,const DGField &,double,unsigned int) const; \
  template std::shared_ptr<PhaseFieldFault<dim>::DGHistory> PhaseFieldFault<dim>::advance_dg_transport(const DGTransport &,const DGHistory &,const DGField &,const DGField &,double,double) const;
    ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
  }
}
