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
    void
    PhaseFieldFault<dim>::set_material_exterior_stress(
      const std::string &identity, const ExteriorStress &provider)
    {
      AssertThrow(material_prescribed_inflow && material_direct_stress
                  && !dg_material_trial && provider && identity == material_exterior_identity,
                  ExcMessage("Exterior stress provider requires the selected direct-stress inflow, matching identity, and no live material trial."));
      material_exterior_stress = provider;
    }

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
          if (this->get_phase_field_handler().has_associated_particle_manager())
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
    std::shared_ptr<const typename PhaseFieldFault<dim>::AcceptedSpinSamples>
    PhaseFieldFault<dim>::sample_accepted_bulk_spin(
      const FEStrainState &accepted, const LinearAlgebra::BlockVector &velocity, const double dt) const
    {
      const auto &manager = this->get_reconstructed_fault_manager();
      AssertThrow(mature_frictional_fault && manager.get_faults().size()==1
                  && manager.slip_rates_are_initialized() && this->get_timestep_number()>0,
                  ExcMessage("Bulk spin requires one mature fault and the converged initialized committed rates."));
      auto samples = std::make_shared<AcceptedSpinSamples>();
      samples->parent = accepted.bulk_history.tensor;
      samples->velocity = &velocity;
      samples->step = this->get_timestep_number();
      samples->geometry_version = manager.get_fault(0).geometry_version();
      samples->quadrature_order = std::max(8u,fe_bulk_normal_quadrature_points);
      samples->committed_rates = manager.get_timestep_committed_slip_rate(0);
      const QGauss<dim> rule(samples->quadrature_order);
      FEValues<dim> values(this->get_mapping(),this->get_fe(),rule,
                           update_values|update_gradients|update_quadrature_points|update_JxW_values);
      std::vector<Tensor<2,dim>> gradients(rule.size());
      std::vector<double> phase(rule.size());
      for (const auto &cell:accepted.cells)
        {
          values.reinit(cell.second);
          values[this->introspection().extractors.velocities].get_function_gradients(velocity,gradients);
          values[FEValuesExtractors::Scalar(this->introspection().phase_field().first_component_index)]
          .get_function_values(accepted.bulk_history.native,phase);
          auto &rates=samples->angular_rates[cell.first];
          rates.resize(rule.size());
          for (unsigned int q=0; q<rule.size(); ++q)
            {
              const auto sample=sample_mature_localization(manager,values.quadrature_point(q),phase[q],accepted.I);
              const auto slip=sample.slip_gradient(samples->committed_rates);
              if constexpr (dim==2)
                {
                  const double total=ReconstructedFaultUtilities::plane_strain_angular_rate(gradients[q]);
                  const double fault=ReconstructedFaultUtilities::plane_strain_angular_rate(slip);
                  // Subtract native pointwise full gradients before rotation.
                  rates[q]=ReconstructedFaultUtilities::plane_strain_angular_rate(gradients[q]-slip);
                  const std::array<double,3> spin= {{total,fault,rates[q]}};
                  for (unsigned int k=0; k<3; ++k)
                    {
                      samples->maximum[k]=std::max(samples->maximum[k],std::abs(spin[k]));
                      samples->l2[k]+=values.JxW(q)*spin[k]*spin[k];
                    }
                }
              else AssertThrow(false,ExcMessage("Bulk spin requires plane strain."));
            }
        }
      for (unsigned int k=0; k<3; ++k)
        {
          samples->maximum[k]=Utilities::MPI::max(samples->maximum[k],this->get_mpi_communicator());
          samples->l2[k]=std::sqrt(Utilities::MPI::sum(samples->l2[k],this->get_mpi_communicator()));
        }
      this->get_pcout()<<std::setprecision(17)<<"   Accepted bulk spin: step="<<samples->step
                       <<" total_max="<<samples->maximum[0]<<" fault_max="<<samples->maximum[1]<<" bulk_max="<<samples->maximum[2]
                       <<" total_l2="<<samples->l2[0]<<" fault_l2="<<samples->l2[1]<<" bulk_l2="<<samples->l2[2]
                       <<" dt_bulk_max="<<dt *samples->maximum[2]<<std::endl;
      return samples;
    }

    template <int dim>
    typename PhaseFieldFault<dim>::MaterialHistoryInputs
    PhaseFieldFault<dim>::prepare_material_history(
      const LinearAlgebra::BlockVector &accepted_velocity, const double time, const double dt)
    {
      AssertThrow(!(legacy_bounded_profile_identity || material_trace_inflow || material_prescribed_inflow)
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
      AssertThrow(!material_prescribed_inflow || material_exterior_stress,
                  ExcMessage("Prescribed exterior stress provider is missing; attach it before material preparation, including after restart."));
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
      if (constitutive_spin==ConstitutiveSpin::bulk_slip_subtracted)
        transaction->accepted_spin=sample_accepted_bulk_spin(*transaction->accepted_history,accepted_velocity,dt);
      if (this->get_phase_field_handler().has_associated_particle_manager())
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
                                   velocity, order, relax ? fe_strain_state->G / fe_strain_state->eta : 0.);
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
  template std::shared_ptr<const PhaseFieldFault<dim>::AcceptedSpinSamples> PhaseFieldFault<dim>::sample_accepted_bulk_spin(const FEStrainState &,const LinearAlgebra::BlockVector &,double) const; \
  template void PhaseFieldFault<dim>::set_material_exterior_stress(const std::string &,const ExteriorStress &); \
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
