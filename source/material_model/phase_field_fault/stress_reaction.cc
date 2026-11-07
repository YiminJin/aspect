/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/material_model/phase_field_fault/material_history_internal.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/material_model/phase_field_fault/tensor_history_internal.h>
#include <aspect/material_model/phase_field_fault/stress_law.h>
#include <cstdlib>
#include <iomanip>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

namespace aspect
{
  namespace MaterialModel
  {
    template <int dim>
    std::shared_ptr<typename PhaseFieldFault<dim>::FEStrainUpdate>
    PhaseFieldFault<dim>::make_material_reaction(const FEStrainState &state, const LinearAlgebra::BlockVector &bulk,
                                                 const std::vector<double> &rates, const bool publication_candidate) const
    {
      AssertThrow(dg_material_transport && state.objective.tensor,
                  ExcMessage("Projected material reaction requires its complete transported anchor."));
      TimerOutput::Scope timer(this->get_computing_timer(), "Material reaction projection");
      const double dt = this->get_timestep_number() > 0 ? this->get_timestep() : initial_time_step;
      auto update = std::make_shared<FEStrainUpdate>();
      update->bulk_history.native = state.bulk_history.native;
      update->I = state.I;
      update->Z.resize(state.Z.size());
      AssertDimension(rates.size(), state.Z.size());
      for (unsigned j = 0; j < state.Z.size(); ++j)
        {
          const auto coefficients = compute_maxwell_coefficients(material_viscosity(state.vertices[j], state.eta), state.G, dt);
          update->Z[j] = coefficients.beta * state.Z[j] + coefficients.eta_ve * rates[j];
        }
      // The objective anchor is fixed throughout the coupled solve. In direct
      // mode the law is tau+=beta*tau_bar+2*kappa*D_3-kappa*(2*L[V]);
      // eta_ve is kappa (Pa s). Z still advances for diagnostics, but neither B0
      // nor B1 enters mechanics. The separated control retains both T and R.
      const MatureReactionField reaction_field = [&](const Point<dim> &x, const double phase,
                                                     const SymmetricTensor<2, dim> &strain, const SymmetricTensor<2, dim> &anchor_value,
                                                     const SymmetricTensor<2, dim> &separated_residual)
      {
        const auto coefficients = compute_maxwell_coefficients(material_viscosity(x, state.eta), state.G, dt);
        const auto B0 = material_direct_stress ? SymmetricTensor<2, dim>() : localize_surface_values(x, phase, state.Z, state.I);
        const auto B1 = material_direct_stress ? SymmetricTensor<2, dim>() : localize_surface_values(x, phase, update->Z, update->I);
        const auto twice_localized_slip_rate = localize_surface_values(x, phase, rates, state.I);
        if (material_direct_stress)
          return std::array<SymmetricTensor<2, dim>, 2>
          {
            {
              MaterialModel::Rheology::mature_stress_endpoint(coefficients.beta, coefficients.eta_ve, anchor_value, strain, twice_localized_slip_rate),
              SymmetricTensor<2, dim>()
            }
          };
        return std::array<SymmetricTensor<2, dim>, 2>
        {
          {
            coefficients.beta *anchor_value + 2 * coefficients.eta_ve *Utilities::Tensors::consistent_deviator(strain),
            coefficients.beta *(separated_residual + B0) + coefficients.eta_ve *twice_localized_slip_rate - B1
          }
        };
      };
      project_mature_reaction(state, bulk, *update, reaction_field, publication_candidate);
      return update;
    }

    template <int dim>
    void
    PhaseFieldFault<dim>::project_mature_reaction(
      const FEStrainState &state, const LinearAlgebra::BlockVector &bulk,
      FEStrainUpdate &update, const MatureReactionField &field, const bool publication_candidate) const
    {
      // Project the whole endpoint field with one mass matrix per cell. For
      // variable viscosity, projecting beta and kappa separately would define a
      // different discrete law. Only the candidate receives the resulting fields.
      const auto &anchor = *state.objective.tensor;
      const auto &space = *anchor.primary.space;
      const auto comm = this->get_mpi_communicator();
      const double dt = this->get_timestep_number() > 0 ? this->get_timestep() : initial_time_step;
      const unsigned int tensors = material_direct_stress ? 1 : 2;
      const bool energy_ledger = material_direct_stress && std::getenv("ASPECT_TRUE_STRESS_ENERGY");
      double local_energy = 0.;
      const QGauss<dim> rule(std::max(8u, fe_bulk_normal_quadrature_points));
      FEValues<dim> native(this->get_mapping(), this->get_fe(), rule, update_gradients | update_values | update_quadrature_points);
      FEValues<dim> dg(this->get_mapping(), space.element, rule, update_values | update_JxW_values);
      std::vector<SymmetricTensor<2, dim>> strain(rule.size());
      std::vector<double> phase(rule.size());
      const auto projected = TensorField::project(*this, anchor.primary, state.cells, rule, tensors,
                                                  [&](const CellId &id, const std::vector<types::global_dof_index> &,
                                                      const FEValues<dim> &target,
                                                      std::vector<std::vector<SymmetricTensor<2, dim>>> &values)
      {
        native.reinit(state.cells.at(id));
        native[this->introspection().extractors.velocities].get_function_symmetric_gradients(bulk, strain);
        native[FEValuesExtractors::Scalar(this->introspection().phase_field().first_component_index)].get_function_values(state.bulk_history.native, phase);
        const auto anchor_values = evaluate_history_tensor(anchor, id, rule);
        const auto residual_values = anchor.residual ? evaluate_history_tensor(*anchor.residual, id, rule)
                                     : std::vector<SymmetricTensor<2, dim>>(rule.size());
        for (unsigned int q = 0; q < rule.size(); ++q)
          {
            const auto endpoint_values = field(native.quadrature_point(q), phase[q], strain[q], anchor_values[q], residual_values[q]);
            for (unsigned int k = 0; k < tensors; ++k)
              values[k][q] = endpoint_values[k];
            if (energy_ledger)
              local_energy += target.JxW(q) * (scalar_product(endpoint_values[0], endpoint_values[0]) + trace(endpoint_values[0]) * trace(endpoint_values[0])) / (4 * state.G);
          }
      });
      update.bulk_history.tensor = std::make_shared<DGHistory>();
      update.bulk_history.tensor->representation = anchor.representation;
      update.bulk_history.tensor->primary = *projected[0];
      if (!material_direct_stress)
        update.bulk_history.tensor->residual = projected[1];
      if (energy_ledger)
        {
          double projected_energy = 0.;
          for (const auto &entry : state.cells)
            {
              dg.reinit(space.cells.at(entry.first));
              const auto values = evaluate_history_tensor(*update.bulk_history.tensor, entry.first, rule);
              for (unsigned q = 0; q < rule.size(); ++q)
                projected_energy += dg.JxW(q) * (scalar_product(values[q], values[q]) + trace(values[q]) * trace(values[q])) / (4 * state.G);
            }
          local_energy = Utilities::MPI::sum(local_energy, comm);
          projected_energy = Utilities::MPI::sum(projected_energy, comm);
          this->get_pcout() << std::setprecision(17) << "   Direct source energy: step=" << this->get_timestep_number()
                            << " dt=" << dt << " publication=" << publication_candidate
                            << " local=" << local_energy << " projected=" << projected_energy
                            << " loss=" << local_energy - projected_energy << std::endl;
        }
    }

    template <int dim>
    SymmetricTensor<2, dim>
    PhaseFieldFault<dim>::material_reaction_correction(
      const CellId &id, const Point<dim> &x) const
    {
      return evaluate_material_projection_correction(dg_material_reaction.get(), *fe_strain_state, id, x);
    }

    template <int dim>
    SymmetricTensor<2, dim>
    PhaseFieldFault<dim>::evaluate_material_projection_correction(
      const DGMaterialReaction *correction, const FEStrainState &state, const CellId &id, const Point<dim> &x) const
    {
      if (!correction)
        return {};
      const auto &r = *correction;
      const auto cell = state.evaluation_cells.at(id);
      const auto unit = this->get_mapping().transform_real_to_unit_cell(cell, x);
      FEValues<dim> fe(this->get_mapping(), this->get_fe(), Quadrature<dim>({unit}), update_gradients | update_values);
      fe.reinit(cell);
      std::vector<SymmetricTensor<2, dim>> strain(1);
      std::vector<double> phi(1);
      fe[this->introspection().extractors.velocities].get_function_symmetric_gradients(r.bulk, strain);
      fe[FEValuesExtractors::Scalar(this->introspection().phase_field().first_component_index)].get_function_values(state.bulk_history.native, phi);
      const double dt = this->get_timestep_number() > 0 ? this->get_timestep() : initial_time_step;
      const auto coefficients = compute_maxwell_coefficients(material_viscosity(x, state.eta), state.G, dt);
      const auto physical = physical_history_stress(state, *r.candidate->bulk_history.tensor, r.candidate->Z, state.I, id, x);
      const auto anchor = physical_history_stress(state, *state.objective.tensor, state.Z, state.I, id, x);
      const auto direct = MaterialModel::Rheology::mature_stress_endpoint(
                            coefficients.beta, coefficients.eta_ve, anchor, strain[0], localize_surface_values(x, phi[0], r.rates, state.I));
      // Mechanics needs the projected endpoint, while its local tangent uses the
      // point law. This outer-iteration correction supplies exactly that defect.
      return physical - direct;
    }

    template <int dim>
    double
    PhaseFieldFault<dim>::refresh_material_reaction(
      const LinearAlgebra::BlockVector &bulk, const std::vector<double> &rates)
    {
      auto next = std::make_shared<DGMaterialReaction>();
      next->bulk = bulk;
      next->rates = rates;
      next->candidate = make_material_reaction(*fe_strain_state, bulk, rates, false);
      auto previous = dg_material_reaction;
      double difference = 0., scale = 1.;
      const QGauss<dim> rule(std::max(8u, fe_bulk_normal_quadrature_points));
      FEValues<dim> fe(this->get_mapping(), this->get_fe(), rule, update_quadrature_points);
      for (const auto &entry : fe_strain_state->cells)
        {
          fe.reinit(entry.second);
          for (const auto &x : fe.get_quadrature_points())
            {
              const auto before = evaluate_material_projection_correction(previous.get(), *fe_strain_state, entry.first, x);
              const auto after = evaluate_material_projection_correction(next.get(), *fe_strain_state, entry.first, x);
              difference = std::max(difference, (after - before).norm());
              scale = std::max({scale, before.norm(), after.norm(), evaluate_dg_history_stress(*next->candidate->bulk_history.tensor, next->candidate->Z, fe_strain_state->I, entry.first, x).norm()});
            }
        }
      // Install only the correction iterate. Acceptance and physical-history
      // publication remain with the ordinary endpoint transaction.
      dg_material_reaction = next;
      fe_strain_state->observations.stress.clear();
      fe_strain_state->observations.mechanics.reset();
      difference = Utilities::MPI::max(difference, this->get_mpi_communicator());
      scale = Utilities::MPI::max(scale, this->get_mpi_communicator());
      return difference / scale;
    }

#define INSTANTIATE(dim) \
  template void PhaseFieldFault<dim>::project_mature_reaction(const FEStrainState &,const LinearAlgebra::BlockVector &,FEStrainUpdate &,const MatureReactionField &,bool) const; \
  template SymmetricTensor<2,dim> PhaseFieldFault<dim>::evaluate_material_projection_correction(const DGMaterialReaction *,const FEStrainState &,const CellId &,const Point<dim> &) const; \
  template std::shared_ptr<PhaseFieldFault<dim>::FEStrainUpdate> PhaseFieldFault<dim>::make_material_reaction(const FEStrainState &,const LinearAlgebra::BlockVector &,const std::vector<double> &,bool) const; \
  template SymmetricTensor<2,dim> PhaseFieldFault<dim>::material_reaction_correction(const CellId &,const Point<dim> &) const; \
  template double PhaseFieldFault<dim>::refresh_material_reaction(const LinearAlgebra::BlockVector &,const std::vector<double> &);
    ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
  }
}
