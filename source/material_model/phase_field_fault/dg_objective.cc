/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/material_model/phase_field_fault/fe_strain_internal.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/material_model/phase_field_fault/tensor_history_internal.h>
#include <aspect/reconstructed_fault/objective_stress.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

namespace aspect
{
  namespace MaterialModel
  {
    template <int dim>
    SymmetricTensor<2, dim>
    PhaseFieldFault<dim>::localize_surface_values(
      const Point<dim> &point, const double phase, const std::vector<double> &surface_values,
      const std::vector<double> &I,
      const ReconstructedFaultUtilities::NormalProfileProjection *cached) const
    {
      const auto &manager = this->get_reconstructed_fault_manager();
      const auto source = cached ? *cached : manager.project_to_bulk_source(point);
      if (!source.active)
        return {};
      const auto &fault = manager.get_fault(source.fault_index);
      const auto j = source.segment_index;
      const double xi = source.xi;
      const double value = (1 - xi) * surface_values.at(j) + xi * surface_values.at(j + 1);
      const double normalization = (1 - xi) * I.at(j) + xi * I.at(j + 1);
      AssertThrow(std::isfinite(value) && std::isfinite(normalization) && normalization > 0.,
                  ExcMessage("Surface localization requires finite values and positive native normalization."));
      auto tangent = fault.vertex(j + 1) - fault.vertex(j);
      tangent /= tangent.norm();
      Tensor<1, dim> normal;
      if constexpr (dim == 2)
        {
          normal[0] = -tangent[1];
          normal[1] = tangent[0];
        }
      else AssertThrow(false, ExcMessage("Objective DG history is restricted to plane strain."));
      const auto S = manager.get_shear_sense(source.fault_index) * symmetrize(outer_product(tangent, normal));
      const auto localization = evaluate_reconstructed_fault_localization(
                                  source.fault_index, j, xi, phase, phase, "mature DG stress");
      return (2 * value * localization.current_h / normalization) * S;
    }

    template <int dim>
    void
    PhaseFieldFault<dim>::prepare_objective_dg_anchor()
    {
      AssertThrow(jaumann_dg_stress && fe_strain_state && fe_strain_state->bulk_history.tensor
                  && (material_direct_stress || fe_strain_state->bulk_history.tensor->residual), ExcMessage("Missing complete objective DG history."));
      auto &state = *fe_strain_state;
      const auto parent = state.bulk_history.tensor;
      const double dt = this->get_timestep_number() > 0 ? this->get_timestep() : initial_time_step;
      if (state.objective.tensor && state.objective.parent.lock() == parent
          && state.objective.step == this->get_timestep_number() && state.objective.dt == dt)
        return;
      if (material_direct_stress)
        {
          // Initialization and rollback prepare accepted observations outside a
          // material trial. A physical solve gets its rotated/advected anchor
          // from rotate_material_history; do not rotate again on restoration.
          AssertThrow(!dg_material_trial,
                      ExcMessage("Direct material trial is missing its complete objective anchor."));
          state.objective.tensor = parent;
          state.objective.parent = parent;
          state.objective.step = this->get_timestep_number() == 0 ? 0 : numbers::invalid_unsigned_int;
          state.objective.dt = dt;
          return;
        }
      TimerOutput::Scope timer(this->get_computing_timer(), "DG objective anchor");
      auto anchor = std::make_shared<DGHistory>();
      anchor->primary.space = parent->primary.space;
      anchor->residual = std::make_shared<TensorField>();
      anchor->residual->space = parent->primary.space;
      const QGauss<dim> rule(std::max(8u, fe_bulk_normal_quadrature_points));
      FEValues<dim> native(this->get_mapping(), this->get_fe(), rule,
                           update_values | update_gradients | update_quadrature_points);
      FEValues<dim> dg(this->get_mapping(), parent->primary.space->element, rule,
                       update_values | update_JxW_values);
      constexpr unsigned int components = SymmetricTensor<2, dim>::n_independent_components;
      std::array<LinearAlgebra::Vector, 2 * components> owned;
      for (auto &v : owned)
        v.reinit(parent->primary.space->dofs.locally_owned_dofs(), this->get_mpi_communicator());
      const unsigned int n = parent->primary.space->element.n_dofs_per_cell();
      std::vector<types::global_dof_index> indices(n);
      std::vector<Tensor<2, dim>> gradients(rule.size());
      std::vector<double> phase(rule.size());
      for (const auto &entry : state.cells)
        {
          native.reinit(entry.second);
          dg.reinit(parent->primary.space->cells.at(entry.first));
          parent->primary.space->cells.at(entry.first)->get_dof_indices(indices);
          native[this->introspection().extractors.velocities].get_function_gradients(this->get_old_solution(), gradients);
          native[FEValuesExtractors::Scalar(this->introspection().variable("phase_field").first_component_index)]
          .get_function_values(state.bulk_history.native, phase);
          const auto T = evaluate_history_tensor(*parent, entry.first, rule);
          const auto R = evaluate_history_tensor(*parent->residual, entry.first, rule);
          if constexpr (dim == 2)
            if (std::all_of(gradients.begin(), gradients.end(), [](const auto & a)
            {
              return a[1][0] == a[0][1];
              }))
          {
            // Preserve the exact zero-spin identity, including nonzero R.
            for (unsigned int c = 0; c < components; ++c)
              for (const auto i : indices)
                {
                  owned[c][i] = parent->primary.coefficients[c][i];
                  owned[c + components][i] = parent->residual->coefficients[c][i];
                }
            continue;
          }
          FullMatrix<double> mass(n, n);
          std::array<Vector<double>, 2 * components> rhs;
          for (auto &v : rhs)
            v.reinit(n);
          for (unsigned int q = 0; q < rule.size(); ++q)
            {
              const auto B = localize_surface_values(native.quadrature_point(q), phase[q], state.Z, state.I);
              SymmetricTensor<2, dim> rotated_T, rotated_R;
              if constexpr (dim == 2)
                {
                  rotated_T = ReconstructedFaultUtilities::rotate_plane_strain_stress(T[q], gradients[q], dt);
                  // Finite-step complete residual: not just rotation of R.
                  rotated_R = ReconstructedFaultUtilities::rotate_plane_strain_stress(R[q], gradients[q], dt)
                              + ReconstructedFaultUtilities::rotate_plane_strain_stress_increment(B, gradients[q], dt);
                }
              else AssertThrow(false, ExcMessage("Objective DG history requires 2D plane strain."));
              for (unsigned int i = 0; i < n; ++i)
                {
                  const double weight = dg.JxW(q) * dg.shape_value(i, q);
                  for (unsigned int j = 0; j < n; ++j)
                    mass(i, j) += weight * dg.shape_value(j, q);
                  for (unsigned int c = 0; c < components; ++c)
                    {
                      const auto ij = SymmetricTensor<2, dim>::unrolled_to_component_indices(c);
                      rhs[c][i] += weight * rotated_T[ij];
                      rhs[c + components][i] += weight * rotated_R[ij];
                    }
                }
            }
          mass.gauss_jordan();
          for (unsigned int c = 0; c < rhs.size(); ++c)
            {
              Vector<double> values(n);
              mass.vmult(values, rhs[c]);
              for (unsigned int i = 0; i < n; ++i)
                owned[c][indices[i]] = values[i];
            }
        }
      for (unsigned int c = 0; c < components; ++c)
        {
          owned[c].compress(VectorOperation::insert);
          owned[c + components].compress(VectorOperation::insert);
          anchor->primary.coefficients[c].reinit(parent->primary.coefficients[c]);
          anchor->primary.coefficients[c] = owned[c];
          anchor->residual->coefficients[c].reinit(parent->residual->coefficients[c]);
          anchor->residual->coefficients[c] = owned[c + components];
        }
      state.objective.tensor = std::move(anchor);
      state.objective.parent = parent;
      state.objective.step = this->get_timestep_number();
      state.objective.dt = dt;
      state.observations.stress.clear();
      state.observations.mechanics.reset();
    }

#define INSTANTIATE(dim) \
  template SymmetricTensor<2,dim> PhaseFieldFault<dim>::localize_surface_values(const Point<dim> &,double,const std::vector<double> &,const std::vector<double> &,const ReconstructedFaultUtilities::NormalProfileProjection *) const; \
  template void PhaseFieldFault<dim>::prepare_objective_dg_anchor();
    ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
  }
}
