/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/material_model/phase_field_fault/fe_strain_internal.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

namespace aspect
{
  namespace MaterialModel
  {
    template <int dim>
    std::shared_ptr<typename PhaseFieldFault<dim>::TensorField>
    PhaseFieldFault<dim>::project_material_tensor(
      const TensorField &prototype, const FEStrainState &profile_state,
      const LinearAlgebra::BlockVector &accepted_velocity,
      const MaterialTensorField &evaluate_tensor) const
    {
      const auto phase_component = this->introspection().phase_field().first_component_index;
      const QGauss<dim> quadrature(std::max(8u, fe_bulk_normal_quadrature_points));
      TimerOutput::Scope timer(this->get_computing_timer(), "Material tensor projection");

      // Material sampling belongs here: the numerical mass projector receives
      // only evaluated tensor values and never discovers a phase/velocity field.
      FEValues<dim> native(this->get_mapping(), this->get_fe(), quadrature,
                           update_values | update_gradients);
      std::vector<Tensor<2, dim>> gradients(quadrature.size());
      std::vector<double> phase(quadrature.size());
      const typename TensorField::CellEvaluator evaluate_cell =
        [&](const CellId & cell_id,
            const std::vector<types::global_dof_index> &indices,
            const FEValues<dim> &target_values,
            std::vector<SymmetricTensor<2, dim>> &values)
      {
        native.reinit(profile_state.cells.at(cell_id));
        native[this->introspection().extractors.velocities]
        .get_function_gradients(accepted_velocity, gradients);
        native[FEValuesExtractors::Scalar(phase_component)]
        .get_function_values(profile_state.bulk_history.native, phase);
        for (unsigned int q = 0; q < quadrature.size(); ++q)
          values[q] = evaluate_tensor(cell_id, q, indices, quadrature.point(q),
                                      target_values.quadrature_point(q), phase[q], gradients[q]);
      };
      return TensorField::project(*this, prototype, profile_state.cells,
                                  quadrature, evaluate_cell);
    }

#define INSTANTIATE(dim) \
  template std::shared_ptr<PhaseFieldFault<dim>::TensorField> \
  PhaseFieldFault<dim>::project_material_tensor(const TensorField &, const FEStrainState &, const LinearAlgebra::BlockVector &, const MaterialTensorField &) const;
    ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
  }
}
