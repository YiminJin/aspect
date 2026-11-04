/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/material_model/phase_field_fault/dg_history_internal.h>
#include <aspect/material_model/phase_field_fault/fe_strain_internal.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
namespace aspect
{
namespace MaterialModel
{
template <int dim>
std::shared_ptr<typename PhaseFieldFault<dim>::DGHistory>
PhaseFieldFault<dim>::project_material_tensor (
    const DGHistory &incoming, const DGMaterialField &field) const
{
  const auto &space = *incoming.space;
  const auto comm = this->get_mpi_communicator ();
  const auto phase_component
      = this->introspection ().variable ("phase_field").first_component_index;
  const QGauss<dim> rule (std::max (8u, fe_bulk_normal_quadrature_points));

  TimerOutput::Scope timer (this->get_computing_timer (),
                            "Material tensor projection");
  auto result = std::make_shared<DGHistory> ();
  result->space = incoming.space;
  result->representation = incoming.representation;
  std::array<LinearAlgebra::Vector,
             SymmetricTensor<2, dim>::n_independent_components>
      owned;
  for (auto &v : owned)
    v.reinit (space.dofs.locally_owned_dofs (), comm);
  FEValues<dim> dg (this->get_mapping (), space.element, rule,
                    update_values | update_JxW_values
                        | update_quadrature_points);
  FEValues<dim> native (this->get_mapping (), this->get_fe (), rule,
                        update_values | update_gradients);
  std::vector<Tensor<2, dim>> gradients (rule.size ());
  std::vector<double> phase (rule.size ());
  std::vector<types::global_dof_index> ids (9);
  for (const auto &entry : fe_strain_state->cells)
    {
      native.reinit (entry.second);
      dg.reinit (space.cells.at (entry.first));
      space.cells.at (entry.first)->get_dof_indices (ids);
      native[this->introspection ().extractors.velocities]
          .get_function_gradients (this->get_old_solution (), gradients);
      native[FEValuesExtractors::Scalar (phase_component)]
          .get_function_values (fe_strain_state->displacement_history, phase);
      FullMatrix<double> mass (9, 9);
      std::array<Vector<double>,
                 SymmetricTensor<2, dim>::n_independent_components>
          rhs;
      for (auto &v : rhs)
        v.reinit (9);
      for (unsigned q = 0; q < rule.size (); ++q)
        {
          const auto value
              = field (ids, rule.point (q), dg.quadrature_point (q), phase[q],
                       gradients[q]);
          for (unsigned i = 0; i < 9; ++i)
            {
              const double w = dg.JxW (q) * dg.shape_value (i, q);
              for (unsigned j = 0; j < 9; ++j)
                mass (i, j) += w * dg.shape_value (j, q);
              for (unsigned c = 0; c < rhs.size (); ++c)
                rhs[c][i] += w
                             * value[SymmetricTensor<
                                 2, dim>::unrolled_to_component_indices (c)];
            }
        }
      mass.gauss_jordan ();
      for (unsigned c = 0; c < rhs.size (); ++c)
        {
          Vector<double> v (9);
          mass.vmult (v, rhs[c]);
          for (unsigned i = 0; i < 9; ++i)
            owned[c][ids[i]] = v[i];
        }
    }
  for (unsigned c = 0; c < owned.size (); ++c)
    {
      owned[c].compress (VectorOperation::insert);
      result->trial_stress[c].reinit (space.dofs.locally_owned_dofs (),
                                      space.relevant, comm);
      result->trial_stress[c] = owned[c];
    }
  return result;
}
#define INSTANTIATE(dim)                                                      \
  template std::shared_ptr<PhaseFieldFault<dim>::DGHistory>                   \
  PhaseFieldFault<dim>::project_material_tensor (                             \
      const DGHistory &, const DGMaterialField &) const;
ASPECT_INSTANTIATE (INSTANTIATE)
#undef INSTANTIATE
}
}
