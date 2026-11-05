/* Copyright(C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/material_model/history/tensor_field.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
namespace aspect
{
  namespace MaterialModel
  {
    namespace HistoryDiscretization
    {
      template <int dim>
      std::shared_ptr<TensorField<dim>>
      TensorField<dim>::project(
        const SimulatorAccess<dim> &context, const TensorField &incoming,
        const Cells &cells, const Quadrature<dim> &rule,
        const CellEvaluator &evaluate_cell)
      {
        return project(context, incoming, cells, rule, 1,
                       [&](const CellId &id,
                           const std::vector<types::global_dof_index> &indices,
                           const FEValues<dim> &target,
                           std::vector<std::vector<SymmetricTensor<2, dim>>> &values)
        {
          evaluate_cell(id, indices, target, values[0]);
        }).front();
      }

      template <int dim>
      std::vector<std::shared_ptr<TensorField<dim>>>
      TensorField<dim>::project(
        const SimulatorAccess<dim> &context, const TensorField &incoming,
        const Cells &cells, const Quadrature<dim> &rule,
        const unsigned int n_fields, const MultiCellEvaluator &evaluate_cell)
      {
        AssertThrow(n_fields > 0, ExcMessage("Tensor projection needs at least one field."));
        // Project the complete supplied field, including variable coefficients.
        // Separate projection of products would change the discrete material law.
        // Each owned cell builds the same local mass/RHS in the retained order.
        const auto &space = *incoming.space;
        const auto comm = context.get_mpi_communicator();
        std::vector<std::shared_ptr<TensorField>> result(n_fields);
        for (auto &field : result)
          {
            field = std::make_shared<TensorField>();
            field->space = incoming.space;
          }
        const unsigned int n = space.element.n_dofs_per_cell();
        constexpr unsigned int components = SymmetricTensor<2, dim>::n_independent_components;
        std::vector<LinearAlgebra::Vector> owned(n_fields * components);
        for (auto &v : owned)
          v.reinit(space.dofs.locally_owned_dofs(), comm);
        FEValues<dim> dg(context.get_mapping(), space.element, rule,
                         update_values | update_JxW_values
                         | update_quadrature_points);
        std::vector<std::vector<SymmetricTensor<2, dim>>> values(
          n_fields, std::vector<SymmetricTensor<2, dim>>(rule.size()));
        std::vector<types::global_dof_index> ids(n);
        for (const auto &entry : cells)
          {
            dg.reinit(space.cells.at(entry.first));
            space.cells.at(entry.first)->get_dof_indices(ids);
            evaluate_cell(entry.first, ids, dg, values);
            FullMatrix<double> mass(n, n);
            std::vector<Vector<double>> rhs(n_fields * components);
            for (auto &v : rhs)
              v.reinit(n);
            for (unsigned q = 0; q < rule.size(); ++q)
              {
                for (unsigned i = 0; i < n; ++i)
                  {
                    const double w = dg.JxW(q) * dg.shape_value(i, q);
                    for (unsigned j = 0; j < n; ++j)
                      mass(i, j) += w * dg.shape_value(j, q);
                    for (unsigned c = 0; c < rhs.size(); ++c)
                      rhs[c][i] += w * values[c / components][q][SymmetricTensor<2, dim>::unrolled_to_component_indices(c % components)];
                  }
              }
            mass.gauss_jordan();
            for (unsigned c = 0; c < rhs.size(); ++c)
              {
                Vector<double> v(n);
                mass.vmult(v, rhs[c]);
                for (unsigned i = 0; i < n; ++i)
                  owned[c][ids[i]] = v[i];
              }
          }
        for (unsigned c = 0; c < owned.size(); ++c)
          {
            owned[c].compress(VectorOperation::insert);
            result[c / components]->coefficients[c % components].reinit(space.dofs.locally_owned_dofs(),
                                                                        space.relevant, comm);
            result[c / components]->coefficients[c % components] = owned[c];
          }
        return result;
      }
#define INSTANTIATE(dim) \
  template std::shared_ptr<TensorField<dim>> TensorField<dim>::project(const SimulatorAccess<dim> &,const TensorField &,const Cells &,const Quadrature<dim> &,const CellEvaluator &); \
  template std::vector<std::shared_ptr<TensorField<dim>>> TensorField<dim>::project(const SimulatorAccess<dim> &,const TensorField &,const Cells &,const Quadrature<dim> &,unsigned int,const MultiCellEvaluator &);
      ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
    }
  }
}
