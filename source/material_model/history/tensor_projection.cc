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
        // Project the complete supplied field, including variable coefficients.
        // Separate projection of products would change the discrete material law.
        // Each owned cell builds the same local mass/RHS in the retained order.
        const auto &space = *incoming.space;
        const auto comm = context.get_mpi_communicator();
        auto result = std::make_shared<TensorField>();
        result->space = incoming.space;
        constexpr unsigned int components = SymmetricTensor<2, dim>::n_independent_components;
        std::array<LinearAlgebra::Vector, components> owned;
        for (auto &v : owned)
          v.reinit(space.dofs.locally_owned_dofs(), comm);
        FEValues<dim> dg(context.get_mapping(), space.element, rule,
                         update_values | update_JxW_values
                         | update_quadrature_points);
        std::vector<SymmetricTensor<2, dim>> values(rule.size());
        std::vector<types::global_dof_index> ids(9);
        for (const auto &entry : cells)
          {
            dg.reinit(space.cells.at(entry.first));
            space.cells.at(entry.first)->get_dof_indices(ids);
            evaluate_cell(entry.first, ids, dg, values);
            FullMatrix<double> mass(9, 9);
            std::array<Vector<double>, components> rhs;
            for (auto &v : rhs)
              v.reinit(9);
            for (unsigned q = 0; q < rule.size(); ++q)
              {
                const auto &value = values[q];
                for (unsigned i = 0; i < 9; ++i)
                  {
                    const double w = dg.JxW(q) * dg.shape_value(i, q);
                    for (unsigned j = 0; j < 9; ++j)
                      mass(i, j) += w * dg.shape_value(j, q);
                    for (unsigned c = 0; c < rhs.size(); ++c)
                      rhs[c][i] += w * value[SymmetricTensor<2, dim>::unrolled_to_component_indices(c)];
                  }
              }
            mass.gauss_jordan();
            for (unsigned c = 0; c < rhs.size(); ++c)
              {
                Vector<double> v(9);
                mass.vmult(v, rhs[c]);
                for (unsigned i = 0; i < 9; ++i)
                  owned[c][ids[i]] = v[i];
              }
          }
        for (unsigned c = 0; c < owned.size(); ++c)
          {
            owned[c].compress(VectorOperation::insert);
            result->coefficients[c].reinit(space.dofs.locally_owned_dofs(),
                                           space.relevant, comm);
            result->coefficients[c] = owned[c];
          }
        return result;
      }
#define INSTANTIATE(dim) \
  template std::shared_ptr<TensorField<dim>> TensorField<dim>::project(const SimulatorAccess<dim> &,const TensorField &,const Cells &,const Quadrature<dim> &,const CellEvaluator &);
      ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
    }
  }
}
