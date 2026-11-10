/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/material_model/history/tensor_advection.h>
#include <aspect/utilities.h>
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
      TensorAdvection<dim>::project(
        const SimulatorAccess<dim> &context, const TensorAdvection &op, const Field &field, const double time, const unsigned int quadrature_order)
      {
        AssertThrow(quadrature_order >= 3, ExcMessage("DGQ2 field projection requires adequate integration."));
        auto result = std::make_shared<History>();
        result->space = op.space;
        const auto &space = *op.space;
        for (auto &component : result->coefficients)
          component.reinit(space.dofs.locally_owned_dofs(), space.relevant, context.get_mpi_communicator());
        const QGauss<dim> quadrature(quadrature_order);
        FEValues<dim> fe(context.get_mapping(), space.element, quadrature, update_values | update_quadrature_points | update_JxW_values);
        const unsigned int n = space.element.n_dofs_per_cell();
        for (const auto &cell : op.cells)
          {
            fe.reinit(space.cells.at(cell.id));
            std::array<Vector<double>, SymmetricTensor<2, dim>::n_independent_components> rhs;
            for (auto &component : rhs)
              component.reinit(n);
            for (unsigned int q = 0; q < quadrature.size(); ++q)
              {
                const auto value = field(fe.quadrature_point(q), time);
                AssertThrow(std::isfinite(value.norm()), ExcMessage("Nonfinite projected DG history."));
                for (unsigned int c = 0; c < rhs.size(); ++c)
                  for (unsigned int i = 0; i < n; ++i)
                    rhs[c][i] += fe.JxW(q) * fe.shape_value(i, q) * value[SymmetricTensor<2, dim>::unrolled_to_component_indices(c)];
              }
            Vector<double> values(n);
            for (unsigned int c = 0; c < rhs.size(); ++c)
              {
                cell.inverse_mass.vmult(values, rhs[c]);
                for (unsigned int i = 0; i < n; ++i)
                  result->coefficients[c][cell.indices[i]] = values[i];
              }
          }
        for (auto &component : result->coefficients)
          component.compress(VectorOperation::insert);
        for (auto &component : result->coefficients)
            {
              LinearAlgebra::Vector owned(space.dofs.locally_owned_dofs(), context.get_mpi_communicator());
              owned = component;
              component = owned;
            }
        return result;
      }


#define INSTANTIATE(dim) \
  template std::shared_ptr<TensorField<dim>> TensorAdvection<dim>::project(const SimulatorAccess<dim> &,const TensorAdvection &,const Field &,double,unsigned int);
      ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
    }
  }
}
