/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/material_model/history/tensor_history.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
namespace aspect
{
  namespace MaterialModel
  {
    namespace HistoryDiscretization
    {
      template <int dim>
      std::shared_ptr<TensorHistory<dim>>
      TensorHistory<dim>::project (
        const SimulatorAccess<dim> &context, const TensorHistory &incoming, const Cells &cells,
        const LinearAlgebra::BlockVector &profile, const LinearAlgebra::BlockVector &accepted_velocity,
        const unsigned int order, const MaterialField &field)
      {
        const auto &space = *incoming.space;
        const auto comm = context.get_mpi_communicator ();
        const auto phase_component
          = context.introspection ().variable ("phase_field").first_component_index;
        const QGauss<dim> rule (order);

        TimerOutput::Scope timer (context.get_computing_timer (),
                                  "Material tensor projection");
        auto result = std::make_shared<TensorHistory> ();
        result->space = incoming.space;
        result->representation = incoming.representation;
        std::array<LinearAlgebra::Vector,
            SymmetricTensor<2, dim>::n_independent_components>
            owned;
        for (auto &v : owned)
          v.reinit (space.dofs.locally_owned_dofs (), comm);
        FEValues<dim> dg (context.get_mapping (), space.element, rule,
                          update_values | update_JxW_values
                          | update_quadrature_points);
        FEValues<dim> native (context.get_mapping (), context.get_fe (), rule,
                              update_values | update_gradients);
        std::vector<Tensor<2, dim>> gradients (rule.size ());
        std::vector<double> phase (rule.size ());
        std::vector<types::global_dof_index> ids (9);
        for (const auto &entry : cells)
          {
            native.reinit (entry.second);
            dg.reinit (space.cells.at (entry.first));
            space.cells.at (entry.first)->get_dof_indices (ids);
            native[context.introspection ().extractors.velocities]
            .get_function_gradients (accepted_velocity, gradients);
            native[FEValuesExtractors::Scalar (phase_component)]
            .get_function_values (profile, phase);
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
                                   * value[SymmetricTensor <
                                           2, dim >::unrolled_to_component_indices (c)];
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
            result->coefficients[c].reinit (space.dofs.locally_owned_dofs (),
                                            space.relevant, comm);
            result->coefficients[c] = owned[c];
          }
        return result;
      }
#define INSTANTIATE(dim) \
  template std::shared_ptr<TensorHistory<dim>> TensorHistory<dim>::project(const SimulatorAccess<dim> &,const TensorHistory &,const Cells &,const LinearAlgebra::BlockVector &,const LinearAlgebra::BlockVector &,unsigned int,const MaterialField &);
      ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
    }
  }
}
