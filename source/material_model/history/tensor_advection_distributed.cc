/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/material_model/history/tensor_advection.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

namespace aspect
{
  namespace MaterialModel
  {
    namespace HistoryDiscretization
    {
      template<int dim>
      std::shared_ptr<TensorAdvection<dim>>
      TensorAdvection<dim>::assemble(
        const SimulatorAccess<dim> &context, const History &history, const Cells &cells,
        const LinearAlgebra::BlockVector &velocity, const unsigned int order, const double relaxation_rate)
      {
        AssertThrow(dim == 2 && order >= 4,
                    ExcMessage("Complete DG transport requires plane-strain objective history and Gauss4 or higher."));
        auto op = std::make_shared<TensorAdvection>();
        op->space = history.space;
        op->mesh_vertices = context.get_triangulation().get_vertices();
        op->quadrature_order = order;
        op->relaxation_rate = relaxation_rate;
        const auto &space = *op->space;
        const auto &comm = context.get_mpi_communicator();
        const unsigned int n = space.element.n_dofs_per_cell();
        const QGauss<dim> rule(order);
        const QGauss<1> face_rule(order);
        FEValues<dim> fe(context.get_mapping(), context.get_fe(), rule, update_values | update_gradients);
        FEValues<dim> dg(context.get_mapping(), space.element, rule,
                         update_values | update_gradients | update_quadrature_points | update_JxW_values);
        std::vector<Tensor<1, dim>> speed(rule.size());
        std::vector<Tensor<2, dim>> gradient(rule.size());
        for (const auto &entry : cells)
          {
            const auto cell = space.cells.at(entry.first);
            typename TensorAdvection::Cell local;
            local.id = entry.first;
            local.indices.resize(n);
            cell->get_dof_indices(local.indices);
            dg.reinit(cell);
            fe.reinit(entry.second);
            fe[context.introspection().extractors.velocities].get_function_values(velocity, speed);
            fe[context.introspection().extractors.velocities].get_function_gradients(velocity, gradient);
            local.inverse_mass.reinit(n, n);
            FullMatrix<double> self(n, n);
            Tensor<1, dim> maximum_speed;
            for (unsigned int q = 0; q < rule.size(); ++q)
              for (unsigned int i = 0; i < n; ++i)
                for (unsigned int j = 0; j < n; ++j)
                  local.inverse_mass(i, j) += dg.JxW(q) * dg.shape_value(i, q) * dg.shape_value(j, q);
            local.inverse_mass.gauss_jordan();
            for (unsigned int q = 0; q < rule.size(); ++q)
              {
                AssertThrow(std::isfinite(speed[q].norm() + gradient[q].norm()), ExcMessage("Nonfinite material transport velocity."));
                Vector<double> test(n), load(n);
                for (unsigned int i = 0; i < n; ++i)
                  {
                    test[i] = dg.JxW(q) * dg.shape_value(i, q);
                    for (unsigned int j = 0; j < n; ++j)
                      self(i, j) += dg.JxW(q) * dg.shape_value(j, q)
                                    * (speed[q] * dg.shape_grad(i, q) + (trace(gradient[q]) - op->relaxation_rate) * dg.shape_value(i, q));
                  }
                local.inverse_mass.vmult(load, test);
                op->source.push_back({dg.quadrature_point(q), local.indices, std::move(load)});
                for (unsigned int d = 0; d < dim; ++d)
                  maximum_speed[d] = std::max(maximum_speed[d], std::abs(speed[q][d]));
              }
            // Cartesian face patches support both sides of a hanging face. Native
            // continuous FE velocity supplies one normal flux; only incoming tensor
            // values are taken from the neighbor's own polynomial/ghost entries.
            for (const auto f : cell->face_indices())
              {
                AssertThrow(!cell->has_periodic_neighbor(f), ExcMessage("This transport operator requires explicit physical inflow, not periodic faces."));
                std::vector<typename DoFHandler<dim>::active_cell_iterator> neighbors;
                if (cell->at_boundary(f))
                  neighbors.emplace_back();
                else if (cell->neighbor(f)->has_children())
                  for (unsigned int s = 0; s < GeometryInfo<dim>::max_children_per_face; ++s)
                    neighbors.push_back(cell->neighbor_child_on_subface(f, s));
                else neighbors.push_back(cell->neighbor(f));
                for (const auto &neighbor : neighbors)
                  {
                    const bool interior = !cell->at_boundary(f);
                    const unsigned int axis = f / 2, tangential = 1 - axis;
                    double lo = cell->vertex(0)[tangential], hi = cell->vertex(1u << tangential)[tangential];
                    if (interior)
                      {
                        AssertThrow(!neighbor->is_artificial(), ExcMessage("Material DG transport needs a readable face neighbor."));
                        lo = std::max(lo, neighbor->vertex(0)[tangential]);
                        hi = std::min(hi, neighbor->vertex(1u << tangential)[tangential]);
                      }
                    AssertThrow(hi > lo, ExcMessage("Empty Cartesian transport face patch."));
                    Tensor<1, dim> normal;
                    normal[axis] = (f % 2) ? 1. : -1.;
                    std::vector<Point<dim>> unit, physical;
                    for (unsigned int q = 0; q < face_rule.size(); ++q)
                      {
                        Point<dim> point;
                        point[axis] = cell->vertex((f % 2) * (1u << axis))[axis];
                        point[tangential] = lo + (hi - lo) * face_rule.point(q)[0];
                        physical.push_back(point);
                        unit.push_back(context.get_mapping().transform_real_to_unit_cell(cell, point));
                      }
                    FEValues<dim> face_flow(context.get_mapping(), context.get_fe(), Quadrature<dim>(unit), update_values);
                    face_flow.reinit(entry.second);
                    std::vector<Tensor<1, dim>> face_speed(unit.size());
                    face_flow[context.introspection().extractors.velocities].get_function_values(velocity, face_speed);
                    FullMatrix<double> coupling(n, n);
                    std::vector<types::global_dof_index> neighbor_indices(n);
                    if (interior)
                      neighbor->get_dof_indices(neighbor_indices);
                    for (unsigned int q = 0; q < unit.size(); ++q)
                      {
                        const double vn = face_speed[q] * normal;
                        AssertThrow(std::isfinite(vn), ExcMessage("Nonfinite material face flux."));
                        for (unsigned int d = 0; d < dim; ++d)
                          maximum_speed[d] = std::max(maximum_speed[d], std::abs(face_speed[q][d]));
                        Point<dim> exterior;
                        if (interior)
                          exterior = context.get_mapping().transform_real_to_unit_cell(neighbor, physical[q]);
                        Vector<double> test(n), load(n);
                        for (unsigned int i = 0; i < n; ++i)
                          {
                            test[i] = -(hi - lo) * face_rule.weight(q) * vn * space.element.shape_value(i, unit[q]);
                            for (unsigned int j = 0; j < n; ++j)
                              if (vn >= 0.)
                                self(i, j) += test[i] * space.element.shape_value(j, unit[q]);
                              else if (interior)
                                coupling(i, j) += test[i] * space.element.shape_value(j, exterior);
                          }
                        if (vn < 0. && !interior)
                          {
                            local.inverse_mass.vmult(load, test);
                            op->inflow.push_back({physical[q], local.indices, std::move(load)});
                          }
                      }
                    if (interior)
                      {
                        FullMatrix<double> product(n, n);
                        local.inverse_mass.mmult(product, coupling);
                        op->blocks.push_back({local.indices, std::move(neighbor_indices), std::move(product)});
                      }
                  }
              }
            FullMatrix<double> product(n, n);
            local.inverse_mass.mmult(product, self);
            op->blocks.push_back({local.indices, local.indices, std::move(product)});
            double inverse_time = 0.;
            for (unsigned int d = 0; d < dim; ++d)
              inverse_time += maximum_speed[d] / (cell->vertex(1u << d)[d] - cell->vertex(0)[d]);
            op->inverse_advection_time = std::max(op->inverse_advection_time, inverse_time);
            op->cells.push_back(std::move(local));
          }
        op->inverse_advection_time = Utilities::MPI::max(op->inverse_advection_time, comm);
        return op;
      }

      template<int dim>
      std::shared_ptr<TensorField<dim>>
      TensorAdvection<dim>::advance(const SimulatorAccess<dim> &context, const TensorAdvection &op,
                                                const History &incoming, const Field &inflow, const Field &source, const double time, const double dt)
      {
        AssertThrow(incoming.space == op.space

                    && context.get_triangulation().get_vertices() == op.mesh_vertices
                    && std::isfinite(time) && std::isfinite(dt) && dt > 0.
                    && dt * (op.inverse_advection_time + op.relaxation_rate) <= .125,
                    ExcMessage("Material DG SSPRK3 requires the original mesh/space and its explicit CFL <= 0.125."));
        using Fields = std::array<LinearAlgebra::Vector, SymmetricTensor<2, dim>::n_independent_components>;
        const auto &owned = op.space->dofs.locally_owned_dofs();
        const auto &comm = context.get_mpi_communicator();
        Fields initial, stage, rhs, ghost;
        for (unsigned int c = 0; c < initial.size(); ++c)
          {
            initial[c].reinit(owned, comm);
            initial[c] = incoming.coefficients[c];
            stage[c].reinit(owned, comm);
            rhs[c].reinit(owned, comm);
            ghost[c].reinit(owned, op.space->relevant, comm);
          }
        const auto evaluate = [&](const Fields & values, const double t)
        {
          for (unsigned int c = 0; c < values.size(); ++c)
            {
              ghost[c] = values[c];
              rhs[c] = 0.;
              for (const auto &block : op.blocks)
                for (unsigned int i = 0; i < block.rows.size(); ++i)
                  for (unsigned int j = 0; j < block.columns.size(); ++j)
                    rhs[c][block.rows[i]] += block.values(i, j) * ghost[c][block.columns[j]];
            }
          for (unsigned int kind = 0; kind < 2; ++kind)
            for (const auto &load : kind ? op.inflow : op.source)
              {
                const auto value = (kind ? inflow : source)(load.point, t);
                AssertThrow(std::isfinite(value.norm()), ExcMessage("Nonfinite physical source/inflow."));
                for (unsigned int c = 0; c < values.size(); ++c)
                  for (unsigned int i = 0; i < load.indices.size(); ++i)
                    rhs[c][load.indices[i]] += load.weights[i] * value[SymmetricTensor<2, dim>::unrolled_to_component_indices(c)];
              }
          for (auto &v : rhs)
            v.compress(VectorOperation::insert);
        };
        evaluate(initial, time);
        for (unsigned int c = 0; c < stage.size(); ++c)
          {
            stage[c] = initial[c];
            stage[c].add(dt, rhs[c]);
          }
        evaluate(stage, time + dt);
        for (unsigned int c = 0; c < stage.size(); ++c)
          {
            stage[c].add(dt, rhs[c]);
            stage[c].sadd(.25, .75, initial[c]);
          }
        evaluate(stage, time + .5 * dt);
        auto result = std::make_shared<History>();
        result->space = op.space;
        for (unsigned int c = 0; c < stage.size(); ++c)
          {
            stage[c].add(dt, rhs[c]);
            stage[c].sadd(2. / 3., 1. / 3., initial[c]);
            for (const auto i : owned)
              AssertThrow(std::isfinite(stage[c][i]), ExcMessage("Nonfinite transported tensor."));
            result->coefficients[c].reinit(owned, op.space->relevant, comm);
            result->coefficients[c] = stage[c];
          }
        // This operator advances one tensor. T and R generally have different
        // physical inflow/source data; silently reusing T's callbacks for R is wrong.
        return result;
      }

#define INSTANTIATE(dim) \
  template std::shared_ptr<TensorAdvection<dim>> TensorAdvection<dim>::assemble(const SimulatorAccess<dim> &,const History &,const Cells &,const LinearAlgebra::BlockVector &,unsigned int,double); \
  template std::shared_ptr<TensorField<dim>> TensorAdvection<dim>::advance(const SimulatorAccess<dim> &,const TensorAdvection &,const History &,const Field &,const Field &,double,double);
      ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
    }
  }
}
