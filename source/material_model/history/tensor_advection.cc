/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/material_model/history/tensor_history.h>
#include <aspect/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

namespace aspect
{
  namespace MaterialModel
  {
    namespace HistoryDiscretization
    {
      template <int dim>
      std::shared_ptr<TensorAdvection<dim>>
      TensorAdvection<dim>::assemble(const SimulatorAccess<dim> &context,
                                     const History &history, const Cells &cells, const LinearAlgebra::BlockVector &velocity,
                                     const unsigned int quadrature_order, const double relaxation_rate, const bool distributed)
      {
        if (distributed)
          return assemble_distributed(context, history, cells, velocity, quadrature_order, relaxation_rate);
        AssertThrow(dim == 2 && quadrature_order >= 4
                    && Utilities::MPI::n_mpi_processes(context.get_mpi_communicator()) == 1,
                    ExcMessage("PF4 transport requires fixed serial Cartesian DGQ2 and at least Gauss4."));
        auto result = std::make_shared<TensorAdvection>();
        auto &op = *result;
        op.space = history.space;
        op.mesh_vertices = context.get_triangulation().get_vertices();
        op.quadrature_order = quadrature_order;
        op.relaxation_rate = relaxation_rate;
        const auto &space = *op.space;
        const unsigned int n = space.element.n_dofs_per_cell();
        DynamicSparsityPattern pattern(space.dofs.n_dofs());
        for (const auto &entry : cells)
          {
            typename TensorAdvection::Cell cell;
            cell.id = entry.first;
            cell.indices.resize(n);
            const auto dg_cell = space.cells.at(entry.first);
            dg_cell->get_dof_indices(cell.indices);
            for (const auto i : cell.indices)
              for (const auto j : cell.indices)
                pattern.add(i, j);
            for (const auto face : dg_cell->face_indices())
              {
                AssertThrow(!dg_cell->has_periodic_neighbor(face), ExcMessage("PF4 has explicit inflow, not periodic faces."));
                if (!dg_cell->at_boundary(face))
                  {
                    const auto neighbor = dg_cell->neighbor(face);
                    AssertThrow(neighbor->is_active() && !dg_cell->neighbor_is_coarser(face),
                                ExcMessage("PF4 transport does not support hanging faces."));
                    std::vector<types::global_dof_index> neighbor_indices(n);
                    neighbor->get_dof_indices(neighbor_indices);
                    for (const auto i : cell.indices)
                      for (const auto j : neighbor_indices)
                        pattern.add(i, j);
                  }
              }
            op.cells.push_back(std::move(cell));
          }
        op.sparsity.copy_from(pattern);
        op.matrix.reinit(op.sparsity);
        op.divergence_correction.reinit(op.sparsity);
        const QGauss<dim> mass_rule(3), quadrature(quadrature_order);
        const QGauss < dim - 1 > face_rule(quadrature_order);
        FEValues<dim> mass_fe(context.get_mapping(), space.element, mass_rule, update_values | update_JxW_values);
        FEValues<dim> dg(context.get_mapping(), space.element, quadrature, update_values | update_gradients | update_quadrature_points | update_JxW_values);
        FEValues<dim> flow(context.get_mapping(), context.get_fe(), quadrature, update_values | update_gradients);
        FEFaceValues<dim> dg_face(context.get_mapping(), space.element, face_rule,
                                  update_values | update_quadrature_points | update_normal_vectors | update_JxW_values);
        FEFaceValues<dim> flow_face(context.get_mapping(), context.get_fe(), face_rule, update_values);
        std::vector<Tensor<1, dim>> v(quadrature.size()), face_v(face_rule.size());
        std::vector<Tensor<2, dim>> grad_v(quadrature.size());
        const auto extractor = context.introspection().extractors.velocities;
        for (auto &cell : op.cells)
          {
            const auto dg_cell = space.cells.at(cell.id);
            const auto native_cell = cells.at(cell.id);
            mass_fe.reinit(dg_cell);
            cell.inverse_mass.reinit(n, n);
            for (unsigned int q = 0; q < mass_rule.size(); ++q)
              for (unsigned int i = 0; i < n; ++i)
                for (unsigned int j = 0; j < n; ++j)
                  cell.inverse_mass(i, j) += mass_fe.JxW(q) * mass_fe.shape_value(i, q) * mass_fe.shape_value(j, q);
            cell.inverse_mass.gauss_jordan();
            dg.reinit(dg_cell);
            flow.reinit(native_cell);
            flow[extractor].get_function_values(velocity, v);
            flow[extractor].get_function_gradients(velocity, grad_v);
            FullMatrix<double> self(n, n), correction(n, n);
            Tensor<1, dim> speed;
            for (unsigned int q = 0; q < quadrature.size(); ++q)
              {
                AssertThrow(std::isfinite(v[q].norm() + grad_v[q].norm()), ExcMessage("Nonfinite prescribed FE transport velocity."));
                const double divergence = trace(grad_v[q]);
                Vector<double> test(n), load(n);
                for (unsigned int i = 0; i < n; ++i)
                  {
                    test[i] = dg.JxW(q) * dg.shape_value(i, q);
                    for (unsigned int j = 0; j < n; ++j)
                      {
                        correction(i, j) += test[i] * divergence * dg.shape_value(j, q);
                        self(i, j) += dg.JxW(q) * dg.shape_value(j, q) *
                                      (v[q] * dg.shape_grad(i, q) + (divergence - op.relaxation_rate) * dg.shape_value(i, q));
                      }
                  }
                cell.inverse_mass.vmult(load, test);
                op.source.push_back({dg.quadrature_point(q), cell.indices, std::move(load)});
                for (unsigned int d = 0; d < dim; ++d)
                  speed[d] = std::max(speed[d], std::abs(v[q][d]));
              }
            for (const auto face : dg_cell->face_indices())
              {
                dg_face.reinit(dg_cell, face);
                flow_face.reinit(native_cell, face);
                flow_face[extractor].get_function_values(velocity, face_v);
                FullMatrix<double> neighbor_block(n, n);
                std::vector<types::global_dof_index> neighbor_indices(n);
                std::unique_ptr<FEValues<dim>> neighbor_fe;
                if (!dg_cell->at_boundary(face))
                  {
                    const auto neighbor = dg_cell->neighbor(face);
                    neighbor->get_dof_indices(neighbor_indices);
                    std::vector<Point<dim>> points;
                    for (const auto &point : dg_face.get_quadrature_points())
                      points.push_back(context.get_mapping().transform_real_to_unit_cell(neighbor, point));
                    neighbor_fe = std::make_unique<FEValues<dim>>(context.get_mapping(), space.element, Quadrature<dim>(points), update_values);
                    neighbor_fe->reinit(neighbor);
                  }
                for (unsigned int q = 0; q < face_rule.size(); ++q)
                  {
                    const double normal_velocity = face_v[q] * dg_face.normal_vector(q);
                    AssertThrow(std::isfinite(normal_velocity), ExcMessage("Nonfinite transport face velocity."));
                    for (unsigned int d = 0; d < dim; ++d)
                      speed[d] = std::max(speed[d], std::abs(face_v[q][d]));
                    Vector<double> test(n), load(n);
                    for (unsigned int i = 0; i < n; ++i)
                      {
                        test[i] = -dg_face.JxW(q) * normal_velocity * dg_face.shape_value(i, q);
                        for (unsigned int j = 0; j < n; ++j)
                          if (normal_velocity >= 0.)
                            self(i, j) += test[i] * dg_face.shape_value(j, q);
                          else if (neighbor_fe)
                            neighbor_block(i, j) += test[i] * neighbor_fe->shape_value(j, q);
                      }
                    if (normal_velocity < 0. && !neighbor_fe)
                      {
                        cell.inverse_mass.vmult(load, test);
                        op.inflow.push_back({dg_face.quadrature_point(q), cell.indices, std::move(load)});
                      }
                  }
                if (neighbor_fe)
                  {
                    FullMatrix<double> product(n, n);
                    cell.inverse_mass.mmult(product, neighbor_block);
                    op.matrix.add(cell.indices, neighbor_indices, product);
                  }
              }
            FullMatrix<double> product(n, n);
            cell.inverse_mass.mmult(product, self);
            op.matrix.add(cell.indices, product);
            cell.inverse_mass.mmult(product, correction);
            op.divergence_correction.add(cell.indices, product);
            double inverse_time = 0.;
            for (unsigned int d = 0; d < dim; ++d)
              inverse_time += speed[d] / (dg_cell->vertex(1u << d)[d] - dg_cell->vertex(0)[d]);
            op.inverse_advection_time = std::max(op.inverse_advection_time, inverse_time);
          }
        return result;
      }


      template <int dim>
      std::shared_ptr<TensorHistory<dim>>
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
        if (op.distributed)
          for (auto &component : result->coefficients)
            {
              LinearAlgebra::Vector owned(space.dofs.locally_owned_dofs(), context.get_mpi_communicator());
              owned = component;
              component = owned;
            }
        return result;
      }


      template <int dim>
      std::shared_ptr<TensorHistory<dim>>
      TensorAdvection<dim>::advance(
        const SimulatorAccess<dim> &context, const TensorAdvection &op, const History &incoming, const Field &inflow,
        const Field &source, const double time, const double dt)
      {
        if (op.distributed)
          return advance_distributed(context, op, incoming, inflow, source, time, dt);
        AssertThrow(incoming.space == op.space
                    && context.get_triangulation().get_vertices() == op.mesh_vertices,
                    ExcMessage("PF4 transport rejects changed spaces or mesh motion."));
        AssertThrow(std::isfinite(time) && std::isfinite(dt) && dt > 0.
                    && dt * (op.inverse_advection_time + op.relaxation_rate) <= .125,
                    ExcMessage("PF4 explicit transport requires finite positive dt and its bounded diagnostic CFL."));
        using Components = std::array<Vector<double>, SymmetricTensor<2, dim>::n_independent_components>;
        const auto n = op.space->dofs.n_dofs();
        Components initial, stage1, stage2, rhs;
        for (unsigned int c = 0; c < initial.size(); ++c)
          {
            initial[c].reinit(n);
            rhs[c].reinit(n);
            for (unsigned int i = 0; i < n; ++i)
              {
                initial[c][i] = incoming.coefficients[c][i];
                AssertThrow(std::isfinite(initial[c][i]), ExcMessage("Nonfinite incoming DG transport history."));
              }
          }
        const auto evaluate = [&](const Components & values, const double stage_time)
        {
          for (unsigned int c = 0; c < values.size(); ++c)
            op.matrix.vmult(rhs[c], values[c]);
          for (unsigned int kind = 0; kind < 2; ++kind)
            for (const auto &load : kind ? op.inflow : op.source)
              {
                const auto value = (kind ? inflow : source)(load.point, stage_time);
                AssertThrow(std::isfinite(value.norm()), ExcMessage("Nonfinite DG transport source or inflow."));
                for (unsigned int c = 0; c < rhs.size(); ++c)
                  {
                    const double component = value[SymmetricTensor<2, dim>::unrolled_to_component_indices(c)];
                    if (component != 0.)
                      for (unsigned int i = 0; i < load.indices.size(); ++i)
                        rhs[c][load.indices[i]] += load.weights[i] * component;
                  }
              }
        };
        evaluate(initial, time);
        stage1 = initial;
        for (unsigned int c = 0; c < initial.size(); ++c)
          stage1[c].add(dt, rhs[c]);
        evaluate(stage1, time + dt);
        stage2 = stage1;
        for (unsigned int c = 0; c < initial.size(); ++c)
          {
            stage2[c].add(dt, rhs[c]);
            stage2[c].sadd(.25, .75, initial[c]);
          }
        evaluate(stage2, time + .5 * dt);
        for (unsigned int c = 0; c < initial.size(); ++c)
          {
            stage2[c].add(dt, rhs[c]);
            stage2[c].sadd(2. / 3., 1. / 3., initial[c]);
          }
        auto result = std::make_shared<History>();
        result->space = op.space;
        for (unsigned int c = 0; c < stage2.size(); ++c)
          {
            result->coefficients[c].reinit(op.space->dofs.locally_owned_dofs(), op.space->relevant, context.get_mpi_communicator());
            for (unsigned int i = 0; i < n; ++i)
              {
                AssertThrow(std::isfinite(stage2[c][i]), ExcMessage("Nonfinite DG transport candidate."));
                result->coefficients[c][i] = stage2[c][i];
              }
            result->coefficients[c].compress(VectorOperation::insert);
          }
        return result;
      }

#define INSTANTIATE(dim) \
  template std::shared_ptr<TensorAdvection<dim>> TensorAdvection<dim>::assemble(const SimulatorAccess<dim> &,const History &,const Cells &,const LinearAlgebra::BlockVector &,unsigned int,double,bool); \
  template std::shared_ptr<TensorHistory<dim>> TensorAdvection<dim>::project(const SimulatorAccess<dim> &,const TensorAdvection &,const Field &,double,unsigned int); \
  template std::shared_ptr<TensorHistory<dim>> TensorAdvection<dim>::advance(const SimulatorAccess<dim> &,const TensorAdvection &,const History &,const Field &,const Field &,double,double);
      ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
    }
  }
}
