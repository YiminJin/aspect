/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/phase_field/dg_operator.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

namespace aspect
{
  namespace PhaseField
  {
    template <int dim>
    void DGOperator<dim>::reinit_matrix(LinearAlgebra::SparseMatrix &matrix) const
    {
      TrilinosWrappers::SparsityPattern graph(owned, owned, relevant, communicator);
      for (const auto &cell : cells)
        {
          for (const auto i : cell.indices)
            for (const auto j : cell.indices) graph.add(i, j);
          for (const auto f : cell.iterator->face_indices())
            if (!cell.iterator->at_boundary(f))
              {
                std::vector<CellIterator> neighbors;
                if (cell.iterator->neighbor(f)->has_children())
                  for (unsigned int s = 0; s < GeometryInfo<dim>::max_children_per_face; ++s)
                    neighbors.push_back(cell.iterator->neighbor_child_on_subface(f, s));
                else neighbors.push_back(cell.iterator->neighbor(f));
                for (const auto &neighbor : neighbors)
                  for (const auto i : cell.indices)
                    for (const auto j : indices(neighbor)) graph.add(i, j);
              }
        }
      graph.compress();
      matrix.reinit(graph);
    }

    template <int dim>
    double DGOperator<dim>::assemble_sipg(LinearAlgebra::SparseMatrix &matrix,
                                          LinearAlgebra::Vector &rhs, const Coefficient &coefficient,
                                          const double penalty, const Coefficient &dirichlet) const
    {
      AssertThrow(penalty > 0. && std::isfinite(penalty), ExcMessage("SIPG phase penalty must be positive."));
      matrix = 0.;
      rhs = 0.;
      double constant = 0.;
      const QGauss<dim> volume_rule(5);
      const QGauss<1> face_rule(5);
      FEValues<dim> volume(mapping, element, volume_rule,
                           update_values | update_gradients | update_JxW_values | update_quadrature_points);
      for (const auto &cell : cells)
        {
          const auto native = cell.iterator;
          const unsigned int n = cell.indices.size();
          FullMatrix<double> self(n, n);
          volume.reinit(typename Triangulation<dim>::cell_iterator(native));
          for (unsigned int q = 0; q < volume_rule.size(); ++q)
            {
              const double k = coefficient(volume.quadrature_point(q));
              AssertThrow(k > 0. && std::isfinite(k), ExcMessage("SIPG requires positive finite gradient coefficient."));
              for (unsigned int i = 0; i < n; ++i)
                for (unsigned int j = 0; j < n; ++j)
                  self(i, j) += volume.JxW(q) * k * (volume.shape_grad(i, q) * volume.shape_grad(j, q));
            }
          for (const auto f : native->face_indices())
            {
              AssertThrow(!native->has_periodic_neighbor(f), ExcMessage("SIPG phase periodic faces are not qualified."));
              const bool interior = !native->at_boundary(f);
              if (!interior && !dirichlet) continue; // Natural zero normal derivative.
              std::vector<CellIterator> neighbors;
              if (!interior) neighbors.emplace_back();
              else if (native->neighbor(f)->has_children())
                for (unsigned int s = 0; s < GeometryInfo<dim>::max_children_per_face; ++s)
                  neighbors.push_back(native->neighbor_child_on_subface(f, s));
              else neighbors.push_back(native->neighbor(f));
              for (const auto &neighbor : neighbors)
                {
                  const unsigned int axis = f / 2, tangent = 1 - axis;
                  double lo = native->vertex(0)[tangent], hi = native->vertex(1u << tangent)[tangent];
                  if (interior)
                    {
                      AssertThrow(!neighbor->is_artificial(), ExcMessage("SIPG requires readable face neighbors."));
                      lo = std::max(lo, neighbor->vertex(0)[tangent]);
                      hi = std::min(hi, neighbor->vertex(1u << tangent)[tangent]);
                    }
                  const double h = native->vertex(1u << axis)[axis] - native->vertex(0)[axis];
                  const double hn = interior ? neighbor->vertex(1u << axis)[axis] - neighbor->vertex(0)[axis] : h;
                  std::vector<Point<dim>> points, unit, neighbor_unit;
                  for (unsigned int q = 0; q < face_rule.size(); ++q)
                    {
                      Point<dim> x;
                      x[axis] = native->vertex((f % 2) * (1u << axis))[axis];
                      x[tangent] = lo + (hi - lo) * face_rule.point(q)[0];
                      points.push_back(x);
                      unit.push_back(mapping.transform_real_to_unit_cell(native, x));
                      if (interior) neighbor_unit.push_back(mapping.transform_real_to_unit_cell(neighbor, x));
                    }
                  FEValues<dim> face(mapping, element, Quadrature<dim>(unit), update_values | update_gradients);
                  face.reinit(typename Triangulation<dim>::cell_iterator(native));
                  std::unique_ptr<FEValues<dim>> other;
                  if (interior)
                    {
                      other = std::make_unique<FEValues<dim>>(mapping, element, Quadrature<dim>(neighbor_unit), update_values | update_gradients);
                      other->reinit(typename Triangulation<dim>::cell_iterator(neighbor));
                    }
                  FullMatrix<double> coupling(n, n);
                  for (unsigned int q = 0; q < points.size(); ++q)
                    {
                      const double k = coefficient(points[q]);
                      const double sigma = penalty * (element.degree + 1) * (element.degree + 1) * k / std::min(h, hn);
                      const double w = (hi - lo) * face_rule.weight(q), normal = f % 2 ? 1. : -1.;
                      const double average = interior ? .5 : 1.;
                      const double g = interior ? 0. : dirichlet(points[q]);
                      if (!interior) constant += .5 * w * sigma * g * g;
                      for (unsigned int i = 0; i < n; ++i)
                        {
                          const double vi = face.shape_value(i, q), di = normal * face.shape_grad(i, q)[axis];
                          if (!interior) rhs[cell.indices[i]] += w * g * (sigma * vi - k * di);
                          for (unsigned int j = 0; j < n; ++j)
                            {
                              const double vj = face.shape_value(j, q), dj = normal * face.shape_grad(j, q)[axis];
                              // Row on this cell: jump(test)=test. The adjacent
                              // rank/cell assembles its own rows, never this row.
                              self(i, j) += w * (-average * k * (di * vj + dj * vi) + sigma * vi * vj);
                              if (interior)
                                coupling(i, j) += w * (.5 * k * di * other->shape_value(j, q)
                                                       - .5 * k * normal * other->shape_grad(j, q)[axis] * vi
                                                       - sigma * vi * other->shape_value(j, q));
                            }
                        }
                    }
                  if (interior)
                    {
                      const auto columns = indices(neighbor);
                      for (unsigned int i = 0; i < n; ++i)
                        for (unsigned int j = 0; j < n; ++j)
                          matrix.add(cell.indices[i], columns[j], coupling(i, j));
                    }
                }
            }
          for (unsigned int i = 0; i < n; ++i)
            for (unsigned int j = 0; j < n; ++j) matrix.add(cell.indices[i], cell.indices[j], self(i, j));
        }
      matrix.compress(VectorOperation::add);
      rhs.compress(VectorOperation::insert);
      return Utilities::MPI::sum(constant, communicator);
    }

#define INSTANTIATE(dim) \
  template void DGOperator<dim>::reinit_matrix(LinearAlgebra::SparseMatrix &) const; \
  template double DGOperator<dim>::assemble_sipg(LinearAlgebra::SparseMatrix &,LinearAlgebra::Vector &,const Coefficient &,double,const Coefficient &) const;
    ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
  }
}
