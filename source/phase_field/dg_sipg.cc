/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/phase_field/dg_operator.h>
#include <aspect/simulator/assemblers/scalar_dg.h>
#include <aspect/phase_field/mapped_face.h>
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
      const QGauss<dim-1> face_rule(5);
      MappedFaceScratch<dim> face_scratch(mapping, element, face_rule);
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
              for (unsigned int patch = 0; patch < neighbors.size(); ++patch)
                {
                  const auto &neighbor = neighbors[patch];
                  const auto &face = face_scratch.reinit(native, f,
                                                         neighbors.size() > 1 ? patch : numbers::invalid_unsigned_int);
                  const auto &points = face.get_quadrature_points();
                  const double minimum_jacobian = minimum_cell_jacobian(mapping, element, native);
                  const double neighbor_jacobian = interior ? minimum_cell_jacobian(mapping, element, neighbor) : minimum_jacobian;
                  std::vector<Point<dim>> neighbor_unit;
                  if (interior)
                    {
                      AssertThrow(!neighbor->is_artificial(), ExcMessage("SIPG requires readable face neighbors."));
                      for (const auto &point : points)
                        neighbor_unit.push_back(mapping.transform_real_to_unit_cell(neighbor, point));
                    }
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
                      const double surface_jacobian = face.JxW(q) / face_rule.weight(q);
                      const double h_inverse = surface_jacobian * neighbors.size() / minimum_jacobian;
                      // Fine cells see the full coarse neighbor face; its reference
                      // face measure is twice the fine patch measure in 2-D.
                      const double neighbor_scale = interior && native->neighbor_is_coarser(f) ? 2. : 1.;
                      const double neighbor_h_inverse = surface_jacobian * neighbor_scale / neighbor_jacobian;
                      const double sigma = penalty * (element.degree + 1) * (element.degree + 1)
                                           * k * std::max(h_inverse, neighbor_h_inverse);
                      const double w = face.JxW(q);
                      const auto normal = face.normal_vector(q);
                      const double average = interior ? .5 : 1.;
                      const double g = interior ? 0. : dirichlet(points[q]);
                      if (!interior) constant += .5 * w * sigma * g * g;
                      for (unsigned int i = 0; i < n; ++i)
                        {
                          const double vi = face.shape_value(i, q), di = normal * face.shape_grad(i, q);
                          if (!interior) rhs[cell.indices[i]] += w * g * (sigma * vi - k * di);
                          for (unsigned int j = 0; j < n; ++j)
                            {
                              const double vj = face.shape_value(j, q), dj = normal * face.shape_grad(j, q);
                              // Row on this cell: jump(test)=test. The adjacent
                              // rank/cell assembles its own rows, never this row.
                              self(i, j) += w * Assemblers::scalar_symmetric_penalty(vi, average*k*di, vj, average*k*dj, sigma);
                              if (interior)
                                coupling(i, j) += w * Assemblers::scalar_symmetric_penalty(vi, .5*k*di,
                                                                                           -other->shape_value(j,q), .5*k*(normal*other->shape_grad(j,q)), sigma);
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
