/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/phase_field/dg_operator.h>
#include <aspect/phase_field/mapped_face.h>
#include <aspect/phase_field/assemblers.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/mapping_cartesian.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_tools.h>

namespace aspect
{
  namespace PhaseField
  {
    template <int dim>
    DGOperator<dim>::DGOperator(const Mapping<dim> &mapping_, const DoFHandler<dim> &dofs_,
                                const unsigned int component_, const types::global_dof_index offset_,
                                const IndexSet &owned_, const IndexSet &relevant_, MPI_Comm communicator_)
      : mapping(mapping_), dofs(dofs_), component(component_), offset(offset_),
        owned(owned_), relevant(relevant_), communicator(communicator_),
        element(dofs.get_fe().base_element(dofs.get_fe().component_to_base_index(component).first)),
        bounds(element)
    {
      AssertThrow(dim == 2 && element.n_dofs_per_face() == 0,
                  ExcMessage("DG phase operators require a two-dimensional discontinuous Q2 field."));
      for (unsigned int i = 0; i < element.n_dofs_per_cell(); ++i)
        local_indices.push_back(dofs.get_fe().component_to_system_index(component, i));
      AssertThrow(dynamic_cast<const MappingQ1<dim> *>(&mapping) != nullptr
                  || dynamic_cast<const MappingCartesian<dim> *>(&mapping) != nullptr,
                  ExcMessage("Native phase currently supports static bilinear MappingQ1 meshes."));
      const QGaussLobatto<dim> rule(4);
      FEValues<dim> fe(mapping, element, rule, update_values | update_JxW_values);
      for (const auto &cell : dofs.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            Cell data;
            data.iterator = cell;
            data.indices = indices(cell);
            data.volume = 0.;
            const auto n = data.indices.size();
            data.inverse_mass.reinit(n, n);
            data.mean_weights.reinit(n);
            fe.reinit(typename Triangulation<dim>::cell_iterator(cell));
            for (unsigned int q = 0; q < rule.size(); ++q)
              {
                data.volume += fe.JxW(q);
                for (unsigned int i = 0; i < n; ++i)
                  {
                    data.mean_weights[i] += fe.JxW(q) * fe.shape_value(i, q);
                    for (unsigned int j = 0; j < n; ++j)
                      data.inverse_mass(i, j) += fe.JxW(q) * fe.shape_value(i, q) * fe.shape_value(j, q);
                  }
              }
            data.mean_weights /= data.volume;
            data.inverse_mass.gauss_jordan();
            cells.push_back(std::move(data));
          }
    }

    template <int dim>
    std::vector<types::global_dof_index> DGOperator<dim>::indices(const CellIterator &cell) const
    {
      std::vector<types::global_dof_index> all(dofs.get_fe().n_dofs_per_cell()), result;
      cell->get_dof_indices(all);
      for (const auto i : local_indices) result.push_back(all[i] - offset);
      return result;
    }

    template <int dim>
    void DGOperator<dim>::assemble_transport(const Velocity &velocity, const Assemblers::Interface<dim> *cell_assembler)
    {
      blocks.clear();
      boundary.clear();
      sources.clear();
      mean_rate = 0.;
      const QGaussLobatto<dim> rule(5);
      const QGaussLobatto<dim-1> face_rule(5);
      MappedFaceScratch<dim> face_scratch(mapping, element, face_rule);
      FEValues<dim> fe(mapping, element, rule, update_values | update_gradients | update_JxW_values | update_quadrature_points);
      for (const auto &cell : cells)
        {
          const auto native = cell.iterator;
          const unsigned int n = cell.indices.size();
          fe.reinit(typename Triangulation<dim>::cell_iterator(native));
          std::vector<Tensor<1, dim>> speed(rule.size());
          std::vector<double> divergence(rule.size());
          velocity(native, fe.get_quadrature_points(), speed, divergence);
          FullMatrix<double> self(n, n);
          double face_loss = 0.;
          const double minimum_jacobian = minimum_cell_jacobian(mapping, element, native);
          double minimum_divergence = *std::min_element(divergence.begin(), divergence.end());
          for (unsigned int q = 0; q < rule.size(); ++q)
            {
              SourceSample sample;
              sample.indices = cell.indices;
              sample.weight = fe.JxW(q) * divergence[q];
              sample.shape.reinit(n);
              for (unsigned int i = 0; i < n; ++i)
                {
                  sample.shape[i] = fe.shape_value(i, q);

                }
              sources.push_back(std::move(sample));
            }
          if (cell_assembler)
            {
              internal::Assembly::Scratch::PhaseTransport<dim> scratch(fe, speed, divergence);
              internal::Assembly::CopyData::PhaseTransport<dim> copy(self);
              scratch.cell = native;
              cell_assembler->execute(scratch, copy);
            }
          else
            Assemblers::assemble_phase_transport_cell(fe, speed, divergence, self);
          for (const auto f : native->face_indices())
            {
              AssertThrow(!native->has_periodic_neighbor(f), ExcMessage("DG phase currently requires explicit exterior data, not periodic faces."));
              std::vector<CellIterator> neighbors;
              if (native->at_boundary(f)) neighbors.emplace_back();
              else if (native->neighbor(f)->has_children())
                for (unsigned int s = 0; s < GeometryInfo<dim>::max_children_per_face; ++s)
                  neighbors.push_back(native->neighbor_child_on_subface(f, s));
              else neighbors.push_back(native->neighbor(f));
              double maximum_outflow = 0.;
              for (unsigned int patch = 0; patch < neighbors.size(); ++patch)
                {
                  const auto &neighbor = neighbors[patch];
                  const bool interior = !native->at_boundary(f);
                  const auto &face = face_scratch.reinit(native, f,
                                                         neighbors.size() > 1 ? patch : numbers::invalid_unsigned_int);
                  const auto &physical = face.get_quadrature_points();
                  std::vector<Point<dim>> unit;
                  for (const auto &point : physical)
                    unit.push_back(mapping.transform_real_to_unit_cell(native, point));
                  if (interior)
                    AssertThrow(!neighbor->is_artificial(), ExcMessage("DG phase needs readable face neighbors."));
                  std::vector<Tensor<1, dim>> face_speed(physical.size());
                  std::vector<double> unused(physical.size());
                  velocity(native, physical, face_speed, unused);
                  FullMatrix<double> coupling(n, n);
                  for (unsigned int q = 0; q < physical.size(); ++q)
                    {
                      const double vn = face_speed[q] * face.normal_vector(q);
                      const double weight = face.JxW(q);
                      // Undo the subface reference scaling when bounding a full face.
                      const double surface_jacobian = weight / face_rule.weight(q) * neighbors.size();
                      maximum_outflow = std::max(maximum_outflow, vn * surface_jacobian);
                      const auto exterior = interior ? mapping.transform_real_to_unit_cell(neighbor, physical[q]) : Point<dim>();
                      Vector<double> load(n), mass_load(n), trace(n);
                      for (unsigned int i = 0; i < n; ++i)
                        {
                          trace[i] = element.shape_value(i, unit[q]);
                          load[i] = -weight * vn * trace[i];
                          for (unsigned int j = 0; j < n; ++j)
                            if (vn >= 0.) self(i, j) += load[i] * element.shape_value(j, unit[q]);
                            else if (interior) coupling(i, j) += load[i] * element.shape_value(j, exterior);
                        }
                      if (!interior)
                        {
                          cell.inverse_mass.vmult(mass_load, load);
                          boundary.push_back({native->face(f)->boundary_id(), physical[q], cell.indices, std::move(mass_load), std::move(trace), weight * vn});
                        }
                    }
                  if (interior)
                    {
                      FullMatrix<double> product(n, n);
                      cell.inverse_mass.mmult(product, coupling);
                      blocks.push_back({cell.indices, indices(neighbor), std::move(product)});
                    }
                }
              // A nonnegative reference Q2 polynomial satisfies the exact
              // Lobatto3 trace bound int_face(phi) <= 6 int_cell(phi).
              // Weight by max(J_face v.n+)/min(J_cell). This also bounds
              // split-face quadrature and the complementary upper-gap loss.
              face_loss += 6. * maximum_outflow / minimum_jacobian;
            }
          mean_rate = std::max(mean_rate, face_loss + std::max(0., -minimum_divergence));
          FullMatrix<double> product(n, n);
          cell.inverse_mass.mmult(product, self);
          blocks.push_back({cell.indices, cell.indices, std::move(product)});
        }
      mean_rate = Utilities::MPI::max(mean_rate, communicator);
    }

    template <int dim>
    void DGOperator<dim>::evaluate(const LinearAlgebra::Vector &state, LinearAlgebra::Vector &rhs,
                                   const BoundaryField &inflow, double time, double &source, double &flux) const
    {
      LinearAlgebra::Vector ghost(owned, relevant, communicator);
      ghost = state;
      rhs = 0.;
      source = 0.;
      flux = 0.;
      for (const auto &block : blocks)
        for (unsigned int i = 0; i < block.rows.size(); ++i)
          for (unsigned int j = 0; j < block.columns.size(); ++j)
            rhs[block.rows[i]] += block.matrix(i, j) * ghost[block.columns[j]];
      for (const auto &sample : sources)
        for (unsigned int i = 0; i < sample.indices.size(); ++i)
          source += sample.weight * sample.shape[i] * ghost[sample.indices[i]];
      for (const auto &load : boundary)
        {
          double value = 0.;
          if (load.flux_weight < 0.)
            {
              value = inflow(load.boundary_id, load.point, time);
              for (unsigned int i = 0; i < load.indices.size(); ++i)
                rhs[load.indices[i]] += load.values[i] * value;
            }
          else for (unsigned int i = 0; i < load.indices.size(); ++i)
              value += load.trace[i] * ghost[load.indices[i]];
          flux += load.flux_weight * value;
        }
      rhs.compress(VectorOperation::insert);
      source = Utilities::MPI::sum(source, communicator);
      flux = Utilities::MPI::sum(flux, communicator);
    }

    template <int dim>
    double DGOperator<dim>::integral(const LinearAlgebra::Vector &state) const
    {
      // This is a diagnostic reduction, not a correction of the field. Extra
      // accumulation precision avoids turning summation noise into an apparent
      // balance defect over many short physical steps.
      long double sum = 0.;
      for (const auto &cell : cells)
        for (unsigned int i = 0; i < cell.indices.size(); ++i)
          sum += cell.volume * cell.mean_weights[i] * state[cell.indices[i]];
      return Utilities::MPI::sum(static_cast<double>(sum), communicator);
    }

    template <int dim>
    void DGOperator<dim>::limit(LinearAlgebra::Vector &state, double lower, double upper, TransportStatistics &stats) const
    {
      unsigned long long count = 0;
      double alpha = 1.;
      unsigned int invalid = 0;
      for (const auto &cell : cells)
        {
          Vector<double> values(cell.indices.size());
          for (unsigned int i = 0; i < values.size(); ++i) values[i] = state[cell.indices[i]];
          const double mean = values * cell.mean_weights;
          try
            {
              const double scaling = bounds.limit(values, mean, lower, upper);
              if (scaling < 1.) ++count;
              alpha = std::min(alpha, scaling);
              for (unsigned int i = 0; i < values.size(); ++i) state[cell.indices[i]] = values[i];
            }
          catch (const std::exception &)
            {
              invalid = 1;
            }
        }
      AssertThrow(Utilities::MPI::max(invalid, communicator) == 0,
                  ExcMessage("DG phase bound scaling rejected an inadmissible mean collectively."));
      state.compress(VectorOperation::insert);
      stats.limited_cells += Utilities::MPI::sum(count, communicator);
      stats.minimum_alpha = std::min(stats.minimum_alpha, Utilities::MPI::min(alpha, communicator));
    }

    template <int dim>
    typename DGOperator<dim>::TransportStatistics DGOperator<dim>::transport(
      const LinearAlgebra::Vector &accepted, LinearAlgebra::Vector &candidate, const Field &inflow,
      double time, double dt, double lower, double upper, bool limiting) const
    {
      return transport(accepted, candidate,
                       BoundaryField([&](types::boundary_id, const Point<dim> &point, double stage_time)
      {
        return inflow(point, stage_time);
      }), time, dt, lower, upper, limiting);
    }

    template <int dim>
    typename DGOperator<dim>::TransportStatistics DGOperator<dim>::transport(
      const LinearAlgebra::Vector &accepted, LinearAlgebra::Vector &candidate, const BoundaryField &inflow,
      double time, double dt, double lower, double upper, bool limiting) const
    {
      AssertThrow(std::isfinite(dt) && dt > 0., ExcMessage("DG phase transport needs positive finite dt."));
      TransportStatistics stats;
      LinearAlgebra::Vector current(owned, communicator), start(owned, communicator), stage(owned, communicator), rhs(owned, communicator);
      current = accepted;
      stats.initial_integral = integral(current);
      if (limiting) limit(current, lower, upper, stats);
      stats.substeps = std::max(1u, static_cast<unsigned int>(std::ceil(dt * mean_rate / .8)));
      AssertThrow(stats.substeps <= 65536, ExcMessage("DG phase transport exceeds its bounded subcycle budget."));
      const double ds = dt / stats.substeps;
      const BoundaryField checked_inflow = [&](types::boundary_id boundary, const Point<dim> &x, double t)
      {
        const double value = inflow(boundary, x, t);
        AssertThrow(std::isfinite(value) && (!limiting || (value >= lower && value <= upper)),
                    ExcMessage("DG phase exterior data violate physical bounds."));
        return value;
      };
      for (unsigned int step = 0; step < stats.substeps; ++step)
        {
          const double t = time + step * ds;
          start = current;
          double source0, source1, source2, flux0, flux1, flux2;
          evaluate(start, rhs, checked_inflow, t, source0, flux0);
          current.add(ds, rhs);
          if (limiting) limit(current, lower, upper, stats);
          evaluate(current, rhs, checked_inflow, t + ds, source1, flux1);
          current.add(ds, rhs);
          current.sadd(.25, .75, start);
          if (limiting) limit(current, lower, upper, stats);
          evaluate(current, rhs, checked_inflow, t + .5 * ds, source2, flux2);
          current.add(ds, rhs);
          current.sadd(2. / 3., 1. / 3., start);
          if (limiting) limit(current, lower, upper, stats);
          stats.integrated_source += ds * (source0 / 6. + source1 / 6. + 2.*source2 / 3.);
          stats.integrated_boundary_flux += ds * (flux0 / 6. + flux1 / 6. + 2.*flux2 / 3.);
        }
      candidate = current;
      stats.final_integral = integral(current);
      stats.balance_error = stats.final_integral - stats.initial_integral - stats.integrated_source + stats.integrated_boundary_flux;
      return stats;
    }

    template <int dim>
    std::size_t DGOperator<dim>::memory_consumption() const
    {
      std::size_t bytes = sizeof(*this);
      for (const auto &b : blocks) bytes += b.matrix.memory_consumption() + sizeof(types::global_dof_index) * (b.rows.size() + b.columns.size());
      for (const auto &c : cells) bytes += c.inverse_mass.memory_consumption() + c.mean_weights.memory_consumption();
      for (const auto &s : sources) bytes += s.shape.memory_consumption() + sizeof(types::global_dof_index) * s.indices.size();
      for (const auto &b : boundary) bytes += b.values.memory_consumption() + b.trace.memory_consumption();
      return bytes;
    }

#define INSTANTIATE(dim) template class DGOperator<dim>;
    ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
  }
}
