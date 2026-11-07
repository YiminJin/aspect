/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/phase_field/dg_operator.h>
#include <aspect/phase_field/mapped_face.h>
#include <aspect/simulator/assemblers/advection.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/mapping_cartesian.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_tools.h>
#include <numeric>

namespace aspect
{
  namespace PhaseField
  {
    template <int dim>
    DGOperator<dim>::DGOperator(const Mapping<dim> &mapping_, const DoFHandler<dim> &dofs_,
                                const unsigned int component_, const types::global_dof_index offset_,
                                const IndexSet &owned_, const IndexSet &relevant_, MPI_Comm communicator_, unsigned int velocity_degree)
      : mapping(mapping_), dofs(dofs_), component(component_), offset(offset_),
        owned(owned_), relevant(relevant_), communicator(communicator_),
        element(dofs.get_fe().base_element(dofs.get_fe().component_to_base_index(component).first)),
        bounds(element), quadrature(element.degree,velocity_degree)
    {
      AssertThrow(dim == 2 && element.n_dofs_per_face() == 0,
                  ExcMessage("DG phase operators require a two-dimensional discontinuous Q2 field."));
      for (unsigned int i = 0; i < element.n_dofs_per_cell(); ++i)
        local_indices.push_back(dofs.get_fe().component_to_system_index(component, i));
      AssertThrow(dynamic_cast<const MappingQ1<dim> *>(&mapping) != nullptr
                  || dynamic_cast<const MappingCartesian<dim> *>(&mapping) != nullptr,
                  ExcMessage("Native phase currently supports static bilinear MappingQ1 meshes."));
      const QGaussLobatto<dim> rule(quadrature.mass_lobatto);
      FEValues<dim> fe(mapping, element, rule, update_values | update_JxW_values);
      for (const auto &cell : dofs.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            Cell data;
            data.iterator = cell;
            data.indices = indices(cell);
            data.volume = 0.;
            const auto n = data.indices.size();
            internal::Assembly::CopyData::AdvectionSystem<dim> mass(element,false);
            internal::Assembly::Scratch::ScalarCell<dim> mass_scratch(fe);
            mass_scratch.mass = 1.;
            data.mean_weights.reinit(n);
            fe.reinit(typename Triangulation<dim>::cell_iterator(cell));
            Assemblers::AdvectionSystem<dim>::assemble_spatial(mass_scratch,mass);
            data.inverse_mass = mass.local_matrix;
            for (unsigned int q = 0; q < rule.size(); ++q)
              {
                data.volume += fe.JxW(q);
                for (unsigned int i = 0; i < n; ++i)
                  {
                    data.mean_weights[i] += fe.JxW(q) * fe.shape_value(i, q);
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
    void DGOperator<dim>::assemble_transport(const Velocity &velocity, const Assemblers::ScalarAdvectionForm form)
    {
      if (transport_matrix.m() == 0) reinit_matrix(transport_matrix);
      transport_matrix = 0.;
      boundary.clear();
      sources.clear();
      mean_rate = 0.;
      const QGaussLobatto<dim> rule(quadrature.transport_lobatto);
      const QGaussLobatto<dim-1> face_rule(quadrature.transport_lobatto);
      FEValues<dim> fe(mapping,element,rule,update_values | update_gradients | update_JxW_values | update_quadrature_points);
      std::vector<unsigned int> shapes(element.n_dofs_per_cell());
      std::iota(shapes.begin(),shapes.end(),0);
      internal::Assembly::Scratch::ScalarFace<dim> face(mapping,element,face_rule,0,shapes);
      face.form = form;
      face.indices = [&](const auto &cell) { return indices(cell); };
      face.coefficients = [&](const auto &values,const auto &cell,unsigned int,bool)
      {
        std::vector<Tensor<1,dim>> speed(values.n_quadrature_points);
        std::vector<double> divergence(values.n_quadrature_points);
        velocity(cell,values.get_quadrature_points(),speed,divergence);
        typename internal::Assembly::Scratch::ScalarFace<dim>::Coefficients result(speed.size());
        for (unsigned int q = 0; q < speed.size(); ++q) result[q].velocity = speed[q];
        return result;
      };
      // Stage-time values are applied later to these spatial boundary loads.
      face.boundary_value = [](types::boundary_id,const Point<dim> &) { return 0.; };
      for (const auto &cell : cells)
        {
          fe.reinit(typename Triangulation<dim>::cell_iterator(cell.iterator));
          internal::Assembly::Scratch::ScalarCell<dim> scratch(fe);
          scratch.velocity.resize(rule.size());
          scratch.divergence.resize(rule.size());
          scratch.advection = 1.;
          scratch.form = form;
          velocity(cell.iterator,fe.get_quadrature_points(),scratch.velocity,scratch.divergence);
          internal::Assembly::CopyData::AdvectionSystem<dim> data(element,true);
          data.local_dof_indices = cell.indices;
          if (cell_assembler) cell_assembler->execute(scratch,data);
          else Assemblers::AdvectionSystem<dim>::assemble_spatial(scratch,data);
          for (unsigned int q = 0; q < rule.size(); ++q)
            {
              SourceSample sample;
              sample.indices = cell.indices;
              sample.weight = fe.JxW(q) * scratch.divergence[q];
              sample.shape.reinit(cell.indices.size());
              for (unsigned int i = 0; i < cell.indices.size(); ++i) sample.shape[i] = fe.shape_value(i,q);
              sources.push_back(std::move(sample));
            }
          face.cell = cell.iterator;
          std::vector<double> maximum_outflow(cell.iterator->n_faces(),0.);
          face.observe = [&](const auto &,unsigned int f,const auto &values,const auto &coefficients,unsigned int patches)
          {
            for (unsigned int q = 0; q < values.n_quadrature_points; ++q)
              maximum_outflow[f] = std::max(maximum_outflow[f],
                (coefficients[q].velocity * values.normal_vector(q)) * values.JxW(q) / face_rule.weight(q) * patches);
          };
          face.boundary_load = [&](const auto &native,const auto &values,unsigned int q,const auto &load,const auto &trace,double flux)
          {
            Vector<double> mass_load(load.size());
            cell.inverse_mass.vmult(mass_load,load);
            boundary.push_back({native->face(face.face_number)->boundary_id(),values.quadrature_point(q),
                                cell.indices,std::move(mass_load),trace,flux});
          };
          assemble_faces(face,data,transport_matrix);
          const double jacobian = minimum_cell_jacobian(mapping,element,cell.iterator);
          const double trace_bound = (element.degree+1)*element.degree;
          const double loss = trace_bound * std::accumulate(maximum_outflow.begin(),maximum_outflow.end(),0.) / jacobian;
          const double divergence = *std::min_element(scratch.divergence.begin(),scratch.divergence.end());
          mean_rate = std::max(mean_rate,loss+std::max(0.,-divergence));
        }
      transport_matrix.compress(VectorOperation::add);
      mean_rate = Utilities::MPI::max(mean_rate,communicator);
    }

    template <int dim>
    void DGOperator<dim>::evaluate(const LinearAlgebra::Vector &state, LinearAlgebra::Vector &rhs,
                                   const BoundaryField &inflow, double time, double &source, double &flux) const
    {
      LinearAlgebra::Vector ghost(owned, relevant, communicator);
      ghost = state;
      transport_matrix.vmult(rhs,state);
      for (const auto &cell : cells)
        {
          Vector<double> local(cell.indices.size()), product(cell.indices.size());
          for (unsigned int i = 0; i < local.size(); ++i) local[i] = rhs[cell.indices[i]];
          cell.inverse_mass.vmult(product,local);
          for (unsigned int i = 0; i < local.size(); ++i) rhs[cell.indices[i]] = -product[i];
        }
      // Complete the cellwise overwrite collectively before ranks with inflow
      // add boundary loads. Mixing Insert/Add only on boundary ranks can enter
      // Trilinos' implicit assembly collectives in different orders.
      rhs.compress(VectorOperation::insert);
      source = 0.;
      flux = 0.;
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
      rhs.compress(VectorOperation::add);
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
      bytes += transport_matrix.memory_consumption();
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
