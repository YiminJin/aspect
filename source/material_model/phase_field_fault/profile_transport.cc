/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/material_model/phase_field_fault/material_history_internal.h>
#include <aspect/phase_field.h>
#include <deal.II/numerics/vector_tools_evaluate.h>

namespace aspect
{
  namespace MaterialModel
  {
    template <int dim>
    LinearAlgebra::BlockVector
    PhaseFieldFault<dim>::transport_material_profile(const MaterialHistoryInputs &input) const
    {
      const auto &old = input.accepted;
      const auto comm = this->get_mpi_communicator();
      const auto component = this->introspection().variable("phase_field").first_component_index;
      const auto velocity_component = this->introspection().component_indices.velocities[0];
      const double dt = input.dt;
      LinearAlgebra::BlockVector result(this->introspection().index_sets.system_partitioning, comm);
      result = old.bulk_history.native;
      if (this->get_phase_field_handler().uses_dg_phase())
        {
          this->get_phase_field_handler().transport_dg_phase(old.bulk_history.native,input.accepted_velocity,
                                                            input.end_time-dt,dt,result);
          return result;
        }
      std::map<types::global_dof_index, Point<dim>> locations;
      for (const auto &entry : old.cells)
        for (const auto vertex : entry.second->vertex_indices())
          {
            const auto index = entry.second->vertex_dof_index(vertex, component);
            if (result.locally_owned_elements().is_element(index))
              locations[index] = entry.second->vertex(vertex);
          }
      std::vector<Point<dim>> points;
      for (const auto &entry : locations)
        points.push_back(entry.second);
      Utilities::MPI::RemotePointEvaluation<dim> lookup;
      auto &cache = this->get_phase_field_handler().get_grid_cache();
      lookup.reinit(cache, points);
      const auto speeds = VectorTools::point_values<dim>(lookup, this->get_dof_handler(), input.accepted_velocity,
                                                         VectorTools::EvaluationFlags::avg, velocity_component);
      auto departure = points;
      for (unsigned i = 0; i < points.size(); ++i)
        departure[i] -= dt * speeds[i];
      lookup.reinit(cache, departure);
      const auto forward = VectorTools::point_values<1>(lookup, this->get_dof_handler(), old.bulk_history.native,
                                                        VectorTools::EvaluationFlags::avg, component);
      std::vector<double> lower(points.size(), 0.), upper(points.size(), 0.);
      if (material_bounded_profile)
        {
          // Local physical bounds belong to the accepted donor cells, including
          // all cells admitted by the same remote-point search convention.
          using Bounds = Tensor<1, 2>;
          const auto bounds = lookup.template evaluate_and_process<Bounds>(
                                [&](const ArrayView<Bounds> &values, const auto & data)
          {
            for (const auto c : data.cell_indices())
              {
                const auto cell = data.get_active_cell_iterator(c)->as_dof_handler_iterator(this->get_dof_handler());
                Bounds bound;
                bound[0] = std::numeric_limits<double>::max();
                bound[1] = 0.;
                for (const auto v : cell->vertex_indices())
                  {
                    const double phi = old.bulk_history.native[cell->vertex_dof_index(v, component)];
                    bound[0] = std::min(bound[0], phi);
                    bound[1] = std::max(bound[1], phi);
                  }
                for (auto &value : data.get_data_view(c, values))
                  value = bound;
              }
          });
          const auto &ptr = lookup.get_point_ptrs();
          for (unsigned i = 0; i < points.size(); ++i)
            if (ptr[i] < ptr[i + 1])
              {
                lower[i] = std::numeric_limits<double>::max();
                for (unsigned j = ptr[i]; j < ptr[i + 1]; ++j)
                  {
                    lower[i] = std::min(lower[i], bounds[j][0]);
                    upper[i] = std::max(upper[i], bounds[j][1]);
                  }
              }
        }
      unsigned i = 0;
      for (const auto &entry : locations)
        result[entry.first] = forward[i++];
      result.compress(VectorOperation::insert);
      if (material_bounded_profile)
        {
          LinearAlgebra::BlockVector predicted(old.bulk_history.native);
          predicted = result;
          auto return_points = points;
          for (unsigned j = 0; j < points.size(); ++j)
            return_points[j] += dt * speeds[j];
          lookup.reinit(cache, return_points);
          const auto reverse = VectorTools::point_values<1>(lookup, this->get_dof_handler(), predicted,
                                                            VectorTools::EvaluationFlags::avg, component);
          const auto &ptr = lookup.get_point_ptrs();
          i = 0;
          for (const auto &entry : locations)
            {
              // Where reverse characteristics leave coverage there is no error
              // estimate: retain the causal first-order forward update.
              const double corrected = ptr[i] < ptr[i + 1]
                                       ? forward[i] + .5 * (old.bulk_history.native[entry.first] - reverse[i]) : forward[i];
              result[entry.first] = std::clamp(corrected, lower[i], upper[i]);
              ++i;
            }
          result.compress(VectorOperation::insert);
        }
      for (const auto &entry : locations)
        AssertThrow(std::isfinite(result[entry.first]) && result[entry.first] >= 0. && result[entry.first] < 1.,
                    ExcMessage("Material profile transport violated physical bounds."));
      // In the admitted growth-disabled configuration this very same transported
      // field is the material irreversibility obstacle. No Eulerian max is taken.
      return result;
    }
#define INSTANTIATE(dim) \
  template LinearAlgebra::BlockVector PhaseFieldFault<dim>::transport_material_profile(const MaterialHistoryInputs &) const;
    ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
  }
}
