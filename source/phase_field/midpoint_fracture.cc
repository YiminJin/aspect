/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#include <aspect/phase_field/midpoint_fracture.h>
#include <deal.II/fe/fe_values.h>

namespace aspect::PhaseField
{
  FractureRows::FractureRows(const SimulatorAccess<2> &sim, const unsigned int composition)
  {
    const auto &model = sim.get_parameters().surface_energy_model;
    AssertThrow(&model.geometry == &CrackGeometry::select("AT1"),
                ExcMessage("Growing fault requires the AT1 surface energy."));
    const double gradient_coefficient = 2. * model.length_scale * model.length_scale;

    const auto &system = sim.get_fe();
    const auto component = sim.introspection().component_indices.compositional_fields[composition];
    const auto block = sim.introspection().block_indices.compositional_fields[composition];
    const auto block_start = sim.get_solution().get_block_indices().block_start(block);
    const auto &element = system.base_element(system.component_to_base_index(component).first);

    const QGauss<2> volume_rule(5);
    const QGauss<1> face_rule(5);
    const auto flags = update_values | update_gradients;
    FEValues<2> volume(sim.get_mapping(), element, volume_rule, flags | update_JxW_values);
    FEFaceValues<2> face(sim.get_mapping(), element, face_rule,
                         flags | update_JxW_values | update_normal_vectors | update_quadrature_points);
    const auto phase_block_indices = [&](const auto &cell)
    {
      std::vector<Index> global(system.n_dofs_per_cell()), indices(element.n_dofs_per_cell());
      cell->get_dof_indices(global);
      for (unsigned int i=0; i<indices.size(); ++i)
        indices[i] = global[system.component_to_system_index(component,i)] - block_start;
      return indices;
    };

    for (const auto &cell : sim.get_dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const auto rows = phase_block_indices(cell);
          volume.reinit(Triangulation<2>::cell_iterator(cell));
          for (unsigned int q=0; q<volume_rule.size(); ++q)
            for (unsigned int i=0; i<rows.size(); ++i)
              {
                mass[rows[i]] += volume.JxW(q) * volume.shape_value(i,q);
                for (unsigned int j=0; j<rows.size(); ++j)
                  stiffness[ {rows[i],rows[j]}] += volume.JxW(q) * gradient_coefficient
                                                   * (volume.shape_grad(i,q) * volume.shape_grad(j,q));
              }

          // Assemble only this cell's test rows. The neighbor assembles its
          // own rows using the opposite normal. Exterior faces are natural.
          for (const auto f : cell->face_indices())
            if (!cell->at_boundary(f))
              {
                const auto neighbor = cell->neighbor(f);
                AssertThrow(neighbor->is_active() && !cell->neighbor_is_coarser(f),
                            ExcMessage("Growing fault requires conforming faces."));
                face.reinit(Triangulation<2>::cell_iterator(cell),f);
                std::vector<Point<2>> neighbor_points;
                for (const auto &point : face.get_quadrature_points())
                  neighbor_points.push_back(sim.get_mapping().transform_real_to_unit_cell(neighbor,point));
                FEValues<2> neighbor_values(sim.get_mapping(),element,Quadrature<2>(neighbor_points),flags);
                neighbor_values.reinit(Triangulation<2>::cell_iterator(neighbor));
                const auto neighbor_columns = phase_block_indices(neighbor);
                const double penalty = sim.get_parameters().phase_discontinuous_penalty * std::pow(element.degree+1,2)
                                       * gradient_coefficient * cell->face(f)->measure()
                                       / std::min(cell->measure(),neighbor->measure());
                for (unsigned int q=0; q<face_rule.size(); ++q)
                  for (unsigned int i=0; i<rows.size(); ++i)
                    {
                      const double test = face.shape_value(i,q);
                      const double average_test_flux = .5 * gradient_coefficient
                                                       * (face.normal_vector(q) * face.shape_grad(i,q));
                      for (unsigned int j=0; j<rows.size(); ++j)
                        {
                          // The trial jump is the interior trace minus the exterior trace.
                          const double interior_jump = face.shape_value(j,q);
                          const double exterior_jump = -neighbor_values.shape_value(j,q);
                          stiffness[ {rows[i],rows[j]}] += face.JxW(q) *
                                                           ((penalty*test - average_test_flux)*interior_jump
                                                            - .5*gradient_coefficient*test*(face.normal_vector(q)*face.shape_grad(j,q)));
                          stiffness[ {rows[i],neighbor_columns[j]}] += face.JxW(q) *
                                                                       ((penalty*test - average_test_flux)*exterior_jump
                                                                        - .5*gradient_coefficient*test*(face.normal_vector(q)*neighbor_values.shape_grad(j,q)));
                        }
                    }
              }
        }
  }
}
