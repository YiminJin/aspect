#include "runtime.h"
#include <aspect/boundary_velocity/interface.h>
#include <aspect/geometry_model/box.h>
#include <aspect/plugins.h>
#include <deal.II/fe/fe_values.h>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>

namespace aspect
{
namespace BP3Restore
{
std::string bottom_velocity_constraint = "full";
Tensor<1, 2> bottom_tangent;

template <int dim>
void
constrain_bottom (const SimulatorAccess<dim> &sim,
                  AffineConstraints<double> &constraints)
{
  if (bottom_velocity_constraint == "full")
    return;
  AssertThrow (dim == 2,
               ExcMessage ("Rotated BP3 bottom is two-dimensional."));
  const auto &box
      = Plugins::get_plugin_as_type<const GeometryModel::Box<dim>> (
          sim.get_geometry_model ());
  const auto bottom = box.translate_symbolic_boundary_name_to_id ("bottom");
  const auto &bc = sim.get_boundary_velocity_manager ();
  AssertThrow (
      !bc.get_prescribed_boundary_velocity_indicators ().count (bottom)
          && !bc.get_zero_boundary_velocity_indicators ().count (bottom)
          && !bc.get_tangential_boundary_velocity_indicators ().count (bottom),
      ExcMessage ("Remove bottom from all ordinary velocity boundary lists "
                  "for fault parallel loading."));
  const auto &fe = sim.get_fe ();
  using Pair = std::array<types::global_dof_index, 2>;
  const auto point_less = [] (const Point<dim> &a, const Point<dim> &b)
    { return a[0] < b[0] || (a[0] == b[0] && a[1] < b[1]); };
  std::map<Point<dim>, Pair, decltype (point_less)> trace (point_less);
  std::vector<types::global_dof_index> dofs (fe.n_dofs_per_cell ());
  for (const auto &cell : sim.get_dof_handler ().active_cell_iterators ())
    if (!cell->is_artificial ())
      for (unsigned face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
        if (cell->face (face)->at_boundary ()
            && cell->face (face)->boundary_id () == bottom)
          {
            cell->get_dof_indices (dofs);
            for (unsigned j = 0; j < dofs.size (); ++j)
              if (fe.has_support_on_face (j, face))
                for (unsigned d = 0; d < 2; ++d)
                  if (fe.system_to_component_index (j).first
                      == sim.introspection ().component_indices.velocities[d])
                    {
                      const auto p
                          = sim.get_mapping ().transform_unit_to_real_cell (
                              cell, fe.get_unit_support_points ()[j]);
                      if (std::abs (p[1] - box.get_origin ()[1]) > 1e-10)
                        continue;
                      auto entry
                          = trace
                                .emplace (
                                    p, Pair{ { numbers::invalid_dof_index,
                                               numbers::invalid_dof_index } })
                                .first;
                      entry->second[d] = dofs[j];
                    }
          }
  std::map<types::global_dof_index, types::global_dof_index> mate;
  for (const auto &entry : trace)
    {
      AssertThrow (entry.second[0] != numbers::invalid_dof_index
                       && entry.second[1] != numbers::invalid_dof_index,
                   ExcMessage ("Incomplete bottom support-point pair."));
      mate[entry.second[0]] = entry.second[1];
    }
  unsigned rows = 0, corners = 0, hanging = 0, covered = 0;
  const double tx = bottom_tangent[0], ty = bottom_tangent[1];
  const auto is_corner = [&] (const Point<dim> &p)
    {
      return std::abs (p[0] - box.get_origin ()[0]) < 1e-10
             || std::abs (p[0] - box.get_origin ()[0] - box.get_extents ()[0])
                    < 1e-10;
    };
  // ASPECT also builds homogeneous velocity constraints during initial field
  // setup. Mirror the existing side-boundary lift rather than guessing from
  // the nonlinear iteration (the coupled solver can request a physical lift
  // while that counter is nonzero). Side loading is nonzero at both corners.
  unsigned physical_corners = 0, homogeneous_corners = 0;
  for (const auto &[p, pair] : trace)
    if (is_corner (p)
        && sim.get_dof_handler ().locally_owned_dofs ().is_element (pair[1]))
      {
        const auto prescribed = loading (sim, p);
        double physical_error = 0., homogeneous_error = 0.;
        for (unsigned d = 0; d < 2; ++d)
          {
            AssertThrow (
                constraints.is_constrained (pair[d])
                    && constraints.get_constraint_entries (pair[d])->empty (),
                ExcMessage ("Bottom corner must retain full side loading."));
            const auto value = constraints.get_inhomogeneity (pair[d]);
            physical_error
                = std::max (physical_error, std::abs (value - prescribed[d]));
            homogeneous_error = std::max (homogeneous_error, std::abs (value));
          }
        AssertThrow (physical_error < 1e-22 || homogeneous_error < 1e-22,
                     ExcMessage ("Bottom corner has neither physical nor "
                                 "homogeneous side loading."));
        physical_corners += (physical_error < 1e-22);
        homogeneous_corners += (homogeneous_error < 1e-22);
      }
  const auto mpi = sim.get_mpi_communicator ();
  physical_corners = Utilities::MPI::sum (physical_corners, mpi);
  homogeneous_corners = Utilities::MPI::sum (homogeneous_corners, mpi);
  AssertThrow ((physical_corners == 2 && homogeneous_corners == 0)
                   || (physical_corners == 0 && homogeneous_corners == 2),
               ExcMessage ("Missing or inconsistent bottom corner lift."));
  const double lift_scale = physical_corners ? 1. : 0.;
  for (const auto &[p, pair] : trace)
    {
      const auto ix = pair[0], iy = pair[1];
      if (!constraints.can_store_line (ix) || !constraints.can_store_line (iy))
        continue;
      const bool owned
          = sim.get_dof_handler ().locally_owned_dofs ().is_element (iy);
      covered += owned;
      const auto prescribed = loading (sim, p);
      const double gt = lift_scale * (tx * prescribed[0] + ty * prescribed[1]);
      if (is_corner (p))
        {
          AssertThrow (
              constraints.is_constrained (ix)
                  && constraints.is_constrained (iy)
                  && constraints.get_constraint_entries (ix)->empty ()
                  && constraints.get_constraint_entries (iy)->empty (),
              ExcMessage ("Bottom corner must retain full side loading."));
          AssertThrow (std::abs (tx * constraints.get_inhomogeneity (ix)
                                 + ty * constraints.get_inhomogeneity (iy)
                                 - gt)
                           < 1e-22,
                       ExcMessage ("Incompatible corner loading."));
          corners += owned;
          continue;
        }
      if (constraints.is_constrained (ix) || constraints.is_constrained (iy))
        {
          // Boundary hanging rows must be identical scalar interpolants in
          // both components. Their independent masters receive the relation.
          AssertThrow (
              constraints.is_constrained (ix)
                  && constraints.is_constrained (iy),
              ExcMessage ("Unpaired pre-existing bottom constraint."));
          const auto &cx = *constraints.get_constraint_entries (ix),
                     &cy = *constraints.get_constraint_entries (iy);
          AssertThrow (!cx.empty () && cx.size () == cy.size (),
                       ExcMessage ("Unexpected bottom constraint conflict."));
          for (const auto &x : cx)
            {
              AssertThrow (mate.count (x.first),
                           ExcMessage ("Hanging bottom master is not in the "
                                       "locally relevant trace."));
              const auto y
                  = std::find_if (cy.begin (), cy.end (), [&] (const auto &v)
                                    { return v.first == mate.at (x.first); });
              AssertThrow (
                  y != cy.end () && std::abs (y->second - x.second) < 1e-14,
                  ExcMessage ("Mismatched hanging bottom interpolants."));
            }
          hanging += owned;
          continue;
        }
      constraints.add_line (iy);
      constraints.add_entry (iy, ix, -tx / ty);
      constraints.set_inhomogeneity (iy, gt / ty);
      rows += owned;
    }
  covered = Utilities::MPI::sum (covered, mpi);
  rows = Utilities::MPI::sum (rows, mpi);
  corners = Utilities::MPI::sum (corners, mpi);
  hanging = Utilities::MPI::sum (hanging, mpi);
  if (sim.get_pcout ().is_active ())
    {
      const auto path
          = sim.get_output_directory () + "bottom_constraint_rows.csv";
      const bool exists = std::ifstream (path).good ();
      std::ofstream out (path, std::ios::app);
      if (!exists)
        out << "step,support_points,independent_rows,corners,hanging,tx,ty,"
               "physical_lift\n";
      out << std::setprecision (17) << sim.get_timestep_number () << ','
          << covered << ',' << rows << ',' << corners << ',' << hanging << ','
          << tx << ',' << ty << ',' << lift_scale << '\n';
    }
}
template void constrain_bottom (const SimulatorAccess<2> &,
                                AffineConstraints<double> &);
template void constrain_bottom (const SimulatorAccess<3> &,
                                AffineConstraints<double> &);
}
}
