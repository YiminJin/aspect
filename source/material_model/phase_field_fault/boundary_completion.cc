/* Copyright (C) 2026 by the authors of ASPECT.
 * This file is part of ASPECT and is distributed under the GNU GPL v2 or later.
 */
#include <algorithm>
#include <aspect/geometry_model/box.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/plugins.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/reconstructed_fault/surface_system.h>
#include <cstdlib>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/mapping_cartesian.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_q1.h>
#include <fstream>
#include <iomanip>
#include <limits>

namespace aspect
{
  namespace internal
  {
    void throw_if_history_error (const std::string &, const MPI_Comm);
  }
  namespace MaterialModel
  {
    template <int dim>
    void
    PhaseFieldFault<dim>::prepare_automatic_boundary_completion ()
    {
      auto &manager = this->get_reconstructed_fault_manager ();
      AssertThrow (dim == 2 && mature_frictional_fault && !evolve_phase_field,
                   ExcMessage ("Automatic boundary completion requires a permanently prescribed frozen "
                               "mature 2-D field."));
      AssertThrow (
        boundary_normalization_completion_file.empty ()
        && !std::getenv ("ASPECT_IH_BOTTOM_COMPLETION_DIAGNOSTIC"),
        ExcMessage ("Legacy and automatic normalization completion are mutually exclusive."));
      AssertThrow (
        !this->get_parameters ().mesh_deformation_enabled,
        ExcMessage ("Automatic exterior Q1 continuation does not support mesh deformation."));
      const auto &mapping = this->get_mapping ();
      AssertThrow (Plugins::plugin_type_matches<GeometryModel::Box<dim>> (this->get_geometry_model ())
                   && (typeid (mapping) == typeid (MappingCartesian<dim>)
                       || typeid (mapping) == typeid (MappingQ1<dim>)
                       || (typeid (mapping) == typeid (MappingQ<dim>)
                           && static_cast<const MappingQ<dim> &> (mapping).get_degree () == 1)),
                   ExcMessage ("Automatic exterior data require an axis-aligned affine Box Q1 lattice; "
                               "other boundary mappings need a specified exterior mesh."));
      manager.prepare_boundary_contacts ();
      const auto &contacts = manager.get_boundary_contacts ();
      const auto &faces = manager.get_boundary_faces ();
      const auto &prescribed = manager.get_prescribed_faults ();
      const auto communicator = this->get_mpi_communicator ();
      const auto &box
        = Plugins::get_plugin_as_type<const GeometryModel::Box<dim>> (this->get_geometry_model ());
      const double tolerance = ReconstructedFaultUtilities::boundary_contact_tolerance (faces);
      const auto uniform = [] (const auto &values)
      {
        return std::all_of (values.begin (), values.end (),
                            [&] (const double x)
        {
          return x == values.front ();
        });
      };
      AssertThrow (
        contacts.empty ()
        || (uniform (elastic_shear_moduli) && uniform (cohesions)
            && uniform (critical_energy_release_rates)),
        ExcMessage ("Automatic exterior material data are unqualified: this implementation requires "
                    "identical degradation/profile coefficients in every material phase."));
      if (boundary_continuation_generation != manager.get_boundary_contact_generation ())
        {
          std::vector<BoundaryContinuation> candidate (contacts.size ());
          std::vector<double> extents (contacts.size ());
          for (unsigned int c = 0; c < contacts.size (); ++c)
            {
              const auto &contact = contacts[c];
              auto &data = candidate[c];
              const auto &fault = prescribed[contact.fault_index];
              const double core = contact.endpoint == 0 ? fault.core_phase_field_values.front ()
                                  : fault.core_phase_field_values.back ();
              auto profiles = this->get_phase_field_handler ().get_phase_field_profiles (core);
              data.profile = std::move (profiles.front ());
              data.lattice_origin = box.get_origin ();
              bool found = false;
              for (const auto &face : faces)
                if (face.boundary_id == contact.boundary_id)
                  {
                    const auto edge = face.vertices[1] - face.vertices[0];
                    const double xi
                      = (contact.position - face.vertices[0]) * edge / edge.norm_square ();
                    if (xi >= -tolerance / edge.norm () && xi <= 1. + tolerance / edge.norm ()
                        && std::abs ((contact.position - face.vertices[0]) * face.inward_normal)
                        <= tolerance)
                      {
                        data.spacing = face.cell_upper - face.cell_lower;
                        found = true;
                        break;
                      }
                  }
              AssertThrow (found,
                           ExcMessage ("No exterior Q1 lattice at the detected boundary contact."));
              double padding = 0.;
              for (const auto &cell : this->get_triangulation ().active_cell_iterators ())
                if (cell->is_locally_owned ())
                  {
                    const auto bounds = cell->bounding_box ().get_boundary_points ();
                    double width = 0.;
                    for (unsigned int d = 0; d < dim; ++d)
                      width += (bounds.second[d] - bounds.first[d]) * std::abs (contact.normal[d]);
                    padding = std::max (padding, width);
                  }
              padding = Utilities::MPI::max (padding, communicator);
              data.extent = data.profile->get_coordinate_values ().back () + padding;
              extents[c] = data.extent;
              const double a = contact.inward_boundary_normal * contact.inward_tangent;
              const double influence
                = data.extent * std::abs (contact.inward_boundary_normal * contact.normal) / a;
              for (unsigned int j = 0; j + 1 < fault.vertices.size (); ++j)
                {
                  const double lo = (fault.vertices[j] - contact.position) * contact.inward_tangent;
                  const double hi = (fault.vertices[j + 1] - contact.position) * contact.inward_tangent;
                  if (std::min (lo, hi) <= influence + tolerance && std::max (lo, hi) >= -tolerance)
                    AssertThrow (fault.core_phase_field_values[j] == core
                                 && fault.core_phase_field_values[j + 1] == core,
                                 ExcMessage ("Exterior continuation requires constant prescribed core "
                                             "phi in its terminal neighborhood."));
                }
              // The boundary trace determines a unique, uniform local ghost grid.
              // Refinement elsewhere is allowed; a changing boundary grid needs
              // an explicitly specified exterior discretization instead.
              for (const auto &face : faces)
                if (face.boundary_id == contact.boundary_id)
                  {
                    auto tangent = face.vertices[1] - face.vertices[0];
                    tangent /= tangent.norm ();
                    const double x = (face.vertices[0] - contact.position) * tangent;
                    const double y = (face.vertices[1] - contact.position) * tangent;
                    if (std::min (x, y) > data.extent / a || std::max (x, y) < -data.extent / a)
                      continue;
                    for (unsigned int d = 0; d < dim; ++d)
                      AssertThrow (
                        std::abs (face.cell_upper[d] - face.cell_lower[d] - data.spacing[d])
                        <= tolerance
                        && std::abs ((face.cell_lower[d] - data.lattice_origin[d])
                                     / data.spacing[d]
                                     - std::round ((face.cell_lower[d] - data.lattice_origin[d])
                                                   / data.spacing[d]))
                        <= tolerance / data.spacing[d],
                        ExcMessage ("Nonuniform or misaligned boundary ghost-Q1 lattice at fault "
                                    + std::to_string (contact.fault_index)
                                    + "; exterior mesh data are required."));
                  }
            }
          double cell_padding = 0.;
          for (const auto &cell : this->get_triangulation ().active_cell_iterators ())
            if (cell->is_locally_owned ())
              cell_padding = std::max (cell_padding, cell->diameter ());
          cell_padding = Utilities::MPI::max (cell_padding, communicator);
          std::vector<double> branch_extents;
          for (const auto &fault : prescribed)
            {
              AssertThrow (uniform (fault.core_phase_field_values),
                           ExcMessage ("Automatic prescribed-field qualification requires constant core phi per fault."));
              auto profiles = this->get_phase_field_handler ().get_phase_field_profiles (
                                fault.core_phase_field_values.front ());
              branch_extents.push_back (profiles.front ()->get_coordinate_values ().back () + cell_padding);
            }
          manager.enable_automatic_source_continuation (extents, branch_extents);
          boundary_continuations = std::move (candidate);
          boundary_continuation_generation = manager.get_boundary_contact_generation ();
          for (unsigned int f = 0; f < prescribed.size (); ++f)
            {
              bool any = false;
              for (unsigned int c = 0; c < contacts.size (); ++c)
                if (contacts[c].fault_index == f)
                  {
                    any = true;
                    this->get_pcout ()
                        << "Automatic boundary contact: fault=" << f << ", contact=" << c
                        << ", endpoint=" << contacts[c].endpoint
                        << ", boundary=" << contacts[c].boundary_id
                        << ", position=" << contacts[c].position
                        << ", influence=" << contacts[c].influence_length << ", treatment="
                        << (contacts[c].influence_length == 0. ? "perpendicular, zero correction"
                            : "paired completion")
                        << std::endl;
                  }
              if (!any)
                this->get_pcout () << "Automatic boundary contact: fault=" << f
                                   << ", interior/no nonperiodic contact" << std::endl;
            }
        }

      // Flags alone cannot qualify an H-driven initialization. Every phase DoF
      // must be constrained, and physical Q1 nodal data throughout the domain
      // must match the same stationary profile used outside.
      const auto &fe = this->get_fe ();
      const unsigned int component
        = this->introspection ().variable ("phase_field").first_component_index;
      const auto &constraints = this->get_current_constraints ();
      std::vector<std::unique_ptr<PhaseField::PhaseFieldProfile>> prescribed_profiles;
      std::vector<std::array<bool, 2>> extended (prescribed.size (), { { false, false } });
      for (const auto &contact : contacts)
        extended[contact.fault_index][contact.endpoint] = true;
      for (const auto &fault : prescribed)
        {
          AssertThrow (
            uniform (fault.core_phase_field_values),
            ExcMessage (
              "Automatic prescribed-field qualification currently requires constant core phi "
              "per fault; varying interior profiles require a defined compatible evaluator."));
          auto profiles = this->get_phase_field_handler ().get_phase_field_profiles (
                            fault.core_phase_field_values.front ());
          prescribed_profiles.push_back (std::move (profiles.front ()));
        }
      std::vector<types::global_dof_index> dofs (fe.n_dofs_per_cell ());
      unsigned int compatible = 1;
      unsigned int affine = 1;
      double error = 0.;
      for (const auto &cell : this->get_dof_handler ().active_cell_iterators ())
        if (cell->is_locally_owned ())
          {
            const auto bounds = cell->bounding_box ().get_boundary_points ();
            for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
              for (unsigned int d = 0; d < dim; ++d)
                affine &= std::abs (cell->vertex (v)[d]
                                    - ((v & (1u << d)) ? bounds.second[d] : bounds.first[d]))
                          <= tolerance;
            cell->get_dof_indices (dofs);
            for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
              compatible
              &= constraints.is_constrained (dofs[fe.component_to_system_index (component, v)]);
            for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
              {
                double expected = 0.;
                unsigned int active = 0;
                for (unsigned int f = 0; f < prescribed.size (); ++f)
                  {
                    const auto &vertices = prescribed[f].vertices;
                    double distance = std::numeric_limits<double>::max ();
                    for (unsigned int j = 0; j + 1 < vertices.size (); ++j)
                      {
                        const auto tangent = vertices[j + 1] - vertices[j];
                        double xi = (cell->vertex (v) - vertices[j]) * tangent / tangent.norm_square ();
                        if (j != 0 || !extended[f][0])
                          xi = std::max (0., xi);
                        if (j + 2 != vertices.size () || !extended[f][1])
                          xi = std::min (1., xi);
                        distance = std::min (distance,
                                             cell->vertex (v).distance (vertices[j] + xi * tangent));
                      }
                    const double value = prescribed_profiles[f]->value (distance);
                    active += value > 0.;
                    expected = std::max (expected, value);
                  }
                // No superposition law is introduced for overlapping profiles.
                compatible &= active <= 1;
                const double actual
                  = this->get_solution ()[dofs[fe.component_to_system_index (component, v)]];
                if (!std::isfinite (actual))
                  compatible = 0;
                error = std::max (error, std::abs (actual - expected));
              }
          }
      compatible = Utilities::MPI::min (compatible, communicator);
      affine = Utilities::MPI::min (affine, communicator);
      error = Utilities::MPI::max (error, communicator);
      AssertThrow (
        affine,
        ExcMessage ("Automatic exterior continuation requires axis-aligned affine physical cells."));
      AssertThrow (compatible && error <= 2e-11,
                   ExcMessage ("Automatic boundary completion requires verified compatible fully "
                               "prescribed Q1 phase data; "
                               "unconstrained/H-driven initialization or profile mismatch (max error="
                               + std::to_string (error)
                               + ") needs the separate phase-field boundary treatment. Freezing alone "
                               "is insufficient."));
      this->get_reconstructed_fault_surface_system ().enable_bulk_work_measure ();
    }

    template <int dim>
    void
    PhaseFieldFault<dim>::apply_automatic_boundary_completion (
      const std::vector<NormalizationProfile> &profiles, std::vector<double> &integrals) const
    {
      std::string error;
      try
        {
          const auto &contacts = this->get_reconstructed_fault_manager ().get_boundary_contacts ();
          const QGauss<1> quadrature (8);
          std::ofstream out (
            this->get_output_directory () + "ih_automatic_completion_rank"
            + std::to_string (Utilities::MPI::this_mpi_process (this->get_mpi_communicator ()))
            + ".csv");
          out.exceptions (std::ios::failbit | std::ios::badbit);
          out << std::setprecision (17) << "id,fault,contact,inside,outside,completed\n";
          for (unsigned int p = 0; p < profiles.size (); ++p)
            for (unsigned int c = 0; c < contacts.size (); ++c)
              {
                const auto &profile = profiles[p];
                const auto &contact = contacts[c];
                if (profile.fault_index != contact.fault_index || contact.influence_length == 0.)
                  continue;
                const auto &data = boundary_continuations[c];
                const double s = (profile.origin - contact.position) * contact.inward_tangent;
                if (s < 0. || s > contact.influence_length)
                  continue;
                const double b = contact.inward_boundary_normal * profile.normal;
                if (b == 0.)
                  continue;
                const double cut
                  = -(profile.origin - contact.position) * contact.inward_boundary_normal / b;
                double lo = -data.extent, hi = data.extent;
                if (b > 0.)
                  hi = std::min (hi, cut);
                else
                  lo = std::max (lo, cut);
                if (lo >= hi)
                  continue;
                const auto value = [&] (const double r)
                {
                  const Point<dim> point = profile.origin + r * profile.normal;
                  Point<dim> lower;
                  Tensor<1, dim> fraction;
                  for (unsigned int d = 0; d < dim; ++d)
                    {
                      const double index
                        = std::floor ((point[d] - data.lattice_origin[d]) / data.spacing[d]);
                      lower[d] = data.lattice_origin[d] + index * data.spacing[d];
                      fraction[d] = (point[d] - lower[d]) / data.spacing[d];
                    }
                  double phi = 0.;
                  for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
                    {
                      auto node = lower;
                      double weight = 1.;
                      for (unsigned int d = 0; d < dim; ++d)
                        if (v & (1u << d))
                          {
                            node[d] += data.spacing[d];
                            weight *= fraction[d];
                          }
                        else
                          weight *= 1. - fraction[d];
                      phi += weight
                             * data.profile->value (
                               std::abs ((node - contact.position) * contact.normal));
                    }
                  return normalization_integrand (
                           phi,
                           this->get_phase_field_handler ().energetic_degradation (
                             profile.material_fractions, phi),
                           "automatic exterior Q1 completion");
                };
                std::vector<double> cuts { lo, hi };
                for (unsigned int d = 0; d < dim; ++d)
                  if (profile.normal[d] != 0.)
                    {
                      double left = profile.origin[d] + lo * profile.normal[d],
                             right = profile.origin[d] + hi * profile.normal[d];
                      if (left > right)
                        std::swap (left, right);
                      const long begin = std::floor ((left - data.lattice_origin[d]) / data.spacing[d]);
                      const long end = std::ceil ((right - data.lattice_origin[d]) / data.spacing[d]);
                      for (long i = begin; i <= end; ++i)
                        {
                          const double r
                            = (data.lattice_origin[d] + i * data.spacing[d] - profile.origin[d])
                              / profile.normal[d];
                          if (r > lo && r < hi)
                            cuts.push_back (r);
                        }
                    }
                std::sort (cuts.begin (), cuts.end ());
                cuts.erase (std::unique (cuts.begin (), cuts.end ()), cuts.end ());
                const auto panel = [&] (const double a, const double b)
                {
                  double sum = 0.;
                  for (unsigned int q = 0; q < quadrature.size (); ++q)
                    sum += quadrature.weight (q) * value (a + (b - a) * quadrature.point (q)[0]);
                  return (b - a) * sum;
                };
                // The legacy BP3 ghost-Q1 table uses cell-edge cuts and an 8-point
                // rule versus two half panels at this fixed verification accuracy.
                const auto integrate = [&] (auto &&self, const double a, const double b,
                                            const unsigned int depth) -> double
                {
                  const double low = panel (a, b), mid = .5 * (a + b),
                  high = panel (a, mid) + panel (mid, b);
                  if (std::abs (high - low) <= 1e-11 * std::max (1., std::abs (high)))
                    return high;
                  AssertThrow (
                    depth < 20,
                    ExcMessage ("Automatic exterior Q1 completion did not resolve a panel."));
                  return self (self, a, mid, depth + 1) + self (self, mid, b, depth + 1);
                };
                double outside = 0.;
                for (unsigned int i = 1; i < cuts.size (); ++i)
                  outside += integrate (integrate, cuts[i - 1], cuts[i], 0);
                const double inside = integrals[p];
                integrals[p] += outside;
                out << profile.id << ',' << profile.fault_index << ',' << c << ',' << inside << ','
                    << outside << ',' << integrals[p] << '\n';
              }
        }
      catch (const std::exception &exception)
        {
          error = exception.what ();
        }
      aspect::internal::throw_if_history_error (error, this->get_mpi_communicator ());
    }

#define INSTANTIATE(dim)                                                                           \
  template void PhaseFieldFault<dim>::prepare_automatic_boundary_completion ();                    \
  template void PhaseFieldFault<dim>::apply_automatic_boundary_completion (                        \
      const std::vector<NormalizationProfile> &, std::vector<double> &) const;
    ASPECT_INSTANTIATE (INSTANTIATE)
#undef INSTANTIATE
  }
}
