/* Copyright (C) 2026 by the authors of ASPECT.
 * This file is part of ASPECT and is distributed under the GNU GPL v2 or later.
 */
#include <algorithm>
#include <aspect/geometry_model/interface.h>
#include <aspect/reconstructed_fault/manager.h>
#include <deal.II/base/mpi.h>
#include <set>

namespace aspect
{
  template <int dim>
  void
  ReconstructedFaultManager<dim>::prepare_boundary_contacts ()
  {
    if (!automatic_boundary_completion)
      return;
    std::vector<std::uint64_t> versions;
    for (const auto &fault : reconstructed_faults)
      versions.push_back (fault.geometry_version ());
    if (boundary_contacts_valid && versions == boundary_fault_versions)
      return;
    AssertThrow (dim == 2,
                 ExcMessage ("Automatic boundary completion does not support 3D contacts."));
    std::set<types::boundary_id> periodic;
    for (const auto &pair : this->get_geometry_model ().get_periodic_boundary_pairs ())
      {
        periodic.insert (pair.first.first);
        periodic.insert (pair.first.second);
      }
    std::vector<FaultBoundaryFace<dim>> local;
    for (const auto &cell : this->get_triangulation ().active_cell_iterators ())
      if (cell->is_locally_owned ())
        for (const auto face : cell->face_indices ())
          if (cell->face (face)->at_boundary ())
            {
              FaultBoundaryFace<dim> entry;
              entry.boundary_id = cell->face (face)->boundary_id ();
              entry.periodic = periodic.count (entry.boundary_id);
              for (unsigned int v = 0; v < 2; ++v)
                entry.vertices[v] = cell->face (face)->vertex (v);
              const auto tangent = entry.vertices[1] - entry.vertices[0];
              entry.inward_normal[0] = -tangent[1];
              entry.inward_normal[1] = tangent[0];
              entry.inward_normal /= entry.inward_normal.norm ();
              if (entry.inward_normal * (cell->center () - entry.vertices[0]) < 0.)
                entry.inward_normal *= -1.;
              const auto bounds = cell->bounding_box ().get_boundary_points ();
              entry.cell_lower = bounds.first;
              entry.cell_upper = bounds.second;
              local.push_back (entry);
            }
    boundary_faces.clear ();
    for (const auto &faces : Utilities::MPI::all_gather (this->get_mpi_communicator (), local))
      boundary_faces.insert (boundary_faces.end (), faces.begin (), faces.end ());
    boundary_geometry_tolerance
      = ReconstructedFaultUtilities::boundary_contact_tolerance (boundary_faces);
    boundary_contacts
      = ReconstructedFaultUtilities::detect_boundary_contacts (prescribed_faults, boundary_faces);
    for (unsigned int c = 0; c < boundary_contacts.size (); ++c)
      {
        const auto &contact = boundary_contacts[c];
        AssertThrow (
          contact.unsupported_reason.empty (),
          ExcMessage ("Automatic boundary completion: fault " + std::to_string (contact.fault_index)
                      + ", contact " + std::to_string (c) + ", boundary "
                      + std::to_string (contact.boundary_id) + ": " + contact.unsupported_reason
                      + ". No endpoint/history remapping is performed."));
      }
    boundary_fault_versions = std::move (versions);
    boundary_contacts_valid = true;
    automatic_source_ready = false;
    ++boundary_contact_generation;
    invalidate_stokes_qp_projection_cache ();
  }

  template <int dim>
  void
  ReconstructedFaultManager<dim>::enable_automatic_source_continuation (
    const std::vector<double> &extents, const std::vector<double> &branch_extents)
  {
    AssertThrow (automatic_boundary_completion && boundary_contacts_valid,
                 ExcMessage ("Prepare automatic boundary geometry before enabling continuation."));
    ReconstructedFaultUtilities::qualify_boundary_contact_support (
      boundary_contacts, prescribed_faults, boundary_faces, extents, branch_extents);
    for (unsigned int c = 0; c < boundary_contacts.size (); ++c)
      {
        const auto &contact = boundary_contacts[c];
        AssertThrow (contact.unsupported_reason.empty (),
                     ExcMessage ("Automatic boundary completion: fault "
                                 + std::to_string (contact.fault_index) + ", contact "
                                 + std::to_string (c) + ": " + contact.unsupported_reason + "."));
      }
    if (!automatic_source_ready)
      invalidate_stokes_qp_projection_cache ();
    automatic_source_ready = true;
  }

#define INSTANTIATE(dim)                                                                           \
  template void ReconstructedFaultManager<dim>::prepare_boundary_contacts ();                      \
  template void ReconstructedFaultManager<dim>::enable_automatic_source_continuation (             \
      const std::vector<double> &, const std::vector<double> &);
  ASPECT_INSTANTIATE (INSTANTIATE)
#undef INSTANTIATE
}
