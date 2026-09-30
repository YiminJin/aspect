/* Copyright (C) 2026 by the authors of ASPECT.
 * This file is part of ASPECT and is distributed under the GNU GPL v2 or later.
 */
#ifndef _aspect_reconstructed_fault_boundary_contact_h
#define _aspect_reconstructed_fault_boundary_contact_h

#include <array>
#include <aspect/reconstructed_fault/utilities.h>
#include <deal.II/base/types.h>

namespace aspect
{
  /** One physical boundary facet, with the geometry of its adjacent cell. */
  template <int dim> struct FaultBoundaryFace
  {
    types::boundary_id boundary_id;
    std::array<Point<dim>, 2> vertices;
    Tensor<1, dim> inward_normal;
    Point<dim> cell_lower, cell_upper;
    bool periodic = false;

    template <class Archive>
    void
    serialize (Archive &ar, const unsigned int)
    {
      ar &boundary_id &vertices[0] & vertices[1] & inward_normal;
      ar &cell_lower &cell_upper &periodic;
    }
  };

  /** Geometry only: no profile evaluator, coefficients, or completed integrals. */
  template <int dim> struct FaultBoundaryContact
  {
    unsigned int fault_index, segment_index;
    /** 0/1 denotes first/last endpoint; invalid denotes an unrepresented crossing. */
    unsigned int endpoint = numbers::invalid_unsigned_int;
    types::boundary_id boundary_id;
    Point<dim> position;
    Tensor<1, dim> inward_tangent, normal, inward_boundary_normal;
    double straight_length = 0.;
    unsigned int terminal_segments = 0;
    /** A geometric enclosure of the full Q1 support, never an activation cutoff. */
    double transverse_extent = 0., influence_length = 0.;
    std::string unsupported_reason;
  };

  namespace ReconstructedFaultUtilities
  {
    /** 256 eps times the boundary coordinate/diameter scale (units of length). */
    template <int dim>
    double boundary_contact_tolerance (const std::vector<FaultBoundaryFace<dim>> &faces);

    /** Inspect all segments; deduplicate shared facet/polyline vertices. */
    template <int dim>
    std::vector<FaultBoundaryContact<dim>>
                                        detect_boundary_contacts (const std::vector<PrescribedInitialFault<dim>> &faults,
                                                                  const std::vector<FaultBoundaryFace<dim>> &faces);

    /** Check straight support, other branches/faces and overlapping contact regions. */
    template <int dim>
    void qualify_boundary_contact_support (std::vector<FaultBoundaryContact<dim>> &contacts,
                                           const std::vector<PrescribedInitialFault<dim>> &faults,
                                           const std::vector<FaultBoundaryFace<dim>> &faces,
                                           const std::vector<double> &transverse_extents,
                                           const std::vector<double> &branch_extents);
  }
}
#endif
