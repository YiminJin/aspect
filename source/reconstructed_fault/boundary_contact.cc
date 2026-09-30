/* Copyright (C) 2026 by the authors of ASPECT.
 * This file is part of ASPECT and is distributed under the GNU GPL v2 or later.
 */
#include <algorithm>
#include <aspect/global.h>
#include <aspect/reconstructed_fault/boundary_contact.h>
#include <cmath>
#include <limits>

namespace aspect
{
  namespace ReconstructedFaultUtilities
  {
    template <int dim>
    double
    boundary_contact_tolerance (const std::vector<FaultBoundaryFace<dim>> &faces)
    {
      double scale = 0.;
      for (const auto &face : faces)
        for (const auto &p : face.vertices)
          scale = std::max ({ scale, p.norm (), (face.cell_upper - face.cell_lower).norm () });
      AssertThrow (scale > 0., ExcMessage ("Boundary contact detection requires a physical boundary."));
      return 256. * std::numeric_limits<double>::epsilon () * scale;
    }

    template <int dim>
    std::vector<FaultBoundaryContact<dim>>
                                        detect_boundary_contacts (const std::vector<PrescribedInitialFault<dim>> &faults,
                                                                  const std::vector<FaultBoundaryFace<dim>> &faces)
    {
      AssertThrow (dim == 2, ExcMessage ("Automatic fault boundary contacts are unsupported in 3D."));
      const double tolerance = boundary_contact_tolerance (faces);
      std::vector<FaultBoundaryContact<dim>> contacts;
      const auto cross
        = [] (const Tensor<1, dim> &u, const Tensor<1, dim> &v)
      {
        return u[0] * v[1] - u[1] * v[0];
      };
      for (unsigned int f = 0; f < faults.size (); ++f)
        {
          const auto &vertices = faults[f].vertices;
          for (unsigned int j = 0; j + 1 < vertices.size (); ++j)
            for (const auto &face : faces)
              {
                if (face.periodic)
                  continue;
                const auto u = vertices[j + 1] - vertices[j];
                const auto v = face.vertices[1] - face.vertices[0];
                const double length = u.norm (), face_length = v.norm ();
                AssertThrow (length > 0. && face_length > 0.,
                             ExcMessage ("Degenerate boundary/contact segment."));
                const double denominator = cross (u, v);
                double xi;
                const bool parallel
                  = std::abs (denominator) <= tolerance * std::min (length, face_length);
                if (parallel)
                  {
                    if (std::abs ((vertices[j] - face.vertices[0]) * face.inward_normal) > tolerance)
                      continue;
                    const double lo = (face.vertices[0] - vertices[j]) * u / (length * length);
                    const double hi = (face.vertices[1] - vertices[j]) * u / (length * length);
                    const double left = std::max (0., std::min (lo, hi));
                    const double right = std::min (1., std::max (lo, hi));
                    if (left > right + tolerance / length)
                      continue;
                    xi = std::clamp (left, 0., 1.);
                  }
                else
                  {
                    const auto offset = face.vertices[0] - vertices[j];
                    xi = cross (offset, v) / denominator;
                    const double eta = cross (offset, u) / denominator;
                    if (xi < -tolerance / length || xi > 1. + tolerance / length
                        || eta < -tolerance / face_length || eta > 1. + tolerance / face_length)
                      continue;
                    xi = std::clamp (xi, 0., 1.);
                  }
                FaultBoundaryContact<dim> contact;
                contact.fault_index = f;
                contact.segment_index = j;
                contact.boundary_id = face.boundary_id;
                contact.position = vertices[j] + xi * u;
                contact.inward_boundary_normal = face.inward_normal;
                const auto tangent = u / length;
                contact.normal[0] = -tangent[1];
                contact.normal[1] = tangent[0];
                if (contact.position.distance (vertices.front ()) <= tolerance)
                  contact.endpoint = 0;
                else if (contact.position.distance (vertices.back ()) <= tolerance)
                  contact.endpoint = 1;
                contact.inward_tangent = contact.endpoint == 1 ? -tangent : tangent;
                const double a = contact.inward_tangent * face.inward_normal;
                if (parallel || std::abs (a) <= 256. * std::numeric_limits<double>::epsilon ())
                  contact.unsupported_reason = "tangential contact";
                else if (contact.endpoint == numbers::invalid_unsigned_int)
                  contact.unsupported_reason
                    = "segment crossing needs endpoint/topology and history remapping";
                else if (a <= 0.)
                  contact.unsupported_reason = "endpoint does not enter the physical domain";
                else
                  {
                    for (unsigned int k = 0; k + 1 < vertices.size (); ++k)
                      {
                        const unsigned int i = contact.endpoint == 0 ? k : vertices.size () - 1 - k;
                        const unsigned int next = contact.endpoint == 0 ? i + 1 : i - 1;
                        const auto step = vertices[next] - vertices[i];
                        const auto offset = vertices[next] - contact.position;
                        if (step * contact.inward_tangent <= 0.
                            || (offset - (offset * contact.inward_tangent) * contact.inward_tangent)
                            .norm ()
                            > tolerance)
                          break;
                        contact.straight_length = offset * contact.inward_tangent;
                        contact.terminal_segments = k + 1;
                      }
                  }
                auto existing = std::find_if (contacts.begin (), contacts.end (),
                                              [&] (const auto &other)
                {
                  return other.fault_index == f
                         && other.position.distance (contact.position)
                         <= tolerance;
                });
                if (existing == contacts.end ())
                  contacts.push_back (contact);
                else if (existing->boundary_id != contact.boundary_id
                         || (existing->inward_boundary_normal - contact.inward_boundary_normal).norm ()
                         > 1e-12)
                  existing->unsupported_reason = "corner or nonplanar boundary contact";
                else if (!contact.unsupported_reason.empty ())
                  existing->unsupported_reason = contact.unsupported_reason;
              }
        }
      std::sort (contacts.begin (), contacts.end (),
                 [] (const auto &a, const auto &b)
      {
        return std::make_tuple (a.fault_index, a.endpoint, a.segment_index, a.boundary_id)
               < std::make_tuple (b.fault_index, b.endpoint, b.segment_index,
                                  b.boundary_id);
      });
      return contacts;
    }

    template <int dim>
    void
    qualify_boundary_contact_support (std::vector<FaultBoundaryContact<dim>> &contacts,
                                      const std::vector<PrescribedInitialFault<dim>> &faults,
                                      const std::vector<FaultBoundaryFace<dim>> &faces,
                                      const std::vector<double> &extents,
                                      const std::vector<double> &branch_extents)
    {
      AssertDimension (faults.size (), branch_extents.size ());
      for (const double extent : branch_extents)
        AssertThrow (std::isfinite (extent) && extent >= 0.,
                     ExcMessage ("Invalid diffuse branch support enclosure."));
      AssertDimension (contacts.size (), extents.size ());
      const double tolerance = boundary_contact_tolerance (faces);
      for (unsigned int c = 0; c < contacts.size (); ++c)
        {
          auto &contact = contacts[c];
          if (!contact.unsupported_reason.empty ())
            continue;
          AssertThrow (std::isfinite (extents[c]) && extents[c] >= 0.,
                       ExcMessage ("Invalid contact support enclosure."));
          contact.transverse_extent = extents[c];
          const double a = contact.inward_boundary_normal * contact.inward_tangent;
          const double b = contact.inward_boundary_normal * contact.normal;
          contact.influence_length = extents[c] * std::abs (b) / a;
          if (contact.influence_length > contact.straight_length + tolerance)
            contact.unsupported_reason = "curved/short terminal neighborhood within completion support";
          // A perpendicular crossing has exactly zero completion measure.
          if (b == 0.)
            continue;
          const auto intersects = [&] (const Point<dim> &p, const Point<dim> &q, const double padding)
          {
            double lo = 0., hi = 1.;
            for (unsigned int d = 0; d < 2; ++d)
              {
                const auto &axis = d == 0 ? contact.inward_tangent : contact.normal;
                const double radius = (d == 0 ? contact.influence_length : extents[c]) + padding;
                const double x = (p - contact.position) * axis, delta = (q - p) * axis;
                if (delta == 0.)
                  {
                    if (std::abs (x) > radius + tolerance)
                      return false;
                  }
                else
                  {
                    double left = (-radius - x) / delta, right = (radius - x) / delta;
                    if (left > right)
                      std::swap (left, right);
                    lo = std::max (lo, left);
                    hi = std::min (hi, right);
                  }
              }
            return lo <= hi;
          };
          for (const auto &face : faces)
            if (!face.periodic && face.boundary_id != contact.boundary_id
                && intersects (face.vertices[0], face.vertices[1], 0.))
              contact.unsupported_reason = "completion support reaches a second boundary face";
          for (unsigned int f = 0; f < faults.size (); ++f)
            for (unsigned int j = 0; j + 1 < faults[f].vertices.size (); ++j)
              {
                const auto &p = faults[f].vertices[j], &q = faults[f].vertices[j + 1];
                if (f == contact.fault_index
                    && (contact.endpoint == 0
                        ? j < contact.terminal_segments
                        : j >= faults[f].vertices.size () - 1 - contact.terminal_segments))
                  continue;
                // Reject a diffuse branch reaching the completion enclosure, even
                // when its centerline remains outside. No nearest-branch rule is added.
                if (intersects (p, q, branch_extents[f]))
                  contact.unsupported_reason = "ambiguous projection onto another fault branch";
              }
        }
      // Separating-axis test of support rectangles: conservative rejection avoids
      // assigning an overlapping completion to a nearest contact without a law.
      for (unsigned int i = 0; i < contacts.size (); ++i)
        for (unsigned int j = 0; j < i; ++j)
          {
            auto &a = contacts[i];
            auto &b = contacts[j];
            if (a.influence_length == 0. || b.influence_length == 0.)
              continue;
            bool separated = false;
            for (const auto &axis :
            {
              a.inward_tangent, a.normal, b.inward_tangent, b.normal
            })
            {
              const double width = a.influence_length * std::abs (axis * a.inward_tangent)
                                   + a.transverse_extent * std::abs (axis * a.normal)
                                   + b.influence_length * std::abs (axis * b.inward_tangent)
                                   + b.transverse_extent * std::abs (axis * b.normal);
              separated |= std::abs ((a.position - b.position) * axis) > width + tolerance;
            }
            if (!separated)
              a.unsupported_reason = b.unsupported_reason = "overlapping completion regions";
          }
    }

#define INSTANTIATE(dim)                                                                           \
  template double boundary_contact_tolerance (const std::vector<FaultBoundaryFace<dim>> &);        \
  template std::vector<FaultBoundaryContact<dim>> detect_boundary_contacts (                       \
      const std::vector<PrescribedInitialFault<dim>> &,                                            \
      const std::vector<FaultBoundaryFace<dim>> &);                                                \
  template void qualify_boundary_contact_support (                                                 \
      std::vector<FaultBoundaryContact<dim>> &, const std::vector<PrescribedInitialFault<dim>> &,  \
      const std::vector<FaultBoundaryFace<dim>> &, const std::vector<double> &, \
      const std::vector<double> &);
    ASPECT_INSTANTIATE (INSTANTIATE)
#undef INSTANTIATE
  }
}
