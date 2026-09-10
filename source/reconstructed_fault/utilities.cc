/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include <aspect/reconstructed_fault/utilities.h>
#include <aspect/utilities.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/quadrature_lib.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace aspect
{
  namespace
  {
    // -----------------------------------------------------------------------------
    // Fault reconstruction helpers
    // -----------------------------------------------------------------------------

    template <int dim>
    std::vector<Point<dim>>
    resample_polyline(const std::vector<Point<dim>> &vertices,
                      const double spacing)
    {
      AssertThrow(dim == 2, ExcNotImplemented());
      AssertThrow(vertices.size() >= 2, ExcMessage("A reference fault requires at least two vertices."));
      AssertThrow(std::isfinite(spacing) && spacing > 0.0,
                  ExcMessage("The fault structural point spacing must be positive and finite."));

      std::vector<double> cumulative_length(vertices.size(), 0.0);
      for (unsigned int i = 1; i < vertices.size(); ++i)
        {
          const double segment_length = vertices[i].distance(vertices[i-1]);
          AssertThrow(std::isfinite(segment_length) && segment_length > 0.0,
                      ExcMessage("The reference fault contains a degenerate segment."));
          cumulative_length[i] = cumulative_length[i-1] + segment_length;
        }

      const double length = cumulative_length.back();
      const unsigned int n_segments = std::max(1U, static_cast<unsigned int>(std::ceil(length / spacing)));
      const double structural_spacing = length / n_segments;
      std::vector<Point<dim>> points(n_segments + 1);
      unsigned int input_segment = 0;
      for (unsigned int i = 0; i <= n_segments; ++i)
        {
          const double s = (i == n_segments ? length : i * structural_spacing);
          while (input_segment + 1 < cumulative_length.size() - 1
                 && s > cumulative_length[input_segment+1])
            ++input_segment;
          const double local_coordinate =
            (s - cumulative_length[input_segment])
            / (cumulative_length[input_segment+1] - cumulative_length[input_segment]);
          points[i] = vertices[input_segment]
                      + local_coordinate * (vertices[input_segment+1] - vertices[input_segment]);
        }
      return points;
    }


    // -----------------------------------------------------------------------------
    // Projection linear algebra helpers
    // -----------------------------------------------------------------------------

    std::pair<std::vector<double>, std::vector<double>>
    factor_tridiagonal_utility_system(const std::vector<double> &diagonal,
                                      const std::vector<double> &off_diagonal)
    {
      AssertThrow(!diagonal.empty(), ExcMessage("A projection system must not be empty."));
      AssertThrow(off_diagonal.size() + 1 == diagonal.size(),
                  ExcMessage("A tridiagonal projection system requires one fewer "
                             "off-diagonal entry than diagonal entries."));
      AssertThrow(std::all_of(diagonal.begin(), diagonal.end(),
                              [](const double value)
      {
        return std::isfinite(value);
      })
      && std::all_of(off_diagonal.begin(), off_diagonal.end(),
                     [](const double value)
      {
        return std::isfinite(value);
      }),
      ExcMessage("The particle-to-fault projection matrix is non-finite."));
      const double scale = *std::max_element(diagonal.begin(), diagonal.end());
      AssertThrow(scale > 0.0,
                  ExcMessage("The particle-to-fault projection matrix has no positive diagonal."));
      const double tolerance = std::numeric_limits<double>::epsilon()
                               * std::max(1.0, static_cast<double>(diagonal.size())) * scale;

      std::vector<double> factor_diagonal(diagonal.size());
      std::vector<double> factor_lower(off_diagonal.size());
      factor_diagonal[0] = diagonal[0];
      AssertThrow(std::isfinite(factor_diagonal[0]) && factor_diagonal[0] > tolerance,
                  ExcMessage("The particle-to-fault projection matrix is singular at its first vertex."));
      for (unsigned int i = 1; i < diagonal.size(); ++i)
        {
          factor_lower[i-1] = off_diagonal[i-1] / factor_diagonal[i-1];
          factor_diagonal[i] = diagonal[i] - factor_lower[i-1] * off_diagonal[i-1];
          AssertThrow(std::isfinite(factor_diagonal[i]) && factor_diagonal[i] > tolerance,
                      ExcMessage("The particle-to-fault projection matrix is singular at vertex "
                                 + Utilities::int_to_string(i) + "."));
        }
      return {factor_diagonal, factor_lower};
    }


    std::vector<double>
    solve_tridiagonal_utility_factors(const std::vector<double> &factor_diagonal,
                                      const std::vector<double> &factor_lower,
                                      const ArrayView<const double> &rhs)
    {
      std::vector<double> solution(rhs.begin(), rhs.end());
      for (unsigned int i = 1; i < solution.size(); ++i)
        solution[i] -= factor_lower[i-1] * solution[i-1];
      for (unsigned int i = 0; i < solution.size(); ++i)
        solution[i] /= factor_diagonal[i];
      for (unsigned int i = solution.size() - 1; i > 0; --i)
        solution[i-1] -= factor_lower[i-1] * solution[i];
      AssertThrow(std::all_of(solution.begin(), solution.end(),
                              [](const double value)
      {
        return std::isfinite(value);
      }),
      ExcMessage("The tridiagonal projection solve produced a non-finite result."));
      return solution;
    }

  }

  namespace ReconstructedFaultUtilities
  {
    // -----------------------------------------------------------------------------
    // Prescribed-fault input and geometry
    // -----------------------------------------------------------------------------

    template <int dim>
    std::vector<PrescribedInitialFault<dim>>
    parse_prescribed_faults(const std::string &file_contents,
                            const std::string &filename)
    {
      std::vector<PrescribedInitialFault<dim>> faults;
      PrescribedInitialFault<dim> fault;
      std::istringstream input(file_contents);
      std::string line;
      unsigned int line_number = 0;

      const auto finish_fault = [&]()
      {
        AssertThrow(fault.vertices.size() >= 2,
                    ExcMessage("Prescribed-fault file <" + filename
                               + "> contains a fault with fewer than two vertices near line "
                               + Utilities::int_to_string(line_number) + "."));
        faults.push_back(std::move(fault));
        fault = PrescribedInitialFault<dim>();
      };

      while (std::getline(input, line))
        {
          ++line_number;
          const std::size_t comment_position = line.find('#');
          if (comment_position != std::string::npos)
            line.erase(comment_position);

          std::istringstream line_stream(line);
          std::string first_entry;
          if (!(line_stream >> first_entry))
            continue;

          if (first_entry == "---")
            {
              std::string extra_entry;
              AssertThrow(!(line_stream >> extra_entry),
                          ExcMessage("Fault separator on line "
                                     + Utilities::int_to_string(line_number)
                                     + " of <" + filename + "> must contain only `---'."));
              AssertThrow(!fault.vertices.empty(),
                          ExcMessage("Empty fault before separator on line "
                                     + Utilities::int_to_string(line_number)
                                     + " of <" + filename + ">."));
              finish_fault();
              continue;
            }

          Point<dim> point;
          try
            {
              point[0] = Utilities::string_to_double(first_entry);
            }
          catch (const std::exception &)
            {
              AssertThrow(false,
                          ExcMessage("Could not read the first coordinate on line "
                                     + Utilities::int_to_string(line_number)
                                     + " of prescribed-fault file <" + filename + ">."));
            }
          for (unsigned int d = 1; d < dim; ++d)
            AssertThrow(line_stream >> point[d],
                        ExcMessage("Line " + Utilities::int_to_string(line_number)
                                   + " of prescribed-fault file <" + filename
                                   + "> contains fewer than "
                                   + Utilities::int_to_string(dim) + " coordinates."));
          double phi_hat;
          AssertThrow(line_stream >> phi_hat,
                      ExcMessage("Line " + Utilities::int_to_string(line_number)
                                 + " of prescribed-fault file <" + filename
                                 + "> is missing the core phase-field value."));
          std::string extra_entry;
          AssertThrow(!(line_stream >> extra_entry),
                      ExcMessage("Line " + Utilities::int_to_string(line_number)
                                 + " of prescribed-fault file <" + filename
                                 + "> contains too many entries."));
          for (unsigned int d = 0; d < dim; ++d)
            AssertThrow(std::isfinite(point[d]),
                        ExcMessage("A coordinate on line "
                                   + Utilities::int_to_string(line_number)
                                   + " of prescribed-fault file <" + filename
                                   + "> is not finite."));
          AssertThrow(std::isfinite(phi_hat),
                      ExcMessage("The core phase-field value on line "
                                 + Utilities::int_to_string(line_number)
                                 + " of prescribed-fault file <" + filename
                                 + "> is not finite."));
          fault.vertices.push_back(point);
          fault.core_phase_field_values.push_back(phi_hat);
        }

      if (!fault.vertices.empty())
        finish_fault();
      AssertThrow(!faults.empty(),
                  ExcMessage("Prescribed-fault file <" + filename
                             + "> does not contain any faults."));
      return faults;
    }


    template <int dim>
    std::pair<double, double>
    closest_point_distance_and_core_phase_field(const PrescribedInitialFault<dim> &fault,
                                                const Point<dim> &position)
    {
      AssertThrow(dim == 2,
                  ExcMessage("Prescribed initial fault geometry is currently implemented only in 2D."));
      AssertThrow(fault.vertices.size() >= 2,
                  ExcMessage("A prescribed initial fault requires at least two vertices."));
      AssertThrow(fault.core_phase_field_values.size() == fault.vertices.size(),
                  ExcMessage("A prescribed initial fault requires one core phase-field value per vertex."));
      for (unsigned int d = 0; d < dim; ++d)
        AssertThrow(std::isfinite(position[d]),
                    ExcMessage("Closest-point evaluation requires a finite position."));
      AssertThrow(std::all_of(fault.core_phase_field_values.begin(),
                              fault.core_phase_field_values.end(),
                              [](const double value)
      {
        return std::isfinite(value);
      }),
      ExcMessage("A prescribed initial fault contains a non-finite core phase-field value."));

      double min_r_squared = std::numeric_limits<double>::infinity();
      double phi_hat = numbers::signaling_nan<double>();

      for (unsigned int segment = 0; segment < fault.vertices.size() - 1; ++segment)
        {
          const Tensor<1,dim> segment_vector =
            fault.vertices[segment+1] - fault.vertices[segment];
          const double squared_segment_length = segment_vector.norm_square();
          AssertThrow(std::isfinite(squared_segment_length) && squared_segment_length > 0.0,
                      ExcMessage("Prescribed initial fault segment "
                                 + Utilities::int_to_string(segment) + " is degenerate."));

          const double segment_coordinate =
            std::clamp(((position - fault.vertices[segment]) * segment_vector)
                       / squared_segment_length,
                       0.0,
                       1.0);
          const Point<dim> closest_point = fault.vertices[segment]
                                           + segment_coordinate * segment_vector;
          const double r_squared = position.distance_square(closest_point);

          if (r_squared < min_r_squared)
            {
              min_r_squared = r_squared;
              phi_hat =
                (1.0 - segment_coordinate) * fault.core_phase_field_values[segment]
                + segment_coordinate * fault.core_phase_field_values[segment+1];
            }
        }

      return {std::sqrt(min_r_squared), phi_hat};
    }


    // -----------------------------------------------------------------------------
    // Normal-profile projection implementation
    // -----------------------------------------------------------------------------

    namespace internal
    {
      template <int dim>
      void
      validate_normal_profile_projection_geometry(
        const std::vector<ReconstructedFault<dim>> &faults,
        const std::vector<std::vector<double>> &half_widths)
      {
        AssertThrow(dim == 2, ExcNotImplemented());
        AssertThrow(half_widths.size() == faults.size(),
                    ExcMessage("Particle projection requires one half-width vector per fault."));

        for (unsigned int fault_index = 0; fault_index < faults.size(); ++fault_index)
          {
            const ReconstructedFault<dim> &fault = faults[fault_index];
            AssertThrow(fault.n_vertices() >= 2,
                        ExcMessage("Particle projection requires faults with at least two vertices."));
            AssertThrow(fault.vertex(0) != fault.vertex(fault.n_vertices()-1),
                        ExcMessage("Closed-loop faults are unsupported by particle projection."));
            AssertThrow(half_widths[fault_index].size() == fault.n_vertices(),
                        ExcMessage("Particle projection requires one influence half-width "
                                   "per fault vertex."));
            AssertThrow(std::all_of(half_widths[fault_index].begin(),
                                    half_widths[fault_index].end(),
                                    [](const double width)
            {
              return std::isfinite(width) && width > 0.0;
            }),
            ExcMessage("Particle projection requires positive finite influence half-widths."));
          }
      }


      template <int dim>
      void
      validate_normal_profile_projection_position(const Point<dim> &position)
      {
        for (unsigned int d = 0; d < dim; ++d)
          AssertThrow(std::isfinite(position[d]),
                      ExcMessage("Particle projection requires a finite position."));
      }


      template <int dim>
      ReconstructedFaultUtilities::NormalProfileProjection
      project_to_normal_profiles_unchecked(
        const std::vector<ReconstructedFault<dim>> &faults,
        const std::vector<std::vector<double>> &half_widths,
        const Point<dim> &position)
      {
        ReconstructedFaultUtilities::NormalProfileProjection result;
        unsigned int admitted_faults = 0;

        for (unsigned int fault_index = 0; fault_index < faults.size(); ++fault_index)
          {
            const ReconstructedFault<dim> &fault = faults[fault_index];
            bool admitted_to_fault = false;
            double smallest_distance = std::numeric_limits<double>::infinity();
            ReconstructedFaultUtilities::NormalProfileProjection candidate;
            for (unsigned int segment = 0; segment < fault.n_cells(); ++segment)
              {
                const Tensor<1,dim> segment_vector = fault.vertex(segment+1) - fault.vertex(segment);
                const double length_squared = segment_vector.norm_square();
                const double xi = ((position - fault.vertex(segment)) * segment_vector) / length_squared;
                if (xi < 0.0 || xi > 1.0)
                  continue;

                const double width = (1.0-xi) * half_widths[fault_index][segment]
                                     + xi * half_widths[fault_index][segment+1];
                Tensor<1,dim> tangent = segment_vector / std::sqrt(length_squared);
                const Tensor<1,dim> normal({-tangent[1], tangent[0]});
                const Point<dim> projected_point = fault.vertex(segment) + xi * segment_vector;
                const double signed_distance = (position - projected_point) * normal;
                const double distance = std::abs(signed_distance);
                if (distance <= width && distance < smallest_distance)
                  {
                    admitted_to_fault = true;
                    smallest_distance = distance;
                    candidate = {true, fault_index, segment, xi, signed_distance};
                  }
              }

            if (admitted_to_fault)
              {
                ++admitted_faults;
                result = candidate;
              }
          }

        AssertThrow(admitted_faults <= 1,
                    ExcMessage("A particle lies in the influence regions of multiple reconstructed faults. "
                               "Overlapping fault influence regions are unsupported."));
        return result;
      }
    }
  }

  namespace ReconstructedFaultUtilities
  {
    // -----------------------------------------------------------------------------
    // Normal-profile projection
    // -----------------------------------------------------------------------------

    template <int dim>
    NormalProfileProjection
    project_to_normal_profiles(
      const std::vector<ReconstructedFault<dim>> &faults,
      const std::vector<std::vector<double>> &half_widths,
      const Point<dim> &position)
    {
      internal::validate_normal_profile_projection_geometry(faults, half_widths);
      internal::validate_normal_profile_projection_position(position);
      return internal::project_to_normal_profiles_unchecked(
               faults, half_widths, position);
    }


    // -----------------------------------------------------------------------------
    // Projection linear algebra
    // -----------------------------------------------------------------------------

    std::vector<double>
    solve_tridiagonal_system(const std::vector<double> &diagonal,
                             const std::vector<double> &off_diagonal,
                             const std::vector<double> &rhs)
    {
      AssertThrow(rhs.size() == diagonal.size(),
                  ExcMessage("A tridiagonal projection right-hand side must match "
                             "the system dimension."));
      AssertThrow(std::all_of(rhs.begin(), rhs.end(),
                              [](const double value)
      {
        return std::isfinite(value);
      }),
      ExcMessage("The projection right-hand side is non-finite."));
      const auto factors =
        factor_tridiagonal_utility_system(diagonal, off_diagonal);
      return solve_tridiagonal_utility_factors(
               factors.first, factors.second, make_array_view(rhs));
    }


    // -----------------------------------------------------------------------------
    // Fault reconstruction utilities
    // -----------------------------------------------------------------------------

    template <int dim>
    std::vector<Point<dim>>
    resample_reference_fault(const std::vector<Point<dim>> &vertices,
                             const double structural_spacing)
    {
      return resample_polyline(vertices, structural_spacing);
    }


    std::vector<double>
    solve_normal_offsets(const std::vector<double> &matrix_values,
                         const std::vector<double> &rhs_values,
                         const double total_weight,
                         const double ridge_coefficient)
    {
      const unsigned int n_points = rhs_values.size();
      AssertThrow(n_points > 0 && matrix_values.size() == n_points*n_points,
                  ExcMessage("The normal-offset matrix and right-hand side dimensions do not match."));
      AssertThrow(std::isfinite(total_weight) && total_weight > 0.0,
                  ExcMessage("The total phase-field reconstruction weight must be positive."));
      AssertThrow(std::isfinite(ridge_coefficient) && ridge_coefficient >= 0.0,
                  ExcMessage("The fault reconstruction ridge coefficient must be nonnegative."));
      AssertThrow(std::all_of(matrix_values.begin(), matrix_values.end(),
                              [](const double value)
      {
        return std::isfinite(value);
      })
      && std::all_of(rhs_values.begin(), rhs_values.end(),
                     [](const double value)
      {
        return std::isfinite(value);
      }),
      ExcMessage("The normal-offset system contains a non-finite value."));

      FullMatrix<double> system(n_points, n_points);
      Vector<double> rhs(n_points), offsets(n_points);
      for (unsigned int i = 0; i < n_points; ++i)
        {
          rhs[i] = rhs_values[i] / total_weight;
          for (unsigned int j = 0; j < n_points; ++j)
            system(i,j) = matrix_values[i*n_points+j] / total_weight;
        }
      for (unsigned int j = 1; j + 1 < n_points; ++j)
        for (unsigned int a = 0; a < 3; ++a)
          for (unsigned int b = 0; b < 3; ++b)
            {
              const double d[3] = {1.0, -2.0, 1.0};
              system(j-1+a,j-1+b) += ridge_coefficient * d[a] * d[b];
            }

      system.gauss_jordan();
      system.vmult(offsets, rhs);
      std::vector<double> result(offsets.begin(), offsets.end());
      AssertThrow(std::all_of(result.begin(), result.end(),
                              [](const double value)
      {
        return std::isfinite(value);
      }),
      ExcMessage("The normal-offset solve produced a non-finite result."));
      return result;
    }
  }
}

namespace aspect
{
  // --- Full-domain surface quadrature
  namespace
  {
    using DomainPolygon = std::vector<Point<2>>;

    struct DomainPlane
    {
      Tensor<1,2> normal;
      double offset;
    };

    // Complementary half-planes share the very same computed intersections.
    // Equality belongs to the boundary, not to a finite-width tolerance band.
    std::pair<DomainPolygon,DomainPolygon>
    split_domain_polygon(const DomainPolygon &polygon, const DomainPlane &plane)
    {
      double minimum=std::numeric_limits<double>::max(), maximum=-minimum;
      std::vector<double> distance;
      for (const auto &point : polygon)
        {
          distance.push_back(plane.normal*point-plane.offset);
          minimum=std::min(minimum,distance.back());
          maximum=std::max(maximum,distance.back());
        }
      if (maximum<=0) return {polygon,{}};
      if (minimum>=0) return {{},polygon};
      std::pair<DomainPolygon,DomainPolygon> halves;
      for (unsigned int i=0;i<polygon.size();++i)
        {
          const unsigned int j=(i+1)%polygon.size();
          if (distance[i]<=0) halves.first.push_back(polygon[i]);
          if (distance[i]>=0) halves.second.push_back(polygon[i]);
          if ((distance[i]<0 && distance[j]>0) || (distance[i]>0 && distance[j]<0))
            {
              const Point<2> intersection=polygon[i]+distance[i]/(distance[i]-distance[j])
                                           *(polygon[j]-polygon[i]);
              halves.first.push_back(intersection);
              halves.second.push_back(intersection);
            }
        }
      return halves;
    }

    double domain_polygon_area(const DomainPolygon &polygon)
    {
      if (polygon.size()<3) return 0;
      long double twice_area=0;
      for (unsigned int i=1;i+1<polygon.size();++i)
        {
          const auto a=polygon[i]-polygon[0], b=polygon[i+1]-polygon[0];
          twice_area+=static_cast<long double>(a[0])*b[1]-static_cast<long double>(a[1])*b[0];
        }
      return .5*std::abs(twice_area);
    }

    Point<2> domain_polygon_center(const DomainPolygon &polygon)
    {
      Point<2> center;
      for (const auto &point : polygon) center+=point;
      center/=polygon.size();
      return center;
    }

    std::vector<DomainPolygon>
    partition_domain(std::vector<DomainPolygon> pieces, const std::vector<DomainPlane> &planes)
    {
      for (const auto &plane : planes)
        {
          std::vector<DomainPolygon> next;
          for (const auto &piece : pieces)
            {
              auto halves=split_domain_polygon(piece,plane);
              if (domain_polygon_area(halves.first)>0) next.push_back(std::move(halves.first));
              if (domain_polygon_area(halves.second)>0) next.push_back(std::move(halves.second));
            }
          pieces=std::move(next);
        }
      return pieces;
    }

    struct DomainSegment
    {
      Point<2> origin;
      Tensor<1,2> tangent,normal;
      double length,offset;
    };

    std::vector<DomainPlane>
    distance_bisectors(const DomainSegment &a, const DomainSegment &b)
    {
      return {{a.normal-b.normal,a.offset-b.offset},
              {a.normal+b.normal,a.offset+b.offset}};
    }

    std::vector<ReconstructedFaultUtilities::DomainQuadraturePoint>
    polyline_domain_quadrature(const DomainPolygon &vertices,
                               const ReconstructedFault<2> &fault,
                               const unsigned int order,
                               ReconstructedFaultUtilities::DomainQuadratureStatistics *statistics)
    {
      // Work near the parent domain to avoid subtracting large plane offsets.
      const Point<2> origin=vertices[0];
      DomainPolygon polygon, nodes;
      for (const auto &vertex : vertices) polygon.emplace_back(vertex-origin);
      for (const auto &vertex : fault.get_vertices()) nodes.emplace_back(vertex-origin);
      std::vector<DomainSegment> segments;
      std::vector<unsigned int> candidates;
      std::vector<DomainPlane> end_planes;
      for (unsigned int j=0;j<fault.n_cells();++j)
        {
          const auto delta=nodes[j+1]-nodes[j];
          const double length=delta.norm();
          const auto tangent=delta/length;
          const Tensor<1,2> normal({-tangent[1],tangent[0]});
          segments.push_back({nodes[j],tangent,normal,length,normal*nodes[j]});
          double first=std::numeric_limits<double>::max(), last=-first;
          for (const auto &point : polygon)
            {
              const double u=tangent*(point-nodes[j]);
              first=std::min(first,u); last=std::max(last,u);
            }
          if (last>=0 && first<=length)
            {
              candidates.push_back(j);
              end_planes.push_back({tangent,tangent*nodes[j]});
              end_planes.push_back({tangent,tangent*nodes[j+1]});
            }
        }
      if (statistics != nullptr)
        {
          statistics->segment_tests += fault.n_cells();
          statistics->candidate_segments += candidates.size();
        }
      std::vector<ReconstructedFaultUtilities::DomainQuadraturePoint> result;
      for (const auto &piece : partition_domain({polygon},end_planes))
        {
          const auto center=domain_polygon_center(piece);
          std::vector<unsigned int> active;
          for (const unsigned int j : candidates)
            {
              const double u=segments[j].tangent*(center-segments[j].origin);
              if (u>=0 && u<=segments[j].length) active.push_back(j);
            }
          if (!active.empty())
            {
              // Signed distances are affine; both factors of d_i^2-d_j^2
              // give straight bisectors. The winner is constant in each piece.
              std::vector<DomainPlane> bisectors;
              for (unsigned int i=0;i<active.size();++i)
                for (unsigned int j=i+1;j<active.size();++j)
                  for (const auto &plane : distance_bisectors(segments[active[i]],segments[active[j]]))
                    bisectors.push_back(plane);
              for (const auto &part : partition_domain({piece},bisectors))
                {
                  const auto point=domain_polygon_center(part);
                  unsigned int owner=active.front();
                  double distance=std::numeric_limits<double>::max();
                  for (const unsigned int j : active)
                    {
                      const double d=std::abs(segments[j].normal*point-segments[j].offset);
                      if (d<distance) { distance=d; owner=j; }
                    }
                  // Reuse the validated straight-segment integrator on this
                  // exact polygon. First and second moments stay distinct.
                  const ReconstructedFault<2> segment_fault({nodes[owner],nodes[owner+1]});
                  auto quadrature=ReconstructedFaultUtilities::domain_quadrature(part,segment_fault,order,statistics);
                  for (auto &q : quadrature) q.segment_index=owner;
                  result.insert(result.end(),quadrature.begin(),quadrature.end());
                }
            }
          else
            {
              // Fill only the finite-projection gaps. A vertex Voronoi partition
              // preserves the entire corner/tip measure with constant Q1 data.
              double radius=0, nearest=std::numeric_limits<double>::max();
              for (const auto &point : piece) radius=std::max(radius,point.distance(center));
              for (const auto &node : nodes) nearest=std::min(nearest,node.distance(center));
              std::vector<unsigned int> vertices_in_range;
              for (unsigned int i=0;i<nodes.size();++i)
                if (nodes[i].distance(center)<=nearest+2*radius) vertices_in_range.push_back(i);
              for (const unsigned int i : vertices_in_range)
                {
                  auto corner=piece;
                  for (const unsigned int j : vertices_in_range)
                    if (i!=j)
                      {
                        const auto delta=nodes[j]-nodes[i];
                        const DomainPlane plane={delta,delta*(nodes[i]+.5*delta)};
                        corner=split_domain_polygon(corner,plane).first;
                        if (domain_polygon_area(corner)==0) break;
                      }
                  if (domain_polygon_area(corner)==0) continue;
                  const unsigned int left=i==0 ? 0 : i-1;
                  const unsigned int right=i==nodes.size()-1 ? left : i;
                  for (const auto &part : partition_domain({corner},distance_bisectors(segments[left],segments[right])))
                    {
                      const auto point=domain_polygon_center(part);
                      const double dl=std::abs(segments[left].normal*point-segments[left].offset);
                      const double dr=std::abs(segments[right].normal*point-segments[right].offset);
                      const unsigned int owner=dl<=dr ? left : right;
                      result.push_back({owner,i==owner ? 0.0 : 1.0,domain_polygon_area(part)});
                    }
                }
            }
        }
      return result;
    }
  }

  namespace ReconstructedFaultUtilities
  {
    std::vector<DomainQuadraturePoint>
    domain_quadrature(const std::vector<Point<2>> &vertices,
                      const ReconstructedFault<2> &fault,
                      const unsigned int order,
                      DomainQuadratureStatistics *statistics)
    {
      AssertThrow(vertices.size() >= 3, ExcMessage("Surface quadrature requires a domain polygon."));
      const Point<2> origin = fault.vertex(0);
      Tensor<1,2> tangent = fault.vertex(fault.n_vertices()-1)-origin;
      const double length = tangent.norm();
      tangent /= length;
      Tensor<1,2> normal;
      normal[0] = -tangent[1];
      normal[1] = tangent[0];
      const double tolerance = 128*std::numeric_limits<double>::epsilon()*length;
      std::vector<double> nodes(fault.n_vertices());
      bool straight=true;
      for (unsigned int i=0; i<nodes.size(); ++i)
        {
          const Tensor<1,2> offset = fault.vertex(i)-origin;
          nodes[i] = offset*tangent;
          straight=straight && std::abs(offset*normal)<=tolerance
                   && (i==0 || nodes[i]>nodes[i-1]);
        }
      if (!straight)
        {
          if (statistics != nullptr) ++statistics->general_calls;
          return polyline_domain_quadrature(vertices,fault,order,statistics);
        }
      if (statistics != nullptr) ++statistics->straight_calls;

      // Between projected polygon vertices the transverse width is linear.
      // Additional fault-node cuts keep every surface basis segment-local.
      std::vector<Point<2>> polygon;
      std::vector<double> cuts;
      for (const auto &vertex : vertices)
        {
          const auto offset = vertex-origin;
          polygon.emplace_back(offset*tangent, offset*normal);
          cuts.push_back(polygon.back()[0]);
        }
      const auto bounds = std::minmax_element(cuts.begin(), cuts.end());
      const double first = *bounds.first, last = *bounds.second;
      for (const double node : nodes)
        if (node>first && node<last)
          cuts.push_back(node);
      std::sort(cuts.begin(), cuts.end());
      cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
      const QGauss<1> gauss(order);
      std::vector<DomainQuadraturePoint> result;
      for (unsigned int interval=1; interval<cuts.size(); ++interval)
        {
          const double a=cuts[interval-1], b=cuts[interval];
          // Ignore only intervals too narrow to represent interior Gauss points.
          if (b-a <= 8*std::numeric_limits<double>::epsilon()*std::max(std::abs(a),std::abs(b)))
            continue;
          const double center = .5*(a+b);
          const unsigned int segment = center<=0 ? 0 : center>=length ? fault.n_cells()-1
            : std::upper_bound(nodes.begin(),nodes.end(),center)-nodes.begin()-1;
          for (unsigned int q=0; q<gauss.size(); ++q)
            {
              const double s = a+(b-a)*gauss.point(q)[0];
              double bottom=std::numeric_limits<double>::max();
              double top=-bottom;
              for (unsigned int edge=0; edge<polygon.size(); ++edge)
                {
                  const auto &v=polygon[edge];
                  const auto &w=polygon[(edge+1)%polygon.size()];
                  if (s>std::min(v[0],w[0]) && s<std::max(v[0],w[0]))
                    {
                      const double y=v[1]+(s-v[0])/(w[0]-v[0])*(w[1]-v[1]);
                      bottom=std::min(bottom,y);
                      top=std::max(top,y);
                    }
                }
              AssertThrow(top>=bottom, ExcMessage("Invalid transverse domain cross-section."));
              // Constant continuation retains portions beyond true tips without
              // extrapolating constrained surface fields or joining endpoint DoFs.
              const double xi = std::max(0.0,std::min(1.0,
                (s-nodes[segment])/(nodes[segment+1]-nodes[segment])));
              result.push_back({segment,xi,(b-a)*gauss.weight(q)*(top-bottom)});
            }
        }
      return result;
    }
  }

#define INSTANTIATE(dim) \
  namespace ReconstructedFaultUtilities::internal \
  { \
    template void \
    validate_normal_profile_projection_geometry( \
                                                 const std::vector<ReconstructedFault<dim>> &, \
                                                 const std::vector<std::vector<double>> &); \
    template void \
    validate_normal_profile_projection_position(const Point<dim> &); \
    template NormalProfileProjection \
    project_to_normal_profiles_unchecked( \
                                          const std::vector<ReconstructedFault<dim>> &, \
                                          const std::vector<std::vector<double>> &, const Point<dim> &); \
  } \
  namespace ReconstructedFaultUtilities \
  { \
    template std::pair<double, double> \
    closest_point_distance_and_core_phase_field( \
                                                 const PrescribedInitialFault<dim> &, const Point<dim> &); \
    template std::vector<PrescribedInitialFault<dim>> \
    parse_prescribed_faults( \
                             const std::string &, const std::string &); \
    template std::vector<Point<dim>> \
    resample_reference_fault( \
                              const std::vector<Point<dim>> &, const double); \
    template NormalProfileProjection \
    project_to_normal_profiles( \
                                const std::vector<ReconstructedFault<dim>> &, \
                                const std::vector<std::vector<double>> &, const Point<dim> &); \
  }

  ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
}
