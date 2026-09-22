/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#ifndef _aspect_reconstructed_fault_utilities_h
#define _aspect_reconstructed_fault_utilities_h

#include <aspect/reconstructed_fault/fault.h>

#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/base/signaling_nan.h>

#include <string>
#include <utility>
#include <vector>

namespace aspect
{
  /**
   * Prescribed geometry and core phase-field values for one initial fault.
   * Core values are specified at the polyline vertices and interpolated
   * linearly along each segment.
   */
  template <int dim>
  struct PrescribedInitialFault
  {
    std::vector<Point<dim>> vertices;
    std::vector<double> core_phase_field_values;
  };

  namespace ReconstructedFaultUtilities
  {
    /** Q1 interpolation of nonnegative absolute slip rates. Preserve the nodal
     * convex hull and exact endpoint/contact values in floating-point arithmetic.
     * This does not impose V_min or repair inadmissible nodal inputs. */
    double interpolate_slip_rate(const double left, const double right, const double xi);

    /**
     * @name Prescribed-fault input and geometry
     * @{
     */
    /** Parse prescribed faults from the documented ASCII representation. */
    template <int dim>
    std::vector<PrescribedInitialFault<dim>>
    parse_prescribed_faults(const std::string &file_contents,
                            const std::string &filename);

    /** Distance and interpolated core value at the closest point on a fault. */
    template <int dim>
    std::pair<double, double>
    closest_point_distance_and_core_phase_field(const PrescribedInitialFault<dim> &fault,
                                                const Point<dim> &position);
    /** @} */

    /**
     * @name Fault reconstruction
     * @{
     */
    /** Resample an ordered polyline at approximately uniform arc length. */
    template <int dim>
    std::vector<Point<dim>>
    resample_reference_fault(const std::vector<Point<dim>> &vertices,
                             const double structural_spacing);

    /** Solve the globally assembled, total-weight-normalized ridge system. */
    std::vector<double>
    solve_normal_offsets(const std::vector<double> &matrix,
                         const std::vector<double> &rhs,
                         const double total_weight,
                         const double ridge_coefficient);
    /** @} */

    /**
     * @name Normal-profile projection
     * @{
     */
    /** Result of projecting a point into the normal-profile strips. */
    struct NormalProfileProjection
    {
      bool active = false;
      unsigned int fault_index = numbers::invalid_unsigned_int;
      unsigned int segment_index = numbers::invalid_unsigned_int;
      double xi = numbers::signaling_nan<double>();
      double signed_distance = numbers::signaling_nan<double>();
    };

    /** Associate a point with at most one open 2-D fault normal profile. */
    template <int dim>
    NormalProfileProjection
    project_to_normal_profiles(
      const std::vector<ReconstructedFault<dim>> &faults,
      const std::vector<std::vector<double>> &half_widths,
      const Point<dim> &position);
    /** @} */

    /**
     * @name Projection linear algebra
     * @{
     */
    /** Solve a symmetric positive-definite tridiagonal system. */
    std::vector<double>
    solve_tridiagonal_system(const std::vector<double> &diagonal,
                             const std::vector<double> &off_diagonal,
                             const std::vector<double> &rhs);
    /** @} */

    /** Domain-integrated Q1 coordinate and physical area weight. */
    struct DomainQuadraturePoint
    {
      unsigned int segment_index;
      double xi;
      double weight;
    };

    /** Optional local work counters; straight calls include partition subpieces. */
    struct DomainQuadratureStatistics
    {
      unsigned long long straight_calls = 0;
      unsigned long long general_calls = 0;
      unsigned long long segment_tests = 0;
      unsigned long long candidate_segments = 0;
    };

    /** Full convex-domain quadrature on an open 2-D polyline, without support clipping. */
    std::vector<DomainQuadraturePoint>
    domain_quadrature(const std::vector<Point<2>> &vertices,
                      const ReconstructedFault<2> &fault,
                      const unsigned int order = 3,
                      DomainQuadratureStatistics *statistics = nullptr);

    namespace internal
    {
      /*
       * Implementation-only entry points used by manager-owned caches to
       * validate stable geometry once before projecting many sample points.
       */
      template <int dim>
      void
      validate_normal_profile_projection_geometry(
        const std::vector<ReconstructedFault<dim>> &faults,
        const std::vector<std::vector<double>> &half_widths);

      template <int dim>
      void
      validate_normal_profile_projection_position(const Point<dim> &position);

      template <int dim>
      NormalProfileProjection
      project_to_normal_profiles_unchecked(
        const std::vector<ReconstructedFault<dim>> &faults,
        const std::vector<std::vector<double>> &half_widths,
        const Point<dim> &position);
    }
  }
}

#endif
