/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#ifndef _aspect_reconstructed_fault_h
#define _aspect_reconstructed_fault_h

#include <aspect/global.h>
#include <aspect/simulator_access.h>

#include <deal.II/base/point.h>

#include <cstdint>
#include <vector>

namespace aspect
{
  template <int dim>
  class PhaseFieldHandler;

  template <int dim>
  class Simulator;

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

    /**
     * Initialize the crack-driving-force particle property from prescribed
     * initial faults. This function updates locally owned particles and is
     * currently implemented only in 2D.
     */
    template <int dim>
    void
    initialize_crack_driving_force(PhaseFieldHandler<dim> &phase_field_handler,
                                   const std::vector<PrescribedInitialFault<dim>> &faults);
  }

  /**
   * An application-owned representation of a reconstructed fault.
   *
   * In two dimensions, the fault is an ordered polyline. Consecutive
   * vertices define the fault cells implicitly: cell @p i connects vertices
   * @p i and @p i+1. Committed vertices are append-only and can only be
   * accessed through const interfaces.
   *
   * This geometry-only container is intentionally independent of the bulk
   * mesh, particles, phase-field reconstruction, and constitutive data.
   */
  template <int dim>
  class ReconstructedFault
  {
    public:
      /** Construct an empty reconstructed fault. */
      ReconstructedFault() = default;

      /** Construct a fault from an ordered sequence of committed vertices. */
      explicit ReconstructedFault(const std::vector<Point<dim>> &vertices);

      /** Return whether the fault contains no vertices. */
      bool
      empty() const;

      /** Return the number of fault vertices. */
      unsigned int
      n_vertices() const;

      /** Return the number of fault cells. */
      unsigned int
      n_cells() const;

      /** Return vertex @p index. */
      const Point<dim> &
      vertex(const unsigned int index) const;

      /** Return the complete ordered sequence of fault vertices. */
      const std::vector<Point<dim>> &
      get_vertices() const;

      /** Append one committed vertex to the fault. */
      void
      append_vertex(const Point<dim> &vertex);

      /**
       * Append an ordered sequence of committed vertices to the fault.
       * Appending an empty sequence does not change the geometry version.
       */
      void
      append_vertices(const std::vector<Point<dim>> &new_vertices);

      /**
       * Return the geometry version. Each non-empty append operation advances
       * this counter once.
       */
      std::uint64_t
      geometry_version() const;

    private:
      std::vector<Point<dim>> vertices;
      std::uint64_t current_geometry_version = 0;
  };


  /** Diagnostics produced by the direct phase-field ridge reconstruction. */
  struct FaultReconstructionDiagnostics
  {
    double total_weight = 0.0;
    std::vector<double> offsets;
    std::vector<double> structural_support;
  };


  /** Owns prescribed and reconstructed faults and their reconstruction lifecycle. */
  template <int dim>
  class ReconstructedFaultManager : public SimulatorAccess<dim>
  {
    public:
      explicit ReconstructedFaultManager(const Simulator<dim> &simulator);

      static void declare_parameters(ParameterHandler &prm);
      void parse_parameters(ParameterHandler &prm);

      void initialize_crack_driving_force(
        PhaseFieldHandler<dim> &phase_field_handler,
        const std::vector<PrescribedInitialFault<dim>> &faults);

      void reconstruct_initial_faults(PhaseFieldHandler<dim> &phase_field_handler);

      const std::vector<ReconstructedFault<dim>> &get_faults() const;
      const std::vector<FaultReconstructionDiagnostics> &get_diagnostics() const;

    private:
      double structural_spacing = numbers::signaling_nan<double>();
      double ridge_coefficient = 1.0;
      std::string prescribed_faults_filename;
      bool initial_reconstruction_complete = false;
      std::vector<PrescribedInitialFault<dim>> prescribed_faults;
      std::vector<ReconstructedFault<dim>> reconstructed_faults;
      std::vector<FaultReconstructionDiagnostics> diagnostics;
  };
}

#endif
