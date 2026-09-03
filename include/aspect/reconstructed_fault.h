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

#include <deal.II/base/array_view.h>
#include <deal.II/base/point.h>

#include <boost/serialization/access.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/vector.hpp>

#include <cstdint>
#include <map>
#include <vector>

namespace aspect
{
  template <int dim>
  class PhaseFieldHandler;

  template <int dim>
  class Simulator;

  template <int dim>
  class ReconstructedFaultManager;

  template <int dim>
  class ReconstructedFault;

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
   * The container is independent of the bulk mesh, particles, phase-field
   * reconstruction, and particular constitutive models. Runtime-defined
   * vertex properties can store material-independent fault data.
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

      /** Return all property components stored at vertex @p vertex_index. */
      ArrayView<double>
      get_properties(const unsigned int vertex_index);

      /** Return all property components stored at vertex @p vertex_index. */
      ArrayView<const double>
      get_properties(const unsigned int vertex_index) const;

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
      friend class ReconstructedFaultManager<dim>;
      friend class boost::serialization::access;

      template <class Archive>
      void serialize(Archive &ar, const unsigned int)
      {
        ar & vertices;
        ar & n_property_components;
        ar & property_values;
        ar & current_geometry_version;
      }

      void initialize_properties(const unsigned int n_components);

      std::vector<Point<dim>> vertices;
      unsigned int n_property_components = 0;
      std::vector<double> property_values;
      std::uint64_t current_geometry_version = 0;
  };


  namespace ReconstructedFaultUtilities
  {
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

    /** Solve a symmetric positive-definite tridiagonal system. */
    std::vector<double>
    solve_tridiagonal_system(const std::vector<double> &diagonal,
                             const std::vector<double> &off_diagonal,
                             const std::vector<double> &rhs);
  }


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
      /** Metadata for one runtime-defined property. */
      struct PropertyInformation
      {
        std::string name;
        unsigned int n_components;
        unsigned int position;

        template <class Archive>
        void serialize(Archive &ar, const unsigned int)
        {
          ar & name;
          ar & n_components;
          ar & position;
        }
      };

      /** Map particle-property components to a registered fault property. */
      struct ParticlePropertyProjection
      {
        std::string particle_property_name;
        unsigned int first_particle_component = 0;
        std::string fault_property_name;
        unsigned int first_fault_component = 0;
        unsigned int n_components = 1;
      };

      /** Coverage information for one reconstructed fault. */
      struct ParticleProjectionDiagnostics
      {
        std::vector<double> weighted_support;
        unsigned int n_contributing_particles = 0;
      };

      /** Residual diagnostics for projecting one particle scalar to a fault. */
      struct ParticleScalarProjectionDiagnostics
      {
        double weighted_rms_residual = numbers::signaling_nan<double>();
        double maximum_absolute_residual = numbers::signaling_nan<double>();
        double normalized_weighted_rms_residual = numbers::signaling_nan<double>();
        double normalized_maximum_absolute_residual = numbers::signaling_nan<double>();
      };

      /** Result of a constitutively neutral particle-scalar projection. */
      struct ParticleScalarProjectionResult
      {
        std::vector<std::vector<double>> nodal_values;
        std::vector<ParticleScalarProjectionDiagnostics> diagnostics;
      };

      ReconstructedFaultManager() = default;
      explicit ReconstructedFaultManager(const Simulator<dim> &simulator);

      static void declare_parameters(ParameterHandler &prm);
      void parse_parameters(ParameterHandler &prm);

      void initialize_crack_driving_force(
        PhaseFieldHandler<dim> &phase_field_handler,
        const std::vector<PrescribedInitialFault<dim>> &faults);

      void reconstruct_initial_faults(PhaseFieldHandler<dim> &phase_field_handler);

      /** Add one complete reconstructed fault and its normal-profile half widths. */
      unsigned int add_reconstructed_fault(
        const std::vector<Point<dim>> &vertices,
        const std::vector<double> &projection_half_widths);

      /**
       * Register a property shared by every reconstructed fault. Properties
       * must be registered before reconstructed geometry exists.
       */
      unsigned int register_property(const std::string &name,
                                     const unsigned int n_components);

      bool has_property(const std::string &name) const;
      unsigned int get_property_index(const std::string &name) const;
      const std::vector<PropertyInformation> &get_property_information() const;

      /** Return whether every reconstructed fault has an initialized slip rate. */
      bool slip_rates_are_initialized() const;

      /** Initialize the nodal slip rate of one reconstructed fault. */
      void initialize_slip_rate(const unsigned int fault_index,
                                const std::vector<double> &values);

      /** Return the active nodal slip rate: trial if active, otherwise current Newton. */
      const std::vector<double> &get_slip_rate(const unsigned int fault_index) const;

      /** Return the timestep-committed nodal slip rate used for persistent output. */
      const std::vector<double> &
      get_timestep_committed_slip_rate(const unsigned int fault_index) const;

      /** Q1-interpolate the current slip rate on one fault segment. */
      double interpolate_slip_rate(const unsigned int fault_index,
                                   const unsigned int segment_index,
                                   const double xi) const;

      /** Initialize the current Newton iterate from the timestep-committed state. */
      void begin_slip_rate_nonlinear_solve();

      /** Commit the converged current Newton iterate to the timestep state. */
      void commit_slip_rate_nonlinear_solve();

      /** Discard all nonlinear work and restore the timestep-committed state. */
      void rollback_slip_rate_nonlinear_solve();

      /** Begin a line-search trial from the current accepted Newton iterate. */
      void begin_slip_rate_trial();

      /** Set V_trial = V_current + step_length * delta_V without accumulation. */
      void set_slip_rate_trial(const std::vector<std::vector<double>> &delta_V,
                               const double step_length);

      /** Accept the trial as the current Newton iterate, without timestep commit. */
      void accept_slip_rate_trial();

      /** Discard the trial and retain the current accepted Newton iterate. */
      void rollback_slip_rate_trial();

      void project_particle_properties(
        const std::vector<ParticlePropertyProjection> &projections);

      /**
       * Interpolate one registered Q1 fault property at every active locally
       * owned particle's cached fault coordinate. The returned values are
       * addressed by stable particle ID; inactive particles are omitted.
       */
      std::map<types::particle_index, std::vector<double>>
      interpolate_property_at_particle_projections(
        const unsigned int property_index);

      /**
       * Project one caller-computed scalar per active locally owned particle
       * to the replicated fault Q1 spaces. Values are addressed by stable
       * particle ID; inactive particles may be omitted.
       */
      ParticleScalarProjectionResult
      project_particle_scalar(
        const std::map<types::particle_index, double> &locally_owned_values);

      void invalidate_particle_projection_cache();
      const std::vector<ParticleProjectionDiagnostics> &
      get_particle_projection_diagnostics() const;

      /** Associate a point with the manager-owned fault normal profiles. */
      ReconstructedFaultUtilities::NormalProfileProjection
      project_to_normal_profiles(const Point<dim> &position) const;

      const std::vector<ReconstructedFault<dim>> &get_faults() const;
      ReconstructedFault<dim> &get_fault(const unsigned int fault_index);
      const ReconstructedFault<dim> &get_fault(const unsigned int fault_index) const;
      const std::vector<FaultReconstructionDiagnostics> &get_diagnostics() const;

    private:
      friend class boost::serialization::access;

      template <class Archive>
      void save(Archive &ar, const unsigned int) const
      {
        ar & initial_reconstruction_complete;
        ar & reconstructed_faults;
        ar & projection_half_widths;
        ar & property_information;
        ar & n_property_components;
        ar & timestep_committed_slip_rates;
        ar & slip_rate_initialized;
      }

      template <class Archive>
      void load(Archive &ar, const unsigned int)
      {
        ar & initial_reconstruction_complete;
        ar & reconstructed_faults;
        ar & projection_half_widths;
        ar & property_information;
        ar & n_property_components;
        ar & timestep_committed_slip_rates;
        ar & slip_rate_initialized;

        rebuild_after_deserialization();
      }

      BOOST_SERIALIZATION_SPLIT_MEMBER()

      struct ParticleProjectionCacheEntry
      {
        types::particle_index particle_id = numbers::invalid_unsigned_int;
        Point<dim> position;
        double particle_domain_volume = numbers::signaling_nan<double>();
        bool active = false;
        unsigned int fault_index = numbers::invalid_unsigned_int;
        unsigned int segment_index = numbers::invalid_unsigned_int;
        double xi = numbers::signaling_nan<double>();
      };

      struct ProjectionSystem
      {
        std::vector<double> diagonal;
        std::vector<double> off_diagonal;
        std::vector<double> factor_diagonal;
        std::vector<double> factor_lower;
      };

      bool particle_projection_cache_is_valid() const;
      void rebuild_particle_projection_cache();
      std::vector<double> solve_projection_system(
        const unsigned int fault_index,
        const ArrayView<const double> &rhs) const;

      double structural_spacing = numbers::signaling_nan<double>();
      double ridge_coefficient = 1.0;
      std::string prescribed_faults_filename;
      bool initial_reconstruction_complete = false;
      std::vector<PrescribedInitialFault<dim>> prescribed_faults;
      std::vector<ReconstructedFault<dim>> reconstructed_faults;
      std::vector<std::vector<double>> projection_half_widths;
      std::uint64_t projection_metadata_version = 0;
      std::vector<PropertyInformation> property_information;
      std::map<std::string, unsigned int> property_indices;
      unsigned int n_property_components = 0;
      std::vector<FaultReconstructionDiagnostics> diagnostics;
      bool particle_projection_cache_valid = false;
      std::uint64_t cached_projection_metadata_version = 0;
      std::vector<std::uint64_t> cached_fault_geometry_versions;
      std::vector<ParticleProjectionCacheEntry> particle_projection_cache;
      std::vector<ProjectionSystem> projection_systems;
      std::vector<ParticleProjectionDiagnostics> particle_projection_diagnostics;
      std::vector<std::vector<double>> timestep_committed_slip_rates;
      std::vector<std::vector<double>> current_newton_slip_rates;
      std::vector<std::vector<double>> trial_slip_rates;
      std::vector<bool> slip_rate_initialized;
      bool slip_rate_nonlinear_solve_active = false;
      bool slip_rate_trial_active = false;

      void rebuild_after_deserialization();
  };
}

#endif
