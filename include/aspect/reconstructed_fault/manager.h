/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#ifndef _aspect_reconstructed_fault_manager_h
#define _aspect_reconstructed_fault_manager_h

#include <aspect/reconstructed_fault/fault.h>
#include <aspect/reconstructed_fault/utilities.h>
#include <aspect/simulator_access.h>

#include <deal.II/base/quadrature.h>
#include <deal.II/grid/cell_id.h>
#include <deal.II/particles/property_pool.h>

#include <boost/serialization/access.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/vector.hpp>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace aspect
{
  template <int dim>
  class Simulator;

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
      /**
       * @name Construction and parameter handling
       * @{
       */
      ReconstructedFaultManager() = default;
      explicit ReconstructedFaultManager(const Simulator<dim> &simulator);

      static void declare_parameters(ParameterHandler &prm);
      void parse_parameters(ParameterHandler &prm);
      /**
       * @}
       */

      /**
       * @name Fault initialization and reconstruction
       * @{
       */
      void initialize_crack_driving_force(
        const std::vector<PrescribedInitialFault<dim>> &faults);

      void reconstruct_initial_faults();

      /** Add one complete reconstructed fault and its normal-profile half widths. */
      unsigned int add_reconstructed_fault(
        const std::vector<Point<dim>> &vertices,
        const std::vector<double> &projection_half_widths);
      /**
       * @}
       */

      /**
       * @name Generic reconstructed-fault properties
       * @{
       */
      /** Metadata for one runtime-defined property. */
      struct PropertyInformation
      {
        std::string name;
        unsigned int n_components;
        unsigned int position;

        template <class Archive>
        void serialize(Archive &ar, const unsigned int)
        {
          ar &name;
          ar &n_components;
          ar &position;
        }
      };

      /**
       * Register a property shared by every reconstructed fault. Properties
       * must be registered before reconstructed geometry exists.
       */
      unsigned int register_property(const std::string &name,
                                     const unsigned int n_components);

      bool has_property(const std::string &name) const;
      unsigned int get_property_index(const std::string &name) const;
      const std::vector<PropertyInformation> &get_property_information() const;
      /**
       * @}
       */

      /**
       * @name Slip-rate nonlinear state
       * @{
       */
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
      void validate_slip_rate_nonlinear_commit() const;

      /** Commit after successful validation without allocation or failure. */
      void commit_slip_rate_nonlinear_solve() noexcept;

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
      /**
       * @}
       */

      /**
       * @name Particle-to-fault projection
       * @{
       */
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

      /** Geometry and weight of one locally owned particle/fault association. */
      struct ParticleFaultAssociation
      {
        types::particle_index particle_id = numbers::invalid_unsigned_int;
        Point<dim> position;
        double particle_domain_volume = numbers::signaling_nan<double>();
        bool active = false;
        unsigned int fault_index = numbers::invalid_unsigned_int;
        unsigned int segment_index = numbers::invalid_unsigned_int;
        double xi = numbers::signaling_nan<double>();
      };

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

      /**
       * Return cached associations in locally owned particle iteration order.
       * The reference remains valid only until the projection cache is
       * invalidated or rebuilt.
       */
      const std::vector<ParticleFaultAssociation> &
      get_locally_owned_particle_fault_associations();

      void invalidate_particle_projection_cache();
      const std::vector<ParticleProjectionDiagnostics> &
      get_particle_projection_diagnostics() const;

      /** Associate a point with the manager-owned fault normal profiles. */
      ReconstructedFaultUtilities::NormalProfileProjection
      project_to_normal_profiles(const Point<dim> &position) const;
      /**
       * @}
       */

      /**
       * @name Stokes quadrature-point geometry cache
       * @{
       */
      struct StokesQPFaultAssociation
      {
        bool active = false;
        unsigned int fault_index = numbers::invalid_unsigned_int;
        unsigned int segment_index = numbers::invalid_unsigned_int;
        double xi = numbers::signaling_nan<double>();
        double shape_0 = numbers::signaling_nan<double>();
        double shape_1 = numbers::signaling_nan<double>();
        double signed_distance = numbers::signaling_nan<double>();
        Point<dim> position;
        Tensor<1,dim> tangent;
        Tensor<1,dim> normal;
      };

      struct StokesQPCacheDiagnostics
      {
        unsigned int n_active_q_points = 0;
        unsigned int rebuild_count = 0;
      };

      /** Build the cache with the production Stokes velocity quadrature. */
      void prepare_stokes_qp_projection_cache();

      /**
       * Return one cell's QP associations and verify the exact quadrature
       * identity and QP ordering in debug mode.
       */
      const std::vector<StokesQPFaultAssociation> &
      get_stokes_qp_fault_associations(
        const CellId &cell_id,
        const Quadrature<dim> &quadrature,
        const std::vector<Point<dim>> &quadrature_points) const;

      void invalidate_stokes_qp_projection_cache();

      const StokesQPCacheDiagnostics &
      get_stokes_qp_cache_diagnostics() const;
      /**
       * @}
       */

      /**
       * @name Fault access and diagnostics
       * @{
       */
      const std::vector<ReconstructedFault<dim>> &get_faults() const;
      ReconstructedFault<dim> &get_fault(const unsigned int fault_index);
      const ReconstructedFault<dim> &get_fault(const unsigned int fault_index) const;
      const std::vector<FaultReconstructionDiagnostics> &get_diagnostics() const;
      /**
       * @}
       */

    private:
      friend class boost::serialization::access;

      /**
       * @name Serialization and restart
       * @{
       */
      template <class Archive>
      void save(Archive &ar, const unsigned int) const
      {
        ar &initial_reconstruction_complete;
        ar &reconstructed_faults;
        ar &projection_half_widths;
        ar &property_information;
        ar &n_property_components;
        ar &timestep_committed_slip_rates;
        ar &slip_rate_initialized;
      }

      template <class Archive>
      void load(Archive &ar, const unsigned int)
      {
        ar &initial_reconstruction_complete;
        ar &reconstructed_faults;
        ar &projection_half_widths;
        ar &property_information;
        ar &n_property_components;
        ar &timestep_committed_slip_rates;
        ar &slip_rate_initialized;

        rebuild_after_deserialization();
      }

      BOOST_SERIALIZATION_SPLIT_MEMBER()

      void rebuild_after_deserialization();

      void reconstruct_initial_fault(
        const unsigned int fault_index,
        const double reconstruction_radius,
        const std::vector<double> &all_reconstruction_radii,
        const std::vector<double> &prescribed_half_widths,
        const double phase_field_activation_threshold);
      /**
       * @}
       */

      /**
       * @name Particle-projection cache
       * @{
       */
      struct ProjectionSystem
      {
        std::vector<double> diagonal;
        std::vector<double> off_diagonal;
        std::vector<double> factor_diagonal;
        std::vector<double> factor_lower;
      };

      using FaultNodalValues = std::vector<std::vector<double>>;

      bool particle_projection_cache_is_valid() const;
      void rebuild_particle_projection_cache();
      std::vector<unsigned int> fault_vertex_offsets() const;
      std::vector<FaultNodalValues>
      reduce_and_solve_projection_rhs(
        const std::vector<double> &local_rhs,
        const unsigned int n_components) const;
      /**
       * @}
       */

      bool stokes_qp_projection_cache_is_valid() const;
      void rebuild_stokes_qp_projection_cache();

      // Persistent prescribed-fault and reconstruction state.
      double structural_spacing = numbers::signaling_nan<double>();
      double ridge_coefficient = 1.0;
      std::string prescribed_faults_filename;
      bool initial_reconstruction_complete = false;
      std::vector<PrescribedInitialFault<dim>> prescribed_faults;
      std::vector<ReconstructedFault<dim>> reconstructed_faults;
      std::vector<std::vector<double>> projection_half_widths;
      std::uint64_t projection_metadata_version = 0;

      // Generic vertex-property registry.
      std::vector<PropertyInformation> property_information;
      std::map<std::string, unsigned int> property_indices;
      unsigned int n_property_components = 0;
      std::vector<FaultReconstructionDiagnostics> diagnostics;

      // Particle/fault projection cache. These members are reconstructible.
      bool particle_projection_cache_valid = false;
      std::uint64_t cached_projection_metadata_version = 0;
      std::vector<std::uint64_t> cached_fault_geometry_versions;
      std::vector<ParticleFaultAssociation> particle_projection_cache;
      std::vector<ProjectionSystem> projection_systems;
      std::vector<ParticleProjectionDiagnostics> particle_projection_diagnostics;

      // Bulk-cell/QP geometry cache. Constitutive coefficients are not stored here.
      bool stokes_qp_projection_cache_valid = false;
      std::uint64_t cached_stokes_qp_projection_metadata_version = 0;
      std::vector<std::uint64_t> cached_stokes_qp_fault_geometry_versions;
      std::vector<Point<dim>> cached_stokes_quadrature_points;
      std::vector<double> cached_stokes_quadrature_weights;
      std::map<CellId, std::vector<StokesQPFaultAssociation>>
      stokes_qp_projection_cache;
      StokesQPCacheDiagnostics stokes_qp_cache_diagnostics;

      // Distinguished slip-rate nonlinear state.
      std::vector<std::vector<double>> timestep_committed_slip_rates;
      std::vector<std::vector<double>> current_newton_slip_rates;
      std::vector<std::vector<double>> trial_slip_rates;
      std::vector<bool> slip_rate_initialized;
      bool slip_rate_nonlinear_solve_active = false;
      bool slip_rate_trial_active = false;
  };

}

#endif
