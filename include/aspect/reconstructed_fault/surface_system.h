/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#ifndef _aspect_reconstructed_fault_surface_system_h
#define _aspect_reconstructed_fault_surface_system_h

#include <aspect/simulator_access.h>

#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/base/timer.h>
#include <deal.II/particles/property_pool.h>

#include <memory>
#include <functional>
#include <array>
#include <vector>

namespace aspect
{
  namespace internal { class FaultNormalFilter; }
  namespace MaterialModel
  {
    template <int dim>
    class PhaseFieldFault;
  }

  /** Surface residual assembled on the replicated reconstructed faults. */
  struct ReconstructedFaultSurfaceResidual
  {
    std::vector<std::vector<double>> values;
    double weighted_rms = numbers::signaling_nan<double>();
    std::vector<double> per_fault_weighted_rms;
    /** Weak terms from this evaluation, before any history publication. */
    std::vector<std::vector<double>> shear_traction, cohesive_traction,
      friction_traction, damping_traction, normal_traction;
    /** Actual constitutive sample extrema, not extrema of a projected field. */
    double minimum_normal_traction = std::numeric_limits<double>::infinity();
    double maximum_normal_traction = -std::numeric_limits<double>::infinity();
    std::vector<std::vector<double>> mass_diagonal, mass_off_diagonal;
    /** Populated only by the explicitly selected normal-filter experiment.
     * Raw means the current mechanical input including background, before filtering. */
    std::vector<std::vector<double>> raw_normal_traction, normal_filter_coefficients;
    double minimum_raw_normal_traction = std::numeric_limits<double>::infinity();
    double maximum_raw_normal_traction = -std::numeric_limits<double>::infinity();
  };


  using ReconstructedFaultVector = std::vector<std::vector<double>>;


  /** Bound-active reconstructed-fault vertices, indexed by fault and vertex. */
  using ReconstructedFaultActiveSet = std::vector<std::vector<bool>>;


  /** Semantic solve used by reconstructed-fault condensation. */
  template <int dim>
  class ReconstructedFaultSurfaceLinearSolve
  {
    public:
      virtual ~ReconstructedFaultSurfaceLinearSolve() = default;

      /** Overwrite @p solution with the solution of the current surface system. */
      virtual void
      solve(const ReconstructedFaultVector &rhs,
            ReconstructedFaultVector &solution) const = 0;
  };


  /**
   * Assemble and solve the particle/Q1 reconstructed-fault surface system.
   * Bulk cell weak-form assembly is intentionally outside this class.
   */
  template <int dim>
  class ReconstructedFaultSurfaceSystem : public SimulatorAccess<dim>,
    public ReconstructedFaultSurfaceLinearSolve<dim>
  {
    public:
      using FaultVector = ReconstructedFaultVector;

      explicit ReconstructedFaultSurfaceSystem(const Simulator<dim> &simulator);
      ~ReconstructedFaultSurfaceSystem();

      /** Opt into the common Stokes-QP work measure for a straight, frozen
       * mature fault. Reattach before mechanics; generic projections are unchanged.
       */
      void enable_bulk_work_measure();

      /** Experimental normal input: raw (default/bypass), projected, helmholtz.
       * Length is physical metres; zero length still projects. Configure outside
       * mechanics, including on restart. Currently requires true-normal bulk work. */
      void set_normal_stress_filter(const std::string &mode, double length);

      /** Output-only snapshot of the true-normal, bulk-QP work evaluation.
       * Loads are globally reduced; samples retain unique local cell ownership.
       * A consumer may publish this only after successful mechanical acceptance.
       * Nothing here is checkpointed or used in a residual or Jacobian.
       */
      struct NormalTractionDiagnostic
      {
        struct Sample
        {
          std::string cell;
          unsigned int qp, level, fault, segment;
          Point<dim> position, surface_position;
          Tensor<1,dim> normal;
          SymmetricTensor<2,dim> stress;
          double cell_size, xi, pressure, deviatoric, background, total;
          double phase, I_h, chi, JxW, weight;
          SymmetricTensor<2,dim> incoming_stress;
          std::array<SymmetricTensor<2,dim>,3> stress_components;
          Point<dim> reference_position;
          SymmetricTensor<2,dim> particle_interpolated_stress;
          Tensor<2,dim> velocity_gradient;
          double stress_time_step = 0., beta = 0., eta_ve = 0.;
          double friction_coefficient = 0., friction_normal = 0.;
        };
        struct ParticleSample
        {
          std::string cell;
          types::particle_index id;
          unsigned int owner_rank;
          Point<dim> position;
          SymmetricTensor<2,dim> stress;
        };
        unsigned int step;
        double time;
        FaultVector pressure_load, deviatoric_load, background_load, rates;
        FaultVector filter_stiffness_diagonal, filter_stiffness_off_diagonal;
        FaultVector friction_mass_diagonal, friction_mass_off_diagonal;
        std::array<FaultVector,3> deviatoric_component_loads;
        /** Independent owned-QP sums, globally reduced (Pa*m in 2-D). */
        std::array<double,6> integrated_loads = {{0.,0.,0.,0.,0.,0.}};
        std::vector<Sample> samples;
        /** Native point evaluations, not quadrature weights or weak loads. */
        std::vector<Sample> line_samples;
        std::vector<ParticleSample> particles;
        unsigned int unassociated_phase_points = 0;
      };

      /** Select mapped surface positions for raw QPs; empty disables capture.
       * All fault rows are captured regardless of the sample window.
       */
      void set_normal_traction_diagnostic(std::function<bool(const Point<dim> &)> window,
                                         std::vector<std::pair<Point<dim>,Point<dim>>> lines = {});
      const NormalTractionDiagnostic &get_normal_traction_diagnostic() const;

      /** Opt-in, read-only observer of a completed linearization, before any
       * history commit. Receives uniquely owned samples and reduced moments.
       */
      std::function<void(const ReconstructedFaultSurfaceResidual &,
                         const NormalTractionDiagnostic &)> normal_diagnostic_observer;

      ReconstructedFaultSurfaceResidual
      evaluate_surface_residual(const LinearAlgebra::BlockVector &bulk_state,
                                const FaultVector &slip_rate) const;

      /** Assemble and replace the current residual, K_V, and its factors. */
      const ReconstructedFaultSurfaceResidual &
      linearize_surface_system(const LinearAlgebra::BlockVector &bulk_state,
                               const FaultVector &slip_rate);

      /** Apply the inverse of the K_V most recently assembled above. */
      void
      solve(const FaultVector &rhs,
            FaultVector &solution) const override;

      /**
       * Build a semantic inverse of the principal free block of the current
       * K_V. Active rows and columns are disconnected, active right-hand-side
       * entries are ignored, and active solution entries are exactly zero.
       */
      std::unique_ptr<ReconstructedFaultSurfaceLinearSolve<dim>>
      create_restricted_linear_solve(
        const ReconstructedFaultActiveSet &active_set) const;

      /** Overwrite @p result with the current K_V applied to @p direction. */
      void
      apply_surface_jacobian(const FaultVector &direction,
                             FaultVector &result) const;

      /**
       * Return the consistent-Q1 RMS norm of a weak surface residual after
       * projecting out active vertices.
       */
      double
      surface_residual_rms(
        const ReconstructedFaultSurfaceResidual &residual,
        const ReconstructedFaultActiveSet &active_set) const;

      /**
       * Overwrite @p result with G applied to a full-system direction whose
       * pressure component is in physical units.
       */
      void
      apply_G(const LinearAlgebra::BlockVector &physical_bulk_direction,
              FaultVector &result) const;

      /** Independent parent-sampling/domain action for matrix verification. */
      void apply_G_reference(const LinearAlgebra::BlockVector &physical_bulk_direction,
                             FaultVector &result) const;

      /** Generation of the current surface-system state. */
      unsigned int
      get_linearization_generation() const;

      /** Frozen weak balance of the most recent linearization, not reevaluated history. */
      const ReconstructedFaultSurfaceResidual &get_linearization_residual() const;

    private:
      struct SurfaceAssembly;
      struct SurfaceLinearization;

      SurfaceAssembly
      assemble_surface_system(const LinearAlgebra::BlockVector &bulk_state,
                              const FaultVector &slip_rate,
                              const bool assemble_jacobian) const;

      SurfaceAssembly
      assemble_particle_system(const LinearAlgebra::BlockVector &bulk_state,
                               const FaultVector &slip_rate,
                               const bool assemble_jacobian) const;

      SurfaceAssembly
      assemble_bulk_work_system(const LinearAlgebra::BlockVector &bulk_state,
                                const FaultVector &slip_rate,
                                bool assemble_jacobian) const;

#ifdef DEAL_II_WITH_UMFPACK
      /**
       * Prepare the full factors and G lookup from assembled coefficients.
       * The caller retains invalidation, observer invocation and publication;
       * this operation returns a complete, unpublished candidate.
       */
      std::unique_ptr<SurfaceLinearization>
      prepare_surface_linearization(const SurfaceAssembly &assembled);
#endif

      const MaterialModel::PhaseFieldFault<dim> &phase_field_fault;
      GridTools::Cache<dim> grid_cache;
      std::unique_ptr<SurfaceLinearization> surface_linearization;
      /** Rank-local timings; surface failures must not enter timer collectives. */
      std::unique_ptr<TimerOutput> performance_timer;
      unsigned int linearization_generation = 0;
      bool bulk_work_measure = false;
      std::function<bool(const Point<dim> &)> normal_diagnostic_window;
      std::vector<std::pair<Point<dim>,Point<dim>>> normal_diagnostic_lines;
      std::shared_ptr<const NormalTractionDiagnostic> normal_diagnostic;
      std::string normal_filter_mode = "raw";
      double normal_filter_length = 0.;
      mutable std::vector<std::shared_ptr<const internal::FaultNormalFilter>> normal_filter_cache;
  };
}

#endif
