/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#ifndef ASPECT_MATERIAL_HISTORY_INTERNAL_H
#define ASPECT_MATERIAL_HISTORY_INTERNAL_H
#include <aspect/material_model/phase_field_fault/fe_strain_internal.h>

namespace aspect
{
  namespace MaterialModel
  {
    /** Unprojected Eulerian angular rates from one coherent accepted parent.
     * Fixed QGauss order and CellId keys match the native material projector.
     * Never transported, changed by Newton, or serialized in a checkpoint. */
    template <int dim>
    struct PhaseFieldFault<dim>::AcceptedSpinSamples
    {
      std::weak_ptr<const DGHistory> parent;
      const LinearAlgebra::BlockVector *velocity = nullptr;
      unsigned int step = 0, quadrature_order = 0;
      std::uint64_t geometry_version = 0;
      std::vector<double> committed_rates;
      std::map<CellId,std::vector<double>> angular_rates;
      std::array<double,3> maximum {}, l2 {}; // total, fault, bulk (1/s)
    };

    /**
     * Borrowed inputs for one ordinary simulator material-history attempt.
     * The accepted parent and old velocity outlive all synchronous preparation
     * calls. They are not a Newton candidate. No large FE vectors are copied
     * into this view, and it is not stored or serialized.
     */
    template <int dim>
    struct PhaseFieldFault<dim>::MaterialHistoryInputs
    {
      /** Accepted bulk/surface/profile parent; its history fields stay immutable. */
      const FEStrainState &accepted;
      /** Ghost-readable accepted velocity (m/s), including the lagged spin source. */
      const LinearAlgebra::BlockVector &accepted_velocity;
      /** End-of-step physical time (s); advection starts at end_time-dt. */
      double end_time;
      /** Attempt interval length (s), which can change on rejection/retry. */
      double dt;
      /** Simulator step number associated with this attempt. */
      unsigned int step;
    };

    /**
     * Attempt-only transported material tensors, never accepted or serialized.
     * Exactly one of physical_stress and separated_trial is present. The
     * separated branch additionally transports R+P(B); it cannot transport R
     * alone when the localization/profile moves. All tensors are in the spatial
     * frame before the subsequent objective rotation.
     */
    template <int dim>
    struct PhaseFieldFault<dim>::TransportedHistory
    {
      /** Advected physical tau (Pa), direct representation only. */
      std::shared_ptr<DGHistory> physical_stress;
      /** Advected auxiliary T (Pa), separated representation only. */
      std::shared_ptr<DGHistory> separated_trial;
      /** Advected R+P(B) (Pa); never cohesive elastic history. */
      std::shared_ptr<DGHistory> separated_total_memory;
      /** Number of explicit SSPRK3 transport substeps in this attempt. */
      unsigned int substeps = 0;
      /** Whether the opt-in observational energy ledger is being collected. */
      bool energy_ledger = false;
      /** Pre-advection physical elastic energy (J/m in plane strain). */
      double energy_before = 0.;
      /** Post-advection energy in the same quadrature; not an acceptance gate. */
      double energy_advected = 0.;
    };

    /**
     * Frozen projection-defect data for one inner coupled solve. The outer
     * solver rebuilds it from the same objective anchor at its latest bulk/slip
     * iterate. Nothing here is an independent accepted history or checkpoint.
     */
    template <int dim>
    struct PhaseFieldFault<dim>::DGMaterialReaction
    {
      /** Ghost-readable bulk iterate whose gradient formed the local response. */
      LinearAlgebra::BlockVector bulk;
      /** Surface slip rates (m/s), in current material-label order. */
      std::vector<double> rates;
      /** Projected endpoint history evaluated at bulk/rates, still unaccepted. */
      std::shared_ptr<FEStrainUpdate> candidate;
    };

    /**
     * Material rollback owner, coordinated by ordinary Simulator::solve_timestep.
     * Once bound, mechanics sees candidate_history and the staged live manager.
     * Acceptance is still exclusively the outer timestep controller's decision.
     * Candidate publication happens once before this owner's rollback data is
     * released. Rejection swaps back geometry and restores the accepted parent.
     */
    template <int dim>
    struct PhaseFieldFault<dim>::DGMaterialTrial
    {
      /** Immutable physical parent retained through every trial of this attempt. */
      std::shared_ptr<FEStrainState> accepted_history;
      /** Privately staged profile/geometry and rotated anchor used by mechanics. */
      std::shared_ptr<FEStrainState> candidate_history;
      /** Advected tensors and optional diagnostics for this attempt. */
      TransportedHistory transport;
      /** Accepted bulk spin sampled before any live geometry/profile binding. */
      std::shared_ptr<const AcceptedSpinSamples> accepted_spin;
      /**
       * Before binding: staged manager clone. After swap/binding: saved accepted
       * geometry, generic surface properties and rates, available for rollback.
       */
      std::shared_ptr<ReconstructedFaultManager<dim>> manager;
      /** Accepted per-label normalization integrals (m) saved before rebuilding. */
      std::vector<std::vector<double>> normalization;
      /** Accepted per-label temperature (K) cache, restored with geometry. */
      std::vector<std::vector<double>> temperatures;
      /** Local particle diagnostic properties saved by particle id for rollback. */
      std::map<types::particle_index, std::vector<double>> particle_properties;
      /** Step key at preparation; used to audit the attempt. */
      unsigned int step = 0;
      /** Interval length (s) used for this candidate, not a global fixed timestep. */
      double dt = 0.;
      /** True only after swapping the staged geometry into the live manager. */
      bool bound = false;
    };
  }
}
#endif
