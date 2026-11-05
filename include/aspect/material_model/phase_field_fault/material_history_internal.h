/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#ifndef ASPECT_MATERIAL_HISTORY_INTERNAL_H
#define ASPECT_MATERIAL_HISTORY_INTERNAL_H
#include <aspect/material_model/phase_field_fault/fe_strain_internal.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/material_model/history/tensor_history.h>

namespace aspect
{
  namespace MaterialModel
  {
    /** Read-only, nonowning mathematical inputs. The accepted parent and velocity
     * outlive this attempt. In particular, rotation uses this velocity gradient,
     * never the current Newton iterate. */
    template<int dim> struct PhaseFieldFault<dim>::MaterialHistoryInputs
    {
      const FEStrainState &accepted;
      const LinearAlgebra::BlockVector &accepted_velocity;
      double time, dt;
      unsigned int step;
    };

    /** Attempt-only transport variables, never serialized. Exactly one of
     * physical_stress / separated_trial is present. The latter carries the
     * separate total memory R+P(B), not a physical cohesive elastic history. */
    template<int dim> struct PhaseFieldFault<dim>::TransportedHistory
    {
      std::shared_ptr<DGHistory> physical_stress;
      std::shared_ptr<DGHistory> separated_trial;
      std::shared_ptr<DGHistory> separated_total_memory;
      unsigned int substeps = 0;
      bool energy_ledger = false;
      double energy_before = 0., energy_advected = 0.;
    };

    /** Frozen projection-defect cache for one coupled outer iteration. Its
     * candidate is not accepted history; it is rebuilt from the same anchor. */
    template<int dim> struct PhaseFieldFault<dim>::DGMaterialReaction
    {
      LinearAlgebra::BlockVector bulk;
      std::vector<double> rates;
      std::shared_ptr<FEStrainUpdate> candidate;
    };

    /** One rollback owner. Accepted physical coefficients stay unchanged.
     * Derived caches on the shared FE state may be invalidated. The candidate
     * holds the moved geometry/profile and per-attempt rotated anchor, and is
     * bound to mechanics until final outer acceptance or rollback. */
    template<int dim> struct PhaseFieldFault<dim>::DGMaterialTrial
    {
      std::shared_ptr<FEStrainState> accepted_history, candidate_history;
      TransportedHistory transport;
      std::shared_ptr<ReconstructedFaultManager<dim>> manager;
      std::vector<std::vector<double>> normalization, temperatures;
      std::map<types::particle_index, std::vector<double>> particle_properties;
      unsigned int step = 0;
      double dt = 0.;
      bool bound = false;
    };
  }
}
#endif
