# Reconstructed-fault and particle-free RSF development

## Authority and task scope

- Work in `aspect-pf-rsf-particle-free` on `pf-rsf-particle-free`; verify root,
  branch, revision, and local changes before editing. Preserve manual work and
  other worktrees, builds, jobs, outputs, and checkpoints.
- The current user instruction determines the task. Plans describe future work;
  historical stage selections do not authorize continuation. Complete routine
  work within the requested scope without repeated confirmation.
- Under `doc/reconstructed_fault/`, `current_design.md` and `specification.tex`
  govern established algorithms, architecture, ownership, and MPI design.
  Source governs existing APIs. The particle-free derivation is a proposal to
  verify, not an automatic replacement for the specification. Stop affected
  implementation and report scientific/design conflicts; continue independent
  useful checks.
- Follow `refactoring.md` for code quality and boundaries. Refactoring-only
  freezes apply to refactoring tasks, not separately authorized scientific work.
  Reuse ASPECT/deal.II infrastructure; avoid duplicate parameters and unnecessary
  abstractions. Do not use core-phase-field or old PhaseFieldRSF architecture
  as the reconstructed-fault design basis.

## Read the relevant references

Paths below are relative to `doc/reconstructed_fault/`:

| Purpose | Files |
|---|---|
| Existing implementation and P2 baseline | `CURRENT_STATUS.md`, `propagation/STATUS.md`, `propagation/P2_CLOSEOUT.md` |
| Inherited cleanup | `propagation/P2_cleanup.md`; verify what is actually present |
| Particle-free task | `particle_free/STATUS.md`, requested phase in `particle_free/PLAN.md` |
| Equations and checks | Relevant sections of `particle_free/particle_free_history.tex` and `particle_free/VERIFICATION.md` |

The historical cleanup reference `41272d263` is not an assumed current base.
Inspect callers, headers, and tests before changing interfaces.

## Retained P2 baseline

- The P2 reference is the restricted native FE consistent-endpoint option with
  reduced Newton: fixed mesh/geometry, homogeneous matched coefficients, no
  transport or birth. FGMRES remains default; direct mechanics is experimental,
  serial, and default off. Unqualified coupling/search experiments stay inactive.
- Use the selected recoverable total-energy formulation and physical phase
  irreversibility; `H+` is diagnostic. Initialization follows its documented rules.
- Preserve governing energy, Maxwell evolution, material/loading conventions,
  normalization, physical bounds, and final KKT/energy acceptance. Keep native
  consistent-deviator history distinct from legacy full-gradient history.
- During physical steps, every trial uses the same accepted bulk history,
  surface cohesive history, and physical phase lower bound. Publish once after acceptance;
  rejection restores trial geometry and histories. Checkpoint conversion is
  outside the restricted baseline.
- Existing evidence covers restricted operators, retained entries, MPI/recovery
  controls, prototype paths, and shorter native onset. It does not establish
  full native evolution, full temporal/spatial qualification, coupled restart,
  transport, or fault propagation. Solver/KKT convergence alone does not prove
  physical validity or stability.

## Particle-free scientific requirements

- Distinguish exact variable transformations, discrete replacements, and
  constitutive changes. Preserve the audited rate/spin convention and physical
  inputs unless the user selects a change.
- For constant matched parameters, the cohesive candidate stores trial stress
  and physical elastic slip; reconstruction implies
  `Q_geo = -2 G C[s_elas]`. Do not drop evolving-profile derivatives.
- Mature evolution with `Q_geo = 0` requires direct stress transport or the
  appropriate residual history. Auxiliary `s_mem` has no cohesive energy or
  resistance; zero initial residual does not justify omitting its evolution.
- Use the full 3D plane-strain convention. Retain native FE strain information
  or quantify its projection. Keep localization, normalization, measures, and
  quadrature consistent across energy, residual, tangent, and work adjoint.
- Separate physical phase redistribution from numerical remapping. Material
  irreversibility, surface birth, and changing geometry need explicit rules.
  Keep physical tips distinct from computational-boundary completion; do not
  invent fault continuation, normalization floors, or phase caps.
- Removing particle stress, replacing CPDI phase integration, and removing
  particle-derived geometry are separate changes. Accurate quadrature cannot
  replace missing history degrees of freedom. Report remaining dependencies.
- WENO is optional; establish need against a smooth no-limiter control.
  Historical `elasticity.cc` requires verification before reuse. Limit
  independent history, not the reconstructed localization.

## Implementation, verification, and handoff

- Preserve legacy/default behavior and failed evidence. Introduce selectable
  experimental paths; avoid unrelated cleanup, solver redesign, parameter
  tuning, tolerance relaxation, or compensating dissipation.
- Use existing tests and bounded local runs in this worktree's build/output
  directories. Add MPI, refinement, or longer runs for a concrete remaining
  question. Production jobs and full trajectories require a selected task.
- Verify changed behavior with independent references and measured errors,
  including acceptance/rollback and relevant transfer/restart semantics.
  Record revision, inputs, commands, norms, tolerances, results, and limitations.
  Keep diagnostics compact; do not qualify physics from a smoother plot.
- Maintain `particle_free/STATUS.md` with scope, baseline, evidence links,
  unresolved decisions, and the next action. Use `SOURCE_AUDIT.md` and one
  report per phase in that directory for details; retain original propagation
  evidence. Reconcile status with actual files after resuming.
- Follow existing commit/push policy. Report completed scope and concrete next
  steps; do not automatically resume historical experiments.
