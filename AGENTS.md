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
| Particle-free task | `particle_free/STATUS.md` and the current user-selected instructions; local plans are proposals, not required tracked entry points |
| Equations and checks | Relevant sections of `particle_free_history.tex` and `particle_free/VERIFICATION.md` |

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

## Research review and objective stress update (2026-10-30)

- Read `doc/reconstructed_fault/particle_free/RESEARCH_HANDOFF.md` and
  `REVIEW_PLAN.md` in that directory at entry. Verify current source, dirty work,
  binary identity, and data availability; handoff revisions are references, not
  reset targets. Preserve the two qualified binaries before a rebuild overwrites
  them. Keep their required runtime libraries available.
- The paper roadmap contains BP3, a controlled viscoelastic-crust experiment,
  and fault propagation. This thread actively covers BP3 review, the shared
  mathematical/implementation maps, and an opt-in stress-spin implementation
  with bounded local qualification. The other two physical campaigns remain
  deferred. No realistic subduction application is planned.
- The user's request selects addition of objective stress rotation. It is a
  scientific feature task, not refactoring. The earlier rule to preserve the
  spin convention permits this selected change. Preserve the legacy/default
  path and existing production inputs as comparison controls. Do not silently
  resume an old trajectory under a changed constitutive law.
- Begin with evidence review and variable/timestep/energy maps. Trace both
  production BP3 and the newer cohesive-interface route far enough to identify
  shared code and different constitutive assumptions. Do not merge their
  histories, fracture inventories, validation claims, or pressure conventions.
  Use one current formulation table rather than duplicate specifications.
- Follow `OBJECTIVE_STRESS_SPEC.md` for the spin audit and local qualification.
  Establish the stress measure, material derivative, velocity-gradient index
  convention, constitutive spin, tensor frame, and treatment of localized slip
  before editing. A corotational symbol does not by itself select Jaumann spin.
  Do not subtract a slip spin or substitute a different objective rate without
  deriving that choice from the selected kinematics and stored energy.
- Add the selected terms to every stress/history path needed for the declared
  enabled configuration, including reconstruction, residual, tangent/outer
  iteration, observers, rejection, and restart. A spin-enabled label must not
  hide an unrotated auxiliary tensor or an unqualified formulation. Leave
  unsupported combinations explicitly disabled with a useful diagnostic.
- Use finite-rotation integration of the selected spin where compatible with
  existing time integration. Derive forcing and transport ordering; rotating
  the old tensor alone is not a complete constitutive update. Preserve accepted
  histories through trials and commit once. Retain the full 3D plane-strain
  tensor convention. Do not claim finite-strain hyperelasticity from objectivity.
- Independent rotation/relaxation, superposed-motion, transport, shear, energy,
  tangent, and lifecycle tests are required where applicable. Use measured
  timestep convergence, not a universal total-strain threshold. Document the
  finite-rotation and constitutive limits separately.
- Bounded local tests, source builds (normally at most four local workers),
  and analysis needed for these deliverables are in scope; proceed without
  repeatedly requesting confirmation. No server submission, long event replay,
  full parameter sweep, new propagation campaign, or production checkpoint
  conversion is included. Missing remote BP3 data must not block independent
  code mapping and local rotation tests.
- If the requested spin requires a substantive unselected constitutive or
  geometry change, identify the exact conflicting equations and stop only
  that dependent change. Continue independent work. Do not silently repair
  physics through extra damping, filtering, tolerance changes, or parameter tuning.
- Keep `STATUS.md`, `SOURCE_AUDIT.md`, the active plan, and compact reports
  consistent. Distinguish implemented, locally verified, physically compared,
  and production-ready. Retain failed evidence and cleanup availability losses.
  Do not recreate deleted historical campaigns; keep only necessary new output.

## Transported objective state contract (Stage A, 2026-11-02)

- For the user-selected transported extension, use
  `particle_free/TRANSPORTED_OBJECTIVE_SPEC.md` and the state table in
  `IMPLEMENTATION_MAPS.md`; task-state/report distinguish design from enabled code.
- Bulk material transport, physical interface-label motion and tangential node
  redistribution are distinct. Remap complete physical state and its obstacle;
  never infer age/slip transport from relative slip or rotate normals by stress spin.
- A transported selector requires complete tensor/native-remainder/lifting state,
  matching work adjoints, and one geometry/history acceptance transaction.
  Checkpoint metadata must identify representation, law/integrator and material
  identities; no automatic old-to-new conversion. Keep legacy controls.
- Qualification must include independent moving-profile/variable-eta references
  and geometry/flux work where applicable. Stage A algebra alone enables no
  configuration; the current user instruction determines which later stage runs.


## Native objective controls (Stage C, 2026-11-03)

- Distinguish stationary reaction controls from a complete transported native
  step. The mature T/R owner and schema6, and cohesive formed-piece/side tensor
  owner with retained native/lifting memory, have local evidence only within
  their stated restrictions. Defaults and legacy checkpoint laws stay intact.
- Cut history partitions belong to formed material, not temporary candidate
  tails. Persist partition/basis/native state and select the same material side
  for tensor and lifting observations. Rebuild matching checker DSOs after
  history-layout changes; do not infer binary identity from HEAD.
- Keep failed uncut-space, physical-energy and tip-admission evidence visible.
  A transport-operator pass does not authorize moving curve/profile/surface
  publication or deformed cohesive shape work. Use the active Stage C report
  and task state for the current implementation/qualification boundary.
