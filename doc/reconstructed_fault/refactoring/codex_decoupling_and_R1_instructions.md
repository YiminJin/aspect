# Codex task: declare module boundaries and qualify the R1 baseline

Work in the `pf-rsf-refactor` worktree. This task authorizes the documentation updates below and R1 baseline qualification. R0 is accepted as a dependency audit. Do not begin source extraction or numerical fixes.

## 1. Adopt these responsibility boundaries

| Module | Owns | Current refactoring scope |
|---|---|---|
| M1 — Particle domains / CPDI | Particle-domain construction, integration weights, shape-function information, domain lifecycle | Freeze the current implementation |
| M2 — Phase-field method | Generic phase-field equations, discretization, solution, and material capability interface | Freeze the current implementation |
| M3 — Reconstructed-fault infrastructure | Geometry, connectivity, reconstruction, bulk–fault associations, generic projections, property storage, persistence, and the existing distinguished slip-rate storage/lifecycle | Active scope after R1 |
| M4 — Fault material physics | Maxwell/cohesive/friction responses, localization and current I_h, material parameters, and constitutive history semantics and updates | Active scope after R1 |
| M5 — Coupled mechanical system | Surface/bulk assembly, MPI reduction, K_V/B/G operations, restricted surface solves, condensation, preconditioning, nonlinear iteration, acceptance and rollback | Active scope after R1 |

These are responsibility boundaries, not five new libraries or plugin hierarchies. Keep existing names and file locations during this task. M3 geometry/property storage must not interpret friction laws or constitutive histories. Reconstruction and projection adapters may use M1/M2 without making every geometry operation depend on them.

M4 owns the physical meaning of I_h and its use in localization. Its numerical implementation may be extracted into private implementation files later. Generic sampling/projection services remain reusable. Do not move material-specific normalization into M2 or M3 merely to shorten the material model.

The surface system belongs conceptually to M5 even while its files remain under `reconstructed_fault/`. Preserve the concrete material-response interface and the canonical simulator-owned surface/coupling objects. No new abstract constitutive base class is requested.

For example: M3 stores Theta as a generic property; M4 computes its physical update; M5 determines when successful convergence permits M4 to publish the update. Preserve current ownership of committed/current/trial V, history candidates, collective validation, and terminal publication.

M1/M2 must eventually work without concrete dependencies on reconstructed-fault physics. For now, record reverse dependencies, including the `PhaseFieldFault` mature-mode check in `PhaseFieldHandler::evolve_phase_field()`. Do not remove or relocate them in R1. Narrow interface changes at those boundaries require a separately selected task; broad M1/M2 cleanup remains out of scope.

## 2. Update existing guidance, without repeating R0

Read the current AGENTS.md, scientific contracts, refactoring guidance, selected staged plan, R0 report, and relevant current status. Preserve user edits and use current files rather than replacing them with earlier copies.

Make small, consistent documentation updates:

- Put the module table and dependency rules in `doc/reconstructed_fault/refactoring.md`.
- Add a short scope/reference paragraph in AGENTS.md: M1/M2 frozen; M3–M5 active; only the user-selected stage is authorized.
- Use the already selected `doc/reconstructed_fault/refactoring/refactoring_plan.md` as the authoritative refactoring roadmap. Reconcile links in the other guidance. Clearly mark an older competing roadmap as superseded; preserve unrelated historical content.
- Keep R0–R8 stage names. Mark R0 complete as an audit, R1 active, and later stages pending. R2a remains normalization relocation within M4; R4 concerns M5. Defer prerequisite cleanup/PR preparation for M1/M2.
- Add a short decision summary to the existing R0 report. Keep detailed inventories as reference. Record remaining boundary dependencies and future upstream decisions as deferred, not immediate blockers.

Do not rewrite mathematical, MPI, or checkpoint contracts to match a desired implementation. This declaration changes refactoring scope and responsibility labels; it does not change scientific algorithms. Record any substantive contract conflict for separate resolution.

Historical scope marker: the first reconstruction commit is
`76a1e68e062be11bf007e541712d0c5251bd410e`; its parent is
`e3ef3ae89ade2a16db04fcabecccc5db3a7f0ced`.
For a targeted historical check, compare that parent with the selected current baseline so the first commit is included. The commit itself also changes phase-field/material code: do not treat chronology as a clean module boundary, revert prerequisites, or repeat the exhaustive upstream audit.

## 3. Carry out R1 on the unchanged scientific implementation

Start by recording worktree, branch, HEAD, local changes and exact source state. R0 recorded `3af7c2aa2590aa8e39eaf2a1e46df0bc6c9d9700`; verify rather than assuming it is still current. Document intervening source changes. Never discard local work or silently compare different source states.

Use separate refactor build, plugin-build and output directories. The scientific worktree is a read-only source of selected immutable fixture inputs; do not rebuild, execute, or overwrite its saved runs. Record compiler, deal.II, Trilinos, MPI, Voro capability, build flags, ranks, threads, PRMs and numerical environment switches.

Use 1–2 MPI ranks, one thread per rank, and modest build parallelism (start with two jobs). Configure the existing test infrastructure and use its actual test names. Run a bounded local suite; do not launch a production BP3 run, full-resolution first event, or full ASPECT test campaign.

Required evidence:

1. **Timestep parameter behavior.** With the installed parser, check the default, literal `infinity`, zero, and one valid positive bound. Check the default's numeric round-trip and whether disabled settings actually bypass the optional limiter. Record the selected trajectory's effective setting. Do not change the production parser or limiter. If the discrepancy prevents qualification, report a separate minimal fix task; continue unaffected baseline checks.
2. **Normalization.** Include both `[phase_field_fault_ih_accuracy]` and `[phase_field_fault_ih_cache]`, relevant 1/2-rank lifecycle tests, and the no-composition case. Qualify both normalization backends through existing small harnesses where available. Time expensive cases before broadening; record missing coverage explicitly.
3. **Mechanics and rollback.** Run the existing focused Maxwell/cohesive and coupled-algebra tests relevant to the planned extraction, plus a reachable forced-failure rollback case. An input that fails before reaching the intended assertion is not passing rollback evidence.
4. **Short evolving trajectory and restart.** Prefer the existing six-physical-step BP3 output-cleanup fixture using `fixture-convex/`, its unchanged physical settings and full bottom constraint. Copy/rebind inputs and old library paths to the new matching builds. Recreate checkpoints; compare a split restart through the final states with the uninterrupted baseline. Verify actual checkpoint step numbering. Preserve original evidence and report any omitted observational plugin.
5. **Ordinary ASPECT behavior.** Use the proposed small disabled-feature particle case and ordinary particle checkpoint/resume case. Do not requalify all of M1/M2.

Small test-only harness/comparison additions are authorized when necessary to reuse the fixture or close a specific evidence gap. Preserve production source, numerical defaults, scientific inputs, and acceptance tolerances. Keep documentation changes and harness additions distinguishable.

Retain a reproducible reference executable, matching plugins, configuration, fixture inputs, logs and baseline checkpoint for later candidate comparisons. Compare physical fields/history and solver decisions; report field-specific absolute/relative errors rather than normalizing tiny slip rates by one. Same-rank restart agreement and cross-rank agreement are separate checks.

Existing compatibility failures are pre-existing findings, not permission to fix algorithms or weaken tests. R2a requires usable passing evidence for normalization and the relevant history/restart/rollback paths, or an explicitly reviewed disposition. Baseline agreement establishes reproducibility, not physical validation of production BP3.

## 4. Stop and report concisely

Update the existing rolling report/status. Do not create a new report for every test.

End with at most roughly 300 words, plus this compact table:

| Check | PASS / FAIL / BLOCKED / NOT RUN | Evidence or reason |
|---|---|---|

State:
- Which guidance/harness files changed; confirm whether production source stayed unchanged.
- The exact qualified source revision/state.
- Any blockers that matter for R2a.
- One recommendation: proceed to R2a, complete one missing check, or select one separate fix.

Keep commands, logs and detailed inventories linked below the summary. Stop after R1; do not start normalization extraction, reorganize modules, or prepare upstream PRs. Follow the existing local commit policy; do not push, rebase, merge, or change the scientific worktree as part of this task.
