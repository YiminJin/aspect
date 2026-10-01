# Codex instructions: revised R4 coupled-solver refactoring

## Active task and review points

R3 is committed. Update the refactoring guidance to reflect the decisions below, then implement **R4a only** in the `pf-rsf-refactor` worktree. R4b and R4c describe the roadmap; they are not authorized for implementation by this instruction.

Complete the guideline updates and R4a, verify the result, and provide a concise review report. Stop before the next pass so Yimin can review and adjust the plan. Resolve routine implementation details without repeated permission questions.

Preserve user edits and the separate scientific-test worktree. Do not restart R0–R3 or perform a scientific experiment.

## 1. Record the agreed architecture

Keep the existing user-facing scheme, `single Advection, iterated Newton Stokes`, and its current reconstructed-fault dispatch. Do not add a new nonlinear-scheme enum, parameter choice, or timestep ordering.

Retain `Simulator::solve_reconstructed_fault_stokes()` as the dedicated coupled nonlinear driver. Do not merge its algorithm into `do_one_defect_correction_Stokes_step()` or import the ordinary solver's Picard startup, stabilization fallback, residual criteria, or state-publication behavior.

The intended organization has three levels:

| Level | Responsibility |
|---|---|
| Coupled nonlinear driver | Prepare one mechanics solve; own iteration state and the outer loop; evaluate convergence; publish the converged state; restore state on failure |
| Coupled iteration operations | Evaluate residuals; construct the consistent linearization; handle slip-rate bounds and active-set solves; recover increments; perform the joint bulk/slip-rate line search |
| Linear solution | Solve the supplied condensed linear system, using existing Stokes preconditioner components where applicable |

A private `do_one_reconstructed_fault_stokes_step()` is a possible organization of the middle level. Its purpose is readability and explicit responsibilities, not hypothetical reuse by other schemes. Extract it only if the boundary is clean after the substantial operations have been separated. A readable driver calling several focused operations is an acceptable outcome.

Keep M1 particle domains/CPDI and M2 phase-field implementation frozen. This task concerns M5 coupled mechanics; preserve M3 infrastructure ownership and M4 material-history interfaces established in R3. Do not create a new persistent state owner, solver framework, plugin family, or abstract constitutive interface.

Update the existing documents rather than replacing them:

- Record these architecture decisions in `doc/reconstructed_fault/refactoring.md`.
- Revise R4 in `doc/reconstructed_fault/refactoring/refactoring_plan.md` to distinguish R4a, R4b, and the separately selected R4c below.
- Update active status and references as needed. Keep `AGENTS.md` changes minimal; it should point to the detailed plan rather than duplicate it.
- Preserve historical reports and their actual outcomes. Label superseded proposals where necessary.

Read the relevant scientific specification and current design before source edits. These instructions revise software organization, not the mathematical model. Report substantive conflicts instead of silently changing scientific contracts.

## 2. Establish the comparison reference

Inspect branch, HEAD, staged/unstaged changes, current definitions, callers, tests, and build rules. Record the actual accepted post-R3 baseline and qualified executable/plugin manifests. The current roadmap identifies the qualified Maxwell naming cleanup as part of that baseline; verify its local evidence rather than assuming an older R3a revision is the reference.

Preserve the reference executable, plugins, configurations, checkpoints, and outputs. Build the candidate separately and ensure candidate tests load candidate plugins. Record any intervening changes that affect the comparison.

Use current source as the authority for names and interfaces. This plan was informed by refactor commit `d7b88b25e6e157206b6bdcaf714dac434701f97c`; it is a reviewed snapshot, not an instruction to reset or check out that commit.

## 3. R4a — Relocate the existing coupled driver

Move the complete definition of `Simulator::solve_reconstructed_fault_stokes()` from `source/simulator/solver.cc` into a dedicated implementation file under `source/simulator/solver/`. A suitable name is `reconstructed_fault_stokes.cc`; follow existing naming conventions if another name fits better.

Keep it a Simulator member. Move exclusive helpers with it. Preserve the function body, local lambdas, numerical expressions, diagnostic capture points, and execution order.

Handle shared definitions explicitly:

- Inventory helpers/classes used by both the ordinary and fault solver, including the Stokes operator wrapper and Schur-preconditioner support where still applicable.
- Do not duplicate their implementations to make the move compile.
- If necessary, mechanically relocate existing shared definitions into a narrowly scoped internal header/source using the repository's conventions. Retain their existing interfaces and behavior.
- Distinguish this necessary definition movement from the later consolidation of duplicated preconditioner construction.
- Do not widen Simulator's public API or introduce broad accessors to avoid member access.

Adjust includes, source discovery/build entries, and explicit template instantiations only as needed. Keep exactly one definition/instantiation where required. Each translation unit must compile with its own includes; do not rely on unity-build ordering. Check the supported 2D/3D instantiations even though fault mechanics remains restricted to 2D.

R4a does not extract the one-step helper, redesign contexts, merge linear-solve paths, remove diagnostics, rename unrelated variables, change assertions, or reformat whole functions. Preserve small necessary build adjustments as an identifiable part of the diff.

**R4a acceptance:** the general solver file no longer contains the complete fault Newton algorithm; the relocated implementation builds independently and behaves like the qualified reference.

## 4. R4b — Separate substantial coupled-solver operations

This pass requires a later user selection. Start with the condensed linear solve/preconditioner setup and trial residual evaluation. Reuse existing reconstructed-fault linear, nonlinear, and condensed-system interfaces where they already express the required operations.

Before each extraction, identify its inputs, outputs, mutated state, collective operations, and required lifetimes. Keep these contracts short and concrete.

Then assess whether one coupled iteration can be represented cleanly by a private `do_one_reconstructed_fault_stokes_step()`. If so:

- Keep preparation, the outer iteration loop, convergence/publication decisions, and whole-solve failure restoration in the driver.
- Let the step coordinate linearization, active-set stabilization, linear solution, increment recovery, and acceptance of one coupled iterate.
- Preserve the existing convergence-check locations and order. The caller must be able to finish without forcing an unnecessary increment; do not move convergence checks merely to fit the helper.
- Distinguish acceptance of an iterate from commitment of timestep histories.
- Do not reuse `DefectCorrectionResiduals` or add `use_picard` merely to match the ordinary solver's signature.

Use explicit arguments where practical. A small private scratch structure is acceptable when its members share the lifetime of one solve and clarify ownership. It must not duplicate canonical manager/surface-system data, become serialized state, or turn into a container for every Simulator dependency.

If a one-step extraction requires a large mutable context, many unrelated arguments, or a broad callback interface, keep the focused operations and readable driver. Report that decision; creating the named function is not an acceptance requirement.

Implement R4b in reviewable subpasses, with one extraction selected at a time. Do not bundle all proposed extractions into the first R4b change.

## 5. R4c — Share demonstrated linear-solver duplication

This is a separately selected pass after the coupled operations are clear. Inspect both implementations and propose the smallest useful shared operation, starting with common preconditioner construction where appropriate.

The coupled bulk increment solves

```text
C delta_x = -R_bulk + B K_V^{-1} R_Gamma
C         = A - B K_V^{-1} G
delta_V   = K_V^{-1}(R_Gamma + G delta_x)
```

Here A is the bulk Stokes Jacobian. During an active-set solve, retain the implementation's corresponding restricted free-surface inverse and residual treatment.

Do not replace C by A or lag the coupling term to call the existing `solve_stokes()`. That would change the algorithm. An operator-aware overload or internal shared helper is possible if justified by the actual extracted dependencies; neither the name `solve_stokes()` nor a single universal entry point is mandatory.

Keep physical state handling and fault iteration policy out of generic linear algebra. A shared linear operation should not commit fault/material histories or publish a nonlinear iterate.

Preserve caller-specific behavior, including:

- Ordinary cheap/expensive solve stages versus the fault solver's existing iteration budget and fresh-residual verification/restart behavior.
- Conditional pressure-complement projection and compatibility checks.
- Fault GMG use as a velocity preconditioner with the assembled fine coupled operator; it must not be redirected into the ordinary full matrix-free Stokes solve.
- Existing solver signals/observers, tolerances, counters, and failure semantics.

Do not introduce a generic policy framework to eliminate minor duplication. Keep ordinary Stokes, melt, direct-solver, and matrix-free behavior unchanged; verify affected paths when common code is actually modified.

## 6. Contracts to preserve throughout R4

Preserve the existing ordering and semantics of:

- Frozen material-history preparation once per mechanics solve; no persistent constitutive-history writes during residual/Jacobian evaluation or line search.
- Manager-owned committed/current/trial slip rates; private accepted bulk iterates; final publication and pre-commit rollback.
- One canonical surface system and coupling implementation; linearization generation validity; restricted inverse, matrix, and constraint lifetimes.
- Physical versus homogeneous increment constraints; pressure scaling; temporary assembly flags and their restoration.
- Bulk and surface residual scales, precision allowances, active-set rebuilding, exact absolute slip-rate values at bound contact, and joint Armijo decisions.
- Algebraic pressure-complement projection separately from permitted physical pressure normalization.
- MPI ownership, collective order and failure propagation; exception paths and observer timing.
- Existing defaults, checkpoint compatibility, and supported solver modes.

Do not claim stronger atomicity or exception guarantees than the current publication implementation provides. Treat suspected numerical defects as separately reported tasks.

## 7. Bounded verification

Reuse qualified small fixtures, comparison tools, ranks, stack, and tolerances. Select checks according to the actual diff:

| Pass | Evidence to obtain |
|---|---|
| R4a | Candidate build/link plus focused independent-compilation checks; an existing short coupled comparison and relevant rollback check; ordinary disabled-feature smoke if shared definitions move |
| R4b | Existing checks for the extracted operation: condensation/fresh linear residual, active bounds and Armijo, trial restoration, relevant pressure modes; short coupled comparison covering final history publication |
| R4c | Small frozen AMG/GMG comparison for affected paths, retained true-residual checks, and existing ordinary Stokes/melt or other backend smoke cases whose common code changed |

Use an existing small multi-rank check where moved or extracted code contains MPI collectives. Compilation checks are not a request for 3D simulations or a full compiler matrix. Reuse existing restart evidence unless changed code reaches restart initialization or restoration.

For unchanged arithmetic, attempt exact comparison of deterministic physical/history outputs and solver decisions under matching ranks/stacks, excluding timing and path metadata. Investigate differences; do not loosen tolerances or regenerate expected answers to manufacture a pass.

Keep known failures separately identified. Do not repair the pre-existing Stage-J nonlinear-convergence case inside R4. No production earthquake run, large performance campaign, or repeated M1/M2 qualification is requested. Add a test only for a concrete gap introduced by the change. Record blocked or unrun checks honestly.

## 8. Report and stop after the selected pass

Keep guideline edits, mechanical movement, structural extraction, and any later shared-code change distinguishable in diffs or commits. Follow the established local commit policy; no push, merge, or modification of the scientific worktree is requested.

Update the rolling review/status and provide a user-facing summary of roughly 250 words:

1. What changed and why.
2. Whether responsibilities or interfaces changed.
3. Baseline/candidate revisions and verification results.
4. Remaining limitations or decisions.
5. One proposed next bounded task.

Include a compact PASS / FAIL / BLOCKED / NOT RUN table with evidence paths. Keep detailed logs and symbol/file inventories in supporting evidence rather than the main summary.

For this instruction, finish **guideline updates plus R4a only**. Recommend the first R4b extraction from the resulting code, then stop for review.

