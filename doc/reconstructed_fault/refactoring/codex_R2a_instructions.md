# Codex task: update the layout guidelines and implement R2a

R1 is completed. In the `pf-rsf-refactor` worktree, update the existing guidance and perform **R2a only: relocate normalization implementation without changing behavior**. This instruction supersedes the earlier proposed normalization-file destination.

## 1. Record the layout decision

Use:

`source/material_model/phase_field_fault/normalization.cc`

This directory contains implementation support for the PhaseFieldFault material model (M4). It is not a new plugin family or a generic material-model interface. Keep the existing plugin entry file:

`source/material_model/phase_field_fault.cc`

Preserve the five-module division: M1 particle domains/CPDI and M2 phase-field method remain frozen; M3 owns generic fault infrastructure; M4 owns material physics; M5 owns surface/bulk assembly and coupled solution.

Normalization currently combines material policy and numerical machinery. M4 retains the integrand, material-mixture choices, admissibility/singularity rules, current-I_h validity decisions, and current/previous-I_h semantics. A future, separately selected R2b may assess extracting model-independent profile integration, sampling, or projection into `reconstructed_fault/`. Such a component must not depend on PhaseFieldFault. R2a does not perform that extraction or introduce a helper class.

Read the current AGENTS.md, refactoring guidance, authoritative scientific contracts, selected roadmap, and R1 report. Then:

- Update `doc/reconstructed_fault/refactoring.md` with this layout/ownership rule.
- Update the R2a destination in the selected `doc/reconstructed_fault/refactoring/refactoring_plan.md`. Keep R1's actual completion evidence and mark R2a active until verified.
- Reconcile active references in AGENTS.md or the rolling review only where needed. Preserve historical reports; label an obsolete proposal as superseded rather than rewriting past evidence.
- Make later proposed M4 implementation paths consistent with this directory convention, but create no future-stage files.
- Preserve the scientific/MPI/history contracts. No equation or algorithm change is authorized.

Do not repeat R0, restart R1, or replace the current guidance with an older copy.

## 2. Establish the exact comparison reference

Record branch, HEAD, local modifications, and the R1-qualified source state, executable, plugins, fixtures, configuration and evidence locations. Use the actual R1 result; do not assume an earlier SHA or that every proposed R1 test passed.

Preserve user edits and the scientific worktree. Keep the qualified reference executable/plugins/checkpoints intact and build the candidate separately. If source has changed since qualification, identify the changes and their relevance; do not silently compare unmatched implementations.

A material missing baseline check or unresolved failure must be reported explicitly. Complete independent authorized work, but do not claim R2a qualification without evidence for its affected paths. Do not repair unrelated numerical defects inside this pass.

## 3. Relocate complete definitions

Inspect current definitions, callers and test access before editing; names below come from R0 and must be checked against the current tree.

Move these normalization definitions from the main material source into the new file:

- `compute_normalization_integrals()`
- `invalidate_normalization_cache()`
- `normalization_effective_phase_field()`
- `validate_normalization_phase_field_minimum()`
- `normalization_integrand()`
- `integrate_cell_normalization_profiles()`
- `integrate_normalization_profiles()`
- `project_surface_chemical_compositions()`
- `NormalizationPointLookupCache::get()`
- `evaluate_normalization_points()`
- `build_owned_normalization_profiles()`
- `project_normalization_integrals_to_fault()`

Move their exclusive anonymous-namespace helpers: `normalization_search_enclosure`, `NormalizationSideState`, `NormalizationEvaluationRequest`, and `normalization_profile_point`, where still applicable.

Keep `interpolate_surface_chemical_compositions` with its cohesive-initialization caller; it is distinct from projection onto the fault. Keep shared mechanics/history helpers, the completion-file setter, and other unrelated methods in their current locations. Do not duplicate shared helpers or mixture preparation to make the split convenient.

Retain PhaseFieldFault's existing namespace, class, declarations, state and visibility in `include/aspect/material_model/phase_field_fault.h`. No new public API, exported helper header, manager method, state owner, or abstract constitutive base class is needed.

Move whole bodies with their comments and diagnostics. Adjust only translation-unit scaffolding, includes and necessary template instantiation. Avoid simultaneous renaming, reformatting, assertion cleanup, cache consolidation, parameter changes or diagnostic extraction.

Preserve arithmetic and evaluation order; MPI ownership and collective order; both normalization backends/fallbacks; consistent Q1 projection; fixed profile mixtures; phase clamps and raw undershoot checks; endpoint completion; geometric lookup versus value-cache validity; mesh-deformation handling; and restart invalidation/restoration. Keep constitutive history and solver code unchanged.

## 4. Handle compilation and registration explicitly

Keep exactly one `ASPECT_REGISTER_MATERIAL_MODEL(PhaseFieldFault, ...)` in the original plugin file.

Inspect the registration macro and explicitly instantiate moved template members as needed for dimensions 2 and 3, including static/private members and the nested cache method used by tests. Do not add a second whole-class instantiation as a shortcut, widen visibility, or rely on unity-build ordering to supply definitions.

Use the existing recursive source discovery if it still covers the new file; no new library target or build framework is requested. Make each translation unit include its own dependencies. Verify ordinary separate compilation/linking, using a focused non-unity check if the qualified build uses unity; check the normal candidate configuration too. These are symbol/build checks, not a request for 3D simulations or a compiler matrix.

## 5. Verify against the completed R1 baseline

Reuse the qualified harness, settings, comparison rules and resource budget. Rebuild required candidate plugins against the candidate build. Do not overwrite reference outputs or let a test target rebuild the reference against edited source.

Repeat the selected checks that protect the moved code:

- Normalization accuracy and cache tests, including `[phase_field_fault_ih_accuracy]` and `[phase_field_fault_ih_cache]`.
- Relevant 1/2-rank I_h lifecycle and no-composition cases.
- Both normalization backends through the existing qualified small cases.
- The qualified short history-evolving trajectory and relevant rollback check.
- Loading a preserved baseline-created checkpoint with the candidate, followed by comparison with the uninterrupted reference over the matching continuation window.
- One qualified ordinary disabled-feature smoke case.

Compare I_h, relevant physical fields/history, cache hit/rebuild behavior and solver decisions at matching ranks/stacks. For unchanged arithmetic, attempt exact comparisons excluding timing and path metadata. Investigate differences; do not weaken tolerances or update expected output to manufacture a pass.

Reuse existing tests. Add or adjust a small test/harness only for a concrete gap caused by the split, such as access to a moved template symbol. Do not repeat unrelated parameter probes, full CPDI/phase-field qualification, or production benchmarks. Record any required check that cannot run.

## 6. Review and stop

Inspect the final diff for unintended changes and preserve a clear distinction between guideline updates, source relocation and any necessary build/test scaffolding. Follow the existing local commit policy; do not push, rebase, merge or change the scientific worktree.

Update the existing rolling report and status. Keep the user-facing summary within roughly 300 words, plus:

| Check | PASS / FAIL / BLOCKED / NOT RUN | Evidence or reason |
|---|---|---|

Report:
- The files/methods moved and before/after line counts.
- Any changes beyond relocation, with reasons.
- The exact baseline/candidate states and comparison results.
- What remains coupled: the material still owns the normalization data and interface.
- One recommendation for the next bounded task.

If verified, mark R2a complete. Describe it accurately as separation into translation units, not completed architectural decoupling. Stop before R2b, helper-class extraction, generic integration services, or R3.
