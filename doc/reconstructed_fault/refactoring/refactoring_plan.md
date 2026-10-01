# Reconstructed-fault refactoring: a staged collaboration plan

Prepared for Yimin Jin — 28 September 2026, America/Los_Angeles

Status: R0–R3 and the separate restart correction/Maxwell cleanup are complete.
R4a is accepted under [revised R4 instructions](codex_R4_instructions.md).
R4a used the accepted post-R3 `eta_ve` state, including the original frozen-stress
implementation: `build-refactor-r3b/aspect-maxwell-qualified` and the source
manifest in [cleanup evidence](../../../benchmarks/reconstructed_fault/maxwell_cleanup/README.md).
That reference HEAD was `d7b88b25e`; its difference from `82e43c266` is the
user's removal of a temporary review document. Preserve local changes and
reference artifacts. The first R4b condensed-solve subpass is accepted against
R4a commit `0c7ed1a0b`. The selected second subpass extracts only the coupled
trial residual operation, using the qualified first-subpass executable and
its accepted uncommitted patch as reference, not plain R4a. The second subpass
is accepted; see the rolling review and residual evidence for exact comparisons.
Both R4b subpasses and the recommendation to retain the existing driver with
two focused operations are accepted and saved together. No whole-iteration
helper was implemented. The qualified second-subpass executable and manifests
recorded in the residual evidence are the baseline for subsequent work.
R4b is committed as `983d57e28`. The selected R4c inventory proposes only shared
Schur-wrapper construction; see the rolling review for dependencies and checks.
The selected R4c implementation and focused verification are accepted:
53 focused comparisons and 372 BP3 field groups match. The historical frozen-GMG
probe has an identical baseline/candidate hierarchy-setup failure; its AMG half
and the supported actual GMG case pass. See the rolling review for that limitation.
The separately selected next task repairs only the historical frozen AMG/GMG
comparison fixture. No further consolidation or R5–R8 work is selected.
Historical outcomes remain in [the rolling review](../refactor_review.md).
Module boundaries and frozen M1/M2 scope are defined in [refactoring.md](../refactoring.md).

Reviewed source: YiminJin/aspect, branch `pf-rsf`, commit `0fc1ce782c48b78f79eff674724686b8277c8618`. The remote branch still pointed to this commit when checked for this plan. The user's local checkout may contain newer work. Inspect and preserve that work before selecting an implementation baseline.

## 1. Goal and definition of success

Make the implementation understandable, separable into reviewable contributions, and easier to maintain in ASPECT. Preserve the scientific algorithms during structural refactoring.

Success means:

- A developer can identify the owners of geometry, constitutive state, trial iterates, assembly, and history publication.
- General ASPECT code contains a small number of explicit integration points; substantial fault-specific algorithms live in feature-specific files.
- The material-model entry point is readable without tracing distributed normalization integration or CSV output.
- The nonlinear driver visibly follows preparation, residual evaluation, linearization, solution, trial acceptance, and final publication.
- The selected numerical comparisons, failure/rollback checks, restart checks, and ordinary non-fault checks match their recorded baseline.
- The contribution can be organized into buildable, documented PRs, with research outputs and unrelated experiments excluded from the submission.

File length is a diagnostic, not an acceptance criterion. Moving a function to another file improves organization but does not, by itself, reduce coupling or the upstream diff.

Equivalence to the research baseline does not establish that the underlying model is scientifically correct. Existing convergence and stress-concentration questions remain separate investigations.

## 2. How the three of us work together

| Participant | Primary responsibility | What they need to see |
|---|---|---|
| Yimin | Select the next bounded task; review scientific ownership and proposed interface changes; accept or revise each stage | A concise report, the relevant diff, numerical comparisons, and explicit unresolved questions |
| This conversation | Review Codex's evidence; check the architecture against the mathematics and upstream goal; refine the next task | Stage report plus diff/commit and any changed interface or numerical evidence; access alone is not assumed |
| Coding Codex session | Inspect local source, implement the selected task, build/test, and produce reproducible evidence | Repository instructions, selected baseline, this plan, and one active task envelope |

Use one implementation task at a time. A stage can contain several small passes, but Codex must finish the selected pass and report before expanding its scope. It can resolve ordinary implementation details within the agreed scope; it should raise questions when scientific behavior, public APIs, ownership, restart formats, or the agreed test budget would change.

Suggested cycle:

1. Select a stage and its exact subtask.
2. Codex identifies source, callers, tests, and the intended diff before making that change.
3. Codex implements that subtask, runs the relevant checks, and reports.
4. Yimin shares the report and diff here; we review the evidence and adjust the plan.
5. Yimin explicitly selects the next task. The next Codex prompt carries the updated scope.

Do not make oversight depend on frequent low-level permission questions. The review points are between meaningful units of work.

## 3. Branches, baseline, and concurrent research

- Keep `pf-rsf` available for scientific experiments and as a reference.
- Create `pf-rsf-refactor` from an explicitly selected research commit, in a separate worktree and build directory when practical.
- Preserve uncommitted files, simulation output, and checkpoints. Never reset, clean, overwrite, or automatically stash them to prepare the refactor.
- If uncommitted scientific changes must be included, first identify them and establish the intended baseline with Yimin. Record both the source revision and any required local patch.
- Keep an immutable baseline executable/configuration for comparisons. Do not overwrite its build directory while testing the candidate.
- Research fixes, such as a boundary-condition experiment, stay in separate commits. If one must enter the refactor baseline, integrate it explicitly and rerun only the affected baseline checks. Record the baseline change.
- Later create contribution branches from `geodynamics/aspect:main`, rather than assuming the fork's `main` matches upstream.
- Local commits can be created when included in the selected task authorization; remote pushes, PR creation, and changes to shared branches are separate tasks.

The snapshot reviewed during planning was shallow. R0 found full history in the
local refactor worktree and established merge base `42464facd7e0e41ba91a32eb061d97c21d97000b`
with the locally stored upstream main. See [the accepted audit](../refactor_review.md).
A tip-to-tip diff is still unsuitable for attributing project changes.

## 4. Authority and invariants

Follow `AGENTS.md`, `doc/reconstructed_fault/current_design.md`, `doc/reconstructed_fault/specification.tex`, and the applicable code-quality guidance in `doc/reconstructed_fault/refactoring.md`. Use `CURRENT_STATUS.md` to distinguish historical reports from newly executed evidence. The source tree defines actual APIs.

The older refactoring note contains obsolete paths and already-completed items. For example, the geometry/property sentinel query is now encapsulated, and the production material header already uses a test-access declaration. Audit current code before scheduling those items again.

Report material disagreements between the authoritative specification and implementation. Do not resolve them by quietly changing either the science or its documentation.

Preserve these contracts during R2–R7:

| Contract | Required preservation |
|---|---|
| Geometry | Application-owned ordered 2-D fault representation, persistent IDs/order, append-only committed geometry, existing versioning |
| Generic storage | Runtime property schema; do not hard-code Theta, cohesive traction, or a particular friction law into the geometry class |
| MPI | Replicated fault and distributed bulk; existing owned contributions, collective order, and collective failure propagation |
| Projection | Current particle-domain weights, Q1 projection, endpoint treatment, support, and quadrature |
| Normalization | Definition of I_h, material mixture, integration tolerances, completion treatment, and cache validity rules |
| Slip rate | Manager-owned committed/current/trial semantics; exact accepted absolute values at lower-bound contact |
| Constitutive histories | Residual/Jacobian evaluation does not publish Theta, cohesive history, previous I_h, particle stress, or H |
| Publication | Candidate construction and collective validation precede terminal persistent writes |
| Linearization | Canonical A/B/G/K_V lifetime; restricted free-block inverse; existing signs and pressure scaling |
| Nonlinear decisions | Existing active-set rules, residual scales, precision allowances, Armijo criteria, and failure behavior |
| Restart | Existing property names/order and serialization; transient caches rebuild as specified |
| Configuration | Existing physical parameters, defaults, supported modes, and timestep policy |

Specific numerical expressions must survive movement intact, including the stable `expm1` Maxwell coefficient evaluation. Preserve mature-frictional and cohesive paths where affected, and stateful/stateless friction. Do not replace the Maxwell implementation with another elasticity model merely because its software organization is attractive.

Do not introduce new filtering, interpolation, line-search strategies, constitutive regularization, propagation, or preconditioner algorithms in a structural pass. Report potential bugs separately and select a separate fix task when warranted.

## 5. Intended responsibility boundaries

| Component | Owns | Boundary to preserve or improve |
|---|---|---|
| `ReconstructedFault` | Geometry and generic vertex property storage | No material-law interpretation or bulk-solver logic |
| `ReconstructedFaultManager` | Geometry lifecycle, generic projections/caches, distinguished V lifecycle, persistence | No Maxwell/cohesive/RSF formulas |
| `PhaseFieldFault` | Material parameters, pointwise mechanics, current I_h, constitutive history semantics | No B/G operator application or Stokes Krylov implementation |
| `Rheology::FaultFriction` | Existing friction-law calculations | Reuse the existing law interface |
| Surface system and Stokes coupling | Weak forms, MPI assembly/reduction, surface factors, B/G actions | Retain canonical simulator-owned instances |
| Condensed system | Coupled block algebra and one linearization lifetime | Reference the existing components; no duplicate constitutive state |
| Nonlinear driver | Solve orchestration, trial acceptance, commit/rollback orchestration | General solver entry point should make a clear dispatch |
| Benchmark/diagnostic support | Model setup, specialized probes, formatting/output | Read exact numerical values at suitable hooks without changing acceptance |

The current specification deliberately uses the concrete PhaseFieldFault capability; it does not authorize an abstract constitutive hierarchy. A proposed interface generalization requires a concrete benefit, maintainer discussion where relevant, and an explicitly selected architecture task.

## 6. Stage overview

| Stage | Main outcome | Review decision |
|---|---|---|
| R0 | Dependency map and core-change inventory | Which changes are essential, independent improvements, or research-only? |
| R1 | Reproducible local baseline and selected tests | Is there enough evidence to begin structural edits? |
| R2 | Normalization implementation isolated | Is the split coherent and behavior preserved? |
| R3 | Material/history implementation organized | Are semantic ownership and publication order still clear? |
| R4 | Fault nonlinear driver extracted and simplified | Are core integration and coupled-solve responsibilities narrower? |
| R5 | Manager and surface assembly organized | Do proposed internal boundaries simplify real dependencies? |
| R6 | Diagnostics and configuration separated | Are numerical options preserved and observational code isolated? |
| R7 | Core interface review and integrated qualification | Is the refactored branch ready for scientific use and upstream preparation? |
| R8 | Upstream discussion and incremental port | What are the agreed buildable PR boundaries? |

Schedule R0 and R1 first. Reorder later stages only when the dependency inventory gives a concrete reason. R4–R6 may each need more than one pass. Estimate effort after R0; do not promise a fixed completion date before measuring build/test cost.

### R0 — Inventory and dependency audit

Source changes: none. Read source and write the audit report.

Tasks:

1. Record worktree status, branch, SHA, remotes, local changes, and shallow/full-history status.
2. Identify the common ancestor with upstream. Fetch history only as needed; do not rebase or merge the research branch during the audit.
3. Inventory all project changes, including prerequisites: particle-domain support, phase field, material mechanics, reconstructed faults, solvers, tests, and benchmark tools.
4. For each modified existing core file, record the changed symbol, callers, reason for the change, its owner, and the smallest plausible retained integration point.
5. Inventory new accessors, signals, parameters, environment switches, mutable shared state, and direct dependencies on PhaseFieldFault.
6. Identify current test dependencies, required local-only fixtures, and proposed first extraction boundaries.
7. Mark stale instructions and specification/source discrepancies; keep these distinct from new architectural proposals.

Core audit columns:

`file/symbol | project responsibility | reason | caller/dependency | keep/move/independent-PR/exclude | proposed destination | verification`

Read the integration sites identified by the actual diff. Likely candidates include simulator core/parameters/solver dispatch, assembly, checkpoint/restart, simulator access/signals, phase-field integration, and particle manager paths. This is a starting list, not a claim that all changes have already been attributed.

Deliverable: a dependency table and concrete R2 scope, with unresolved questions. Gate: Yimin reviews ownership and scope with this conversation. No numerical implementation is authorized by R0.

### R1 — Local reference baseline

Source changes: only an explicitly justified minimal fixture/comparison addition if existing tests leave a material gap. Do not repair a numerical failure inside this stage.

Tasks:

- Record compiler, deal.II/Trilinos/MPI versions, build mode and relevant floating-point flags, rank/thread counts, parameter files, and relevant environment options.
- Build the selected baseline using existing local tooling. Establish which tests pass, fail, skip, or cannot run because required artifacts are absent.
- Select a small stable coupled fixture that advances beyond initialization and updates histories. Prefer an existing fixture. Run approximately 3–6 accepted steps if that is sufficient to exercise the required lifecycle.
- Produce matched restart evidence with a small checkpoint created by the baseline executable. A candidate must later load it, and a split trajectory must agree with the uninterrupted reference over the selected window.
- Include one ordinary ASPECT case with the new features disabled. Select another baseline case only if R0 exposes a distinct affected core path.
- Record the relevant unit and integration checks from Section 7.

Use 1–2 MPI ranks by default on the laptop and record thread settings. Agree on a practical runtime budget after timing the baseline; aim for an incremental suite that takes minutes. Report a necessary longer build/test instead of launching an unbounded run. Full-resolution first-event simulations and TACC are not prerequisites.

The repository status notes explicitly say the saved source checkpoint was not fully rebuilt and simulated. Treat older reported successes as leads, not proof for the selected baseline. In particular, some historical fixtures already encountered pressure-compatibility problems.

Gate: the small reference trajectory and critical lifecycle tests have usable evidence. Any pre-existing failure has a recorded disposition. A failure in the code path about to be refactored requires a separate fix decision or independent passing evidence before proceeding.

### R2 — Extract normalization implementation

Selected R2a destination: `source/material_model/phase_field_fault/normalization.cc`.
This supersedes the earlier flat `source/material_model/phase_field_fault_normalization.cc`
proposal. For R2a, keep the material header unchanged and the sole plugin registration
in `source/material_model/phase_field_fault.cc`.

Pass R2a: move complete normalization methods and their file-local helpers. Keep the class, public API, parameter names, caches, formulas, MPI operations, and order unchanged. Handle explicit template instantiation carefully so moving definitions does not leave missing or duplicate symbols; keep one material plugin registration.

The group includes profile construction, distributed point evaluation, cell/profile integration, consistent projection, and normalization cache handling. Shared material-mixture preparation should have one clear owner rather than being duplicated for the new file.

Selected pass R2b: extract cell-profile geometry preparation into a private
PhaseFieldFault operation implemented in `phase_field_fault/normalization.cc`.
Preserve backend admission/fallback, clipping/shared-face ownership, interval
ordering, cached DoF indices, invalidation, MPI collective ordering and work
counters. Keep current phase sampling, material evaluation and quadrature in
the integration operation. Cache reuse criteria and M4 ownership do not change;
no generic integration framework is selected. Boundary-completion extraction
and value-cache extraction (proposals 2 and 3) were not selected in proposal 1.

Selected proposal 2: first extract the existing completion file loading,
validation, addition and diagnostics privately within M4, preserving BP3
exactly. Report that result before the second step: geometric per-fault/per-
contact detection and automatic paired completion under the selected instruction.
The second step is a behavior extension, not a pure refactor. Support is gated
on a compatible prescribed frozen field, exterior material/profile data and
consistent mechanical coupling; unsupported contacts fail before assembly.
Retain defaults/legacy behavior and separate the two changes in recorded diffs.
Proposal 3 was not part of that behavior extension. Evolution-compatible phase-field boundary data
are a separate follow-up; M1/M2 implementation remains frozen.

Selected proposal 3: one private helper in PhaseFieldFault captures the existing
cache inputs, tests composition independence and executes the identical MPI
reuse decision. Invalidation and final publication remain in the caller.
Automatic-completion qualification must still run before any cache hit; preserve
all equality rules, all-rank agreement and collective ordering. Missing cache
dependencies, if discovered, must be reported as separate correctness work.
Use the completed proposal-2 baseline, existing hit/miss and MPI cache tests,
and matched short legacy/automatic runs. Require unchanged fields, solver
choices and cache/work counters; elapsed time is not a correctness criterion.

Phase-field boundary prerequisite and follow-up: BP3's `prescribe_phase` signal
constrains every unconstrained phase DoF to `get_phase_field_profiles(core_phi)`;
its particle H initializer uses that same profile and the handler's stationary
H relation. This is a fully prescribed frozen Q1 field. Ordinary prescribed-H
initialization instead solves the CPDI weak equation; it has no oblique-profile
boundary flux term. Freezing its result does not prove exterior compatibility.
Automatic completion must verify compatible prescribed data, not infer it from
H > 0 or from the frozen/mature flags alone. Legacy behavior stays available,
with this limitation recorded.

The separate boundary-condition task must choose exterior physics and specify
initialization, evolution, irreversibility, refinement and restart. A permanent
profile Dirichlet trace fixes the selected boundary patch, not the interior,
and must distinguish physical constraints from homogeneous Newton increments.
It must not be imposed automatically on an initially prescribed evolving fault.
For evolving oblique continuation, a candidate is the inhomogeneous natural
flux `q_phi = c Phi_prime(r) (n dot m_out)` using the actual weak-form gradient
coefficient; the exterior/profile and flux must evolve consistently. Initial
flux frozen forever, or switching initial Dirichlet to homogeneous Neumann,
does not establish that consistency. Do not impose both on the same patch.
Reuse the profile definition for H, boundary data and exterior continuation;
there is no universal pointwise phi(H) law. Determine the boundary footprint
from the profile, never from a binary boundary-H test or just the contact point.
Qualify that later change with oblique initialization versus homogeneous natural
data, then changed driving force demonstrating the intended boundary evolution,
including near-boundary phi/I_h checks. No phase boundary law is implemented here.

Preserve distinctions between geometry lookup caches and cached field values. A valid point lookup does not imply valid cached I_h. All ranks must agree on cache reuse as before.

Checks: selected I_h analytic/projection/cache tests, a relevant MPI case, and the short coupled comparison. Compare I_h and cache rebuild/reuse behavior where relevant, not timing as the correctness criterion.

Gate: explain what moved, what dependency became narrower, and what remains coupled. Do not claim that file movement alone reduced the public interface.

### R3 — Organize constitutive mechanics and history publication

Proposed implementation files:

- `phase_field_fault.cc`: material entry points, parameters, registration, accessors.
- `phase_field_fault/constitutive.cc`: Maxwell/cohesive/localization and point responses.
- `phase_field_fault/history.cc`: history initialization, solve preparation, accepted-state publication, and the existing constitutive timestep query.
- `phase_field_fault/normalization.cc`: the R2 implementation.
- `phase_field_fault/boundary_completion.cc`: retain the completed R2 implementation.

This is a proposed organization, not a requirement to fill four files regardless of actual dependencies. Retain material semantic ownership even when implementation is separated.

R3a moves complete definitions without rewriting expressions, changing header
declarations or ownership, or reorganizing either R2 file. Verify independent
translation-unit compilation and the required 2D/3D member instantiations.
Record storage owners separately from material computation, including preparation,
publication, rollback and restart for each relevant history quantity.

For a separately selected R3b, first establish whether the current implementation
already follows these stages before exposing them in
`commit_reconstructed_fault_mechanical_history()`:

1. Sample the accepted bulk state.
2. Compute/project candidate cohesive state and construct state/particle candidates.
3. Complete collective validation.
4. Publish the existing persistent state through terminal writes.

Keep timestep-zero initialization distinct. Avoid introducing a second committed-state owner, a generic transaction framework, or throwing/MPI work in destructors. Preserve existing preparation and publication order; acceptance of a Newton trial is not acceptance of a timestep.

Moving validation ahead of existing writes, reordering collectives or changing
rollback is a semantic change requiring separate selection. Terminal placement
of writes alone does not establish atomic publication. Timestep acceptance stays
with the solver/simulator; history.cc computes/prepares/publishes when called.
Keep timestep-zero, mature/frozen and cohesive paths distinct.

Checks: Maxwell/cohesive tests; short trajectory beyond timestep zero; rollback after a trial/accepted Newton update; baseline checkpoint read and short restart; stateful/stateless or mature/cohesive coverage as needed for moved paths.

Gate: a compact state-ownership and lifecycle table must match the implementation. Any intentional semantic change is a separate task.

### R4 — Dedicated coupled solver: relocation, decomposition, targeted reuse

The revised architecture retains the existing solver scheme/dispatch and the
dedicated `Simulator::solve_reconstructed_fault_stokes()` driver. It supersedes
any proposal to merge fault Newton orchestration with ordinary defect correction.
The three levels are driver, iteration operations, and supplied-operator linear
solve, as detailed in [standing guidance](../refactoring.md#m5-solver-architecture-and-staged-extraction).

**R4a (accepted): move only.** Relocate the complete driver and exclusive helpers
from `source/simulator/solver.cc` into
`source/simulator/solver/reconstructed_fault_stokes.cc`. Keep bodies, expressions,
lambdas, diagnostics, ordering, declarations and ownership unchanged. Inventory
shared definitions first; mechanically expose Stokes operator/Schur definitions
through a narrowly scoped internal header if needed, without duplication or
public API expansion. Preconditioner setup consolidation belongs to R4c.

R4a checks: candidate build/link, independent affected translation-unit compilation
and 2D/3D instantiations; existing short coupled matched comparison, rollback,
small MPI case, and ordinary feature-disabled smoke if shared definitions move.
Use candidate plugins with the candidate executable. Reuse qualified restart
checks unless initialization/restoration semantics change. Compare deterministic
fields/history/solver decisions exactly, excluding time/path metadata. Keep
pre-existing Stage-J failures separate. No long production/performance campaign.
Gate: general solver no longer contains the full fault Newton algorithm and the
independently compiled relocation matches the accepted post-R3 reference.

**R4b (both extractions and helper assessment accepted): focused private operations.** First extract condensed linear
solve/preconditioner setup, then trial residual evaluation, one reviewed subpass
at a time. Before each extraction record inputs, outputs, mutations, collectives
and object lifetimes. A single private iteration helper is optional after these
operations if it fits without changing convergence checks, unused-direction
avoidance or iterate/timestep acceptance. No ordinary Picard/residual types are
required merely for signature similarity. Preserve canonical owners; use only
narrow solve-lifetime scratch where needed.

The residual helper rolls back temporary trial V but retains assembly flags,
linearization-point and RHS side effects. Preserve existing driver/audit cleanup
boundaries and all five evaluation sites. The completed assessment recommends
no whole-step helper: two driver-owned convergence checks divide the iteration,
and the restricted inverse/linearization, active set, directions and scales span
those boundaries. A whole-step call would need broader mutable context or move
driver decisions. The user accepted retaining the focused operations. See the
rolling review for the dependency table. Further extraction needs selection.

**R4c (shared Schur construction complete for review): demonstrated common linear-solver work.** Select the smallest
shared operation, potentially preconditioner construction. Retain the condensed
operator/RHS/recovery and restricted inverse, pressure complement, physical
normalization, total fault budget and true-residual restarts. Preserve ordinary
cheap/expensive budgets, signals and exception behavior. An operator-aware helper
is permitted only when justified; no generic policy framework. GMG remains a
velocity preconditioner for the assembled coupled operator. Verify ordinary,
melt or other common paths only to the extent that shared implementation changes.

The selected inventory identifies the duplicated BFBT/inverse-weighted-mass
construction branch as the smallest useful shared operation. Propose one
source-private function with explicit dependencies, preserving lazy access to
the BFBT-only lumped-mass block. Retain all caller policies and existing
`simulator.h` dependence on the nested condensed Linearization type. No complete
solver unification or header-interface reorganization is proposed. The rolling
review records the bounded checks; no implementation or runtime campaign was
performed during the inventory. The subsequent selected implementation preserves
ordinary melt and the existing combined fault/melt rejection, and runs those
focused checks against accepted R4b without changing numerical policies.

All passes preserve M3 storage/M4 history interfaces, frozen M1/M2, canonical
surface/coupling generation and constraint lifetimes, temporary flags, exact
bound contact, active sets and Armijo acceptance. No diagnostic removal or
assertion cleanup is bundled. Stop and report scientific conflicts separately.

### R5 — Organize manager and surface-system internals

Select one component per task.

Manager candidates: reconstruction lifecycle; property registration; particle/fault projection; current/trial/committed V; serialization and cache maintenance. Preserve generic storage and serialization. Consider a grouped private cache/state struct only where members share a lifetime.

Surface-system candidates: bulk sampling; local quadrature accumulation; MPI reduction; assembly publication; surface factorization/filter state; linearized G action. Do not merge similar loops unless they demonstrably share the same quadrature, measures, coefficients, and numerical meaning.

The two surface assembly paths may represent distinct supported numerical choices. Preserve both until their role is established; apparent duplication is insufficient grounds for deletion.

Checks: the affected geometry/projection or weak-form/Jacobian tests, rank-consistency evidence, and the short coupled fixture. Retain existing normal-filter identities and boundary/endpoint support tests if those paths move.

Gate: the developer can state the major algorithm steps and the lifetime of every cache. No generic manager/material ownership reversal is introduced.

### R6 — Isolate diagnostics and classify configuration

Build on the R0 switch inventory. Separate three categories:

| Category | Example | Initial treatment |
|---|---|---|
| Observational | Timing, stress-cycle CSVs, detailed residual logs | Extract recording/formatting; preserve capture points |
| Numerical implementation choice | Explicit G, interface preconditioner modes, cache disable/reference path | Preserve existing selection and defaults; review separately |
| Benchmark setup | Fixed background traction, completion input, specialized initial/loading state | Isolate setup where existing extension points suffice; retain mechanics hooks actually required |

Use a small named diagnostic configuration/recorder only if it simplifies real repeated code. Retain exact computation inputs in records instead of recomputing them later. Preserve whether a diagnostic failure currently aborts or continues unless a separate behavior change is selected.

If moving setup/output to a plugin, verify callback timing and information availability first. Prefer existing signals when suitable. Do not add a new signal or change callback order purely to reduce a core file's line count.

Supported parameter migration, removal of environment variables, or deletion of experiments is a separate reviewed subtask. Record compatibility decisions. Assertion cleanup and formatting belong in separate commits from algorithm extraction; update log-matching tests only when an intentional message change is the actual task.

Checks: diagnostics off/on leave the selected numerical trajectory unchanged; touched output/error paths behave as specified; retained numerical modes remain selectable. Do not delete useful oscillation diagnostics before their replacements preserve the evidence needed for ongoing research.

Gate: every live switch has an owner and a documented purpose; production calculations are readable without CSV formatting blocks.

### R7 — Review core integration and qualify the result

Revisit the R0 inventory. For each existing core modification, explain why it remains, what behavior it enables, and which test covers it. Review source-level dependencies, public APIs, mutable aliases, call order, and disabled-feature behavior.

Keep essential hooks for creation, dispatch, persistence, and accepted-state events. Identify broad solver/material changes that can become independent PRs. Do not force elimination of every core hook.

Run the selected integrated qualification set once: unit checks, short coupled evolution, relevant 1/2-rank tests, rollback, checkpoint/restart, and ordinary non-fault cases. Use baseline and candidate built under the same stack. Cover changed solver backends on a small fixture if applicable. Run Debug coverage for touched internal invariants where a Debug build is available; explicitly report missing coverage.

Measure memory/timing only enough to detect concrete regressions, such as accidental cache rebuilding or a new global gather. A noisy laptop timing difference alone is not a performance conclusion.

Gate: a reviewed branch with a known source revision, test evidence, remaining limitations, and a clear map from research features to proposed upstream contributions. Long scientific qualification can follow when server resources return.

### R8 — Maintainer discussion and incremental upstream port

Prepare an architecture summary early, after R0, for Yimin to discuss with maintainers. This task does not authorize sending a message or opening an issue automatically.

Tentative contribution groups: prerequisite particle/phase-field support; generic reconstructed-fault infrastructure; material mechanics and coupled solve; a compact benchmark. The actual dependency graph and maintainer feedback determine boundaries. Every PR should build and have a meaningful purpose and tests.

Port one selected contribution at a time onto current upstream main. Preserve attribution. Do not copy the entire research tree over upstream or import mixed commits blindly. Keep compatibility adjustments separate from numerical changes and repeat the tests affected by upstream API adaptation.

Exclude build products, large outputs, checkpoints, local environment paths, and research-only traces from the contribution. Preserve these locally. Retain appropriate scientific documentation, reproducible tests, and required small fixtures.

Gate: Yimin chooses the PR scope and whether to publish it after the concrete diff and evidence are ready.

## 7. Verification matrix and comparison rules

These are existing source/test families observed in the inspected commit. Codex must discover the actual build/CTest/unit-test invocation in the local checkout rather than invent command names or assume every fixture is runnable.

| Concern | Existing starting points | Main stages |
|---|---|---|
| Maxwell/cohesive formulas | `unit_tests/phase_field_fault_maxwell.cc`, `phase_field_fault_cohesive.cc` | R3 |
| I_h integration, cache and projection | `unit_tests/phase_field_fault_ih.cc`; `tests/phase_field_fault_ih*` | R2 |
| Geometry, generic storage, V lifecycle, lower bound, filter | `unit_tests/reconstructed_fault.cc` | R4–R5 |
| Exact condensation | `unit_tests/reconstructed_fault_condensation.cc` | R4 |
| Coupled residual consistency | `tests/phase_field_fault_residual_consistency*` | R4–R5 |
| Failure restoration | `tests/phase_field_fault_stage_i_rollback*`, pressure-gauge rollback family | R3–R4 |
| History and restart | `tests/phase_field_fault_stage_j*`, including restart create/resume | R3, R7 |
| Pressure and boundary changes | `tests/phase_field_fault_pressure_gauge*`, `phase_field_fault_changed_loading*` | R4 |
| Small backend check | Existing reconstructed-fault frozen-GMG/mechanical-mode fixtures | R4/R7 if affected |
| Ordinary ASPECT behavior | Select existing cases matching touched core paths | R1, R7 |

Use stage-relevant subsets. A file split does not justify rerunning an entire benchmark campaign. Some hidden tests require saved local artifacts; list missing artifacts and choose another adequate existing check where possible.

Compare in this order:

1. **Exact structural facts:** property names/order, supported configuration, accepted/rejected decisions, active/prescribed masks, checkpoint readability, absence of unintended history mutation.
2. **Numerical fields:** V, Theta, I_h, cohesive/normal/shear traction, particle Maxwell stress, H, cumulative slip, and frictional work where recorded.
3. **Solver behavior:** nonlinear/Krylov counts, accepted step lengths, fresh residuals, and time-step sequence. A difference is investigated, not automatically accepted or declared incorrect.

First compare baseline and candidate at the same rank count. Then compare each version across ranks using established MPI tolerances. Never require bitwise equality between unrelated MPI decompositions.

For changed source files with identical arithmetic/order, attempt exact numerical comparison on the same stack, excluding nondeterministic log fields such as elapsed time. If results differ, investigate first. Use existing scientifically justified tolerances for field comparisons, or agree variable-specific absolute/relative scales before examining candidate differences. For tiny slip rates, a generic denominator `max(1, |V|)` is inappropriate because it can hide large relative errors.

A useful normalized error is `max_i |new_i-old_i| / (absolute_tolerance + relative_tolerance*|old_i|)`, with tolerances recorded per quantity. Do not loosen tolerances to make the refactor pass.

Use the same saved dt sequence for a diagnostic replay only when needed to isolate a discrepancy; also verify the natural controller path where changed history/control interfaces could affect it. Changing dt or resolution solely to make the candidate agree is not evidence of equivalence.

## 8. Progress tracking and review packet

Keep one reviewed plan in the repository, a compact entry in the existing `CURRENT_STATUS.md`, and one rolling `doc/reconstructed_fault/refactor_review.md` report. Git history can retain prior reports. Avoid a proliferation of inconsistent progress documents.

Suggested status row:

`stage/pass | not started/in progress/ready for review/accepted/blocked | base SHA | candidate SHA or diff | test summary | next action`

Current status: accepted post-R3/Maxwell reference verified; R4a is accepted and
committed as `0c7ed1a0b`. Fresh build, independent 2D/3D compilation, one/two-rank units
and rollback, ordinary AMG smoke and four short BP3 comparisons pass exactly.
R4b's first condensed-solve extraction is accepted: 31 focused comparisons,
372 BP3 field/history groups and 28 counter/decision checks match R4a. The
second, trial-residual extraction also passes 31 focused comparisons, 372 exact
field/history groups and 28 counter/decision checks against the accepted first
subpass; independent 2D/3D compilation and eight source/protection checks pass.
It is accepted. The optional iteration-helper assessment is also accepted:
retain the driver with the two focused operations. R4b is committed as
`983d57e28`. R4c implements the shared Schur-wrapper construction and passes its
build/source checks, 53 focused comparisons and 372 exact BP3 field/history groups.
The frozen probe's GMG half has a confirmed pre-existing hierarchy-setup failure;
frozen AMG and the supported coupled GMG path pass. R4c is ready for review.
Further
R4b operations and later stages require separate selection. Historical R1 fixture failures,
Stage-J pressure incompatibility and cohesive step-two nonconvergence remain
separate from refactoring equivalence. See CURRENT_STATUS.md and the rolling
review for detailed earlier outcomes.

Every completed task should report:

1. **Outcome:** the selected task and what changed.
2. **Revisions:** base/candidate SHA, remaining local changes, and exact scope.
3. **Responsibility changes:** old/new owner and the core dependencies removed or retained; say explicitly if only files moved.
4. **Evidence:** commands, versions, rank counts, actual pass/fail/skip results, baseline failures, and measured field differences.
5. **Contract check:** algorithms/defaults/serialization unchanged, or each intentional exception listed.
6. **Review material:** diff or accessible commit, changed headers and top-level routines, small comparison tables; logs only where needed.
7. **Next decision:** recommend one bounded next task; identify any blocker requiring a decision.

Yimin's five review questions:

- Can I follow the top-level algorithm and identify the state owners?
- Is the core-facing interface simpler, or has complexity merely moved?
- Does the evidence support equivalence for the paths actually changed?
- Were new parameters, abstractions, or scientific choices introduced unexpectedly?
- Is this small enough to revert or adjust independently?

## 9. Ready-to-copy first instruction to coding Codex

```text
We are preparing the phase-field/reconstructed-fault work for eventual upstream
ASPECT contribution. Perform R0 ONLY: inspect the current repository and produce
a dependency/core-modification audit. Do not refactor source or change numerical
parameters, tests, checkpoint formats, or scientific documentation contracts.

Read AGENTS.md, doc/reconstructed_fault/current_design.md, the relevant sections
of specification.tex, refactoring.md, and CURRENT_STATUS.md. Treat the current
source as authoritative for actual APIs. Identify obsolete instructions rather
than repeating already-completed work.

Start by recording branch, SHA, git status, remotes, and shallow-history status.
Preserve all local source changes, outputs and checkpoints. Inspect the actual
local checkout; do not assume the remote snapshot contains the newest work.
Do not reset, clean, stash, switch branches, rebase, or merge to conduct the audit.

Find the real common ancestor with geodynamics/aspect main, fetching only the
history needed if appropriate. If ancestry cannot be established, label the
inventory provisional and explain the limitation. Do not attribute every
tip-to-tip difference to this project.

Inventory prerequisite particle-domain/phase-field work, reconstructed-fault
infrastructure, constitutive mechanics, coupled solvers, diagnostics and tests.
For each modified existing core file/symbol, record:
  responsibility; reason; callers/dependencies; keep/move/independent-PR/exclude;
  proposed destination or essential hook; relevant verification.
Inventory added public accessors/signals, shared mutable state, parameters and
environment switches. Distinguish observation from numerical implementation
selection and benchmark setup. Do not delete either category.

Inspect available test runners and fixture dependencies. Propose a small local
R1 baseline suite using 1–2 MPI ranks, including a short history-evolving coupled
run, rollback, baseline checkpoint/restart, and an ordinary non-fault case.
Report existing failures described in CURRENT_STATUS as historical/unverified
until actually rerun. Do not launch production simulations or builds in R0.

Write one audit report, preferably doc/reconstructed_fault/refactor_review.md,
and provide the dependency table, core-change inventory, unresolved specification
conflicts, proposed baseline, and precise first normalization-extraction scope.
Writing this report is the only planned working-tree change. Stop after R0 so
Yimin and the reviewing assistant can select the next bounded task.
```

## 10. Reusable implementation-task envelope

```text
Execute only stage/pass [ID and name] from the reviewed refactoring plan.

Selected baseline: [SHA plus any explicitly included local patch].
Goal: [one concrete responsibility or readability improvement].
Allowed files/components: [list].
Preserved contracts: [relevant numerical, MPI, lifecycle and restart invariants].
Excluded changes: [physics, parameter/default changes, unrelated cleanup].
Required checks: [named existing tests and short comparison, exact commands if known].
Local resource budget: [ranks/threads/runtime guidance established in R1].
Commit policy: [whether local commits are authorized; no remote publication].

Inspect relevant callers and tests before editing. Complete the selected work
and its verification without repeated confirmation for routine implementation
choices. Raise a scope question only for a material interface/ownership change,
a specification conflict, a numerical behavior change, or a necessary task
beyond the agreed boundary.

Preserve unrelated and uncommitted work. Keep pure moves, structural edits,
formatting/assertion cleanup, and numerical fixes in distinguishable commits
or diffs. Do not silently repair a baseline failure or update expected numerical
outputs to accommodate changed behavior.

Update the compact status and rolling review report. Return the base/candidate
revisions, diff, measured verification results, remaining limitations, and one
recommended next task. Stop at the end of this selected pass for review.
```

## 11. Source references

- [Reviewed source revision](https://github.com/YiminJin/aspect/tree/0fc1ce782c48b78f79eff674724686b8277c8618)
- [Repository instructions](https://github.com/YiminJin/aspect/blob/0fc1ce782c48b78f79eff674724686b8277c8618/AGENTS.md)
- [Current design](https://github.com/YiminJin/aspect/blob/0fc1ce782c48b78f79eff674724686b8277c8618/doc/reconstructed_fault/current_design.md)
- [Specification](https://github.com/YiminJin/aspect/blob/0fc1ce782c48b78f79eff674724686b8277c8618/doc/reconstructed_fault/specification.tex)
- [Existing refactoring guidance](https://github.com/YiminJin/aspect/blob/0fc1ce782c48b78f79eff674724686b8277c8618/doc/reconstructed_fault/refactoring.md)
- [Status and evidence qualifications](https://github.com/YiminJin/aspect/blob/0fc1ce782c48b78f79eff674724686b8277c8618/doc/reconstructed_fault/CURRENT_STATUS.md)
- [ASPECT contribution guidance](https://github.com/geodynamics/aspect/blob/db01934c865eb60b26090c254084b497f4ee2dd1/CONTRIBUTING.md)

This plan is based on source inspection. No C++ compilation, numerical simulation,
or regression test was executed while preparing it.
