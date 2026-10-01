# Reconstructed-fault refactoring review

## Maxwell coefficient naming follow-up (complete; ready for review)

The user selected the `kappa` → `eta_ve` rename after R3b and then requested
retaining the original efficient frozen-stress implementation. The rename covers
Maxwell coefficients, material responses, coupling data and C++ callers/tests.
The stable `-eta*expm1(-dt*G/eta)` expression is unchanged. Existing diagnostic CSV
column names remain compatible with archived output; the design/specification
identify `eta_ve` with the existing mathematical kappa notation.

`evaluate_frozen_maxwell_stress` is byte-identical to its pre-task implementation.
It prepares the material coefficients and returns only beta times retained stress.
The full `compute_maxwell_stress` operation includes the current-strain term;
ordinary Stokes assembly supplies that contribution separately. No delegation
through a zero-strain tensor remains. No ownership, checkpoint, algorithm or
parameter change is included, and prior R3b edits are preserved.

Core and selected plugin builds pass. Existing tests pass 20,816 assertions in
26 cases per rank on one/two ranks. Frozen-stress/temperature fixtures yield 12
exact comparisons. Four short legacy/automatic BP3 runs and a cross-rank old-
checkpoint continuation yield 405 exact field groups and 30 matching cache/work
checks. No numerical differences were observed. No 3D or long/full BP3 run was
performed; updated historical/BP5 diagnostic callers were not run, and earlier
cohesive limitations remain separate. Details and qualified executable hashes:
[cleanup evidence](../../benchmarks/reconstructed_fault/maxwell_cleanup/README.md).

This cleanup is saved separately from R3b commit `4d2f14f53` for future R4 work. Proposed next task remains separately
selected R4a solver-method relocation; no R4 work has begun.

## R3b — expose the existing history lifecycle (complete; ready for review)

Reference is corrected R3a `bef79b31a`, executable
`build-restart-fix/aspect-r3a-corrected-qualified`. Before editing, the suspected
R3a deletion was checked: the entire R3a section in this document, including its
ownership/lifecycle table and equivalence report, matched the committed section
exactly. Nothing needed restoration. It remains intact below; local temporary
review/plan files and the R2b review move are preserved.

The pre-edit ordering check confirmed the proposed four broad later-time stages.
`commit_reconstructed_fault_mechanical_history()` now keeps its entry validation,
timestep-zero return, timestep/property/mapping checks and calls four private
operations in `history.cc`:

| Operation | Responsibility and data boundary |
|---|---|
| `sample_accepted_history` | Accepted bulk state and existing association/FE data → velocity gradients, temperature and current/previous phase samples in unchanged association order |
| `compute_history_candidates` | Samples, timestep and particle property/composition offsets → cohesive Q1 projection, Theta and particle stress/H candidates; frozen surface state and retained histories are read through the existing material/manager interfaces |
| `validate_history_candidates` | Candidate completeness MPI minimum, cohesive validation, local particle-ID checks and profile diagnostics, in their original order |
| `publish_history_candidates` | Existing cohesive/previous-I_h writes, followed by Theta and particle stress/H writes; no timestep acceptance decision |

Two private forward-declared scratch types are defined only in `history.cc` and
instantiated locally in the driver. They introduce no persistent member,
serialized field, public API or second committed-state owner. The sampling cache
and diagnostic streams retain their lifetime through publication. Buffer members
replace local variables; manager/particle aliases are obtained by their consuming
operation. Exact expression checks permit these declaration/reference adaptations
and verify the numerical bodies unchanged.

The first collective error propagation remains before cohesive projection; the
second remains after Theta/particle candidate construction. They have not been
moved to the third stage. Projection still precedes the H update. Local validation
and diagnostics retain their positions before writes; the Theta property lookup
remains after cohesive publication, as before. The extraction adds no new atomicity
or noexcept guarantee. Solver acceptance, V publication, bulk swaps and rollback
are untouched. Timestep zero, mature/frozen and cohesive/evolving paths stay
distinct, and both R2 files and all other history/constitutive methods are intact.

The ten-row R3a ownership/lifecycle table was verified before editing and again
after extraction. It still applies: material computes, manager/particles store
and serialize, simulator/solver accepts. Evidence includes `lifecycle-before.md`,
`lifecycle-after.md`, and an exact copy of the preserved R3a section. Source checks
confirm only `history.cc` and private header declarations changed, with 5,822 other
entry source/header/test files and the corrected reference executable unchanged.

Verification against the corrected R3a reference:

| Check | Result |
|---|---|
| Full Release build; independent compilation | PASS; all four helpers and the driver instantiated for 2D/3D |
| Selected units on reference/candidate, one/two ranks | PASS: 20,828 assertions in 28 cases per rank, including restart/prescribed-rate regression, Maxwell/cohesive, limiter and Stage-I tests |
| Temperature, frozen stress, original/open-top accepted-Newton rollback | PASS on reference/candidate, one/two ranks; actual restoration markers retained |
| Lifecycle comparisons | PASS: all 40 outcomes, decisions, markers and statistics checks match |
| Cohesive accepted checkpoint and same-rank restart | PASS: 15 exact comparisons; original restored-state assertions pass; step-one mesh/particle payloads and physical fingerprint match |
| Short legacy/automatic BP3 and old-checkpoint cross-rank continuation | PASS: 405 exact field groups and 30 matching cache/work checks, including solver decisions |

The original Stage-J pressure-compatibility failure and supplemental cohesive
step-two nonconvergence remain separate reference limitations. Fresh and restarted
step-two traces match exactly; no accepted step-two state exists for comparison.
No physical parameter, solver tolerance or history assertion was changed. The
prior cross-rank cohesive observer limitation was not repaired. No full 3D,
200-km BP3 or long production run was performed. Elapsed times are excluded.

Commands, source proof, lifecycle tables and individual results are in the
[R3b harness](../../benchmarks/reconstructed_fault/refactoring_r3b/README.md).
Qualified candidate: `build-refactor-r3b/aspect-r3b-qualified` (SHA256
`fbd3a74e0238c0fcf5026c9ec8b554f5296302fe14f3d0de4d986096d2c109db`).
R3b is complete and saved in its own local commit. Proposed next task: separately
select R4a, moving the existing coupled solver method without restructuring it.
No R4 work has begun. The sections below preserve earlier pass records.

## Minimal restart correction (complete; corrected R3a reference)

The user separately authorized the correctness fix after the diagnosis below.
`ReconstructedFaultManager::rebuild_after_deserialization()` now assigns one
empty prescribed-rate map per reconstructed fault beside its other transient
resets. A short comment explains the layout and caller reapplication contract.
This is the only production change: no serialized member, header/API, ownership,
material history operation, MPI ordering, tolerance or physical parameter changed.

The new `[fault_slip_restart]` unit regression round-trips both an unprescribed
manager and one with runtime prescriptions, verifies that prescriptions are not
restored, opens an absolute-value trial without calling the setter, then checks
accepted-Newton rollback. It reapplies a prescribed row as BP3 does, checks its
mask/exact rate, rejects a violating candidate without changing the trial, and
commits the valid values. The new test crashes with the original manager object
(exit 139) and passes with the fix. Existing prescribed-rate/checkpoint/Stage-I
coverage plus this regression passes 20,173 assertions/17 cases per rank on
one and two ranks.

The original one-rank cohesive checkpoint and its observer are retained.
Restart now passes the original restored history/V/geometry/bulk assertions and
reaches the same step-two nonconvergence as the uninterrupted run. Both accept
two Newton updates then exhaust line-search candidates at relative bulk residual
`4.105631e-06`. Phase-field target doubles, iterations, linear residuals,
nonlinear residuals and decisions match exactly, also against the pre-fix fresh
run. Comparison normalizes stream whitespace/scientific notation and separately
checks nondeterministically interleaved stdout/stderr error messages; it applies
no numerical tolerance. Accepted step-one checkpoint meshes/particle payloads,
physical fingerprint and fresh-run statistics match. All 15 cohesive checks
pass. Step-two nonconvergence remains a separate limitation; no accepted step-two
physical state exists for comparison.

Four short BP3 runs (legacy/automatic, one/two ranks) plus the automatic old-
checkpoint one-to-two-rank continuation pass. All 405 field groups and 30 cache/
work checks match the qualified R3a reference exactly. The short maintained BP3
callback reapplies empty maps; nonempty prescribed-row enforcement after caller
reapplication is tested through the same manager API in the new regression.
The full 200-km BP3 prescribed-row run and 3D simulations were not run. The prior
cross-rank cohesive observer failure remains unmodified.

The move-only R3a and investigation record are committed as `8490431b5`, with
this correctness correction in a separate commit. Candidate compilation/link
reuses preserved R3a objects except the rebuilt manager and unit-test groups.
All 5,822 other entry source/header/test files remain unchanged. The corrected
R3a reference for R3b is `build-restart-fix/aspect-r3a-corrected-qualified`;
its SHA256, plugins, source hashes, commands and outcomes are recorded in the
[correction harness](../../benchmarks/reconstructed_fault/restart_fix/README.md).
R3b is not implemented. Proposed next task: expose the existing history lifecycle
against this corrected baseline, preserving validation/write/MPI ordering and
solver-owned timestep acceptance. The investigation below is a historical record.

## Separate cohesive restart investigation (historical pre-fix record)

The user selected diagnosis after R3a, not a source correction or R3b. The
preserved R3a executable reproduces the crash on **one rank**, from a checkpoint
created on one rank with the same existing cohesive/open-top fixture settings.
The original restored-history/V/geometry/bulk assertions pass before SIGSEGV.
The original two-rank checkpoint and all R3a evidence remain intact.

The source-located backtrace is:

```text
ReconstructedFaultManager<2>::set_slip_rate_trial_values   manager.cc:1077
  for (const auto &entry : prescribed_slip_rates[f])
Simulator<2>::solve_reconstructed_fault_stokes::<residual lambda> solver.cc:1126
  fault_manager.set_slip_rate_trial_values(slip_rate);
Simulator<2>::solve_reconstructed_fault_stokes             solver.cc:1161
  initial_residual = evaluate_coupled_residual(working_x, initial_slip_rate);
```

GDB inspected the manager after deserialization and immediately before the first
absolute trial assignment. There is one fault with eight vertices:

| State | After restart rebuild | Immediately before failing call |
|---|---|---|
| Committed V | outer size 1, inner size 8; all 1e-12 | unchanged |
| Current V | outer size 1, inner size 8; equals committed | unchanged |
| Trial V | outer size 1, empty inner vector | outer size 1, inner size 8; equals current |
| Incoming absolute candidate | not yet constructed | outer size 1, inner size 8; all 1e-12 |
| Solve/trial active | false / false | true / true |
| `prescribed_slip_rates` | **outer size 0** | **outer size 0** |

The slip-rate histories and trial lifecycle are valid. The crash is an
out-of-bounds access to `prescribed_slip_rates[0]`, not an invalid candidate or
premature trial call. Fresh `add_reconstructed_fault()` creates one empty
prescribed-rate map per fault (`manager.cc:769`). This boundary-configuration
container is intentionally not serialized, but `rebuild_after_deserialization()`
reconstructs current/trial vectors without recreating its per-fault layout.
`set_slip_rate_trial_values()` nevertheless indexes it unconditionally. Thus the
normal restart lifecycle leaves a required transient layout absent.

For the diagnostic comparison only, a copied observer plugin disconnects
`verify_restored_state` from `start_timestep`. All original history assertion
code and the normal Stage-I/Stage-J callbacks remain unchanged; the original
plugin/case is retained. Both variants exit 139 and have exactly the same
pre-call vectors/flags/empty prescribed container. The observer-disabled GDB
run also captures the empty container on return from deserialization, before any
observer callback. This excludes the optional observer as the cause of this crash.
BP3 avoids the defect because its normal prepare callback explicitly calls
`set_prescribed_slip_rates(std::vector<std::map<unsigned int,double>>(1))` on
every entry, including restart (`bp3/plugin/bp3.cc:211`).

**Smallest proposed correction:** alongside the other transient-vector resets
in `ReconstructedFaultManager::rebuild_after_deserialization()`, add:

```cpp
prescribed_slip_rates.assign(reconstructed_faults.size(), {});
```

This restores the fresh-manager invariant of one empty map per fault. It does
not serialize boundary configuration or invent prescribed rates; callers still
reapply actual prescribed rows after restart, as specified. Do not add a BP3-like
setup call to the cohesive fixture to hide the missing manager initialization.
A focused regression should extend the existing manager checkpoint round-trip
test to begin a solve/trial and call `set_slip_rate_trial_values()` on the newly
loaded manager without any prescribed-rate setter. That path is currently absent
from the archive test, which checks restored values but does not open a new
absolute-value trial.

The correction/regression are **proposed only**, not applied or qualified.
No physical parameters, solver tolerances or history assertions were adjusted.
The initial two-to-one-rank checkpoint attempt stops earlier at the observer's
entry-2 fingerprint assertion; this additional cross-rank observer limitation
was preserved, not loosened. The same-rank checkpoint creation retains the known
second-step convergence failure after successfully saving the first real update.
Neither issue was repaired in this investigation.

Evidence: [restart investigation](../../benchmarks/reconstructed_fault/restart_investigation/README.md),
especially `evidence/manager-states.json`, `investigation-verification.json` and
the two `*-gdb.log` files. The diagnostic executable recompiles only the existing
manager/solver unity groups with debug information and unchanged Release
optimization, then links untouched R3a objects. All 5,824 entry source/header/test
files and the original checkpoint hashes are unchanged. Stop for review; proposed
next task is the separately scoped one-line restart fix and targeted regression.

## R3a — constitutive/history relocation (complete; ready for review)

Selected scope: complete method relocation only. Reference is committed R2b
`06b70740f`, executable `build-refactor-r2b-cache/aspect-cache-qualified`.
The user's deletion/move of `refactoring/R2b_review.md` into the local `tmp/`
directory is preserved. R3b restructuring is not part of this pass.

### Ownership/lifecycle table recorded before source edits

The table records the existing implementation, including initialization writes;
it is not a proposed transaction model. Material computation does not imply
ownership of the storage receiving the computed value.

| Quantity | Storage owner | Preparation/update computation | Publication point | Rollback behavior | Restart handling |
|---|---|---|---|---|---|
| Nodal Theta | Manager's generic fault properties | Material preparation projects mapped particle initial Theta only on a fresh timestep-zero model; friction law computes later candidates from current V and committed Theta | Fresh projection; later terminal loop in `commit_reconstructed_fault_mechanical_history` after cohesive publication | Newton/trials do not write it; candidates are local. No undo of published history is supplied by this method | Manager archives property schema/values; later/restarted preparation requires complete positive state, never initializes missing Theta |
| Cohesive traction T_coh and previous I_h | Manager's generic fault properties | Material computes initial q from particle H and FE phi with surface mixture (zero q in mature mode), then projects; later projects accepted cohesive samples before H construction | `commit_cohesive_state`: paired scalar writes after validation, in fresh initialization and at the end of later accepted-history update | No trial mutation; local candidates discarded on pre-publication failure. Fresh initialization has its own publication sequence | Manager archive; current I_h rebuilt, frozen mature restart checks then restores exact saved previous-I_h values as the current snapshot |
| Maxwell stress tau | Associated particle manager's `maxwell stress` property | Particle plugin initializes mapped stress; material computes accepted slip-corrected Maxwell candidates using bulk mixture/temperature and retained particle stress | Final particle loop of accepted-history update; timestep zero returns without evolving stress | Frozen through trials, so no stress restoration is needed on failed solve; no second stress transaction | Existing particle archive/migration; constitutive method reads restored particle history |
| Irreversible H | Associated particle manager's `crack_driving_force` property | Initial particle field; material interpolates projected accepted traction before exact cohesive-work candidate and max with old H; `Evolve phase field` gates H update | Same final particle loop; timestep zero retains initial H; frozen mode retains old H | No trial writes; candidate map is local. No general undo after terminal publication | Existing particle checkpoint; restored H drives the next phase solve |
| Committed/current/trial V | ReconstructedFaultManager's distinct vectors | Fresh material preparation initializes V_min; solver controls manager begin/trial/accept operations | Solver calls material history publication, then manager `commit_slip_rate_nonlinear_solve`, then swaps accepted bulk vectors | Rejected trial discards trial V; failed solve restores current V from committed V in solver catch | Manager archives committed V; load reconstructs current V and clears trial/solve activity |
| Current I_h, reuse/lookup caches | PhaseFieldFault transient members | Existing normalization operations in unchanged R2 file | Existing successful integration/projection and cache-publication points | Existing invalidation/reuse behavior; not committed constitutive history | Resume callback invalidates caches; qualified frozen mature snapshot handling remains in normalization.cc |
| Surface temperature and projected chemical composition | Temperature: material transient vectors; composition: manager generic Q1 properties | Material preparation samples FE temperature; normalization preparation projects particle compositions | Existing preparation writes, before constitutive-state validation | Preparation is not an all-or-nothing history transaction; these are frozen solve inputs, not trial candidates | Temperature resampled; composition properties archived and projected by existing preparation |
| Mature reference geometry; fixed background traction selectors/data | Reference geometry and background data: manager properties; selectors: material members | Fresh preparation fills missing reference geometry and checks exact geometry; caller/benchmark selects initialized background properties | Existing initialization/setup sites; no mechanical-history update of fixed background data | No trial mutation; fresh initialization writes precede later preparation checks | Geometry/property archive; resume checks mode; benchmark reattaches selectors without recalibration |
| Current/previous FE phi; accepted bulk solution | Simulator FE vectors (phase algorithm remains M2) | Phase solve uses retained H; localization derives h_old from old FE phi except timestep zero uses current phi; solver constructs accepted bulk state | Existing simulator/phase lifecycle; solver bulk-vector swaps follow history and V publication | Solver restores bulk/current-linearization state on pre-terminal failure; material never decides timestep acceptance | Existing simulator vectors restored; no additional stored h_old |
| Cohesive/state/particle candidates | Local material-method containers | Accepted-state sampling, projection and constitutive evaluation | Only the publication points above; no persistent candidate owner | Local temporaries expire on failure; no destructor-driven MPI or publication | Not serialized |

The later-time method already has the proposed broad four-stage order: accepted
FE sampling; cohesive sample projection followed by Theta/particle candidates;
collective error propagation and completeness reduction followed by the existing
cohesive and local particle-ID validation/diagnostic output; then cohesive/I_h,
Theta and particle writes. R3a preserves the actual interleaving, including all
checks and collectives. The table does not infer atomicity from the terminal
location of writes. Fresh initialization and the timestep-zero early return are
separate paths. Solver convergence/acceptance and its pre-terminal rollback
remain in `Simulator::solve_reconstructed_fault_stokes`.

Implementation placement: `initialize()` moves intact with history setup so its
file-local initial-state mapping validator keeps one definition shared with
solve preparation. Registration, parameter parsing, ordinary material evaluation
and accessors remain in the entry file. Existing history error propagation moves
with history while retaining its `aspect::internal` symbol used by normalization.
Both R2 files and all header declarations remain unchanged.

### R3a post-move verification and disposition

The table above was rechecked against the moved bodies and unchanged callers,
manager/particle storage, solver acceptance/rollback and serialization. All ten
rows still describe the implementation. Its pre-edit text and post-move check
are retained in the [R3a evidence](../../benchmarks/reconstructed_fault/refactoring_r3a/README.md).
The existing broad four-stage later-time sequence is compatible with a future
readability pass, but this pass neither extracts those stages nor establishes a
new atomicity guarantee. Initial preparation writes and the timestep-zero return
remain distinct; mature/frozen and cohesive branches are unchanged.

Nine constitutive methods moved into `phase_field_fault/constitutive.cc`; ten
initialization/history/preparation/timestep-query methods moved into `history.cc`.
Six existing implementation helpers moved with their callers, with no duplicate
definitions or changed linkage. The material entry file retains ordinary
evaluation, parameters, registration and accessors. No header declaration,
ownership, expression, validation, collective, call order or publication point
changed. Neither R2 file was reorganized. The entry file is now 580 lines;
constitutive/history files contain 508/1,222 lines, including copied include
scaffolding and explicit instantiations. Include cleanup is not part of R3a.

| Check | Result |
|---|---|
| Exact source movement | PASS: 19 complete methods and six helpers byte-identical; retained entry code unchanged apart from vacated scaffolding/spacing; all other 5,835 existing source/header/test/plugin files unchanged |
| Build and instantiation | PASS: fresh Release build; all three affected files separately compile without PCH/unity; all 38 moved-member definitions present for dimensions 2 and 3 |
| Maxwell/cohesive/lifecycle units | PASS on reference/candidate, one/two ranks: 20,791 assertions in 25 cases per rank |
| Surface temperature and frozen Maxwell assembly | PASS on reference/candidate, one/two ranks; original assertions retained |
| Trial/accepted-Newton rollback | PASS: original and traction-free-top cases on reference/candidate, one/two ranks; actual accepted-update and surface/particle/bulk/V restoration markers required |
| Short legacy/automatic BP3 and automatic cross-rank restart | PASS: four steps-0–6 trajectories plus identical reference-checkpoint continuation through steps 5–6; all 405 field groups exact and 30 cache/work checks equal |
| Focused history lifecycle equivalence | PASS: 48 outcome, solver-decision, marker and statistics comparisons, including the separately recorded failing fixtures |
| Evolving cohesive accepted checkpoint | PASS: after the supplemental case's first real update, mesh, mesh metadata, fixed/variable particle/FE data and serialized history/V/geometry/bulk fingerprint all byte-identical (five checks) |
| Reference protection | PASS: all 31 captured qualified reference artifacts unchanged; user's local documentation move and temporary plan preserved |

The initial default-unity build exposed an existing missing declaration of
`ExcNonlinearSolverNoConvergence` in untouched M2 `source/simulator/phase_field.cc`
when adding files shifted unity groups. CMake now explicitly compiles only the
two new files without unity/PCH, as required for independent translation units.
All 55 previous unity groups are exactly unchanged, and the build passes. No M2
include, expression or behavior was modified. The failed build is retained.
An initial standalone-command lookup used the wrong target spelling and was
corrected before compiling; all three actual standalone compilations passed.

Pre-existing runtime limitations remain separate from refactoring equivalence:

- Original Stage-J one/two-rank trajectories fail on both executables at the
  same significant-pressure-incompatibility check. They are not passing tests.
- The separate traction-free-top Stage-J fixture retains every history assertion
  and verifies initialization plus one evolving Theta/H update. Both executables
  then fail to converge at physical step two, with identical solver decisions.
- Reading that same cohesive checkpoint verifies restored histories, V, geometry
  and bulk on both executables, then both segfault (exit 139) in
  `ReconstructedFaultManager<2>::set_slip_rate_trial_values`. This is an unresolved
  reference/fixture failure, not successful cohesive resumed evolution and not
  repaired by this pass. The earlier wrapper attempt failed because its checkpoint
  copy hard-codes output paths; loading the existing shared observer directly
  avoids only that harness side effect. Both attempt logs are preserved.

No assertion or tolerance was relaxed. Mature/frozen BP3 restart succeeds, but
general cohesive resumed evolution remains unqualified. No full 3D simulation,
long production trajectory or unrelated test campaign was run. Elapsed times
are intentionally excluded from equivalence. Candidate executable is preserved
as `build-refactor-r3a/aspect-r3a-qualified`; commands, source proofs, snapshots,
hashes and individual outcomes are in `refactoring_r3a/evidence/`.

R3a is complete and stops for review; no commit was made in this pass. Proposed
next task: review/select R3b's lifecycle-only extraction using this ownership
table and actual ordering, with correctness/fixture fixes kept separately scoped.

Local checkpoint (September 30, 2026): the completed limiter, normalization
refactoring, prescribed boundary completion, test harnesses and guidance are
saved in separate local commits at the user's request. Generated evidence and
binaries remain local. Statements below that no commit was made describe the
earlier verification sessions. Review remains pending; no next pass is selected.


## R2b proposal 3 — normalization reuse decision (complete; ready for review)

Reference is the completed proposal-2 implementation, not the earlier geometry-
only executable: `build-refactor-boundary/aspect-boundary-qualified`. All 25
recorded artifact hashes matched before editing. The existing local edits are
preserved; source changes are confined to `normalization.cc` and the private
portion of `phase_field_fault.h`. No commit or scientific-worktree change was made.

`PhaseFieldFault::prepare_normalization_reuse(previous_cache_valid) const`
privately captures owned phase entries, fault versions/coordinates and required
surface compositions; tests the existing identical-G/cohesion/Gc rule; and runs
the unchanged all-rank MPI minimum of the local reuse predicate. The 48-line
body is byte-identical. Its private return record is defined in `normalization.cc`:
the phase-block index, original borrowed owned-index reference, moved input
vectors, composition-independence flag, global reuse result and timing markers. It adds no persistent cache or public API.

Dependencies remain explicit through the owning material object: introspection
and current solution, reconstructed fault geometry/properties, material coefficients,
degradation revision, mesh-deformation flag, lookup/cell-cache readiness, previous
cache data, the disable-cache environment switch and communicator. Ownership is
unchanged. No missing cache dependency was identified during this bounded
extraction; no correctness fix is included.

The caller retains initial invalidation, automatic-completion qualification,
lookup resets/mapping invalidation, empty-fault handling and composition projection
before invoking the helper. Hit/miss counters and final publication remain in
the caller. The extracted timing markers preserve the diagnostic phase boundaries.
A source verifier substitutes the original body back into the call site, removes
only the new definition/instantiation, and requires byte equality with proposal 2.
This protects compatibility checks before hits, all collective ordering, invalidation
and publication, integration algorithms and work counters. All other 5,834 entry
source/header/test/plugin files remain unchanged.

Qualification uses the existing cache unit and lifecycle cases plus matched short
legacy and automatic BP3 runs on one/two ranks. Both executables load the same
observer plugin, which includes the unchanged lifecycle assertions and records
per-rank counters without invoking normalization. The existing restart-invalidator/failure-recovery checks are included, along
with a short matched one-to-two-rank automatic checkpoint restart. This directly
exercises the phase-block index and composition-independence flag returned for
the unchanged frozen-normalization restoration checks.
See [harness and evidence](../../benchmarks/reconstructed_fault/refactoring_r2b_cache/README.md).


### Proposal-3 verification outcome

The fresh Release build passed after correcting the return record to include
the phase-block index and composition-independence flag used by the unchanged
restart-restoration code. The original compile failure is retained in the log;
it was an extraction omission, not a missing cache dependency or numerical fix.

All 30 runtime invocations passed: the existing cache unit cases on one/two ranks
(126 assertions per rank), six value-cache lifecycle configurations and two warm
cell-traversal configurations per executable, four short legacy/automatic BP3
configurations per executable, and a matched automatic restart pair from one
to two ranks. The comparison reports **405 exact field groups and 96 passing
lifecycle/work checks**. Normalization, phase/history/bulk fields, solver decisions,
completion and mechanical work CSVs, per-rank hit/integration/request counters,
and cell-work counters are unchanged. Only elapsed time is excluded. The value-
cache lifecycle cases retain one hit/eight integration attempts, including the
expected failure/recovery. Warm traversal reuse is exercised with only value
reuse disabled. The restart's cold step has zero hits/one integration and its
next step has one hit/one integration, identically in both executables.

The first comparison-script coverage assertion incorrectly required a prior hit
at the cold restart observation. It was corrected to require both paths over
the whole case; no equality or numerical tolerance was relaxed. Both script
attempts remain in the evidence. No baseline runtime failure was found.
Full production cycles, unrelated solvers and adaptive-mesh qualification were
not rerun. Proposal 2's unsupported-case limitations remain unchanged.

The final source check proves the moved block and restored caller exact, with
all other 5,834 entry source/header/test/plugin files preserved. Candidate hashes,
source diffs and the preserved `aspect-cache-qualified` executable are recorded
in the harness. No commit was made. Stop for review. Proposed next task: a bounded
R3 manager-responsibility assessment, only if separately selected; the phase-field
boundary design remains independent correctness/feature work.

## R2b proposal 2 — boundary completion (complete; ready for review)

Selected instruction: [prescribed boundary completion](refactoring/codex_boundary_completion_instructions.md).
Start state is the completed geometry extraction, including its local changes;
the recorded R2b artifact hashes matched before editing. The first step is
complete: `apply_boundary_normalization_completion()` privately owns the legacy
file loading, validation, addition and diagnostic output in `normalization.cc`.
Its body is byte-identical and runs at the same preprojection point. All 5,810
other source/header/test files were unchanged at this checkpoint. A fresh
Release build passed; short BP3 on one/two ranks matches the geometry-pass
reference exactly in all 186 field groups, including I_h, mechanical fields,
history and solver records. No tolerances or expected outputs changed.

The preserved move-only executable is `build-refactor-boundary/aspect-completion-move`.
Snapshots, separate diffs, hashes and comparison records are in
`benchmarks/reconstructed_fault/refactoring_boundary/evidence/`. Subsequent
automatic treatment is a behavior extension, recorded separately from this move.

The paired interior path remains `ReconstructedFaultManager::project_to_bulk_source`:
legacy bottom/top wedges use constant endpoint surface coordinates, retaining
physical phase, pressure, stress and history at each physical point. The Stokes
QP association cache supplies the common basis to bulk/coupling and the bulk-
work surface rule; particle Maxwell history uses that same source association.
No mechanical assembly is relocated into the material model. BP3 qualifies its
frozen field through full phase-DoF constraints to the handler's stationary
profile, with H initialized from that same profile. H-driven initialization
without compatible oblique boundary data remains unqualified for automatic
completion; the separate phase-boundary follow-up is recorded in the plan.

### Automatic extension and ownership

The second step adds the explicit selector `Fault reconstruction / Boundary
completion = automatic prescribed`; existing inputs default to `legacy`.
It classifies every prescribed fault without fault-zero/bottom assumptions.
`boundary_contact.h`, `boundary_contact.cc` and `boundary_contact_manager.cc`
(M3) own boundary-facet gathering, contact identity, support admission and
lifecycle. `PhaseFieldFault::prepare_automatic_boundary_completion()` and
`apply_automatic_boundary_completion()` in `boundary_completion.cc` (M4) own
profile compatibility, ghost-Q1 constitutive evaluation and integral addition.
`manager.cc` supplies endpoint associations; `surface_system.cc` admits the
existing per-fault bulk-work assembly after automatic qualification (M5).
`bp3/plugin/bp3.cc` selects the new path without attaching a legacy CSV.
The material header declares private operations/data; the manager header exposes
geometry records and geometric extents. No ownership or checkpoint schema changed.

Supported: 2-D fixed prescribed geometry on an axis-aligned affine Box, uniform
aligned boundary ghost lattice, mature/frozen fully constrained compatible Q1
field, constant prescribed core per fault, identical profile/degradation material
coefficients. Either/both transverse endpoints, different faces, independent
faults, and curves outside the straight terminal support are covered. The
selected automatic mode uses the existing bulk-work mechanical measure.

Unsupported cases fail explicitly: corner/tangential/unrepresented crossings,
short curved terminals, intersecting completion envelopes or nearby diffuse
branches, overlapping prescribed profiles, undefined exterior material/lattice
data, mesh deformation, 3D, and unqualified H-driven/evolving fields. Conservative
support rectangles can reject close geometry even when a more elaborate
analysis might establish disjoint physical wedges. That generalization requires
a separate design; no nearest-continuation rule was invented.

Geometric extents include the full discrete Q1 support. They are not an activation
threshold or a new truncation of nonzero mechanical phase support. The material
verifies the entire physical nodal profile and full constraints before use.
Legacy and automatic completion are mutually exclusive. Exterior integration
retains the BP3 ghost-grid/profile convention and ordinary localization law,
adds once before unchanged projection, and keeps all physical mechanics in the
existing assembly/history operations. Free endpoint shape functions and K/B/G
couplings remain active. A tolerance-limited endpoint guard retains an existing
association when raw/resampled frames disagree only at roundoff in s=0.

M1/CPDI, M2/evolution, the limiter edits, proposal 1's cell traversal, completed-
value reuse criteria and proposal 3 remain unchanged. Contact/continuation caches
are transient and follow geometry, mesh and restart invalidation. Runtime adaptive
refinement has not been qualified; uniform boundary spacing is checked after
rebuilding. MPI decisions are collective and corrections retain profile ownership.

### Focused verification

The final Release build, registered CTest, two contact unit cases (56 assertions each rank), ten small fixture
runs, both seven-observation BP3 runs and a one-to-two-rank restart are recorded
in the [isolated harness](../../benchmarks/reconstructed_fault/refactoring_boundary/README.md).
The focused plugin checks actual interior-wedge association and centered
free-endpoint K, pressure G and bulk B derivatives; it uses nonzero free first/last
endpoint directions. Geometry units additionally cover shared-facet deduplication,
periodic-face exclusion, 3D rejection and diffuse branch envelopes.

| Fixture | Contact identity `(fault, endpoint, boundary)` | Treatment |
|---|---|---|
| interior / diffuse support touching boundary | none | no continuation |
| perpendicular | (0,0,bottom), (0,1,top) | zero correction |
| oblique | (0,0,bottom), (0,1,top) | paired, both ends |
| reversed | (0,0,top), (0,1,bottom) | same physical corrections |
| left/top | (0,0,left), (0,1,top) | paired, different faces |
| multiple | (0,0,bottom), (1,0,right) | independent paired contacts |
| curved interior | (0,0,bottom) | paired straight terminal; interior curve retained |
| BP3 | (0,0,bottom), (0,1,top) | paired; 77 affected profiles per end |

Oblique and multiple-fault nodal I_h are exact across one/two ranks; reversal
differs by 8.58e-16 relative. Perpendicular/interior correction files contain no
rows. Free-endpoint derivative relative errors are at most 5.51e-13 (K),
7.78e-14 (G), and 6.01e-15 (B), against the existing 1e-8 criterion.
Corner, tangent, between-vertex crossing, H-driven/unconstrained, and undefined
exterior-material cases reject on two ranks with the intended messages.

The automatic BP3 correction covers 154 of 744 profiles without duplicate IDs.
In-domain integrals are exact versus legacy. Maximum outside-integral difference
is 3.75e-7, or 3.23e-13 of the largest correction; nodal I_h differs by at most
3.35e-7 absolute / 1.44e-13 relative. This is consistent with independently
computed C++ versus Python ghost-Q1 panel integration at 1e-11 panel accuracy.
No legacy tolerance was changed. Final V differs by at most 4.86e-23 m/s,
accumulated slip by 1.67e-20 m, and Theta/phase/H/geometry are unchanged.
Mechanical weak-table differences are below 2.01e-12 of their field scale.
The final executable also reproduces the default legacy one-rank trajectory
exactly. All discrete solver decisions match. Near-zero residual differences are retained
in absolute units in the detailed comparison; their large relative percentages
are not interpreted as trajectory changes. MPI/restart field comparisons pass
1e-10 scaled/absolute checks; cold restarted normalization differs by 4.45e-16
before restoring the validated frozen snapshot. No long cycle was run.

Exploratory fixture failures are preserved, not hidden: inherited unavailable
postprocessor names; an under-resolved new profile giving a negative consistent
projection; an existing stationary-H roundoff assertion for an exact 45-degree
line; and the existing normalization overlap rejection for insufficiently
separated faults. The fixture was resolved and made independent. Production
projection, scientific algorithms, M2 assertions and solver tolerances were not
relaxed. The phase-boundary prerequisite is established by full prescribed DoF
constraints and a nodal profile check, not by the initial H flag.

Stop for review after this task. The remaining separate decision is whether to
select the documented evolution-compatible phase-boundary design; no new
refactoring extraction has begun. Detailed commands, hashes, stage-separated
diffs, contact records and comparison metrics remain in the harness evidence.

## R2b — private cell-profile geometry preparation (complete; ready for review)

Historical proposal-1 review boundary: only geometry preparation had been
selected. Proposals 2 and 3 were unchanged, and boundary generalization was
deferred. The later proposal-2 selection is recorded above; proposal 3 remains
unselected and unchanged.

### Revisions and responsibility boundary

Worktree `/home/ein/repository/aspect-pf-rsf-refactor`, branch `pf-rsf-refactor`,
HEAD `3af7c2aa2590aa8e39eaf2a1e46df0bc6c9d9700`, with the existing uncommitted
R1/limiter/R2a changes preserved. No commit was made. The immediate reference
is the actual R2a entry source and preserved R2a executable/plugins/results,
whose 25 recorded hashes matched before editing. Matched results also compare
against qualified R1. Candidate artifacts and evidence are isolated in
`build-refactor-r2b/` and `benchmarks/reconstructed_fault/refactoring_r2b/`.

`PhaseFieldFault::prepare_cell_normalization_geometry()` now performs backend
admission, physical-box reductions, profile clipping, half-open face ownership,
cell searches, cached DoF-index construction and interval sorting. It remains
private, implemented in `source/material_model/phase_field_fault/normalization.cc`.
Its input is the existing profile vector; mesh/mapping, geometry model,
introspection/FE/DoFHandler, communicator and traversal cache remain dependencies
of the owning material object. It returns a private value record containing the
two endpoint distances per profile and four local geometry work counters.
Cached intervals remain in the existing material-owned cache.

The integration definition shrinks from 366 to 252 lines. It still samples
current phase values, invokes the existing material integrand, performs all
quadrature/tail/coverage operations and emits the existing work diagnostic.
Geometry preparation does not receive the material callback or sample the
solution. The header adds only the private result record and method declaration;
no public interface, persistent member, cache layout, checkpoint field or
ownership changes. Explicit member instantiations retain 2D/3D linking.

Movement verification confirms that geometry statements are exact apart from
qualifying the returned fields and sizing their vector. All earlier methods
(including boundary completion, value reuse and invalidation) are byte-identical.
The integration tail and later methods are identical except for diagnostic
field qualification and the new helper instantiation. MPI reduction ordering,
backend fallback, clipping, interval order and origin/normal reuse criteria
are unchanged. All 5,810 other source/header/test files match entry hashes,
including the user's limiter changes. No numerical fixes or formatting sweep
are included.

### Verification and differences

- Fresh full Release/unity/PCH build and matching lifecycle/BP3 plugins passed
  with the qualified GCC 12.4.0/OpenMPI 5.0.6/deal.II 9.6.2 stack and unchanged
  floating-point flags. The executable contains both 2D and 3D helper symbols.
- All 15 selected runtime invocations passed: accuracy/cache cases on 1/2
  ranks; both backends' lifecycle cases on 1/2 ranks; background-only material
  on one rank; both six-step BP3 trajectories; and paired R2a/R2b warm traversal
  runs on 1/2 ranks. Unit coverage includes four accuracy and two lookup-cache
  cases per rank (59,260 assertions on the one-rank run).
- Each reference comparison passed **186 field groups with zero differences**:
  all saved bulk components, particle properties, fault profiles, final native
  I_h/history and solver records at matching rank counts. This holds against
  both R2a and qualified R1; no cross-rank equality is asserted.
- All **22 lifecycle/work checks** passed in each comparison. Cold construction
  and warm traversal work counters match exactly. With completed-value caching
  disabled, each warm case changes from 21 rebuilt/0 reused profiles to
  0 rebuilt/21 reused and zero cell candidates. The one-rank case retains
  1,345 intervals and 23,688 FE samples; rank-zero diagnostics in the two-rank
  case retain 673 intervals and 12,096 samples. Existing coverage assertions
  run on every rank. Only elapsed geometry time is excluded from work-record
  equality; no performance improvement is claimed.
- The first sandbox MPI probe failed before tests because local sockets were
  denied. The permitted rerun passed. This infrastructure failure is retained
  separately from the passing suite; no test expectations/tolerances changed.
- All 23 untouched R2a artifact hashes and 45 qualified R1 artifact/input/
  checkpoint hashes still match. Source verification and `git diff --check`
  passed. The scientific-test worktree and reference outputs were not modified.

Reproduction and evidence: [R2b harness](../../benchmarks/reconstructed_fault/refactoring_r2b/README.md).
Ignored local evidence includes entry snapshots, the two source diffs, build/run
commands and logs, comparison JSON, source-preservation checks and artifact hashes.
No full test campaign, production BP3, restart/rollback rerun, dedicated cell-cache
AMR/moved-profile/fallback run or 3D simulation was performed. Existing R1 fixture
failures retain their prior dispositions; this extraction makes no additional
scientific qualification claim for the opt-in backend.

Stop for review. Proposed next task: a short design-only review of boundary-
completion generalization, if selected; do not implement it or proposals 2/3
without the user's next selection.

## R2a — normalization translation-unit separation (complete; ready for review)

Selected instruction: [codex_R2a_instructions.md](refactoring/codex_R2a_instructions.md).
R2a is complete. At that pass's review boundary, R2b/helper extraction/generic
numerical services and R3 had not started. This is separation into translation units; the material still
owns normalization data, policy and interface. Proposed next task: a bounded
R2b assessment of model-independent integration/sampling/projection boundaries,
only if selected after review. The R0/R1 material below is historical evidence.

### Revisions and preserved responsibilities

The destination is `source/material_model/phase_field_fault/normalization.cc`.
The earlier flat filename proposal is superseded in guidance and the selected
roadmap; proposed future M4 paths use the same directory convention, without
creating future-stage files. M1/M2 remain frozen. M4 retains the integrand,
mixture/admissibility/singularity policy, cache validity and current/previous-I_h
semantics. No constitutive, solver, parameter or checkpoint algorithm changed.

| File | Before | After | Change |
|---|---:|---:|---|
| `source/material_model/phase_field_fault.cc` | 3,798 lines | 2,199 lines | Complete normalization block and exclusive helpers removed; sole material registration retained |
| `source/material_model/phase_field_fault/normalization.cc` | absent | 1,698 lines | Moved definitions plus includes, namespaces and explicit member instantiations |
| `include/aspect/material_model/phase_field_fault.h` | existing | byte-identical | Class, declarations, visibility, data, caches and test access unchanged |

The 12 moved definitions are `compute_normalization_integrals`,
`invalidate_normalization_cache`, `normalization_effective_phase_field`,
`validate_normalization_phase_field_minimum`, `normalization_integrand`,
`integrate_cell_normalization_profiles`, `integrate_normalization_profiles`,
`project_surface_chemical_compositions`, `NormalizationPointLookupCache::get`,
`evaluate_normalization_points`, `build_owned_normalization_profiles` and
`project_normalization_integrals_to_fault`. The four exclusive helpers are
`normalization_search_enclosure`, `NormalizationSideState`,
`NormalizationEvaluationRequest` and `normalization_profile_point`.
The 1,503-line method block and 103 helper lines match their recorded entry
text byte-for-byte, including diagnostics/comments. No expression, evaluation
order, MPI collective, fallback, projection, cache key or restart path changed.

One necessary change beyond movement: `throw_if_history_error` is shared by
mechanical history and cell normalization. Its body remains unchanged with
one definition in the original file, but gains linkage in `aspect::internal`;
the new file has a private forward declaration and both import the helper name
with a using-declaration. This permits separate compilation without duplicating
the helper or adding a public helper header/class/API. The completion-file
setter and `interpolate_surface_chemical_compositions` stay with their existing
callers. Includes and explicit member instantiations for 2D/3D are the other
translation-unit scaffolding. Registration remains exactly once in the entry
file; no second whole-class instantiation or build target was added. Existing
recursive source discovery includes the new file without a CMake edit.

Guideline edits are in `refactoring.md` and the selected nested roadmap;
status/evidence are in this report and `CURRENT_STATUS.md`. AGENTS.md and
scientific specifications were not changed by R2a. The test-only
[harness](../../benchmarks/reconstructed_fault/refactoring_r2a/README.md) rebinds
R1 paths, records bounded runs, checks exact relocation and compares fields.
No existing numerical test, tolerance or expected output was changed in R2a.

### Exact reference and candidate

Worktree `/home/ein/repository/aspect-pf-rsf-refactor`, branch `pf-rsf-refactor`,
HEAD `3af7c2aa2590aa8e39eaf2a1e46df0bc6c9d9700`; no commit was made.
Qualified reference: the original R1 Release executable in
`build-refactor-baseline/`, matching plugins and immutable convex fixture in
`benchmarks/reconstructed_fault/refactoring_r1/`, and that run's checkpoint
`output-bp3-one/restart/02/` (accepted step 4, 400 s). Reference binary, plugin,
input and checkpoint hashes matched before and after R2a. No reference build
or test target was run against edited source. `../aspect/` was not modified.

Since R1, the only source/test delta at entry was the separately selected
limiter zero-rejection/help/test update, validated in `build-limiter-validation/`.
R2a preserves all four affected files exactly. The selected qualification cases
use positive/default limiter settings, so the added zero assertion is inactive.
The candidate is HEAD plus that recorded limiter delta and this normalization
relocation/linkage scaffolding; precise source patches/hashes are retained below.
User guidance edits, the removed old roadmap and selected instruction files
are preserved. No unrelated source/header/test/build file changed.

Candidate `build-refactor-r2a/` uses the actual R1 toolchain/configuration:
GCC 12.4.0 through OpenMPI 5.0.6, deal.II 9.6.2, Trilinos 14.2.0, Voro++ 0.4.6,
Release with unity/PCH, NetCDF off. Effective C++ flags are exactly equal to R1,
including `-fno-finite-math-only -ffp-contract=off`. Build parallelism is two
jobs (focused separate compilation used one additional compiler). Checks use
1/2 MPI ranks and one thread per rank; numerical environment switches match
R1's respective cases, including explicit B/G for the supplemental/BP3 runs.
Required MPI socket access used sandbox escalation. Both the maintained BP3
plugin and existing lifecycle/rollback plugins were rebuilt against the
candidate configuration. An initial build-target invocation used the root
build instead of its separate `tests/` project; the corrected build passed.
That setup error and log are retained, not classified as a numerical failure.

### Verification and comparisons

| Check | Result | Evidence / scope |
|---|---|---|
| Normal Release unity/PCH build and matching plugins | PASS | Fresh candidate build; unchanged R1 stack/flags |
| Separate compilation and linking | PASS | Both changed files compiled without PCH/unity; focused executable linked with separate objects; all 12 members defined for dimensions 2 and 3 |
| Separately linked normalization unit checks | PASS | 6 cases, 59,260 assertions on one rank |
| `[phase_field_fault_ih_accuracy]`, 1/2 ranks | PASS | Same assertions and printed analytic integral as R1 |
| `[phase_field_fault_ih_cache]`, 1/2 ranks | PASS | Same lookup-hit/rebuild and collective-invalidation assertions as R1 |
| Qualified I_h lifecycle, both backends | PASS | Remote/cell on 1/2 ranks; remote no-composition on 1 rank; unchanged qualified 50-iteration wrappers and original tolerances/assertions |
| Forced rollback | PASS | Original and supplemental open-top cases, 1/2 ranks; two accepted Newton updates and exact restoration assertions; raw markers equal R1 |
| Short evolving BP3, 1/2 ranks | PASS | Six physical 100 s steps; matching-rank fields/history/solver decisions exactly equal |
| Load preserved R1 checkpoint | PASS | Candidate resumes step 4/400 s checkpoint through steps 5/6, exactly matching uninterrupted R1 |
| Ordinary disabled-feature particle case | PASS | `convection_box_particles` CTest and exact statistics comparison |
| Exact movement/local-edit/reference preservation | PASS | Byte-span and entry-hash checks, reference artifact hashes, `git diff --check` |
| Debug/full non-unity build, 3D runs, full suite/production BP3 | NOT RUN | Focused non-unity compile/link was used as requested; no compiler matrix or new physical qualification |

All 19 bounded invocations succeeded. The comparator checked **219 field
groups with zero differences**: every owned bulk component and all particle
properties at steps 0–6 on matching rank counts, surface profiles, final native
V/Theta/C/previous-I_h/composition/background/slip fields, and all recorded
solver decisions. The preserved-baseline continuation matches over steps 5/6.
Maximum absolute errors, pointwise relative errors for nonzero reference values,
and zero-reference errors are all zero. This is same-rank/stack comparison;
it does not conflate R1's already recorded cross-rank differences with a refactor
change. BP3 one/two-rank runs cost about 17.5/17.0 s; restart about 6.4 s.

All 19 separate lifecycle comparisons passed: statistics, verification markers,
cold cell-cache work counts (only geometry time excluded), accepted-update and
rollback markers, and ordinary statistics. Lookup-cache hit/rebuild counts and
mesh/coordinate invalidation are additionally asserted by the unchanged cache
unit tests. The original R1 initialization-budget and stale output-filter/
checkpoint-reference failures retain their recorded dispositions: R2a used the
passing qualified wrappers and raw rollback evidence, without repairs or
rebaselining. No new discrepancy or missing required R2a check remains.

Limits remain those of the selected reference: 2D small fixtures, a mature
fixed-profile BP3 trajectory with evolving stress/Theta/slip, fixed H and zero
C; this is not production BP3 physical validation or a propagation/mesh-motion
campaign. Normalization state, cache types and interface remain in the material
header. Shared-helper linkage and dependencies remain for later review; this
pass does not claim completed architectural decoupling.

Detailed reproduction/evidence:

- [Harness commands](../../benchmarks/reconstructed_fault/refactoring_r2a/README.md),
  [entry/source delta](../../benchmarks/reconstructed_fault/refactoring_r2a/evidence/baseline-check.json),
  [candidate patch](../../benchmarks/reconstructed_fault/refactoring_r2a/evidence/candidate-source.diff),
  [candidate artifact manifest](../../benchmarks/reconstructed_fault/refactoring_r2a/evidence/candidate-artifact-sha256.txt).
- [Configure](../../benchmarks/reconstructed_fault/refactoring_r2a/evidence/configure.log),
  [build](../../benchmarks/reconstructed_fault/refactoring_r2a/evidence/build.log),
  [separate compile commands](../../benchmarks/reconstructed_fault/refactoring_r2a/evidence/separate-commands.json),
  [separate link commands](../../benchmarks/reconstructed_fault/refactoring_r2a/evidence/separate-link-commands.json).
- [Run outcomes/commands/environment](../../benchmarks/reconstructed_fault/refactoring_r2a/evidence/run-summary.json),
  [field errors](../../benchmarks/reconstructed_fault/refactoring_r2a/evidence/state-comparison.json),
  [lifecycle comparisons](../../benchmarks/reconstructed_fault/refactoring_r2a/evidence/lifecycle-comparison.json),
  [preservation checks](../../benchmarks/reconstructed_fault/refactoring_r2a/evidence/final-verification.txt).

---

R0 — dependency and core-modification audit, 28 September 2026.

**R0 accepted as an audit on 29 September 2026. The following R0 evidence is historical; R1 progress is recorded below.**
Only this report was written. No source, parameter file, test, checkpoint,
scientific contract, or existing local edit was changed. No build, simulation,
test execution, commit, branch switch, fetch, merge, or rebase was performed.
The specific R0 instruction requests one report as the only working-tree change;
the status row is therefore here rather than in `CURRENT_STATUS.md`.

| Stage/pass | Status | Source baseline | Candidate/diff | Evidence | Next decision |
|---|---|---|---|---|---|
| R0 | Accepted audit | `3af7c2aa2590aa8e39eaf2a1e46df0bc6c9d9700` | This new report only; source unchanged | Git ancestry, diff, source/header/caller/test-runner inspection | Review classifications and limiter discrepancy; select bounded R1 |

## Scope decision and R1 qualification — 29 September 2026

The user accepted R0 and selected the nested decoupling/R1 instruction. M1
particle domains/CPDI and M2 phase-field implementation are frozen. M3 generic
fault infrastructure, M4 constitutive mechanics/normalization/history and M5
coupled assembly/solve are the active scope for later selected passes. These
labels do not introduce new libraries, move code, or change scientific contracts.
R2a remains private normalization relocation within M4; R4 concerns M5.

Current reverse dependencies (deferred): M2 `evolve_phase_field()` checks
concrete PhaseFieldFault mature mode; phase setup and crack-driving-force
initialization connect to M3, and generic domain unit tests also include M3
quadrature utilities. M3 reconstruction/projection adapters consume M1/M2.
Removing these dependencies or preparing prerequisite PRs is not an R1 task.
The historical marker is `76a1e68e062be11bf007e541712d0c5251bd410e`, parent
`e3ef3ae89ade2a16db04fcabecccc5db3a7f0ced`; chronology does not define a clean
module boundary. No exhaustive upstream audit is repeated.

HEAD remains `3af7c2aa2590aa8e39eaf2a1e46df0bc6c9d9700`; production source,
headers, tests and build definitions match HEAD at R1 entry. User guidance edits,
the deleted old roadmap and untracked selected instructions/report are preserved.
Only documentation and necessary test-only harness additions are authorized.
The known limiter conflict is investigated without changing production code;
unaffected baseline checks continue as explicitly requested.

**R1 execution complete; ready for review with pre-existing failures. R2a is
not authorized or started.** This is a usable conditional reference, not an
all-green regression suite. After R1, the user confirmed the numeric-parser
design, requested comments/documentation correction, then explicitly required
an assertion rejecting zero. `Patterns::Double(0.)` is retained, with an
`AssertThrow` requiring a strictly positive value. The maximum finite double
still disables the stateful limiter; `infinity` is unsupported. The earlier
recommendation to restore that alias is withdrawn. Parameter help, both
specifications and limiter tests now match the selected contract. The other
original regression failures still need review before selecting R2a.

| Check | Result | Evidence / disposition |
|---|---|---|
| Fresh Release core and matching plugins | PASS | Separate baseline, BP3 and probe builds; source unchanged |
| Timestep parameter contract at R1 | FAIL against then-written contract | Historical observations: exact default/bypass, positive bound, `infinity` rejected, zero accepted; subsequent user-selected update below |
| I_h accuracy/cache, 1/2 ranks | PASS | Accuracy: 4 cases; cache: 2 cases on each rank count |
| Original I_h lifecycle, 1/2 ranks and no composition | FAIL | Ten initialization iterations exhaust the budget; no geometry exists at the intended assertion |
| Supplemental I_h lifecycle | PASS | Original tolerance/assertions, initialization budget 50; remote points and cell intervals on 1/2 ranks; remote no-composition on 1 rank |
| Focused mechanics | PASS | 28 unit cases/21,334 assertions; frozen Maxwell load integration on 1/2 ranks |
| Original forced rollback behavior | PASS | Raw logs: two accepted Newton updates, exact bulk/surface/particle/current-and-committed-V restoration on 1/2 ranks |
| Original rollback CTest comparison | FAIL | Shell grep expects old acceptance line without `; alpha=...`; no production behavior failure |
| Six-step BP3, 1/2 ranks | PASS | 0–600 s, six physical 100 s steps, all existing runtime checks pass |
| One-rank split restart from step 4 | PASS | Steps 5/6 physical fields, histories and solver records exactly equal |
| Cross-rank comparison | PASS for execution; measured differences | Fields below; step-1 Krylov count 45 versus 44; not bitwise equivalent |
| Ordinary particle initialization | PASS | `convection_box_particles` |
| Ordinary particle checkpoint runtime/data | PASS | Both nested runs finish; 1,000 final particle rows × 10 columns exactly equal |
| Ordinary checkpoint CTest comparison | FAIL | `checkpoint_03_particles`: existing log/statistics reference mismatches |
| Debug, full suite, production BP3, cross-rank bulk FE remapping | NOT RUN | Outside the bounded qualification; no claim of physical production validation |

### R1 reference and reproducibility

Worktree `/home/ein/repository/aspect-pf-rsf-refactor`, branch `pf-rsf-refactor`,
HEAD `3af7c2aa2590aa8e39eaf2a1e46df0bc6c9d9700`. All 5,820 hashed source,
header, existing test and core build files match their entry hashes. Tracked
BP3 source/inputs also have no diff. `../aspect/` remains unchanged and was only
read for selected immutable fixture inputs. No commit, extraction, production
code edit, scientific-contract edit or numerical fix was made.

Revisions are limited to the scope/reference paragraphs in `AGENTS.md`, the
module table/rules in `refactoring.md`, status/links in the selected nested
roadmap, this rolling report and `CURRENT_STATUS.md`, plus the test-only
[reference harness](../../benchmarks/reconstructed_fault/refactoring_r1/README.md).
The user's deletion of the old root roadmap is retained and explicitly treated
as superseded. Detailed R0 inventories below remain historical reference.
Production responsibilities/interfaces are unchanged; only their agreed module
labels are documented. The harness adds a postprocessor for parser/bypass
observation, input staging, bounded execution and numerical field comparisons.

Core build: `build-refactor-baseline/`; maintained six-source BP3 plugin and
limiter observation plugin: `benchmarks/reconstructed_fault/refactoring_r1/`
`plugin-build/` and `probe-build/`. GCC 12.4.0 through OpenMPI 5.0.6 wrappers,
deal.II 9.6.2 (`6890ca740b`), Trilinos 14.2.0, p4est 2.8.7, Voro++ 0.4.6 enabled;
NetCDF disabled. Release, unity/PCH enabled. Effective flags include `-O3
-DNDEBUG`, subsequently `-O2`, `-march=native`, loop unrolling/strict aliasing,
and explicit `-fno-finite-math-only -ffp-contract=off`. Exact flags, linked
libraries and commands are retained in the build tree and evidence files.
Build parallelism started at two jobs; the small probe used one job. Runs use
one or two MPI ranks with `OMP_NUM_THREADS=DEAL_II_NUM_THREADS=
OPENBLAS_NUM_THREADS=1`, inherited `OMP_PROC_BIND=spread`. BP3 and supplemental
fixtures source the existing BP3 environment (`ASPECT_FAULT_EXPLICIT_B=1`,
`ASPECT_FAULT_EXPLICIT_G=1`); the final limiter-only retry uses ordinary defaults.
MPI required sandbox escalation because local sockets were blocked inside it.

The [harness README](../../benchmarks/reconstructed_fault/refactoring_r1/README.md)
contains build/run commands. Retained artifacts include:

- [Source hashes](../../benchmarks/reconstructed_fault/refactoring_r1/evidence/source-sha256.txt),
  [unchanged-source verification](../../benchmarks/reconstructed_fault/refactoring_r1/evidence/source-verification.txt),
  [binary/plugin/input hashes](../../benchmarks/reconstructed_fault/refactoring_r1/evidence/reference-artifact-sha256.txt),
  [build provenance](../../benchmarks/reconstructed_fault/refactoring_r1/evidence/build-provenance.txt).
- [Fixture origins](../../benchmarks/reconstructed_fault/refactoring_r1/evidence/fixture-provenance.json),
  exact PRMs in `inputs/` and expanded `parameters.prm/json` in each output.
- [Per-command outcomes/timing/environment](../../benchmarks/reconstructed_fault/refactoring_r1/evidence/run-summary.json)
  and adjacent complete logs; original CTest raw and comparison files remain
  under `build-refactor-baseline/tests/output-*/`.
- [Field comparisons](../../benchmarks/reconstructed_fault/refactoring_r1/evidence/state-comparison.json),
  the reference executable/matching plugins, full-state CSVs/native output,
  and the newly generated [step-4 checkpoint manifest](../../benchmarks/reconstructed_fault/refactoring_r1/evidence/checkpoint-step4-sha256.txt)
  for `output-bp3-one/restart/02/`.

These generated artifacts are ignored but retained locally; later candidates
must use separate build/output paths and verify reference hashes.

### Parameter findings and pre-existing test failures

The installed parser's default string is `1.7976931348623157e+308`, exactly
`DBL_MAX` after parsing. Both default and explicit sentinel return the old
law proposal (`2e10 s` in the probe), even with temporarily invalid synthetic
Theta: the optional predictor is genuinely bypassed. `0.1` parses and limits
the probe to `2313.7664336192356 s`; it does not bypass state validation.
`infinity` is rejected by the installed `Patterns::Double(0.)`. Zero passes
both parameter setting and plugin parsing. R1 initially classified these as
contract failures; the user subsequently retained the numeric parser and
rejected the infinity alias, then explicitly requested an assertion rejecting
zero. The R1 observation of accepted zero therefore predates that assertion.
Zero execution was deliberately not attempted. All synthetic Theta
changes are restored. See
[probe log](../../benchmarks/reconstructed_fault/refactoring_r1/evidence/limiter-probe-registered.log)
and [unchanged unit failure](../../benchmarks/reconstructed_fault/refactoring_r1/evidence/unit-limiter.log).
The first observation-plugin invocation lacked an inherited postprocessor
registration and failed before execution; the test-only include was corrected,
rebuilt, and rerun. Both logs are retained. This was a harness issue.

The selected BP3 trajectory has the exact disabled default, recorded in its
expanded PRM, and a global 100 s cap. Its disabled-path reference remains
applicable. The subsequent user-selected update changes parameter help, parsing
descriptions, zero validation and the tests' disabled spelling. The parser
pattern, default, timestep algorithm and history semantics are unchanged. The
retained R1 executable/hashes describe the pre-update source; no reference build,
log or checkpoint was overwritten. Updated validation is recorded below.

The original I_h cases exhaust their ten-iteration phase initialization budget
at relative residual about `1.293e-3` versus tolerance `1e-5`, continue according
to the configured failure strategy, and then fail the uninitialized-property
assertion because geometry was never reconstructed. Supplemental PRMs change
only the initialization iteration budget to 50, not the tolerance, physical
parameters or assertions; all five runs pass in 4–6 s. Accuracy runs cost
2.6–5.6 s; cache runs about 1.7 s. No original test was repaired or rebaselined.

For original rollback cases, inspect `screen-output.tmp`, not only the shell's
filtered `screen-output`: both ranks' runs have two accepted Newton updates
and the plugin's exact restoration assertions pass after forced failure.
The old grep pattern omits today's `alpha` suffix. An early reading of the
filtered output incorrectly suggested Newton was unreachable; raw logs resolve
that uncertainty. Supplemental open-top runs also pass, but are unnecessary
to establish the original rollback path. No pressure-compatibility failure is
claimed for these runs.

`checkpoint_03_particles` completes both ordinary nested simulations. Final
particle data agrees exactly after removing only timestamp comments. Its CTest
still fails saved log/statistics expectations (resume messages/restored rows
and duplicated console lines); neither reference outputs nor tolerances were
edited. This is recorded separately from functional checkpoint evidence.

### Evolving fixture and restart comparison

The convex fixture is copied from the sibling worktree's
`bp3/output-cleanup-evidence/fixture-convex/`; its saved-mesh, profile and bottom
completion inputs are rebound locally. The existing tracked fault geometry is
retained. Physical settings remain 150 × 50 km, 1,875 cells/16,875 particles,
32 fault vertices, length scale 4,000 m, Gc `2e7`, uniform direct effect `0.025`,
normal filter 20 and **full** bottom constraint. The old `map_measurement`
plugin only observed heap/map size and is omitted; maintained full-state
auditing is enabled. No production-resolution first event is attempted.

One- and two-rank runs each take about 18.3 s. Maxwell history evolves from
zero to maximum stress about `0.402 Pa`; Theta and accumulated slip also evolve.
Mature C stays zero; fixed H/geometry/completed I_h are checked at every accepted
step by the maintained plugin. Newton update counts are `2,2,1,1,1,1,1`, all
minimum accepted alphas are one, all 32 vertices remain free, and every fresh
linear check passes. This is history evolution in a mature fixed-profile case,
not coverage of growing phase damage/nonzero evolving cohesive history.

Actual checkpoint metadata after completion: slot `01` is step 6/600 s,
`02` is step 4/400 s, `03` is step 5/500 s. The existing branch tool copied
**slot 02** into a new output directory; the one-rank restart then recomputed
steps 5/6 in about 7.0 s. At both steps every owned bulk component and all
particle properties agree exactly, as do surface profile values and all
recorded solver statistics. Final native fault properties (V, Theta, C,
previous I_h, fractions, background tractions, accumulated slip) also agree
exactly. This checks physical state, not byte equality of checkpoint archives.

Cross-rank measurements are separate. Particles are aligned by their exact
initial physical coordinates, then tracked by stable IDs within each run;
fault rows by fault/node identity. Bulk DoF numbering is not compared across
partitionings. Maximum errors over saved steps 0–6 (final-only for native I_h):

| Field | Maximum absolute difference | Maximum pointwise relative difference, nonzero reference |
|---|---|---|
| V | `3.826e-23 m/s` | `3.826e-14` |
| Theta | `9.313e-10 s` | `1.164e-16` |
| accumulated slip | `1.016e-20 m` | `3.825e-14` |
| weak shear traction | `4.843e-8 Pa` | `1.824e-15` |
| weak normal traction | `2.012e-7 Pa` | `4.023e-15` |
| previous I_h (final) | `9.313e-10` | `3.986e-16` |
| particle tau_xx | `2.438e-11 Pa` | `8.172e-4` |
| particle tau_yy | `2.255e-11 Pa` | `9.196e-4` |
| particle tau_xy | `1.672e-11 Pa` | `1.395e-2` |

Particle H/locations, final C and background tractions are exact. Large relative
stress differences occur near zero; absolute errors and reference magnitudes
are retained rather than masking them by scaling with one. Zero-reference
errors are recorded separately. Time sequence, free/active counts, Newton
counts, alphas and fresh-check decisions agree; the step-1 Krylov count differs
(45 on one rank, 44 on two), with small residual differences. These measured
cross-rank differences are not assigned a newly invented acceptance tolerance
and are not confused with exact same-rank restart agreement.


### User-selected limiter correction and validation — September 29

This follow-up is separate from R1's unchanged-source qualification. The user
retained the numeric parser and explicitly requested rejection of zero.
`source/time_stepping/reconstructed_fault.cc` keeps `Patterns::Double(0.)` and
now uses `AssertThrow(maximum_logarithmic_state_change > 0.0, ...)` during
parameter parsing. The error names the parameter and requires a strictly
positive value. No `Patterns::Anything()` or infinity alias was introduced.
The default, enabled predictor, MPI/state ownership and checkpoint format are
unchanged. Parameter help, both scientific specifications, status and roadmap
are consistent with this selected behavior.

The existing parameter unit test now checks numeric-pattern rejections at
`ParameterHandler::set()`, separately checks zero at plugin parsing, and keeps
exact default round-trip/positive-value coverage. The integration test uses the
maximum finite double for disabled behavior; only its corresponding output
label changed. Analytic-bound tolerances and all remaining assertions are
unchanged. No R1 failures or saved artifacts were rewritten.

Validation passed: `[fault_state_limiter]` (15 assertions, Release), plus the
existing limiter integration fixture on one and two MPI ranks. The latter
checks disabled/default behavior, equilibrium, increasing/decreasing state,
tiny timesteps, overflow, all vertices, committed rather than trial velocity,
nonmutation and MPI agreement. `git diff --check` passed. The full regression
suite and BP3 trajectory were not rerun for this parsing-only change.

To preserve the baseline, `build-limiter-validation/` holds a separate Release
candidate. It recompiles the two affected unity translation units with the
recorded baseline commands and links them with the unchanged baseline objects;
the test plugin is separately rebuilt. Commands and evidence:
[build commands](../../build-limiter-validation/commands.json),
[validation runner](../../build-limiter-validation/run_checks.sh),
[unit results](../../build-limiter-validation/unit-test.log),
[one-rank integration](../../build-limiter-validation/integration-1.log),
[two-rank integration](../../build-limiter-validation/integration-2.log).
Reference binary/plugin/input and step-4 checkpoint hashes were rechecked and
are unchanged. R2a remains unstarted. Proposed next task: review the remaining
R1 fixture/output-comparison dispositions before selecting R2a.

## 1. Checkout and attribution

| Item | Observed value |
|---|---|
| Worktree | `/home/ein/repository/aspect-pf-rsf-refactor` |
| Branch | `pf-rsf-refactor` |
| HEAD | `3af7c2aa2590aa8e39eaf2a1e46df0bc6c9d9700`, “Add opt-in fault-parallel BP3 bottom boundary” |
| Other worktree | `/home/ein/repository/aspect`, branch `pf-rsf`, same HEAD; not modified |
| History | Full (`git rev-parse --is-shallow-repository` → `false`) |
| Local upstream main | `a42a1c4da4bf1a7622dfc8c8266856479cb7943b`, 2026-06-22, merge PR 7002 |
| Unique merge base with that ref | `42464facd7e0e41ba91a32eb061d97c21d97000b`, 2026-02-18, merge PR 6847 |
| Starting unstaged edits | `AGENTS.md`, `doc/reconstructed_fault/refactoring.md` |
| Starting untracked file | `doc/reconstructed_fault/refactoring/refactoring_plan.md` |
| Staged source changes | None |
| Local build directories in this worktree | None matching `build*` |

Remotes (same fetch/push URL for each): `origin` →
`https://github.com/YiminJin/aspect`; `upstream` →
`https://github.com/geodynamics/aspect.git`; `anne-glerum` →
`https://github.com/anne-glerum/aspect`; `jdannberg` →
`https://github.com/jdannberg/aspect`.

The inventory uses **merge-base → HEAD**, not upstream-tip → HEAD. Local
history establishes real ancestry without fetching. The live upstream tip was
not queried; the June upstream ref is not claimed to be current in September.
Consequently attribution to this development lineage is established, while
whether any changes have since landed upstream remains unverified. Refresh
upstream before R8. There is no shallow-history limitation in this checkout.

The plan reviewed `0fc1ce782c48b78f79eff674724686b8277c8618`. The additional
HEAD commit retains the opt-in BP3 bottom constraint and its local test tools.
No uncommitted scientific patch needs to be selected for this source baseline.
The modified guidance and untracked selected plan remain user work.

Reproduce the attribution without changing refs:

```sh
git status --short
git branch --show-current
git rev-parse HEAD
git remote -v
git rev-parse --is-shallow-repository
git worktree list
git merge-base --all HEAD upstream/main
git diff --name-status 42464facd7e0e41ba91a32eb061d97c21d97000b HEAD
git diff --numstat 42464facd7e0e41ba91a32eb061d97c21d97000b HEAD
```

## 2. Complete change population and dependency map

The committed diff contains **1,713 files, 2,112,389 added lines and 87 deleted
lines**. There are 42 modified pre-existing files and 1,671 additions. These
counts include prerequisites, documentation, fixtures and research artifacts;
they are not the size of a proposed upstream contribution.

| Area | Added / modified files | Responsibility and proposed treatment |
|---|---:|---|
| Root | 1 / 2 | `AGENTS.md`; build configuration and archive ignore rule. Keep local workflow separate from upstream scientific code. |
| `cmake/` | 1 / 2 | Voro++ discovery, exported capability and configuration report; prerequisite contribution. |
| `include/aspect/` | 21 / 15 | Interfaces, generic fault storage, material, particle, phase, surface and solver declarations; audit below. |
| `source/` | 22 / 20 | Implementation and integration; audit below. |
| `tests/` | 208 / 1 | Integration plugins, inputs, expected output, filters and shared test access. Select coherent cases with each contribution. |
| `unit_tests/` | 8 / 2 | Formula, geometry, projection, MPI/cache, condensation and domain tests. |
| `cookbooks/reconstructed_fault/` | 3 / 0 | Small prescribed-fault example inputs; candidate reproducible example. |
| `doc/reconstructed_fault/` | 198 / 0 | Current contracts plus substantial investigation history; retain authority and relevant explanation, curate upstream scope later. |
| `benchmarks/reconstructed_fault/` | 1,201 / 0 | BP3 517, BP5 171, uniform shear 417, performance 68, server GMG 20, state limiter 5, plus three shared/root files. Separate maintained fixtures/tools from archived investigations. |
| `tmp/` | 8 / 0 | Old RSF material/property, cumulative-slip and timestep snapshots. Exclude from proposed production contribution; preserve locally. |

Large tracked inputs dominate the line count: restored BP3 `target_cells.txt`
has 542,958 lines; modified-BP3 ell25 and ell50 target lists have 459,696 and
229,890. Restored BP3 `completion.txt` has 69,313. These are presently tracked
fixture inputs, not necessarily disposable output. Preserve them and replace
their role with small reproducible fixtures/generators only in a selected
contribution task. Do not copy all benchmark data into an upstream PR.

| Component/files | Owner | Dependencies and consumers | Boundary |
|---|---|---|---|
| `particle/particle_domain.{h,cc}`, optional Voro++; `voronoi_linear_reconstruction.{h,cc}` | Particle subsystem | Distributed particles, mapping, triangulation, domain volumes/faces/CPDI weights → phase assembly, fault projection, optional interpolator | Prerequisite; retain owned-particle contributions and periodic/domain support. |
| `phase_field.h`, `simulator/phase_field.cc`, mesh-refinement phase plugin, crack-driving-force particle property | `PhaseFieldHandler` / `PhaseFieldModel` | Particle domains, H, material geometric/degradation laws → Q1 phase field, length scale and activation/admissibility APIs | Reuse these APIs. Core-phase extension is not a reconstructed-fault algorithm. |
| `reconstructed_fault/fault.{h,cc}`, `utilities.{h,cc}` | Geometry/generic numerical infrastructure | Ordered append-only polylines, generic property arrays, closest profile/domain quadrature | Keep material interpretation out. Sentinel query already encapsulated. |
| `reconstructed_fault/manager.{h,cc}` | Simulator-owned manager | Phase field and particles → prescribed H, initial reconstruction, property schema, Q1 projection caches, committed/current/trial V, persistence | Manager stores generic material histories without interpreting their constitutive meaning. |
| `material_model/phase_field_fault.{h,cc}` | Material model | Phase handler + manager + EOS + existing friction law → Maxwell/cohesive point response, current I_h, history candidates/publication | R2 normalization remains this class's responsibility. No B/G operator API here. |
| `material_model/rheology/fault_friction.{h,cc}` | Friction law | Surface fractions, V, Theta → friction/derivatives/aging and law timestep | Reuse one Dc/minimum-rate interface; stateful and stateless paths remain distinct. |
| `particle/property/maxwell_stress.{h,cc}` | Particle property initialization/storage | Initial compositions and PhaseFieldFault capability → stress history; material publishes accepted updates | No trial constitutive updates in particle storage. |
| `reconstructed_fault/surface_system.{h,cc}`, `normal_filter_internal.h`, `surface_direct_internal.h` | Canonical simulator-owned surface system | Concrete material response + manager projections/bulk FE sampling → R, K, G, mass norms, restricted pivoted solves and normal filters | Preserve both particle-domain and bulk-work assembly paths; they use different measures. |
| `simulator/assemblers/reconstructed_fault_stokes.{h,cc}`, `sparse_coupling.h` | Canonical simulator-owned coupling | Material bulk coefficients + manager QP/source association → B, slip source and frozen history load | Preserve separate frozen/unknown MPI assembly and matrix/reference actions. |
| `solver/reconstructed_fault_condensed_system.{h,cc}`, `reconstructed_fault_linear.h`, `reconstructed_fault_nonlinear.{h,cc}` | Coupled algebra / nonlinear helpers | Existing A, canonical B/G/K, constraints, pressure scaling → condensed solve, active set and acceptance | Immutable linearization lifetime and principal free-block inverse; no duplicate surface owner. |
| Full driver currently in `simulator/solver.cc`; interface preconditioner and residual-audit headers | Simulator solve orchestration | Material prepare/commit, manager V lifecycle, assembled linear system, optional velocity GMG, diagnostics | R4 candidate extraction; retain Simulator member first. |
| `time_stepping/reconstructed_fault.{h,cc}` | Opt-in timestep plugin | Committed V/Theta, material friction law, existing CFL/global cap | Limiter contract discrepancy below; no new controller proposed. |
| `postprocess/reconstructed_faults.{h,cc}` | Output plugin | Replicated geometry/schema and committed V → rank-zero VTU/PVD | Output aliases/exclusions must not affect checkpoint schema. |
| `implicit_constitutive_stokes.{h,cc}` and generic output type | General implicit constitutive support | Additional material tangent/stress outputs, particle-point shape gradients, Newton assembly | Independent prerequisite candidate; not the architectural basis of reconstructed-fault coupling. |
| Maintained `bp3/plugin/`, BP5/uniform-shear tools and shared stress-only adapter | Model/benchmark plugins | Setup and accepted-state observations, native-output gates, restart metadata | Keep setup/output outside scientific owners where existing hooks suffice. Six source files now build the maintained BP3 library. |

Lifecycle seen in the production path:

```text
distributed particles/H → phase solve → initial reconstruction (step zero)
 → material preparation (surface mixture, I_h, initial-state validation)
 → manager current/trial V + private bulk Newton state
 → canonical surface/coupling linearization → condensed bulk direction
 → noncommitting trial residuals → accepted Newton iterate
 → converged history candidates + collective validation
 → terminal history/V/bulk publication → accepted-state observers
 → checkpoint persistent manager/particle/bulk state; rebuild transient caches
```

This agrees with the later Stage-J fixed-fault ordering. The earlier generic
“Timestep ordering” paragraph describes a broader reconstruction/propagation
sequence and should not be used to reverse the implemented Stage-J ordering.

## 3. Every modified pre-existing core/build file

Paths are repository-relative. Paired header/source rows explicitly cover both
files. `keep`, `move`, `independent-PR`, and `exclude` are review recommendations,
not authorization to change/remove anything. Verification entries are proposed
coverage, **not newly executed passing results**.

| File/symbol | Project responsibility | Reason | Caller/dependency | Disposition | Proposed destination / smallest retained integration | Verification |
|---|---|---|---|---|---|---|
| `CMakeLists.txt`: Voro discovery/link blocks | Particle prerequisite | Link optional tessellation library | Domain/interpolator sources; plugins | independent-PR | Keep capability detection/link hook; existing source glob discovers future splits | Configure with/without Voro; domain tests |
| `cmake/AspectConfig.cmake.in`: `ASPECT_WITH_VORO` | Build export | Match plugin capability | `ASPECT_SETUP_PLUGIN`, test configure | keep with prerequisite | Export boolean only | Plugin build |
| `cmake/write_config.cmake`: Voro report | Build provenance | Record dependency | Build configuration output | keep with prerequisite | One report field | Configuration inspection |
| `include/aspect/config.h.in`: Voro define | Build capability | Compile guards | Particle domain and tests | keep with prerequisite | One macro | With/without Voro build |
| `include/aspect/material_model/interface.h`: `tangent_modulus`, `ImplicitConstitutiveOutputs` | General material mechanics | Implicit stress/tangent output | Implicit assemblers and assembly scratch | independent-PR | Additional-output contract; no concrete fault formulas | Implicit material/assembler tests; ordinary Stokes |
| Same header: `MaterialProperties::operator\|=` | General API correction | Actually assign the OR-ed property mask | Requested-properties updates throughout material assembly | independent-PR | Small isolated correctness change | Property-mask regression; generic material tests |
| `include/aspect/parameters.h`: enable flags, `create_particle_domains` | Configuration | Phase/reconstruction/implicit switches | Simulator and particle subsystem | keep / exclude unused member later | Three active formulation flags; `create_particle_domains` has declaration only in source/include search | Parsing and disabled cases |
| `include/aspect/particle/interpolator/distance_weighted_average.h`; `source/particle/interpolator/distance_weighted_average.cc`: `compute_weight`, declare/parse | Generic interpolation | Optional reciprocal/Shepard weights, regularization | Configured particle-to-field interpolation | independent-PR | Existing plugin owns alternatives; default linear unchanged | Constant transfer and selected-weight cases |
| `include/aspect/particle/manager.h`; `source/particle/manager.cc`: initialization, move constructor, `connect_to_signals`, `advance_timestep`, domain getters, declare/parse | Domain lifecycle | Build/regenerate domains after particle changes | Phase handler, fault projections, Voronoi interpolator | keep prerequisite; move timing later | Domain owner and lifecycle calls; observational timer separate | Domain constants/area, advection, MPI/AMR |
| `include/aspect/particle/property/interface.h`; `source/particle/property/interface.cc`: custom late initialization enum/virtual/dispatch | Generic particle property | Property-specific late particle values | Property manager; inspect override consumers before port | independent-PR | One virtual hook and dispatch; no fault-law special case | Late insertion/component-offset tests (coverage gap to select) |
| `include/aspect/particle/property/initial_composition.h`; `source/particle/property/initial_composition.cc`: field selection/mapping, initialize/update/flags | Selected-field particles | Avoid duplicating stress/state properties; optionally refresh spatial strengthening | Particle manager and BP3 field layout | independent-PR / benchmark option review | Existing plugin with explicit empty-default refresh list | Selected/mixed fields, mapped names, restart and nonmutation |
| `source/particle/integrator/rk_2.cc`: `local_integrate_step` diagnostic dt | Research experiment | Freeze motion without changing constitutive clock | BP5 stress-cycle tools | move/exclude from initial production contribution | Benchmark integrator option using existing plugin boundary; retain until qualified replacement | Frozen/advected A/B and normal RK2 |
| `include/aspect/solution_evaluator.h`; `source/solution_evaluator.cc`: `SolutionEvaluator` phase member/index, reinit/evaluate/get values/gradients | Phase prerequisite | Evaluate added FE component at particles | Particle update and material sampling | keep prerequisite | Conditional phase component evaluation | Phase sampling and phase-disabled particles |
| `include/aspect/postprocess/particles.h`; `source/postprocess/particles.cc`: request flag/API, `execute` gate | Output orchestration | Force/gate native output at accepted states | Benchmark scheduling signals | independent-PR / keep hook review | Generic output request/gate, no BP3 schedules | Output off/on, restart clocks, ordinary particles |
| `include/aspect/postprocess/visualization.h`; `source/postprocess/visualization.cc`: phase metadata, request API, `execute` gate | Phase visualization / output | Correct component metadata; shared native schedule | Phase handler; benchmark output | keep phase metadata; independent-PR gate | Conditional scalar metadata and generic request/gate | Phase/ordinary VTU and output error paths |
| `include/aspect/simulator.h`: handlers, canonical surface/coupling, driver declaration, assembly flag | Simulator ownership | One owner and explicit dispatch | Core constructor, assembly and solver | keep | Owner pointers, member entry point, scoped assembly control | Lifecycle and rollback |
| Same header: `friend Postprocess::MomentCycle` | Research access | Snapshot/replay complete particle-to-FE transfer | `bp5/moment_cycle.cc` | exclude/move after research replacement | Test-only access boundary; no new public API | Moment replay snapshot/restore |
| `include/aspect/simulator/assemblers/interface.h`; `source/simulator/assemblers/interface.cc`: scratch constructors/point evaluator, `local_frozen_fault_rhs` | Implicit prerequisite / fault residual | Particle-point gradients; avoid cancellation before distributed addition | `assembly.cc`, implicit/fault assemblers | keep minimal storage; independent prerequisite separately | Constructor flag and one assembly-lifetime vector | Scratch copy, residual consistency, disabled assembly |
| `source/simulator/assembly.cc`: `set_stokes_assemblers`, preconditioner construction/material requests | Generic implicit support + coupled GMG | Select tangent assemblers; retain assembled blocks with GMG preconditioner | Newton handler, solver | keep dispatch; independent implicit support | Small selection/flag checks | Ordinary Newton, fault AMG/GMG |
| Same file: `local_assemble_stokes_system`, `assemble_stokes_system`, audit global | Fault bulk residual | Constant-velocity removal, canonical coupling execution, separate frozen MPI load | Coupled driver and residual-audit channel | keep numerical hook; move diagnostics later | Call coupling and combine independently assembled loads; preserve operation order | Residual consistency, translation/pressure precision, rollback |
| `source/simulator/checkpoint_restart.cc`: critical parameters save/load, `serialize` | Persistence | Store mode flags and generic fault registry | Snapshot/resume | keep | Conditional manager archive + mode checks | Baseline checkpoint read, split trajectory, ordinary checkpoint |
| Same file: `create_snapshot`, `resume_from_snapshot` signals | Benchmark output / matched clock | Notify completed checkpoint; reduce pending dt safely | BP3 archive and BP5 matched-clock plugins | independent-PR / essential hook review | Post-publication notification and validated opt-in reduction | Restart clock positivity/MPI agreement, default restart |
| `source/simulator/core.cc`: constructor, coupling/sparsity and preconditioner setup | Phase/fault ownership and backend setup | Allocate handlers, parse/init in order, keep assembled matrices for coupled GMG | Simulator lifecycle | keep | Conditional creation and initialization, phase sparsity call, backend guard | Disabled ordinary case; coupled init/GMG |
| `source/simulator/helper_functions.cc`: `select_default_solver_and_averaging` | Backend configuration | Reconstructed default AMG/unaveraged coefficients | Constructor | keep | Conditional default selection | Default vs explicit AMG/GMG; ordinary default |
| `source/simulator/initial_conditions.cc`: `interpolate_particle_properties` | General transfer correctness | MPI ADD/count average shared continuous DoF proposals | Particle-field refresh and initial conditions | independent-PR; move trace blocks later | Deterministic sum/count publication; trace formatting outside later | Continuous/DG transfer, rank agreement, ordinary particle case |
| `source/simulator/newton.cc`: `NewtonHandler::set_assemblers` | Implicit prerequisite | Select implicit constitutive assemblers | Newton schemes | independent-PR | Existing assembler selection boundary | Ordinary Newton + implicit material |
| `source/simulator/parameters.cc`: declare/parse formulation, phase/reconstruction dispatch, solver description, composition selection string | Configuration | Expose prerequisites and enforce enable dependencies | All input readers | keep active controls; review stale selection below | Existing parameter owners, no duplicates | Parse enabled/disabled/invalid combinations |
| `include/aspect/simulator_access.h`; `source/simulator/simulator_access.cc`: five accessor families (six overloads) | Core-facing capability access | Reach canonical owners; diagnostic RHS | Material, phase, solver helpers, plugins and tests | keep required owners; review RHS exposure | Existing access class with assertions, no duplicate objects | Callers listed in §4; disabled guards |
| `include/aspect/simulator_signals.h`: seven added signals | Setup/solve/output/restart observation | Correct timing for plugin setup and accepted output | Emitters and slots in §4 | keep essential hooks; independent-PR/research review | Synchronous narrow events; no constitutive ownership transfer | Failure timing, output/restart, frozen-solve tests |
| `source/simulator/solver.cc`: `solve_reconstructed_fault_stokes`, `current_slip_rate` | Coupled nonlinear orchestration | Preparation, physical lift, residual/active-set solves, Armijo, commit/rollback | Newton dispatch; canonical algebra; material history | move R4a | Dedicated `source/simulator/solver/` implementation, retain Simulator member | Condensation, bound/Armijo, rollback, residuals, short trajectory |
| `include/aspect/simulator/solver/stokes_matrix_free_local_smoothing.h`; `source/simulator/solver/stokes_matrix_free_local_smoothing.cc`: `with_velocity_preconditioner`, `preconditioner_only`, constraint callback gate | Existing GMG reuse | Borrow velocity V-cycle for assembled condensed operator | Fault driver only in production | independent-PR / keep narrow interface | Scoped consumer callback with numbering/constraint checks | Frozen matched solve and ordinary GMG |
| `source/simulator/solver_schemes.cc`: phase evolution/reconstruction calls, coupled dispatch | Lifecycle | Evolve phase from committed H before mechanics | Selected solver schemes | keep | Short calls/dispatch only | Stage J; phase-only; ordinary solver |
| Same file: pre-assembly signal and reordered `post_nonlinear_solver` calls | Generic nonlinear failure semantics | Observers see failure before exception; reset tolerance ordering | All changed schemes, rollback observers | independent-PR | Explicit event-order contract, not fault algorithm code | Failure callbacks and non-fault Newton (coverage gap) |
| `tests/CMakeLists.txt`: `SHOULD_ENABLE_TEST` Voro gate | Test prerequisite | Skip unsupported domain tests | Test configure | keep | One capability gate | `ctest -N` plus capability configuration |
| `unit_tests/CMakeLists.txt`: `phase_field_fault_ih_accuracy_mpi` | MPI verification | Unequal profile ownership | ASPECT Catch test binary | keep | Two-rank tagged test | Actual R1 execution required |
| `unit_tests/particles.cc`: CPDI/domain tests | Particle prerequisite | Constants, ownership, area, endpoint moments | Production particle-domain and fault quadrature utilities | keep; separate prerequisite/fault portions for PRs | Existing unit runner | Tags `particle_domain_constants`, `particle_domain_area`, `fault_domain_quadrature` |
| `.gitignore`: `.benchmark-cleanup-*` | Local evidence preservation | Ignore recoverable local archives | Benchmark cleanup workflow | exclude from scientific contribution unless independently justified | Local workflow | No deletion or cleanup in R0 |

No modified existing core file is classified solely from a likely-site list:
the rows above follow the actual 42-file modified population. New feature files
are covered in §2 and the generated path inventory below.

## 4. Public interfaces, signals and shared state

Added core accessors are `get_system_rhs()` (const vector reference),
`get_phase_field_handler()` (const/nonconst overloads),
`get_reconstructed_fault_manager()`,
`get_reconstructed_fault_surface_system()`, and
`get_reconstructed_fault_stokes_coupling()`. The last three are const member
functions returning **mutable** canonical objects. Consumers include material
preparation/publication, surface/coupling/condensed helpers, postprocessing and
benchmark setup. Test plugins `phase_field_frozen_history.cc` and
`phase_field_fault_stage_j.cc` cast away RHS constness for controlled failure
injection; the RHS accessor is not needed to give normalization a new owner.

Particle manager adds `particle_domains_requested()` and
`get_particle_domain_handler()`. GMG adds the borrowed `VelocityCycle` callback.
Particles/visualization add `request_output() const` backed by mutable flags;
no invocation was found in the inspected source and reconstructed benchmark C++
search, so verify all consumers before retaining that API in an upstream PR.

| Signal | Emitter / exact lifecycle | Located consumers / decision |
|---|---|---|
| `post_simulator_initialization` | End of constructor after assemblers and canonical fault objects exist | BP3 monitor/setup and surface test plugin; keep necessary late setup hook |
| `pre_assemble_stokes_system` | Newton scheme in `solver_schemes.cc` | Generic implicit-state preparation intent; no connect call found in source/reconstructed benchmark search; do not assume every assembly emits it |
| `post_reconstructed_fault_linear_solver` | After fresh condensed-direction verification; borrowed operators/vectors valid only during callback | Frozen-GMG and mechanical-mode test plugins; research/test observation candidate |
| `post_reconstructed_fault_solver` | After terminal accepted history/V/bulk writes, not on failed solve | BP3 accepted-solve counts/mask; preserve callback timing |
| `post_resume_time_step` | After restored user data and clock, before physical update | BP5 matched-clock plugins; decrease-only pending interval, collective validation |
| `post_checkpoint` | After complete checkpoint and last-good publication | BP3 metadata/archive output; callback error is after checkpoint publication |
| `allow_native_output` | Native bulk/particle/fault writer gates | BP3 output scheduler; preserve empty-slot ordinary behavior and error semantics |

New feature APIs that merit continued review, without widening them in R2:

- Manager: property registration/index lookup, mutable `get_fault()`/property
  views, projected particle/FE associations, projection diagnostics and
  invalidation, prescribed V map and all current/trial/committed V transitions,
  shear sense and top/bottom source continuation selectors.
- Material: point/bulk response structures and evaluators, frozen Maxwell
  evaluator, prepare/validate/commit, law/minimum-rate/Dc/timestep queries,
  background traction property selection, mature-mode query, completion-file
  selector, and public `benchmark_retained_stress` callback.
- Surface: residual/linearization/K/G/restricted solve/norm operations,
  `enable_bulk_work_measure()`, `set_normal_stress_filter()`, diagnostic window,
  observer and result access, reference action and generation counter.
- Coupling: assembly, B preparation/application/reference, slip-dependent bulk
  residual, rebuild counter. These belong outside the material model.
- Phase handler: Q1/grid/particle-manager access, geometric/degradation/length
  and revision queries. `pre_extend_core_phase_field` is a leftover declared
  signal, not a reconstructed-fault extension mechanism.

| Shared state | Owner/lifetime | Coupling and risk to preserve |
|---|---|---|
| Ordered geometry, property schema/values, projection widths, shear senses | Fault/manager; persistent | Versioned geometry, stable property order, serialization v1 manager handling |
| V vectors and prescribed rates | Manager; committed/current/trial | Accepted absolute bound-contact values copied unchanged; only timestep commit publishes |
| Previous I_h, Theta, cohesive traction | Material semantics in generic manager storage | Residual/linearization must not publish them |
| Current I_h, value keys, remote batches, cell intervals, minimum raw phase, transient surface temperature | Material; transient preparation/mesh/field lifetimes | Independent geometric and value validity; collective hits; restart invalidation/restoration |
| Particle stress and H | Particle storage, material accepted-history writer | Candidate construction/collective checks precede terminal writes; step zero retains initial histories |
| A, system RHS, constraints, current linearization, Newton scaling/rebuild flags, pressure adjustment | Simulator; temporarily borrowed by driver | Driver snapshots/restores around residuals and failures; physical lift differs from algebraic complement |
| Surface linearization/filter factors and B cache | Canonical simulator-owned helpers | No second surface owner; restricted solve tied to correct generation |
| `internal::fault_residual_audit_channel` | Process-global enum, defined in `assembly.cc` | Driver switches only between synchronous WorkStream calls; worker threads read it; extraction must retain restoration |
| `FaultLinearTiming` recorder | Static thread-local timing storage in feature header | Observational counts/times; should not become another numerical state owner |
| `benchmark_retained_stress`, surface observer/window, MomentCycle friends | Benchmark/test injection | Nonserialized borrowed setup/observation; isolate later without losing replay capability |
| Native `output_requested`, plugin schedule/archives | Writers / benchmark | Output clocks are persistent benchmark state, distinct from constitutive histories |

Direct production `PhaseFieldFault` dependencies are in material implementation,
Maxwell particle property, surface system, reconstructed Stokes assembler,
simulator constructor, phase handler, nonlinear driver, condensed system and
fault timestep plugin. The generic fault and manager implementation do **not**
depend directly on that concrete material. R2 does not remove the header's
manager/remote-evaluation dependencies or narrow a public API; it isolates
implementation only. An abstract constitutive hierarchy is not proposed.

## 5. Configuration and environment inventory

Core parameter additions and defaults are catalogued below from declarations.
Existing global CFL, timestep caps, Newton/linear tolerances and friction Dc
remain authoritative; no replacement controls are proposed.

| Owner / subsection | Added selections (defaults) | Category |
|---|---|---|
| Formulation | Enable phase field (`false`); Reconstruct faults from phase field (`false`); Use implicit constitutive model (`false`) | Feature / solver selection |
| Particles | Generate particle domains, face data, CPDI data (each `false`) | Prerequisite numerical infrastructure |
| Distance weighted average interpolator | Weight type (`linear`); Distance regularization factor (`0.1`) | Numerical interpolation selection |
| Initial composition property | List of field names (empty); Spatially refreshed field names (empty) | Field ownership / opt-in spatial setup |
| Phase field model | Length scale (`1000`); Geometric function type (`AT1`); Degradation curvature parameter (`1`) | Physical/model parameters |
| Phase field solver parameters | Linear tolerance `1e-8`, max linear iterations `1000`, nonlinear tolerance `1e-5`, max nonlinear iterations `10`, max Newton line search iterations `3` | Solver controls |
| Fault reconstruction | Structural point spacing `1000`; Fit prescribed geometry to phase field `true`; Ridge coefficient `1`; Prescribed faults file empty | Geometry/numerical setup |
| Phase field fault material | Conductivity `3.0`, reference temperature `293`, min/max viscosity `1.e17/1.e25`, averaging `harmonic`, reference viscosities `1.e24`, thermal exponents `0`, shear moduli `1e10`, cohesions `1.e7`, initial friction `0.6`, Gc `1.e5`, damping empty | Material physical parameters; composition lists |
| Same material | Activation `0.1`, normal-lock threshold `0.5`, initial artificial dt `1`, adiabatic friction pressure `false`, evolve phase `true`, mode `cohesive` | Model/configuration choices; mature mode explicit |
| Same material | I h backend `remote points`, quadrature/tail tolerances `1e-8`, surface subdivisions `1` | Normalization numerical controls; R2 preserves all |
| Friction law in material subsection | Law `rate state`, V0 `1.e-6`, Vmin `1.e-20`, Dc `0.04`, reference/dynamic friction `0.6/0.4`, weakening rate `1.e-6`, a/b `0.025/0.013`, regularized `true` | Constitutive law, not geometry |
| Reconstructed fault time step | Maximum logarithmic state change (largest finite double) | Optional controller; discrepancy §7 |
| Reconstructed faults output | Excluded properties (empty); native-output gate described in §4 | Observation; exclusions do not change registry |
| Build | `ASPECT_WITH_VORO` requested ON, disabled if not found; `VORO_DIR` hint | Dependency capability |

EOS and existing ASPECT/plugin parameters are reused, not newly invented here.
Benchmark PRMs add setup/loading, mesh/profile/completion files, initial
prestress/state, normal filter, output cadences, event/termination and diagnostic
controls. In maintained BP3, `Bottom velocity constraint = full` remains default;
`fault parallel` is opt-in. These settings belong to the benchmark plugin. The
old BP5 startup controller is not compiled into the maintained BP3 plugin.

All literal environment reads in production feature/core files inspected:

| Names | Reader/owner | Classification and retained behavior |
|---|---|---|
| `ASPECT_FAULT_PERFORMANCE` | Material, particle/fault manager, surface, coupling | Observational timing; presence enables |
| `ASPECT_FAULT_LINEAR_PERFORMANCE` | `linear_performance.h` | Observational rank-local timings |
| `ASPECT_FAULT_NONLINEAR_DIAGNOSTIC`, `ASPECT_FAULT_COMPATIBILITY_DIAGNOSTIC` | Driver | Observational residual/active-bound/compatibility logging |
| `ASPECT_K1_FLOOR_AUDIT` | Driver + residual channel | Shadow assembly audit; temporarily changes assembly channel, restores before solve |
| `ASPECT_STRESS_CYCLE_TRACE` | Particle-to-FE publication and material history | Observational CSVs from actual proposals/evaluations; requires selected-cell files |
| `ASPECT_FAULT_SOURCE_HISTORY_DIAGNOSTIC` | Material history | Observational history/source output |
| `ASPECT_FAULT_HISTORY_AUDIT`, `ASPECT_FAULT_NORMAL_STRESS_DIAGNOSTIC` | Surface | Observation/audit; output failures/checks may abort |
| `ASPECT_FAULT_STRESS_SAMPLE_DIAGNOSTIC` | Surface, coupling, driver | Additional sample diagnostics and standalone discarded residual evaluation |
| `ASPECT_FAULT_EXPLICIT_B`, `ASPECT_FAULT_EXPLICIT_G` | B/G owners | Numerical implementation selection; sparse action instead of reference path; G checks filter applicability |
| `ASPECT_FAULT_INTERFACE_MODES` | Interface preconditioner | Numerical preconditioning selection; parse setting, retain defaults/zero-mode behavior |
| `ASPECT_FAULT_VERIFY_INTERFACE` | Interface preconditioner | Diagnostic verification of selected coarse correction |
| `ASPECT_DISABLE_IH_VALUE_CACHE` | Material normalization | Numerical/reference execution path, exact reuse disabled |
| `ASPECT_IH_BASELINE_GUARDS` | Material normalization | Reference validation path, includes otherwise redundant single-fault support checks |
| `ASPECT_IH_COMPARE_CELL`, `ASPECT_IH_REFERENCE_FACTOR` | Material normalization | Reference integration comparison and reference tolerance multiplier; not plain logging |
| `ASPECT_IH_COMPARE_SAMPLES`, `ASPECT_IH_VERIFY_CELL_QUADRATURE` | Cell normalization | Extra sampling/quadrature verification; can reject mismatch |
| `ASPECT_IH_BOTTOM_COMPLETION_DIAGNOSTIC`, `ASPECT_BP3_UNIFORM_SLIDING` | Material + old reference benchmark | Benchmark completion/setup selection and guard; physical result can change |
| `ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION` | RK2 | Experimental numerical override: uses zero advection dt; never treat as observational |

`bp3/environment.sh` selects explicit B/G and one thread using
`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `DEAL_II_NUM_THREADS`. It unsets
obsolete surface-solver/velocity-GMG selectors. An absent switch differs from
the string `0` for presence-tested options. Record the actual environment in R1.
Benchmark/test-only selector inventory is appended below; names in rejection
lists are not evidence of a live core algorithm.

## 6. Test infrastructure, missing artifacts, and proposed R1

No result in this section is a new pass. `CURRENT_STATUS.md` explicitly says the
source checkpoint was not fully rebuilt/simulated. Historical Stage-I rate/state
closed-box inputs hit pressure-nullspace compatibility before the limiter.
Historical short BP3/restart/plugin checks and server compiler remedies are
leads, not qualification of this SHA under a new build.

The top-level CMake globs `source/*.cc` and `unit_tests/*.cc`, configures separate
`tests` and `unit_tests` projects, and embeds Catch tests in ASPECT (`--test`).
Integration CTest entries invoke `tests.<name>` build targets; they build
same-named `.cc` plugins, create `.x.prm` files, run ASPECT with `# MPI:` ranks,
and filter/compare outputs. `ASPECT_RUN_ALL_TESTS=ON` is needed to register the
non-QUICK cases here; registration does not require running the whole suite.
`# Enable if: ASPECT_WITH_VORO` gates domain cases. `TEST_TIME_LIMIT` defaults
to 600 seconds per integration test. Test scripts regenerate their own output
directories: use a new designated baseline build/test directory, never the
scientific worktree or saved evidence directory.

| Required evidence | Existing starting point | Concrete proposal / dependency |
|---|---|---|
| Normalization formulas/projection/cache | `unit_tests/phase_field_fault_ih.cc`, test access header | Run named/tagged tests, including analytic/Q1 accuracy and lookup invalidation in 1/2 ranks. Q1 study reaches refinement 8; time it first. |
| Small I_h lifecycle | `phase_field_fault_ih`, `_ih_mpi`, `_ih_no_composition` | Existing rank 1/2 PRMs/plugins and tracked `phase_field_fault_ih.txt`; initialization only, not a trajectory. |
| Mechanical formulas/algebra/lifecycle | Unit Maxwell, cohesive, reconstructed fault, condensation | Run relevant name filters; include normal filter, V bounds, free-block inverse and domain constant tests. |
| Failure restoration | `phase_field_fault_stage_i_rollback`, `_rollback_mpi`; pressure-gauge rollback family | Existing forced failure after accepted Newton updates; preserve scripts' acceptance and restoration assertions. Check actual reachability before claiming rollback passed. |
| Existing evolving history/restart smoke | `phase_field_fault_stage_j_restart_create`, `_resume` | Both rank 2; create includes Stage J and advances initialization plus exactly two physical updates; resume copies create's fresh checkpoint and compares histories/V/geometry/bulk against uninterrupted state. Do not count this as 3–6-step evidence. |
| Short history-evolving comparison | Existing local `../aspect/benchmarks/reconstructed_fault/bp3/output-cleanup-evidence/fixture-base.prm`, `after.prm`, `fixture-convex/` | Preferred after the user identified the data worktree: preserve the existing six 100-second physical steps and full bottom loading. Stage copies under this worktree with library/input/output paths rebound to a matching new build. No new physical fixture is needed. |
| Matched six-step BP3 split restart | Same existing output-cleanup fixture and `restart-v5` / `restart-thin` recipes | Recreate baseline checkpoints with the new reference executable; branch the accepted state-4 checkpoint and resume through states 5/6, matching the uninterrupted six-step run. Preserve saved historical checkpoints separately. Capture V/Theta/I_h, stress/H, tractions/slip and solver decisions; verify checkpoint step semantics at runtime. |
| Ordinary disabled-feature case | `convection_box_particles` | Rank 1, End time 0; tests default core initialization, particle transfer and visualization without phase/fault features. |
| Ordinary checkpoint path | `checkpoint_03_particles` | Generic checkpoint layout changed unconditionally, so add this separate existing checkpoint/resume fixture; inspect runtime before broadening further. |
| Conditional later coverage | Ordinary Newton failure/GMG, changed loading, pressure modes, residual consistency | Select when R4/core edits touch those paths. R2 should not trigger a full benchmark campaign. |

Example **future R1** commands, after configuring/building a new baseline at
`build-refactor-baseline` with tests and Voro enabled (not executed):

```sh
ctest --test-dir build-refactor-baseline -N
OMP_NUM_THREADS=1 DEAL_II_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  mpirun -np 1 build-refactor-baseline/aspect-release --test '[phase_field_fault_ih_accuracy]'
OMP_NUM_THREADS=1 DEAL_II_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  mpirun -np 2 build-refactor-baseline/aspect-release --test '[phase_field_fault_ih_accuracy]'
ctest --test-dir build-refactor-baseline/tests --output-on-failure -j1 \
  -R '^phase_field_fault_ih(_mpi|_no_composition)?$'
ctest --test-dir build-refactor-baseline/tests --output-on-failure -j1 \
  -R '^phase_field_fault_stage_i_rollback(_mpi)?$'
ctest --test-dir build-refactor-baseline/tests --output-on-failure -j1 \
  -R '^phase_field_fault_stage_j_restart_(create|resume)$'
ctest --test-dir build-refactor-baseline/tests --output-on-failure -j1 \
  -R '^(convection_box_particles|checkpoint_03_particles)$'
```

Select the actual binary name/MPI installation from the new build, not from this
example. Set thread variables for the CTest launches as well. CTest `DEPENDS`
orders selected tests but does not automatically select create if only resume
is requested; keep both in the regex. Preserve baseline executable/configuration
immutably after qualification. Candidate build must use a different directory.

R1 should first record compiler/deal.II/Trilinos/MPI/build flags and measure the
smallest cases, then agree a minutes-scale incremental budget. The alternative reduced
rotated fixture's historical 10-step A took about 71 seconds; this is not a
prediction for the new baseline. Its runner uses a 900-second cap and a
hardcoded `/opt/openmpi/5.0.6/bin/mpirun`; that executable exists here, but no
compatibility claim follows from existence. Do not reuse a mismatched plugin.

Missing/local-only dependencies in this worktree:

- No core build, reduced `rotated_bottom_local/build` or generated `fixture`
  directory exists **in this refactoring worktree**. All three are available
  in `../aspect/`, as inspected after the user supplied that location. Tracked `prepare_fixture.cc`, CMake and `run.sh prepare`
  can generate it from the tracked restored BP3 inputs. Prepare only under the
  refactoring worktree, with matching build configuration; copying the existing immutable fixture
  inputs from the data worktree avoids regeneration. The stock runner's
  default `build-tmp` is absent here; override `ASPECT_LOCAL_BINARY` in R1.
- Restored BP3 `fixtures/bp3_150x50/{profile,completion,target_cells,fault}.txt`
  inputs are tracked/present. The hidden `[bp3_restore_profile]` unit test is
  therefore potentially usable without a full-resolution simulation.
- Hidden `[.fault_ih_performance]` needs `ASPECT_FAULT_PERFORMANCE_STATE`
  pointing to saved mesh/field data; `[.fault_saved_support]` needs
  `ASPECT_SAVED_FAULT_SUPPORT_AUDIT`. No such selected external dataset has
  been established for R1. Exclude from the initial mandatory suite and record
  as unavailable, not passed. Cache plugin's `ASPECT_IH_SAVED_PHASE` and
  `ASPECT_IH_SAVED_SURFACE` likewise require their saved inputs when enabled.
- Some performance `.cc` plugins have no same-named `tests/*.prm`; their
  benchmark CMake/PRMs supply the harness. Do not assume CTest discovered them.
- Historical reports can exist without their raw output/checkpoints. The
  scientific worktree's binaries and saved trajectories are not a qualified
  refactor baseline and were not overwritten or borrowed for execution.

### Located BP3 data in `../aspect/` (read-only follow-up)

The user identified the scientific data worktree during R0. Read-only inspection
found the following; “present” means filesystem presence, not new execution:

| Absolute-root-relative path under `/home/ein/repository/aspect/` | Located evidence / use |
|---|---|
| `build-tmp/` | `aspect-release`, CMake cache and compile commands; source directory points to this scientific worktree, Release, MPI wrapper `/opt/openmpi/5.0.6/bin/mpic++`, deal.II `/opt/dealii/9.6-local`, Voro ON at `/home/ein/local/voro++/0.4.6`. Do not rebuild this directory. |
| `benchmarks/reconstructed_fault/bp3/rotated_bottom_local/{fixture,build}/` | Four fixture input files, local plugin, `prepare_local`, model test executable. Outputs A/B, half-dt and uniform 1/2-rank cases are present. |
| `benchmarks/reconstructed_fault/bp3/output-cleanup-evidence/` | Existing smaller functional fixture, generator, `fixture-base.prm`, `after.prm`, restart recipes, `comparison.txt`, saved runs and checkpoints. This is now the preferred R1 trajectory lead. |
| Same directory: `after/accepted_steps.csv` | Seven data rows, states 0–6 through 600 seconds; final recorded committed stress about 0.402 Pa and Theta relative error 2.22e-16. This confirms saved evolution evidence exists; it is not a rerun at HEAD. |
| Same directory: `after/restart`, `before-uniform/restart` | Last-good markers and slots 01/02/03; `restart-v5` and `restart-thin` outputs also present. Preserve all original archives. |
| `benchmarks/reconstructed_fault/bp3/filter-test/` | Raw/filter20/filter40 outputs and comparison directory present. |
| `benchmarks/reconstructed_fault/bp3/output-first-event/` | Profiles, accepted summaries, plots and native evidence present; preserve the status note's differing-clock/completeness qualifications. |

The six-step fixture's historical report describes 1,875 cells, 16,875 particles,
32 vertices, ell=4 km, Gc=2e7 and uniform a=0.025. It is a functional mechanics/
history/output regression, not a production-resolution BP3 qualification. Use
these existing settings unchanged in baseline/candidate comparisons.
`fixture-base.prm` selects **`fixture-convex/`**, not the earlier failed
`fixture/`; keep that distinction. `after.prm` also loads a map-measurement test
plugin from `/tmp`, and the local CMake file hardcodes the former runtime library
path. R1 must stage/rebind this harness to the new matched plugin/build and
separate output directories; do not execute the historical PRM in place. Record
any decision to omit observational map measurement and test its noninterference.
The saved `comparison.txt` reports exact old/new plugin and restart matches;
those remain historical evidence.

`git -C ../aspect status --short --untracked-files=no` returned no tracked edits.
No file there was written, copied, regenerated or executed in R0. New baseline
builds/runs remain confined to this refactoring worktree or designated directories.

If Stage J/rollback hits its historical compatibility failure, record the
failure verbatim and its scope. Do not repair it or weaken tolerance under R1.
Require independent passing history/restart/rollback evidence for the intended
extraction, or select a separate fix task. No numerical equivalence or field
error has been measured in R0.

## 7. Discrepancies, stale guidance and unresolved questions

**Historical R0 finding, superseded by the user's September 29 clarification:**
The user retained the numeric-parser design and authorized corrected
comments/specifications, then explicitly requested a zero-rejecting assertion.
R1 confirmed the default round-trip and rejected infinity spelling. The original audit below is retained as history.

**Original source/contract discrepancy:**
`current_design.md` §25's optional limiter text (around line 1625) and
`specification.tex` Stage-J limiter paragraph require a positive bound and
`infinity` as an alias for the disabled maximum-double sentinel.
`source/time_stepping/reconstructed_fault.cc:28` declares `Patterns::Double(0.)`,
`:49` reads `get_double()` without explicit positive validation or alias
normalization, and `:74` bypasses only equality to maximum finite double (or
stateless friction). Zero is not rejected by that declared lower bound. If
positive infinity is admitted by the installed parser, it enters the enabled
path, caps the law proposal by global maximum dt, collects/validates state and
evaluates the predictor, rather than taking the stipulated disabled path.
Parser spelling acceptance and default numeric round-trip were not run here.
This matches the status note's warning that the older sentinel tests predate
the present implementation. **No fix or contract edit is included.** Decide
whether to qualify this unchanged behavior with the limitation or select a
separate parameter-contract fix before baseline acceptance.

Other findings are deliberately distinguished from scientific conflicts:

| Finding | Evidence / meaning | Disposition |
|---|---|---|
| Wrong roadmap path in code-quality guidance | `refactoring.md` points to `doc/reconstructed_fault/refactoring_plan.md`; selected plan is in `refactoring/refactoring_plan.md`. The older root-level plan also exists. | Follow user's explicitly selected nested plan; reconcile links later. |
| Plan's shallow-history and revision assumptions are stale locally | Full history and newer HEAD recorded above | Do not fetch/deepen or revert to old snapshot. |
| Geometry-only/Stage-F plans and recovery “next task” sections are historical | Current source has complete coupled lifecycle; user selected R0 | Do not restart implementation stages or production experiments. |
| Sentinel/test-access cleanups already exist | `fault.h::property_value_is_initialized`; material friend declaration, implementation in `tests/phase_field_fault_test_access.h` | Not new refactoring tasks; retain narrow declarations. |
| Extra formulation documentation/selection appears stale | `parameters.cc` admits string `phase field` in compositional-method pattern/description; the actual phase handler adds a separate FE variable. No corresponding `AdvectionFieldMethod` enum member exists in `parameters.h`, and the parser falls through to `ExcNotImplemented()`. | Configuration/API inconsistency to examine separately; do not use this string for new fixtures. |
| Unused-looking APIs | `Parameters::create_particle_domains` and phase extension signal only declarations in production search; request-output has no located benchmark caller | Candidates for later consumer-complete cleanup, not removal now. |
| Generic checkpoint format differs from ancestor even when faults disabled | Critical-parameter serialization unconditionally adds two booleans | Preserve research-baseline format during refactor; ordinary restart check required. Upstream migration/backward compatibility is a separate R8 decision. |
| Research output APIs after commit can throw | Accepted-solve callbacks emitted after terminal publication | Preserve current semantics; no rollback redesign implied by extraction. |
| Maintained plugin source list changed | Current CMake builds six files including `work_audit.cc` and `bottom_constraint.cc`; older status paragraphs describe four/five | Use current CMake, not historical lists. |
| LLS concern in recovery handoff was resolved | Later CURRENT_STATUS and restored input select all-field unlimited LLS; stress-only routing belongs to a separate controlled comparison | No inferred interpolation fix. |

No other scientific conflict was established by this dependency audit; this is
not an exhaustive mathematical or MPI correctness proof. Existing stress
concentration, convergence, finite-domain loading and server-performance
questions remain separate research work.

Review questions requiring decisions between stages: accept HEAD as the source
baseline; disposition of the limiter mismatch; approve the minimal R1 reduced
run/restart wrappers and test budget; retain general transfer/implicit/output
improvements as separate contribution candidates; and which benchmark modes
must accompany the eventual upstream mechanics contribution.

## 8. Precise proposed first normalization extraction (R2a only)

Prerequisite: accepted R1 evidence for normalization and a short coupled
history/restart fixture, with any baseline failure given an explicit
disposition. This report does not authorize R2.

Allowed change: add
`source/material_model/phase_field_fault_normalization.cc` and move complete
definitions from `phase_field_fault.cc`. Keep `PhaseFieldFault` and its private
members/declarations in the existing header; no new public helper class,
manager method, constitutive base class, physical parameter or cache owner.

Move these existing methods, currently in the contiguous normalization region
around lines 1970–3470:

1. `compute_normalization_integrals()` in full, including value cache keys,
   diagnostics/comparison paths, endpoint completion and validated frozen
   restart restoration.
2. `invalidate_normalization_cache()`, `normalization_effective_phase_field()`,
   `validate_normalization_phase_field_minimum()`, `normalization_integrand()`.
3. `integrate_cell_normalization_profiles()` and
   `integrate_normalization_profiles()`; both backends and fallback behavior.
4. `project_surface_chemical_compositions()` (one definition),
   `NormalizationPointLookupCache::get()`, `evaluate_normalization_points()`,
   `build_owned_normalization_profiles()`,
   `project_normalization_integrals_to_fault()`.

Move only the normalization-exclusive anonymous helpers:
`normalization_search_enclosure`, `NormalizationSideState`,
`NormalizationEvaluationRequest`, and `normalization_profile_point`.
Keep `interpolate_surface_chemical_compositions` with its cohesive-initialization
caller (`evaluate_initial_cohesive_particle_values`); it is different from
projecting bulk chemical fields onto the surface. Keep `interpolate_fault_scalar`
and history validation helpers with mechanics/history. Do not duplicate mixture
preparation in the new file. The completion-file setter can stay in the main
material source as an existing public configuration entry point.

Calls into normalization remain from material preparation/cohesive initialization,
initialization/restart signal handling, parse invalidation, and test-access
wrappers. Retain exact projected-mixture order and the distinction between
current transient I_h and the persistent previous-I_h property. Keep helper
state in the class so public/private header exposure is unchanged in R2a.

The top-level CMake uses `GLOB_RECURSE ... CONFIGURE_DEPENDS "source/*.cc"`;
no new build framework is needed. Leave the **single**
`ASPECT_REGISTER_MATERIAL_MODEL(PhaseFieldFault, ...)` in the original file.
In the new translation unit explicitly instantiate each moved member for
dimensions 2 and 3, including static/private members used by test access and
the nested lookup-cache method. Do not add a second whole-class instantiation
or registration. Inspect the registration macro and emitted symbols during
implementation to avoid missing/duplicate instantiations, including unity and
non-unity build behavior where available. Explicit member instantiation does
not require making private members public.

Preserve every expression and collective/iteration order. Specifically retain
physical `[0,1]` treatment, the I_h-only lower clamp and raw undershoot check,
singularity reporting, consistent Q1 projection, fixed surface mixtures,
profile rank ownership, cell/rejection-cache restrictions, mesh-deformation
bypass, exact all-rank value hits, diagnostic failure behavior and restart
comparison budget. Keep stable Maxwell `expm1` code untouched in its current
file. No cache consolidation, parameter relocation, assertion cleanup or CSV
extraction in this pass.

R2a verification: build/link the split with existing test-access users; repeat
R1's analytic/Q1/cache tests, I_h lifecycle at 1/2 ranks and no-composition case;
exercise both backends using existing benchmark harness or qualified R1
wrapper; compare the short coupled run and restart at matching ranks/stacks.
Attempt exact field/decision comparison for moved arithmetic, excluding timing.
Compare I_h hit/rebuild/integration behavior as well as values. An observed
difference is investigated, not accepted by loosening tolerance.

Expected outcome: normalization implementation can be reviewed separately;
material still owns all normalization semantics and data. Public interface,
scientific algorithm, MPI ownership and serialization remain unchanged. No
claim that a file move alone removes coupling. R2b restructuring is a separate
selection after reviewing R2a.

## 9. R0 evidence and limits

Executed read-only commands: Git status/branch/remotes/worktrees/history and
merge-base/diff; `rg` file/symbol/environment/parameter/caller searches; `cat`
and `sed` on guidance, contracts, implementation, headers, PRMs, test/build
runners and historical reports; small Python standard-library scripts to count
Git diff paths and check artifact existence. No network access was needed to
establish the local common ancestor. No external facts or live upstream status
are asserted.

Algorithms, numerical defaults, scientific documents and checkpoint formats
were unchanged. No implementation test or simulation was run, consistent with
R0. The only validation applicable to the new report is path/count/coverage
and whitespace checking. Final static checks passed: all 42 modified existing paths and all 44 new
production/build paths are represented; ancestry counts sum to 1,713; Markdown
fences balance and the report has no trailing whitespace. `git diff --check`
reported no whitespace errors. Final status contains only the three original
user guidance paths plus this report. Numerical pass/fail/skip measurements,
runtime cost, freshly verified compiler stack and field differences are
deliberately deferred to R1.

Recommended next bounded task: **R1 baseline qualification at the recorded
HEAD**, after review of the limiter conflict and the proposed reduced fixture
wrappers. Stop here; neither R1 execution nor R2 source movement is part of R0.

## Appendix A. New production/build paths

Generated from the ancestry diff. Grouped responsibility and verification are in §§2–3; no production path is omitted from this new-file list.

```text
cmake/modules/FindVORO.cmake
include/aspect/material_model/phase_field_fault.h
include/aspect/material_model/rheology/fault_friction.h
include/aspect/mesh_refinement/phase_field.h
include/aspect/particle/interpolator/voronoi_linear_reconstruction.h
include/aspect/particle/particle_domain.h
include/aspect/particle/property/crack_driving_force.h
include/aspect/particle/property/maxwell_stress.h
include/aspect/phase_field.h
include/aspect/postprocess/reconstructed_faults.h
include/aspect/reconstructed_fault/fault.h
include/aspect/reconstructed_fault/linear_performance.h
include/aspect/reconstructed_fault/manager.h
include/aspect/reconstructed_fault/sparse_coupling.h
include/aspect/reconstructed_fault/surface_system.h
include/aspect/reconstructed_fault/utilities.h
include/aspect/simulator/assemblers/implicit_constitutive_stokes.h
include/aspect/simulator/assemblers/reconstructed_fault_stokes.h
include/aspect/simulator/solver/reconstructed_fault_condensed_system.h
include/aspect/simulator/solver/reconstructed_fault_linear.h
include/aspect/simulator/solver/reconstructed_fault_nonlinear.h
include/aspect/time_stepping/reconstructed_fault.h
source/material_model/phase_field_fault.cc
source/material_model/rheology/fault_friction.cc
source/mesh_refinement/phase_field.cc
source/particle/interpolator/voronoi_linear_reconstruction.cc
source/particle/particle_domain.cc
source/particle/property/crack_driving_force.cc
source/particle/property/maxwell_stress.cc
source/postprocess/reconstructed_faults.cc
source/reconstructed_fault/fault.cc
source/reconstructed_fault/manager.cc
source/reconstructed_fault/normal_filter_internal.h
source/reconstructed_fault/surface_direct_internal.h
source/reconstructed_fault/surface_system.cc
source/reconstructed_fault/utilities.cc
source/simulator/assemblers/implicit_constitutive_stokes.cc
source/simulator/assemblers/reconstructed_fault_stokes.cc
source/simulator/phase_field.cc
source/simulator/reconstructed_fault_interface_preconditioner.h
source/simulator/reconstructed_fault_residual_audit.h
source/simulator/solver/reconstructed_fault_condensed_system.cc
source/simulator/solver/reconstructed_fault_nonlinear.cc
source/time_stepping/reconstructed_fault.cc
```

## Appendix B. Benchmark/test environment readers

These are literal `getenv` reads outside production source, grouped by file. Most select a fixture, saved data, perturbation, replay clock or diagnostic; they are **not all observational**. See each owning test/plugin before selecting it. Presence in archived investigation headers does not imply it is compiled by the maintained BP3 target. Dynamic guard lists in `execution_environment.h` reject incompatible settings and do not activate those algorithms. Launcher/compiler variables and inherited ASPECT variables are not new fault configuration.

| Reader | Literal variables |
|---|---|
| `benchmarks/reconstructed_fault/bp3/bp3.cc` | `ASPECT_BP3_LENGTH_COUPLED_DIAGNOSTIC`, `ASPECT_BP3_LENGTH_FULL_AUDIT_FROM`, `ASPECT_BP5_SHORT_TEST` |
| `benchmarks/reconstructed_fault/bp3/disturbance_diagnostic.h` | `ASPECT_DISTURBANCE_CONTROL`, `ASPECT_DISTURBANCE_DT`, `ASPECT_DISTURBANCE_EPS`, `ASPECT_DISTURBANCE_RATIO_LIMIT`, `ASPECT_DISTURBANCE_REFERENCE`, `ASPECT_DISTURBANCE_WAVELENGTH` |
| `benchmarks/reconstructed_fault/bp3/investigations/cohesion_diagnostic.h` | `ASPECT_BP3_FULLY_FRICTIONAL_REPLAY`, `ASPECT_FAULT_FROZEN_COHESION_DIAGNOSTIC` |
| `benchmarks/reconstructed_fault/bp3/investigations/history_load_diagnostic.h` | `ASPECT_BP3_HISTORY_LOAD_DIAGNOSTIC` |
| `benchmarks/reconstructed_fault/bp3/investigations/history_mechanics_diagnostic.h` | `ASPECT_BP3_EXPORT_CHECKPOINT_BULK`, `ASPECT_BP3_FROZEN_SURFACE` |
| `benchmarks/reconstructed_fault/bp3/investigations/junction_diagnostic.h` | `ASPECT_BP3_EARLY_TRACE_STEP`, `ASPECT_BP3_JUNCTION_DIAGNOSTIC`, `ASPECT_BP3_THETA_EXACT_DIAGNOSTIC`, `ASPECT_BP3_WITHIN_STEP_DIAGNOSTIC`, `ASPECT_BP3_WORK_MEASURE`, `ASPECT_FAULT_FREE_TRACE_DIAGNOSTIC`, `ASPECT_FAULT_HISTORY_FE`, `ASPECT_FAULT_NONCOMMITTING_DIAGNOSTIC`, `ASPECT_FAULT_WITHIN_STEP_STATE` |
| `benchmarks/reconstructed_fault/bp3/investigations/theta_exact_diagnostic.h` | `ASPECT_BP3_THETA_EXACT_DIAGNOSTIC`, `ASPECT_FAULT_NONCOMMITTING_DIAGNOSTIC` |
| `benchmarks/reconstructed_fault/bp3/investigations/theta_history_diagnostic.h` | `ASPECT_FAULT_THETA_HISTORY_DIAGNOSTIC` |
| `benchmarks/reconstructed_fault/bp3/investigations/within_step_diagnostic.h` | `ASPECT_BP3_COUPLED_STATE_INPUT`, `ASPECT_BP3_COUPLED_STATE_REPLAY`, `ASPECT_BP3_NOTCH_BOUNDARY_PROBE`, `ASPECT_BP3_WITHIN_STEP_DIAGNOSTIC`, `ASPECT_FAULT_FREE_TRACE_DIAGNOSTIC` |
| `benchmarks/reconstructed_fault/bp3/reference_200km/bp3.cc` | `ASPECT_BP3_BOTTOM_SOURCE_CONTINUATION`, `ASPECT_BP3_TIMESTEP_SEQUENCE`, `ASPECT_BP3_TOP_SOURCE_EXPERIMENT`, `ASPECT_FAULT_STRESS_SAMPLE_DIAGNOSTIC`, `ASPECT_IH_BOTTOM_COMPLETION_DIAGNOSTIC` |
| `benchmarks/reconstructed_fault/bp3/reference_200km/matched_resolution.h` | `ASPECT_BP3_EXACT_TARGET`, `ASPECT_BP3_EXPECTED_FAULT`, `ASPECT_BP3_MESH_ONLY`, `ASPECT_BP3_TARGET_MESH` |
| `benchmarks/reconstructed_fault/bp3/reference_200km/refresh_check.h` | `ASPECT_BP3_REFRESH_TEST` |
| `benchmarks/reconstructed_fault/bp3/reference_200km/replay_time_step.h` | `ASPECT_BP3_ADAPTIVE_REPLAY`, `ASPECT_BP3_TIMESTEP_SEQUENCE` |
| `benchmarks/reconstructed_fault/bp3/reference_200km/uniform_sliding.h` | `ASPECT_BP3_ALL_SOURCE_QPS`, `ASPECT_BP3_TOP_SOURCE_EXPERIMENT`, `ASPECT_BP3_UNIFORM_SLIDING` |
| `benchmarks/reconstructed_fault/bp3/work_replay.h` | `ASPECT_BP5_SHORT_TEST` |
| `benchmarks/reconstructed_fault/bp5/clean_stress_cycle.cc` | `ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION`, `ASPECT_STRESS_CYCLE_TRACE` |
| `benchmarks/reconstructed_fault/bp5/startup_time_step.cc` | `ASPECT_BP5_TIMESTEP_AUDIT` |
| `benchmarks/reconstructed_fault/bp5/steady_initialization.h` | `ASPECT_BP5_INITIAL_REFERENCE` |
| `benchmarks/reconstructed_fault/bp5/stress_cycle_diagnostic.cc` | `ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION`, `ASPECT_STRESS_CYCLE_TRACE`, `STRESS_TEST_EXPECT_AFFINE` |
| `benchmarks/reconstructed_fault/bp5/test_traction_projection.cc` | `ASPECT_TRACTION_PROJECTION_AUDIT` |
| `benchmarks/reconstructed_fault/uniform_shear/evolving/seam-audit/diagnostic.cc` | `ASPECT_SOURCE_DIR`, `K3_CORRECTED_PERIODIC_AUDIT` |
| `benchmarks/reconstructed_fault/uniform_shear/uniform_shear.cc` | `ASPECT_K4_STATE_GUARD` |
| `tests/bp3_length_scale_checks.h` | `ASPECT_BP3_LENGTH_STUDY`, `ASPECT_BP5_SHORT_TEST` |
| `tests/phase_field_fault_ih_cache.cc` | `ASPECT_IH_SAVED_PHASE`, `ASPECT_IH_SAVED_SURFACE` |
| `tests/phase_field_fault_surface_system.cc` | `ASPECT_FAULT_COMPARE_COUPLING`, `ASPECT_TEST_NORMAL_FILTER`, `ASPECT_TEST_REVERSED_SHEAR` |
| `tests/phase_field_periodic_domains.cc` | `K3_REQUIRE_REMOTE_PERIODIC_IMAGES` |
| `tests/reconstructed_fault_frozen_gmg.cc` | `ASPECT_FAULT_INTERFACE_MODES`, `ASPECT_FROZEN_GMG_NEWTON`, `ASPECT_FROZEN_GMG_STEP` |
| `tests/reconstructed_fault_frozen_profile.h` | `ASPECT_MECHANICAL_EXPORT_PROFILE`, `ASPECT_MECHANICAL_FROZEN_PROFILE`, `ASPECT_MECHANICAL_WIDTH_PROBE` |
| `tests/reconstructed_fault_mechanical_modes.cc` | `ASPECT_BP3_LENGTH_QUALIFICATION`, `ASPECT_BP3_LENGTH_STUDY`, `ASPECT_BP3_TIMESTEP_SEQUENCE`, `ASPECT_BP5_SHORT_TEST`, `ASPECT_MECHANICAL_DECOMPOSITION`, `ASPECT_MECHANICAL_PROBE_NEWTON`, `ASPECT_MECHANICAL_PROBE_STEP`, `ASPECT_MECHANICAL_WIDTH_PROBE` |
| `unit_tests/phase_field_fault_ih.cc` | `ASPECT_FAULT_PERFORMANCE_STATE`, `ASPECT_IH_CARTESIAN`, `ASPECT_IH_LOOKUP_BASELINE` |
| `unit_tests/reconstructed_fault.cc` | `ASPECT_SAVED_FAULT_SUPPORT_AUDIT` |

## Appendix C. Benchmark parameter declaration inventory

Literal parameter names declared in committed benchmark C++ (including historical/reference packages). Defaults and selection logic stay in their owning files. This list supplements the production controls in §5; it is not a request to migrate/remove benchmark options. Model/loading/mesh/state/filter/clock choices are numerical or setup; output/profile/audit/report choices are observational unless their implementation deliberately changes acceptance or stopping. Tests, legacy plugins and maintained `bp3/plugin` must not be loaded together indiscriminately.

| Owning file | Declared parameter names |
|---|---|
| `benchmarks/reconstructed_fault/bp3/bp3.cc` | `Weakening region length`, `Weakening initial state ratio`, `Heavy output slip interval`, `Heavy output time interval`, `Profile slip interval`, `Profile time interval`, `Graceful wall seconds`, `Last accepted step`, `Stop after first event`, `Audit full state every step`, `Mature prestress file`, `Bottom normalization completion file` |
| `benchmarks/reconstructed_fault/bp3/matched_resolution.h` | `Target cells file` |
| `benchmarks/reconstructed_fault/bp3/plugin/mesh.cc` | `Target cells file` |
| `benchmarks/reconstructed_fault/bp3/plugin/monitor.cc` | `Stationary profile file`, `Friction normal input`, `Normal filter length`, `Bottom velocity constraint`, `Local state disturbance`, `Write detailed diagnostics` |
| `benchmarks/reconstructed_fault/bp3/plugin/output.cc` | `Weakening region length`, `Heavy output slip interval`, `Heavy output time interval`, `Profile slip interval`, `Profile time interval`, `Graceful wall seconds`, `Last accepted step`, `Stop after first event`, `Audit full state every step`, `Mature prestress file`, `Bottom normalization completion file` |
| `benchmarks/reconstructed_fault/bp3/reference_200km/bp3.cc` | `Long run output`, `Heavy output slip interval`, `Heavy output time interval`, `Profile slip interval`, `Profile time interval`, `Graceful wall seconds`, `Last accepted step`, `Stop after first event`, `Fault loading configuration`, `First event run`, `Audit full state every step`, `Mature prestress file`, `Committing work-measure replay`, `Bottom normalization completion file` |
| `benchmarks/reconstructed_fault/bp3/restore_150x50.h` | `Stationary profile file`, `Friction normal input`, `Normal filter length` |
| `benchmarks/reconstructed_fault/bp5/clean_stress_cycle.cc` | `Prescribed slip rate`, `Compact output`, `Fault angle`, `Advect particles` |
| `benchmarks/reconstructed_fault/bp5/moment_cycle.cc` | `History mode` |
| `benchmarks/reconstructed_fault/bp5/normal_filter_clock.cc` | `Actual intervals file`, `Halve recorded intervals` |
| `benchmarks/reconstructed_fault/bp5/normal_stress_diagnostic.cc` | `Reference trajectory file`, `Checkpoint accepted step`, `Checkpoint physical time`, `New accepted steps`, `Wall seconds`, `Capture stress split`, `Small windows only`, `Raw every step`, `Filter experiment diagnostics`, `Friction normal input`, `Normal filter length`, `Native centerline`, `Expected clock file` |
| `benchmarks/reconstructed_fault/bp5/startup_time_step.cc` | `Maximum logarithmic state change`, `Record timestep selection` |
| `benchmarks/reconstructed_fault/bp5/stress_cycle_diagnostic.cc` | `Timestep fraction` |
| `benchmarks/reconstructed_fault/bp5/test_restart_clock.cc` | `Mode` |
| `benchmarks/reconstructed_fault/server_gmg/coarse_diagonal_probe.cc` | `Test mode` |
| `benchmarks/reconstructed_fault/uniform_shear/uniform_shear.cc` | `Evolving profile` |
