# Phase-field / RSF current status

## R6 accepted and closed

The user accepted R6c and requested this commit as the closure of R6. R6a
(`1a3b57eda`) extracted material-history diagnostics; R6b (`00ad5ce1c`)
extracted nonlinear-bound presentation; this commit records the accepted R6c
assessment and retained legacy-completion boundary. The switch inventory and
selected diagnostic extractions are complete. No further migration is implied.

The qualified post-R6 implementation is unchanged from R6b:
`build-refactor-r6b/aspect-r6b-qualified`, SHA256
`cdf7e0c58009d11dfe8c216caeb31ba6d8edc08b71360fec9fb63fcd95287596`.
R6c preservation checks cover the same source/artifacts; its environment-guard
regression passes. Numerical evidence and stated coverage limits remain those
of the accepted passes. Local refactoring/tmp files are excluded from the commit.
R7 and legacy-switch compatibility changes remain unselected. Earlier entries
retain their historical review state.


## R6c benchmark setup boundary assessment complete for review

R6b is accepted and committed as `00ad5ce1c`, using the qualified executable
`build-refactor-r6b/aspect-r6b-qualified`, SHA256
`cdf7e0c58009d11dfe8c216caeb31ba6d8edc08b71360fec9fb63fcd95287596`.
Its 170 comparisons and structural/symbol checks were rechecked before commit.

R6c inventories existing plugin setup and assesses legacy completion attachment/
application. The production integration is retained under R6 §5: available
callbacks cannot preserve cache-miss profile data/timing, legacy restart rejection
and file-error ordering. Explicit supported BP3 attachment already uses the
existing preparation callback. No production/plugin source, public API, selector,
checkpoint, ownership or scientific algorithm changes; no new interface needed.

Guidelines/roadmap/inventory and [the assessment](../../benchmarks/reconstructed_fault/refactoring_r6c/README.md)
record the boundary and deferred compatibility decision. Source, benchmark,
specification, qualified-artifact and local-edit preservation checks pass, as do
the unchanged BP3 execution-environment regression and documentation checks.
No new numerical/MPI/frozen/restart campaign for this documentation-only result;
accepted R6b evidence is reused. Earlier scientific/cache/restart gaps remain.

R6c assessment is complete, uncommitted for review. Recommended next bounded
task: R7 core-integration inventory/proposal only. Stop before R7 and any
legacy-switch migration. Earlier entries are historical.


## R6b accepted; R6c selected

The commit containing this entry records accepted R6b and its rechecked qualified
baseline. See [acceptance](../../benchmarks/reconstructed_fault/refactoring_r6b/README.md#accepted-r6b-baseline).
The user selected R6c benchmark setup boundaries. Assess existing callback state,
timing, restart reattachment and errors before any relocation. Retain production
integration where the existing interfaces cannot preserve behavior.
Earlier review entries are historical.


## R6b nonlinear-bound presentation complete for review

R6a is accepted and committed as `1a3b57eda`. The selected R6b extraction adds
three source-private formatting functions; the driver retains stream lifetime,
audit calculations, guards, MPI, decisions and observer timing. No public API,
physical ownership, selector or error-policy change. Guidance and the rolling
switch inventory describe the boundary.

Separate Release/independent builds, nine definitions and eight source/protection
checks pass. All 18 one/two-rank cases and 170 comparisons pass: exact bound rows,
fields/history, decisions, same-selector work counts, off/on physical neutrality,
silent blocked opens and complete rollback markers. Each ordinary on-run has
512 rows across seven timestep files; rollback has 16 active-bound rows.
Diagnostic-on retains its pre-existing extra 16 bulk-work/normal-filter calls
(39 to 55), without changing physical state.

Candidate: `build-refactor-r6b/aspect-r6b-qualified`, SHA256
`cdf7e0c58009d11dfe8c216caeb31ba6d8edc08b71360fec9fb63fcd95287596`.
See [review](refactor_review.md#r6b--nonlinear-bound-presentation-complete-for-review)
and [evidence](../../benchmarks/reconstructed_fault/refactoring_r6b/README.md).
R6b remains uncommitted for review. Local tmp edits and previous qualified
artifacts are preserved. Frozen evidence reused; no Debug/3D runtime, new
restart/long campaign or injected mid-write failure. No numerical/MPI defect
demonstrated. Stop before R6c/R7. Recommended next bounded task: surface
stress-sample/weak-moment CSV formatting only. Earlier entries are historical.


## R6a accepted; R6b selected

The commit containing this entry records accepted R6a and its rechecked qualified
baseline. See [acceptance](../../benchmarks/reconstructed_fault/refactoring_r6a/README.md#accepted-r6a-baseline).
R6b selects nonlinear-bound CSV and summary formatting only. Audit evaluations,
solver decisions, MPI and capture timing remain in the driver. Earlier entries
record their historical review state.

## R6a material-history diagnostics complete for review

Accepted post-R5 source is committed as `3b4ae16dd`. R6a refreshes the existing
switch inventory (87 literal selectors: 23 production and 64 research/test,
plus parameters/APIs/observers) and extracts only the two M4 history trace
streams to source-private history_diagnostics.h/.cc. Per-call candidate storage
owns the streams through publication/unwinding. Switch reads, lazy capture,
source/selected-cell admission, error policy and numerical/history/MPI control
flow stay at their existing sites. Public headers and physical ownership remain
unchanged; historical kappa CSV labels still mean eta_ve.

Release, independent 2D/3D compilation/symbols and 15 source/protection checks pass.
All 20 one/two-rank runtime cases and 258 comparisons pass: off/off and on/on
between binaries, off/on neutrality within each, nonempty exact diagnostic rows,
missing-selection, silent failed-open and accepted-update rollback. Each of six
history updates records 16,875 selected-particle and 146 continued-source rows
across ranks. Recorded physical/history values, decisions and work counts match.
The initial statistics comparison flagged path-dependent padding only; exact
numeric-token comparison passes. No simulation rerun or tolerance change.

Candidate: `build-refactor-r6a/aspect-r6a-qualified`, SHA256
`d373cb3308ecc00fb05a574975cf55f9d65facea003464b48153fc6aecf88f01`.
See [review](refactor_review.md#r6a--material-history-diagnostics-and-switch-inventory-complete-for-review)
and [inventory/evidence](../../benchmarks/reconstructed_fault/refactoring_r6a/README.md).
R6a is uncommitted for review; user's instruction file and local temporary edits
are preserved. No R6b/R6c/R7, Debug/3D runtime, long/restart or injected late-history
failure campaign. Frozen evidence reused; unrelated historical limitations remain.
Recommended next task: one R6b nonlinear-bound reporting extraction, retaining
residual audits, decisions, MPI and timing in the driver. Earlier entries are historical.



## R5 accepted; R6a selected

The commit containing this entry records accepted R5b2 and the qualified
post-R5 baseline. See [baseline evidence](../../benchmarks/reconstructed_fault/refactoring_r5b2/README.md#accepted-post-r5-baseline).
The user selected the R6a switch inventory refresh and material-history trace
extraction. No singular-fixture correction, R6b/R6c or R7 is selected.

## R5b2 candidate preparation complete for review

Accepted R5b1 is committed as `dfb7ad9f2`. R5b2 adds one private preparation
operation in surface_system.cc: the existing complete candidate copy/factor/RPE/
optional sparse-G block, byte-identical, returns unpublished owned data. The
caller retains reset/generation/diagnostic invalidation, assembly, observer and
publication order. Both backend files and all other production source remain
unchanged. No public API/layout, owner, MPI, physical or cache-policy change.

Release and independent compilation pass with unique 2D/3D helper definitions.
Six source/protection checks and 248 matched comparisons pass: 148 surface/filter/
coupled/rollback/short-BP3 checks on one/two ranks against qualified R5b1, plus
100 fresh repaired frozen AMG/GMG checks on four ranks. Inverse/filter units pass
636 assertions per rank. Frozen directions, state, RHS and work counts match;
both runs prove actual backends and reach the complete pass marker before the
intentional stop. Recorded numerical/solver/counter differences are zero.

The original singular fixture's obsolete diagnostic assertion still fails as
on R5b1; the unchanged supplemental probe verifies failed-preparation invalidation
on one/two ranks. No fixture correction or numerical fix is bundled. Initial
sandbox socket denial and symbol-parser correction are documented separately.
No Debug/3D runtime, unsupported-UMFPACK build, restart/long campaign or dedicated
native-QP/line observer test; historical scientific/cache-lifecycle gaps remain.

Candidate: `build-refactor-r5b2/aspect-r5b2-qualified`, SHA256
`edf23c823e86fe579a231110f2e2167dd929fa531996501516759d8faa432cd2`.
See [review](refactor_review.md#r5b2--surface-linearization-candidate-preparation-complete-for-review)
and [evidence](../../benchmarks/reconstructed_fault/refactoring_r5b2/README.md).
R5b2 remains uncommitted for review. Accepted artifacts and unrelated local edits
are preserved. No further extraction or R6 started. Recommended next bounded
task: separately repair the original singular-fixture diagnostic expectation,
retaining all generation/invalidation assertions. Earlier entries are historical.



## R5b1 accepted; R5b2 selected

The commit containing this entry records accepted R5b1. Its qualified executable
and evidence are preserved; see the [accepted baseline](../../benchmarks/reconstructed_fault/refactoring_r5b1/README.md#accepted-r5b1-baseline).
The user selected R5b2: one coherent surface-operation extraction, preserving
ownership, MPI and linearization lifetimes. The singular-fixture text correction
remains separate and is not selected. Earlier entries are historical.

## R5b1 surface assembly separation complete for review

R5a2 is committed as `d29115ada` (R5a1: `c3ce532be`). Its immutable qualified
binary, repaired-fixture evidence and local temporary edits are preserved.
R5b1 moves particle-domain and bulk-work backend bodies unchanged into separate
files; only private SurfaceAssembly gets a source-private shared definition.
Dispatch, configuration, SurfaceLinearization, solves/G, observer timing and
publication/failure semantics stay in surface_system.cc. No owner/public API/
layout change. The header adds one private method and the direct deal.II include
for its existing particle-index field; independent compilation exposed that
pre-existing dependency. No M1/M2 source or existing test was edited.

Full Release and three independent TUs pass; four backend/two dispatcher symbols
are unique across 2D/3D. Eight source checks and 148 matched comparisons pass.
Inverse/filter units pass 636 assertions in three cases per rank on both binaries.
Particle pressure/rate modes, explicit/reference G, bulk-work normal filtering,
restricted/stale views, coupled history/rollback and seven-output automatic BP3
comparisons match on one/two ranks. Recorded values and work counts are unchanged.

The original singular fixture fails its obsolete diagnostic-text expectation
on both binaries before checking invalidation. It is retained unchanged. A
separate probe requires the current GTTRF singular-pivot message, verifies
generation advance/stale-inverse rejection, and reaches its intentional final
failure marker on both binaries/rank counts. This is a fixture issue, not a
numerical correction. Other historical restart/scientific limitations remain.

Candidate: `build-refactor-r5b1/aspect-r5b1-qualified`, SHA256
`fd8c03ab1f1f4e363e2b69cb69d8517a35c648c5a682308004aee4453ed4910c`.
See [review](refactor_review.md#r5b1--surface-assembly-implementation-boundaries-complete-for-review)
and [evidence](../../benchmarks/reconstructed_fault/refactoring_r5b1/README.md).
Uncommitted for review; no R5b2 or manager Stokes-QP pass begun. Recommended next
bounded task: separately repair the original singular-fixture diagnostic
expectation, retaining all lifecycle assertions. Older entries below are historical.


## R5a accepted; R5b1 selected

R5a1 is committed as `c3ce532be`; accepted R5a2 is saved by the commit containing
this entry. Use `build-refactor-r5a2/aspect-r5a2-qualified` and the
[accepted baseline record](../../benchmarks/reconstructed_fault/refactoring_r5a2/README.md#accepted-r5a-baseline).
All source/artifact/protection/checkpoint and local temporary-file hashes match.
The user's active selection is R5b1: inventory the surface implementation, then
move the coherent backend group with private records and lifecycle preserved.
Stokes-QP manager work and R5b2 remain outside this task.


## R5a2 particle-projection move complete for review

Reference is accepted R5a1 `c3ce532be` and its qualified executable. The user
confirmed the reverse-interpolation contract applies to a valid cache. Both
specifications now explicitly allow collective lazy preparation on a miss and
require consistent rank entry; local validity does not establish global agreement.

Ten manager methods and two exclusive helpers moved byte-for-byte into
`manager_particle_projection.cc`. Owner/API/layout, checkpoint, cache keys,
measures, MPI and publication order are unchanged. The new TU uses the existing
independent-build mechanism to preserve baseline unity groups; no M1/M2 source
change. Stokes-QP associations and source continuation remain in place.

Release/independent builds, 20 unique 2D/3D symbols and nine source checks pass.
All 115 matched comparisons pass, including one/two-rank projection/material,
cache, coupled-history/rollback and preserved restart checks. Selected units
pass 834 assertions in 16 cases per rank on both binaries. The added two-rank
cold/warm test records one rebuild then zero, with exact values and support.
The known cohesive step-two nonconvergence is unchanged. Broader cache-lifecycle
gaps remain recorded; no demonstrated MPI correctness defect or numerical fix.

Candidate: `build-refactor-r5a2/aspect-r5a2-qualified`, SHA256
`8f696bedb006147c564f4146b96f765e88b9cd11f68f3fb120c5f0c222a1bf27`.
See [review](refactor_review.md#r5a2--particle-projection-move-complete-for-review)
and [evidence](../../benchmarks/reconstructed_fault/refactoring_r5a2/README.md)
for the initial unity dependency/harness failures and successful checks. Local
edits and reference evidence are preserved. Uncommitted for review; stop before
another pass. Proposed next task: Stokes-QP association inventory/cache-lifetime
assessment only. Entries below are historical, including the resolved contract
question from the initial R5a2 inventory.


## R5a1 committed; R5a2 inventory complete, contract clarification needed

The accepted R5a1 source, guidance and verification harness are committed as
`c3ce532be86765c7c9edcacb2f44377ff62820e1`, after fixture commit `aea2a80b0`. Use `build-refactor-r5a1/aspect-r5a1-qualified`
and the manifest fingerprints in the [accepted baseline](../../benchmarks/reconstructed_fault/refactoring_r5a1/README.md#accepted-r5a1-baseline).
All qualified source/artifact and protection hashes match; no runtime campaign
was repeated. Local temporary review files remain excluded. The user's next
selection follows the recommendation: particle-projection inventory, cache-
lifetime assessment and one coherent move proposal only, before implementation.

The [R5a2 inventory](refactor_review.md#r5a2--particle-projection-inventory-and-move-proposal-no-implementation)
proposes moving ten particle-projection members and two exclusive helpers into
`manager_particle_projection.cc`, preserving all storage, APIs and cache/MPI
behavior. Stokes-QP associations and source continuation remain separate. No
source was edited and no new runtime campaign was run. The assessment found a
contract discrepancy: both specifications describe reverse interpolation as
search/MPI-free, while its public method rebuilds collectively on a stale cache.
Clarify this before implementation; no behavior or specification was silently
changed. Cache-lifetime test gaps and the focused verification proposal are
recorded in the review. Earlier entries below describe their original review state.


## R5a1 manager slip-rate organization complete for review

Accepted fixture repair is committed as `aea2a80b0` after R4c `fa6013678`.
The unchanged qualified R4c executable and repaired-fixture manifests are the
reference; no unrelated R4 runtime campaign was repeated.

R5a1 moves all 16 manager slip-rate definitions byte-for-byte into
`source/reconstructed_fault/manager_slip_rate.cc`. The manager still owns all
state. Headers/API/layout, geometry sizing, generic registration, archive and
restart rebuild (including the R3 prescribed-map sizing fix) are unchanged.
The responsibility/state table was recorded before editing and verified after.
No cache/projection, boundary-contact or surface-system implementation changed.

Fresh Release/link, independent original/new TUs, and 32 unique 2D/3D symbols
pass. Seven source checks and 31 matched comparisons pass. Existing lifecycle,
prescribed/absolute-bound/restart units pass 20,130 assertions in seven cases
per rank on one/two ranks for reference and candidate. Small coupled-history
and accepted-update rollback comparisons match the accepted R4c evidence exactly.
Both binaries restore the preserved one-rank cohesive checkpoint and retain the
same known step-two nonconvergence with exact phase-field/coupled traces; no
numerical workaround was made. All 17 recorded outcomes are expected.

Candidate: `build-refactor-r5a1/aspect-r5a1-qualified`, SHA256
`4159f38bb530c97bed3fddb12009cd892b3428fb28c1e28530a230f050146ec6`.
R5a1 remains uncommitted for review. See [review](refactor_review.md) and
[evidence](../../benchmarks/reconstructed_fault/refactoring_r5a1/README.md).
Projection/cache reorganization and surface-system implementation have not begun.
Next proposed bounded task: R5a2 particle-projection inventory/cache-lifetime
assessment and one coherent move proposal. Earlier entries are historical.


## Frozen fixture accepted; R5a1 selected

The accepted fixture repair is saved in its own commit after R4c `fa6013678`.
Use the unchanged qualified R4c executable and the accepted repaired-fixture
[manifest record](../../benchmarks/reconstructed_fault/frozen_gmg_repair/README.md#accepted-fixture-baseline).
All reference artifacts and local temporary reviews are preserved. Verification
is reused without another runtime campaign. The next selected task is only
R5a1 manager slip-rate definition movement under the revised R5 instructions;
projection/cache and surface-system implementation remain outside this task.


## R4c committed; separate frozen AMG/GMG fixture repaired

Accepted R4c is commit `fa6013678b525b189a1d27ef08465d4a6ef263f6`. Its qualified
baseline remains `build-refactor-r4c/aspect-r4c-verified`, SHA256
`c6811cbd877ff56af9113dc2f110cfd9a8b71998ea63f8e249ff94b292eeed85`.
The accepted source commit excludes the subsequent fixture repair and local
`refactoring/tmp/` edits, which remain preserved.

The separate test-only repair selects the existing `default solver` setting:
mesh construction builds the hierarchy, then reconstructed-fault selection
resolves to AMG. The observer asserts AMG identity and hierarchy availability;
its second solve explicitly applies the GMG velocity cycle on every rank.
No production source/interface, numerical parameter or obsolete switch changed.
The historical launcher also uses the retained research replay plugin and a
new output directory.

Matched four-rank pre-R4c (`983d57e28`) and post-R4c immutable executables both
reach the complete pass marker before the intentional stop. All 226 checks pass:
exact physical snapshots, RHS, operator probes, both returned directions,
fields, decisions and counters. AMG/GMG each take 17 iterations with fresh
residuals `4.4068965535591382e-4` / `4.8708263397949716e-4`, below the unchanged
`1.2258892187472356e-3` target. No step-2 state is published. Historical AMG field
and decision comparisons also match; its failed observer lacks exactly the 22
operator applications now made by the completed GMG solve and preservation audits.

The fixture repair remains uncommitted for review, separate from R4c. See the
[review](refactor_review.md) and [reproduction/evidence](../../benchmarks/reconstructed_fault/frozen_gmg_repair/README.md).
Only this frozen Q2/2-D/four-rank case and prefix are qualified; no broader GMG,
restart or performance claim is made. Stop before R5. Next proposed task: review
this fixture repair, then select the R5 inventory separately. Earlier status
entries are historical checkpoints.


## R4c accepted baseline; separate frozen-fixture repair selected

The accepted R4c source, documentation and harness are committed together over
R4b `983d57e28`. The qualified baseline is
`build-refactor-r4c/aspect-r4c-verified`, SHA256
`c6811cbd877ff56af9113dc2f110cfd9a8b71998ea63f8e249ff94b292eeed85`.
Source/artifact manifest fingerprints and the unchanged verification limitation
are recorded in the [R4c evidence](../../benchmarks/reconstructed_fault/refactoring_r4c/README.md).
Local temporary review edits are excluded. The user selected only the separate
historical frozen AMG/GMG fixture repair before R5; no production redesign or
further refactoring stage is authorized. Earlier entries retain historical status.


## R4c shared Schur construction complete; ready for review

Against accepted R4b `983d57e28`, one source-private constructor helper replaces
the two BFBT/inverse-weighted-mass branches. It preserves conditional lumped-mass
access, ordinary melt pressure-block selection and the fault/melt restriction.
Both callers are otherwise byte-identical; solver policies, MPI/observer order,
ownership and `simulator.h` are unchanged. Nine source/protection checks pass.

Fresh Release/link, independent ordinary/coupled/header compilation and 2D/3D
symbols pass. All 31 coupled and 22 ordinary comparisons pass, including actual
GMG, BFBT, melt and failure histories. Units pass 20,149 assertions/16 cases per
rank on one/two ranks. Four BP3 legacy/automatic runs match all 372 field/history
groups and 28 counter/decision checks exactly.

The historical four-rank frozen probe's AMG results match exactly, but its GMG
half remains blocked on both baseline and candidate: the AMG-selected mesh has
no multigrid hierarchy and the archive's old hierarchy switch is no longer
consumed. No numerical change or assertion relaxation was made. The corrected
replay-plugin pairing and failure diagnosis are recorded in the
[review](refactor_review.md) and [evidence](../../benchmarks/reconstructed_fault/refactoring_r4c/README.md).
Candidate: `build-refactor-r4c/aspect-r4c-verified`, with that explicit limitation.
Changes remain uncommitted for review; local edits/reference artifacts are
preserved. Stop before further refactoring. Proposed next task: separately
modernize the historical frozen-comparison fixture. Earlier entries are
historical checkpoints.

## R4b committed; R4c inventory/proposal ready for review

R4b is committed as `983d57e28`, including its accepted helper assessment and
qualified-baseline record. Local `refactoring/tmp/` files remain excluded.
Use `build-refactor-r4b-residual/aspect-r4b-residual-qualified` and its recorded
source/artifact manifests for subsequent comparisons.

The [R4c inventory](refactor_review.md) proposes one source-private operation:
construct the existing BFBT or inverse-weighted-mass Schur wrapper. Preserve
conditional access to BFBT-only lumped mass, caller-selected matrix blocks and
existing ownership. Keep velocity wrappers, AMG/GMG dispatch, pressure handling,
stopping/restarts and observers at their current callers. The inventory also
records why Simulator's nested Linearization parameter needs the condensed
header; removing that dependency is separate from this proposal.

Only documentation changed after the R4b commit. Existing future verification
cases are identified; no new build/runtime campaign was run. Stop before R4c
implementation. Proposed next selection: the Schur-construction operation only.

## R4b accepted baseline

Both focused extractions and the recommendation to retain the existing driver
without a whole-iteration helper are accepted. The accepted R4b source,
documentation and verification scripts are saved together over R4a `0c7ed1a0b`.
Use `build-refactor-r4b-residual/aspect-r4b-residual-qualified` (SHA256
`6ccdcf81b65ad5cbe0c949cdcd45da6332c3949354e0a034dcc830fa889fe7a7`)
as the qualified reference for subsequent work. Its 5,910 source entries and
21 executable/plugin/input entries were rechecked without mismatches before
commit; manifest fingerprints are recorded in the
[baseline evidence](../../benchmarks/reconstructed_fault/refactoring_r4b_residual/README.md).
Existing verification outcomes remain unchanged. Local temporary files are
excluded from the commit; no new build or runtime campaign was run. Historical
entries below preserve the original review/commit status at each stage.

## R4b extractions accepted; iteration-helper assessment ready for review

Both R4b subpasses are accepted and remain uncommitted over `0c7ed1a0b`.
The small-iteration-helper assessment recommends keeping the two focused
operations and the existing driver: early direction gating and later convergence
publication divide the iteration, while the restricted inverse/linearization,
active mask, directions and scales must survive across those boundaries.
A whole-step helper would need broader state transfer or take over driver
decisions. A line-search-only extraction would be a separate proposal.

Only the status, roadmap and [review](refactor_review.md) changed in this task.
All 5,910 accepted source entries and the qualified second-subpass executable
hash match. No new build/runtime tests were run; accepted numerical evidence
is unchanged. Local edits and historical reports are preserved. Stop for review.
Proposed next task: R4c duplication inventory and a minimal shared-preconditioner
proposal, before any implementation. R4c has not begun. Earlier entries below
are historical checkpoints.

## R4b second subpass complete; ready for review

The first R4b condensed-solve extraction is accepted. Its qualified executable
and accepted uncommitted source patch over `0c7ed1a0b` are the reference for this
second subpass. The existing coupled residual lambda is now the private
`Simulator::evaluate_reconstructed_fault_coupled_residual()` operation with a
two-field result. All five calls remain in place. The pre-edit contract was
verified afterward: exact absolute trial V, bulk/surface evaluation order and
guarded trial rollback are preserved. Assembly flags, linearization point and
RHS retain their existing side effects; driver/audit cleanup remains unchanged.
The accepted condensed solve is byte-identical. No public API or owner changes.

Fresh Release/link, independent compilation, candidate plugins and 2D/3D
instantiations pass. Units pass 20,149 assertions/16 cases per rank on one/two
ranks. All 31 focused comparisons pass, including residual audits, pressure and
history, accepted-update rollback, expected exhaustion and one-rank GMG. Four
short legacy/automatic BP3 trajectories match in all 372 field/history groups
and 28 cache/work/solver-decision checks. Maximum numerical difference is zero.
All 20 selected build/runtime invocations have their expected outcomes; the
two deliberate exhaustion failures retain the original budget and rollback.

Qualified candidate: `build-refactor-r4b-residual/aspect-r4b-residual-qualified`.
Both R4b subpasses remain uncommitted, with an incremental second-subpass patch
recorded in [residual evidence](../../benchmarks/reconstructed_fault/refactoring_r4b_residual/README.md).
See the [review](refactor_review.md). Reference artifacts, local edits and earlier
reports are preserved. Ordinary/restart evidence is reused; no new Debug, 3D or
production/performance campaign, and historical Stage-J/cohesive limitations
remain separate. Stop for review. Proposed next task: assess the optional
iteration-helper boundary without implementing it. No further extraction or
R4c work has begun. Earlier entries below are historical review checkpoints.

## R4b first subpass complete; ready for review

R4a is committed as `0c7ed1a0b`. Against its qualified executable and source,
`Simulator::solve_reconstructed_fault_condensed_system()` now holds the existing
condensed linear solve, pressure compatibility calculation and Schur/AMG/GMG
wrappers. A private five-scalar input record makes residual scales explicit.
Matrix/preconditioner assembly, linearization, active sets, trial evaluation,
convergence, history publication and failure restoration retain their driver
locations. No public interface, persistent owner, scientific algorithm or
checkpoint format changes. The pre-edit contract was checked afterward;
all six source/protection checks pass.

Fresh Release/link, independent compilation, candidate plugins and 2D/3D
instantiations pass. Units pass 20,149 assertions/16 cases per rank on one/two
ranks. All 31 focused comparisons pass: affine/fresh residual diagnostics,
pressure/history, accepted-update rollback, intentional one-iteration-budget
exhaustion, and the existing one-rank GMG-Q1 case. Four BP3 legacy/automatic
trajectories match in 372 exact field/history groups, 24 cache/work checks and
four detailed solver-decision comparisons. No numerical difference was observed.

Candidate: uncommitted first R4b subpass over `0c7ed1a0b`, qualified executable
`build-refactor-r4b-linear/aspect-r4b-linear-qualified`. See the
[review](refactor_review.md) and [R4b evidence](../../benchmarks/reconstructed_fault/refactoring_r4b_linear/README.md).
The initial include-order compile issue was resolved without M3/M4 edits.
Ordinary/restart evidence is reused; no Debug/3D/production/performance campaign
was run, and historical Stage-J/cohesive failures remain separate.
Stop for review. Proposed next selection: the existing non-committing trial
residual operation, with its contract recorded before editing. Remaining R4b
operations and R4c are unimplemented. Earlier stage records follow.

## R4a accepted; committed for the first R4b extraction

The revised [R4 instructions](refactoring/codex_R4_instructions.md) are recorded
in the standing guidance and roadmap, with driver/iteration/linear-solve
responsibilities and separately selected R4a/R4b/R4c passes. HEAD `d7b88b25e`
matches all 5,907 accepted post-R3/Maxwell source-manifest entries. The reference
executable, plugins, outputs and local instruction/tmp files are preserved.

The complete `Simulator::solve_reconstructed_fault_stokes()` body and exclusive
helper move unchanged into `source/simulator/solver/reconstructed_fault_stokes.cc`.
Shared Stokes/Schur declarations and templates move into source-private
`solver/stokes_operators.h`; non-inline StokesBlock definitions stay in the
general solver. There is no public API, ownership or numerical change. Independent
compilation required an explicit `newton.h` include. All 55 unity groups remain
unchanged, and both affected member sets have 2D/3D instantiations.

Fresh Release/link, independent compilation and candidate plugin builds pass.
Units pass 20,149 assertions/16 cases per rank on one/two ranks. Accepted-Newton
rollback passes on both ranks; ordinary feature-disabled AMG fields/statistics
and solver decisions match. Four short legacy/automatic BP3 trajectories match
in 372 field/history groups (zero differences), 24 cache/work checks and four
solver-decision comparisons. All 16 focused comparisons pass.

Candidate: accepted R4a changes over `d7b88b25e`, qualified executable
`build-refactor-r4a/aspect-r4a-qualified`. See the [review](refactor_review.md) and
[R4a evidence](../../benchmarks/reconstructed_fault/refactoring_r4a/README.md).
Restart evidence is reused because initialization/restoration semantics are
unchanged; fresh rollback covers the moved failure path. Known Stage-J/cohesive
failures remain separate, with no tolerance or parameter changes. No 3D,
Debug, full earthquake or large performance run was performed.
Proposed next selection: the private condensed-solve/preconditioner operation,
with its input/mutation/MPI/lifetime contract recorded first. Stop for review;
R4b/R4c remain unimplemented. The following entries are historical stage records.

## Maxwell naming cleanup complete; frozen-stress implementation retained

The Maxwell viscoelastic viscosity is now named `eta_ve` throughout fault C++
coefficients, responses, caches and callers. Its stable `expm1` evaluation is
unchanged. Existing CSV columns retain `kappa`; the design/specification state
the notation equivalence. `evaluate_frozen_maxwell_stress` is restored exactly to
its original direct beta-times-old-stress implementation at the user's request.

Release/plugin builds and existing unit/frozen-stress/temperature tests pass on
one/two ranks (20,816 assertions/26 cases per rank). Short legacy/automatic BP3
and the cross-rank old-checkpoint continuation match R3b: 405 exact field groups,
30 matching cache/work checks and 12 frozen/temperature comparisons. No physical
parameter, tolerance or history behavior changed. R3b and local edits are preserved.
See [cleanup evidence](../../benchmarks/reconstructed_fault/maxwell_cleanup/README.md).
Qualified candidate: `build-refactor-r3b/aspect-maxwell-qualified`. R3b and this
cleanup are saved in separate local commits (R3b: `4d2f14f53`). Use this candidate and the cleanup
source manifest as the R4 reference; no next refactoring stage has begun.

## R3b complete; review pending

The R3a ownership/lifecycle table and full report in `refactor_review.md` were
already intact and matched commit `bef79b31a`; no restoration was necessary.
They remain unchanged. Local documentation moves and temporary files are preserved.

Against corrected R3a, the history driver now calls four private operations:
accepted-state sampling, candidate construction, validation and publication.
Two call-local scratch buffers add no persistent owner or checkpoint state.
Numerical expressions, intermediate collective error checks, cohesive projection
before H, validation/diagnostic order and terminal write order are preserved.
Timestep-zero, mature/frozen and cohesive paths remain distinct. Solver acceptance
and rollback, both R2 files and the restart correction are unchanged.

Fresh Release and independent compilation pass; all affected operations have
2D/3D definitions. Units pass 20,828 assertions/28 cases per rank on one/two ranks.
Temperature, frozen stress and accepted-Newton rollback pass. All 40 lifecycle,
15 cohesive and 30 BP3 cache/work comparisons match; all 405 BP3 field groups
are exact. Original Stage-J pressure incompatibility and cohesive step-two
nonconvergence remain unchanged, with identical solver decisions. No tolerances,
physical parameters or history assertions changed. No 3D or long/full BP3 run
was performed, and the prior cross-rank cohesive observer limitation remains.

Qualified candidate: `build-refactor-r3b/aspect-r3b-qualified`. See the
[review](refactor_review.md) and [evidence](../../benchmarks/reconstructed_fault/refactoring_r3b/README.md).
R3b is saved in its own local commit. Proposed next task is separately selected R4a
solver-method relocation; no R4 work has begun. The entries below are historical
checkpoints and review boundaries.

## Restart correction complete; corrected R3a reference ready

The separately authorized correction rebuilds the transient prescribed-rate map
layout after deserialization. Checkpoint format is unchanged; callers still
reapply prescribed conditions. The new archive/trial regression fails with the
old manager and passes with the fix. Restart/prescribed/checkpoint/Stage-I tests
pass on one/two ranks (20,173 assertions/17 cases per rank).

Cohesive restart now passes the original restored-state assertions and reaches
step two without SIGSEGV. Fresh and restarted step-two solver decisions and
residuals match exactly; both retain the existing nonlinear nonconvergence.
Accepted step-one fields/checkpoint payloads match. No physical parameter,
solver tolerance or history assertion changed. Four short BP3 legacy/automatic
runs and the old-checkpoint cross-rank continuation match R3a in all 405 field
groups and 30 cache/work checks. Nonempty prescribed-row reapplication is
covered by the focused unit regression; the short BP3 fixture reapplies empty
maps, and the full 200-km prescribed-row run was not repeated.

R3a/investigation is committed as `8490431b5`; the correction is separate.
Use `build-restart-fix/aspect-r3a-corrected-qualified` and the source/artifact
manifest in [restart correction evidence](../../benchmarks/reconstructed_fault/restart_fix/README.md)
as the reference for a subsequently selected R3b. R3b remains unimplemented.
The step-two convergence and cross-rank cohesive observer limitations remain
separate; proposed next task is R3b lifecycle extraction with existing ordering,
publication and ownership preserved. The following diagnosis records pre-fix
observations, not the current correction status.

## Separate cohesive restart diagnosis (historical pre-fix record)

The existing crash reproduces on one rank with original restored-history
assertions passing, and with the optional restored-fingerprint observer
disconnected. GDB locates SIGSEGV at `manager.cc:1077`: an unconditional access
to `prescribed_slip_rates[0]`. The manager contains one fault/eight vertices and
valid, identical committed/current/trial/candidate V; solve and trial flags are
both active. The prescribed-rate container has outer size zero immediately after
deserialization and at the failing call. Restart rebuild omits its transient
per-fault initialization; BP3's setup explicitly recreates it and masks the defect.

Proposed smallest correction: add
`prescribed_slip_rates.assign(reconstructed_faults.size(), {});` to the other
transient resets in `rebuild_after_deserialization()`, plus an archive-round-trip
regression that opens an absolute-value trial without prescribing any rows.
Neither correction nor regression is implemented. Production source, original
history assertions, physical parameters, tolerances and original checkpoint data
are unchanged. The separate cross-rank observer check failure and subsequent
cohesive convergence limitation remain unresolved.

See [cause, state table and backtrace](refactor_review.md) and
[diagnostic evidence](../../benchmarks/reconstructed_fault/restart_investigation/README.md).
Stop for review before the targeted correctness fix; R3b remains unimplemented.

## R3a — constitutive/history relocation (complete; review pending)

Against committed R2b `06b70740f`, nine constitutive methods and ten history/setup/
preparation/query methods moved into private implementation files
`phase_field_fault/constitutive.cc` and `history.cc`. All 19 method bodies and six
helpers are exact. Header declarations, both R2 files, ownership, solver acceptance,
initialization/publication/rollback ordering, MPI and checkpoint behavior are
unchanged. The ownership/lifecycle table was written before source edits and
rechecked afterward; see [the rolling review](refactor_review.md).

The Release build passes. All three affected files compile without unity/PCH;
all 38 required 2D/3D moved-member definitions are present. CMake keeps the new
files independent, preserving the original 55 unity groups. This avoids a latent
M2 include dependency exposed by regrouping, without modifying M2. Units
(20,791 assertions/25 cases per rank), temperature/frozen-stress and actual
accepted-Newton rollback checks pass on one/two ranks. Four short legacy/automatic
BP3 runs and a cross-rank reference-checkpoint continuation pass: 405 exact field
groups and 30 matching cache/work checks. Another 48 lifecycle and five accepted
cohesive-checkpoint comparisons match exactly. All 5,835 other entry source/test
files and 31 reference artifacts are unchanged; local documentation edits remain.

Reference failures are preserved: original Stage J has pressure incompatibility;
the supplemental open-top case verifies one physical Theta/H update then fails
at step two. Its checkpoint restores correctly but subsequent trial setup segfaults
in both binaries. This is not successful cohesive restart qualification; no fix
or tolerance change is included. Mature/frozen BP3 restart passes. No 3D simulation
or long production run was performed.

Evidence and commands: [R3a harness](../../benchmarks/reconstructed_fault/refactoring_r3a/README.md).
Qualified executable: `build-refactor-r3a/aspect-r3a-qualified`. R3b is not
implemented. Stop for review; proposed next task is separately selected R3b
lifecycle extraction under the recorded ordering and ownership constraints.

The entries below describe earlier checkpoints and review boundaries.

Local checkpoint (September 30, 2026): the completed limiter, normalization
refactoring, prescribed boundary completion, test harnesses and guidance are
saved in separate local commits at the user's request. Generated evidence and
binaries remain local. Statements below that no commit was made describe the
earlier verification sessions. Review remains pending; no next pass is selected.


## R2b proposal 3 — private cache-reuse decision (complete; review pending)

The selected extraction is complete against the qualified proposal-2 baseline.
`PhaseFieldFault::prepare_normalization_reuse(previous_cache_valid) const` in
`phase_field_fault/normalization.cc` captures the existing inputs, determines
composition independence and performs the identical collective reuse decision.
Invalidation, automatic-completion qualification, composition projection,
counters, frozen-restart validation and final publication remain in the caller.
The moved 48-line block is exact; mechanically restoring it reconstructs the
proposal-2 implementation byte-for-byte. No missing cache dependency was
identified or correctness fix included. No public API or ownership changed.

The Release build and all 30 matched runtime invocations passed. Existing hit/
miss, failure/recovery, restart-invalidator and lookup tests cover one/two ranks;
short legacy and automatic BP3 comparisons and an automatic one-to-two-rank
restart also pass. All 405 field groups and 96 lifecycle/work checks are exact,
including solver decisions and cache/work counters; elapsed time is excluded.
The two production files changed are `normalization.cc` and the private header
declarations in `phase_field_fault.h`. All other 5,834 entry source/header/test/
plugin files, the user's local edits, M1/M2 and proposal-2 reference artifacts
are preserved. No commit was made.

See [review](refactor_review.md) and
[reproduction/evidence](../../benchmarks/reconstructed_fault/refactoring_r2b_cache/README.md).
The initial compile omission and comparison-script cold-restart coverage
assumption are recorded separately; neither changed scientific behavior or
relaxed equality. Long production runs and unrelated qualifications were not
repeated. Stop for review. A bounded R3 assessment is a possible next task only
after user selection; no R3 or phase-boundary implementation has begun.

The following sections retain earlier review boundaries and their evidence.


## R2b proposal 2 — prescribed boundary completion (complete; review pending)

The selected [boundary-completion task](refactoring/codex_boundary_completion_instructions.md)
is complete, starting from the qualified geometry-preparation extraction.
The legacy implementation was first extracted privately with exact one/two-rank
BP3 agreement across 186 field groups. Its executable and source snapshots are
preserved separately from the subsequent behavior extension.

`Boundary completion = automatic prescribed` now detects contacts per fault,
verifies compatible fully prescribed frozen Q1 phase data, and pairs exterior
ghost-Q1 normalization with the existing physical endpoint mechanical source and
bulk-work surface measure. The default remains `legacy`; the final executable's
legacy regression is exact. Geometry belongs to M3, constitutive continuation
to M4, and mechanical residuals/derivatives to M5. No checkpoint schema, M1/CPDI,
M2/evolution, limiter edit, cell-preparation extraction or normalization value-
reuse criterion changed. Proposal 3 and other refactoring stages remain unselected.

The final build, CTest, 56 contact assertions per rank, ten small cases, free-endpoint
K/B/G checks, short BP3 on one/two ranks, and a one-to-two-rank restart passed.
Corner/tangent/unrepresented crossing/H-driven/undefined-material cases reject
with intended diagnostics. Automatic nodal I_h differs from legacy by at most
1.44e-13 relative; solver decisions match. Small oblique/multiple-fault I_h is
exact across rank counts; reversing the polyline changes it by less than 8.58e-16.
The interior diffuse-support-touching case gets no fictitious continuation.

Supported scope is deliberately bounded: transverse 2-D endpoints on planar
nonperiodic Box faces, straight terminal support, an aligned uniform local ghost
lattice, constant core per fault and identical profile/degradation coefficients.
Curves elsewhere, either/both ends and independent faults are qualified.
Overlapping diffuse supports, unsupported exterior data, deformation, 3D and
unqualified/evolving phase fields remain unsupported. Adaptive refinement
invalidation is implemented but not runtime-qualified. Full prescription is
established through actual phase constraints and nodal profile verification,
not inferred from H or the frozen flag.

See [review report and contact table](refactor_review.md) and
[reproduction/evidence](../../benchmarks/reconstructed_fault/refactoring_boundary/README.md).
Exploratory fixture failures and the existing 45-degree stationary-H roundoff
assertion are recorded separately. No assertion or expected result was relaxed.
Stop for review. The next separately selectable design task is the documented
evolution-compatible phase-boundary treatment, not another automatic extraction.

The following sections describe earlier review boundaries and scientific work.

## Refactoring R2b cell-profile geometry extraction — September 29

The user selected proposal 1 only: a private PhaseFieldFault operation in
`source/material_model/phase_field_fault/normalization.cc` prepares cell-profile
geometry. M4 ownership, backend admission/fallback, clipping/shared-face rules,
interval order, DoF indices, cache criteria/invalidation and MPI order stay
unchanged. Phase sampling, material evaluation and quadrature remain in the
integration operation. R2b is complete and ready for review. A fresh Release
build and matching plugins passed, as did all 15 selected runtime invocations,
including cell/remote cases and warm traversal reuse on one/two ranks. Both
six-step BP3 trajectories match preserved R2a and qualified R1 exactly: 186
field groups and 22 lifecycle/work checks per reference, with zero numerical
or work-counter differences (elapsed time excluded). All 5,810 other source/
header/test files match entry hashes. Evidence is in `build-refactor-r2b/` and
`benchmarks/reconstructed_fault/refactoring_r2b/`. No next pass is authorized.

Proposals 2 (boundary-completion extraction) and 3 (value-cache extraction)
remain unselected and unchanged. Boundary-completion generalization is recorded
as a deferred design task requiring separate review. The R2a/R1 entries below
are historical; see the [rolling report](refactor_review.md) for pass evidence.
No new restart/rollback, cell-cache AMR/fallback or 3D simulation qualification
is claimed. Stop for review; the proposed next task is a design-only discussion
of boundary-completion generalization, only if selected.

## Refactoring R2a normalization relocation — September 29

R1 is completed; the user selected only R2a under
[codex_R2a_instructions.md](refactoring/codex_R2a_instructions.md).
R2a is complete and ready for review: complete normalization definitions moved into
`source/material_model/phase_field_fault/normalization.cc`, retaining M4
ownership, the existing class/header/state and the sole plugin registration.
M1/M2 remain frozen. At R2a completion, R2b, helper extraction and R3 were not selected.

Comparison uses the preserved R1 executable/plugins/fixtures/checkpoint in
`build-refactor-baseline/` and `benchmarks/reconstructed_fault/refactoring_r1/`.
Their hashes match. The candidate additionally retains the separately validated
limiter zero-rejection change; all selected R1 inputs use positive/default
settings. Existing lifecycle initialization and output-filter failures keep
their recorded dispositions. Candidate builds and evidence are separate in
`build-refactor-r2a/` and `benchmarks/reconstructed_fault/refactoring_r2a/`.
See the [rolling report](refactor_review.md) for verification and review status.

The main material file shrank from 3,798 to 2,199 lines; the new file has 1,698
lines. All twelve methods and four exclusive helpers are byte-identical to
their entry definitions. Only includes, explicit 2D/3D member instantiations
and necessary shared-helper linkage were added; no public header/API changed.
The shared history-error helper retains one definition in the original file.
Normal Release and focused non-unity/non-PCH compilation/linking passed.
Normalization accuracy/cache, both backends on 1/2 ranks, no-composition,
rollback and ordinary particle smoke checks passed. Both six-step BP3 runs
and the continuation from R1 step 4 match the corresponding R1 physical fields,
histories and solver records exactly: 219 field groups, zero differences.
All 19 lifecycle/cache/rollback/statistics comparisons also passed. Reference
artifacts/checkpoints and pre-existing local source edits remain unchanged.
This is translation-unit separation, not architectural decoupling. Stop for
review; proposed next task is a bounded R2b assessment of model-independent
integration/sampling/projection boundaries, only if the user selects it.

## Refactoring R1 baseline qualification and limiter follow-up — September 29 (historical)

R0 is accepted as a dependency audit. The user selected documentation boundary
updates and R1 on unchanged source `3af7c2aa2590aa8e39eaf2a1e46df0bc6c9d9700`.
M1/M2 are frozen; M3–M5 are the scope for later selected passes. R1 execution is
complete with pre-existing failures; R2a progress is recorded above.
R1 left production source and the scientific worktree `../aspect/` unchanged.
Release reference builds, fixture inputs, logs and checkpoints are retained in
`build-refactor-baseline/` and `benchmarks/reconstructed_fault/refactoring_r1/`.
Normalization accuracy/cache and supplemental lifecycle checks pass on 1/2
ranks and both backends; original lifecycle inputs exhaust initialization.
Mechanics and forced rollback behavior pass (rollback CTest filters are stale).
Six-step BP3 and a step-4 split restart pass, with exact same-rank final fields,
histories and solver records. Ordinary particle restart data also matches
exactly, although its saved log/statistics comparison fails. Cross-rank BP3
differences are measured separately in the report.

After R1, the user confirmed the numeric-parser design and requested corrected
comments/documentation, then explicitly required an assertion rejecting zero.
`Patterns::Double(0.)` stays, followed by `AssertThrow(value > 0)`. The default
maximum finite double still disables the limiter; literal `infinity` remains
unsupported. Parameter help and both specifications now describe this contract.
The old recommendation to restore the infinity alias is withdrawn. Parameter
tests and the existing integration test now use the numeric disabled sentinel;
no numerical tolerance changed. R1 artifacts are preserved in their original
state, including the evidence that zero was previously accepted. Validation in
`build-limiter-validation/` passed all 15 parameter assertions and the existing
limiter integration fixture on one/two ranks; R1 reference hashes are unchanged.
At the end of that follow-up, the other fixture failures still required review
and no extraction had started. The subsequently selected R2a result is above.
See the selected
[roadmap](refactoring/refactoring_plan.md),
[instruction](refactoring/codex_decoupling_and_R1_instructions.md), and
[rolling report](refactor_review.md) for current scope and evidence.

## Local rotated-bottom A/B experiment — September 28

The user-requested `bp3_rotated_bottom_local_test.md` task is complete in
`benchmarks/reconstructed_fault/bp3/rotated_bottom_local/`. See its `REPORT.md`,
`README.md`, resolved `output-*/parameters.prm`, compact `analysis/` CSVs and
PNG/PDF figures. Base commit is `0fc1ce782`; the source checkpoint retains the
opt-in plugin, isolated test inputs/tools and reports. Generated fixtures,
builds and run outputs remain local. No core source, production PRM or existing
checkpoint was changed. The user explicitly retained the current production
BP3 model and requested the new boundary condition as an option only.

The maintained BP3 plugin now has opt-in `Bottom velocity constraint = fault
parallel`, default `full`. It pairs bottom Q2 velocity support points, preserves
side corners, checks existing hanging relations, and inserts the tangent-only
row before constraint closure/sparsity rebuilding. It mirrors ASPECT's physical
or homogeneous side-corner lift, including initial-field setup. The existing
coupled Newton physical-lift/homogenization path and stress-history volume
assembly are reused unchanged. B uses natural zero complementary perturbation
traction, retaining the 50 MPa fault background and the common natural top.

Executed: uniform-creep A/B checks on one rank; B on two ranks; 10-step disturbed
A/B pair with dt=2e5 s; sole follow-up is paired dt=1e5 s, 20 steps, same final
time 2e6 s. All use the verified 6275-cell, 58917-Stokes-DoF ell20/Q2/LLS fixture,
AMG and one thread per rank. Constraint, normal-freedom, flux, actual normal
feedback, exponential-state and fresh linear-residual audits pass. B inserts
123 independent rows and retains two corners; no boundary hanging rows occur
in this fixture. Default plugin build and parse-only production validation pass,
as do 157 exact restored-model C++ comparisons. Local compiler: GNU 16.2.1,
deal.II 9.6.2, OpenMPI 5.0.6; no Intel/GMG/restart qualification is implied.

Result: B reduces the deep 200 m raw-normal peak increment by 12.29% (10.16%
at half dt), but exact arc-length RMS only by 3.38% (2.09% at half dt). Individual
RMS changes under timestep halving are larger, so the area-wide benefit is not
resolved beyond temporal error. The adjacent negative raw-normal excursion
increases; deep creep persists and flux balances. This does not establish a
production stress-concentration fix. Primary A/B wall times were 71/67 s;
half-step runs 131/135 s. All setup-failure artifacts were preserved and the
mandatory checks corrected rather than bypassed; details are in the report.

Next bounded task: a same-quadrature imposed-profile strain versus continued
slip-source diagnostic in the local bottom 200 m, retaining separate pressure,
deviatoric and near-endpoint-node information. Keep full bottom loading in
production pending stronger evidence; a future controlled production restart
comparison remains necessary before adoption. No additional sweep is authorized.

## Bottom velocity from copied mesh snapshots — September 28

Read-only C++/VTK extraction and gnuplot figures are complete for
`bp3/output-first-event/snapshots/solution.pvd`: 13 snapshots, steps 0–563,
0–139.159274 years. All 52 VTU pieces exist. Their Float32 velocities reside on
nine-node Q2 Lagrange visualization cells; 158769 horizontal/true-normal probes
were located and checked without smoothing. Snapshot/profile clocks agree
within 0.004762 s of PVD rounding. No Python or simulations were used.

Reusable tools: `benchmarks/reconstructed_fault/bp3/velocity_profiles/`.
Results: `benchmarks/reconstructed_fault/bp3/output-first-event/bottom_velocity/`,
including `REPORT.md`, CSV samples/metrics, and PNG/PDF figures. The bottom
velocity is unchanged across snapshots; interior normal motion evolves to
0.954% and 1.273% of Vp on cuts 500 m and 1 km above the bottom at the final
mesh snapshot. The continuous loading-reference deviation at the boundary is
0.04090% of Vp, including FE interpolation/export effects. These data expose
interior accommodation excluded at the boundary, but do not establish the
cause of deep-end stress concentration or exclude a discrete source mismatch.
The latter needs a same-quadrature source/strain-rate comparison; a causal
boundary-condition test would require a separately requested simulation.

## Repository checkpoint selection — September 28

The user requested a commit of reusable development work. The selection retains
the current core implementation, tests, maintained BP3/BP5 tools and PRMs,
required restored-BP3 fixture inputs, scientific specifications, recovery notes
and compact reports. Long-run outputs, checkpoints, generated visualizations,
build products, archives and copied source snapshots remain local and unchanged.
Reports can therefore reference local evidence that is intentionally absent
from a fresh checkout; the commit is not a backup of simulation data.

Commit preparation reran the 14 offline BP3 reader/launcher tests and two BP5
stress-cycle analysis tests successfully. No simulations or full C++ rebuild
were run for this checkpoint. Earlier build/solver/limiter evidence below
records the implementation tested at that time, not a new qualification of
every intervening working-tree edit. In particular, the current limiter uses
`maximum_logarithmic_state_change`, `Patterns::Double(0.)` and `get_double()`;
the earlier explicit sentinel parser/validation described below is historical.
Its present parameter-edge behavior was not retested during commit preparation.

## Copied first-event run: fault evolution plots — September 28

Plotted the new data without running simulations or changing numerical code.
The newer index is `benchmarks/reconstructed_fault/bp3/output-first-event/profiles/profiles.csv`,
with payloads in its nested `profiles/` directory: 124 complete snapshots of
2889 vertices, steps 0–594, through 142.399097 model years. The top-level
`profiles.csv`, accepted-step log, event summary and native VTUs are older;
accepted steps/native snapshots reach only step 54 (10.627038 years).
All nine overlapping profile payloads are byte-identical. Later profiles are
internally consistent but cannot be checked against the stale accepted-step log.

PNG/PDF profiles, time–distance maps, selected-node histories, and separately
labeled early native-property ranges are in
`benchmarks/reconstructed_fault/bp3/output-first-event/fault_evolution/`.
See its `REPORT.md` for provenance, reproduction and observations, and
`summary.json` for exact ranges. Saved peak V is 1.04299733037e-9 m/s; these
snapshots show creep/locking evolution, not a recorded fast event. Sparse output
does not rule out an unsaved transient. The reported seven-hour failure is not
diagnosed by this plotting task; the copied log ends during step 55.

The plotting utility now accepts a separate `--profile-run`, writes PDF copies,
and plots histories at existing vertices. Four existing reader tests passed;
the full copied dataset passed index/geometry/finiteness checks and generated
figures were visually inspected. Original outputs and checkpoints are preserved.
Next bounded task, if failure analysis is requested: obtain the matching latest
accepted-step/event summaries and final server stdout/stderr before attributing
the failure to any numerical mechanism. Do not rerun the experiment for context.

## Optional reconstructed-fault Theta timestep limiter — September 26

**Historical implementation and verification:** the infinity alias and strict
positive-bound validation described below predate the user's numeric-parser
revision. The current contract, confirmed September 29, uses
`Patterns::Double(0.)` plus a strictly-positive assertion, rejects `infinity`,
and disables only at the maximum-finite-double sentinel (or for stateless friction). The old test
results below do not qualify the revised parser.

The user explicitly authorized this new timestep policy after the plugin
cleanup. The core `reconstructed fault time step` plugin now declares
`Time stepping / Reconstructed fault time step / Maximum logarithmic state change`.
Following the user's sentinel revision, the default is
`std::numeric_limits<double>::max()`, with `infinity` accepted as an alias.
Both bypass the new limiter, preserving the previous proposal. The mismatched
sentinel references and missing positive-bound validation were corrected;
the member is also initialized to the disabled sentinel.
The sentinel fix passed the Release build, 16 parameter assertions (including
exact maximum-double round-trip), and the expanded one-rank integration check
for both disabled spellings. Evidence is in
`benchmarks/reconstructed_fault/state_limiter/sentinel-fix/`; the earlier
two-rank results below predate this parameter-only correction.
A positive finite value limits the **unweighted** maximum predicted absolute
log-change of Theta over all fault vertices. It reuses the existing exponential
aging law at the latest committed velocity/state and the existing material Dc;
it does not copy the old BP5 b/a weighting or introduce another controller.
Halving/bisection returns a safe proposal without modifying histories. Stateless
friction bypasses it. Ordinary manager/MPI combination and minimum-step semantics
remain unchanged. No production PRM was opted in implicitly.

Verified local Release build, 14 parameter assertions, real-plugin stateful
checks on one/two MPI ranks, and stateless bypass. The tests cover analytic
increasing/decreasing bounds, equilibrium, very small predicted timesteps,
ratio overflow, all vertices, committed versus trial velocity and nonmutation.
The historical closed-box Stage-I rate/state fixture hit its pressure-nullspace
compatibility assertion before reaching the limiter; the new test uses a free
top and leaves core pressure handling unchanged. No production trajectory or
server validation was run. See [implementation and evidence](../../benchmarks/reconstructed_fault/state_limiter/README.md).

Next bounded task: rebuild ASPECT on the matching server stack and explicitly
choose a finite bound in the intended BP3 input if desired. This replaces the
need for a BP5 startup plugin; existing inputs otherwise retain disabled behavior.

## Maintained BP3 diagnostics/output cleanup — September 26

Completed the explicit `BP3_PLUGIN_CLEANUP_INSTRUCTIONS.md` task. The maintained
five-source plugin now separates `work_audit.cc` from output lifecycle. Mandatory
native weak-traction, stable-ID H, geometry/Ih, Theta/compression/boundary and
first-step Maxwell checks remain enabled. Production avoids later particle
replay, incoming-state copies, full-state dumps and initial mesh CSVs. Monitor
profile/filter initialization and callback order remain unchanged. Four live
core experiment guards remain; the two hidden LENGTH output selectors are gone
from this plugin. No core solver or physical production behavior was changed.

Scheduled full-precision profiles are now canonical full-fault history; the
duplicate every-step cumulative-slip table is removed. Slip still integrates
and checkpoints every physical step, without step-zero accumulation. Production
continuation uses 0.1 m / 31557600 s profile/heavy intervals and diagnostics off;
dedicated startup comparisons retain their explicit dense settings. Readers use
instantaneous profile V, with a clearly sparse offline legacy exporter.

Version-5 archive layout is preserved. New PRM intervals apply on restart while
last-written references persist. Output failures propagate collectively;
successful writes precede schedule advancement. Growth headers work in a new
directory; newer/conflicting output is rejected. `branch_output.sh` restores the
selected metadata prefix and indexed profile payloads from the preserved parent.

Verified: Release build and production PRM parse; a six-step reduced functional
fixture comparing the old/new plugin byte-for-byte in accepted summaries,
V/Theta/slip/tractions, events and audits; diagnostics off/on; slip/time schedules;
production initial/final forcing; version-5 baseline checkpoint load; split
restart between profiles with a changed cadence; one/two-rank output-error
checks; ten reader/launcher tests. The stable fixture does not produce a seismic
event; event transitions were tested synthetically. No full production run was
launched. The requested existing small full restored fixture was not found;
the new explicitly coarse fixture is for functional checks, not BP3 resolution
qualification. Its physical settings are identical before/after, and separate
from production inputs.

Evidence: [cleanup report](../../benchmarks/reconstructed_fault/bp3/output-cleanup-evidence/REPORT.md),
[exact comparisons](../../benchmarks/reconstructed_fault/bp3/output-cleanup-evidence/comparison.txt),
[sizes](../../benchmarks/reconstructed_fault/bp3/output-cleanup-evidence/sizes.csv),
and [runtime instructions](../../benchmarks/reconstructed_fault/bp3/plugin/README.md).
At 32 vertices/seven accepted states, baseline profiles plus dense slip were
49,896 B; production-clock profiles are 9,866 B. Checkpoints/native outputs
dominate this small case; no universal reduction factor is inferred. The global
16,875-entry H audit map measured ~1.08 MB heap storage per rank and 202,562 B
serialized; the audit remains intentionally replicated/checkpointed.

Next bounded task: compile the package against the matching Intel 26 server
stack and verify a short preserved-filter20 checkpoint branch using the desired
server timestep caps. The AMG/GMG performance investigation below remains a
separate unresolved task. Supplied server files and all prior changes remain
preserved; no completed production experiment was repeated.

## Latest server AMG / GMG performance audit — September 26

The new `bp3/output-test-gmg/` is directly comparable in saved parameters with
`output-test-amg/`: only backend and output directory differ. Both use 48 ranks,
542,958 cells, 19,149,424 DoFs, ten levels, deal.II 9.7.0, Trilinos 16.2.1 and
AVX512. Both saved runs still use the pre-cleanup BP5 state bound 0.1; the new
files do not exercise the removed controller's absence. GMG completes through
step 2 in the upload, so comparisons stop there.

At that matched state, total wall time is AMG 340 s versus GMG 401 s; condensed
solve time is 173 s versus 235 s; outer iterations are 183 versus 205. GMG setup
is only 5.52 s, nested inside its solve timer. Existing rank-local detailed
profiles show block-preconditioner application totals 88.066 versus 134.683 s,
or 0.481 versus 0.657 s per call. At step 2 the per-call gap is 0.504 versus
0.839 s. Thus both more outer iterations and slower preconditioner applications
contribute. Active-cell ownership is balanced (11,308–11,314 cells/rank).
The cause inside the preconditioner (inner CG iterations, cycle throughput,
copies, pressure work, communication) is not resolved by these outputs.

See [the comparison report](../../benchmarks/reconstructed_fault/bp3/amg_gmg_server_comparison.md)
for exact scopes, earlier four-rank local evidence and limitations. The next
bounded performance task is one matched frozen solve with inner velocity
iterations and preconditioner sub-timings, not another full trajectory. This
audit changed documentation only; outputs and source were preserved and no
simulation or Python script was run.

## GMG and BP3 plugin cleanup — September 26

At the user's request, removed production GMG debug logging, raw-diagonal
observers, diagnostic collectives and scratch-vector probes. The original
diagonal calculation, eigenvalue estimator and velocity GMG cycle are retained.
Standalone reproduction probes remain in `server_gmg/`; their helper is now
benchmark-local and the runner no longer requires removed diagnostic markers.

The maintained BP3 plugin no longer builds/registers `BP5 state startup`.
Its source is retained outside the build in
[`gmg-bp5-cleanup-evidence`](../../benchmarks/reconstructed_fault/bp3/gmg-bp5-cleanup-evidence/).
The raw/filter20/filter40 PRMs and their generator now select convection and
reconstructed-fault timestep controllers only; first-event inherits filter20.
There is no replacement state-change limiter. Standard ASPECT caps are unchanged:
the maintained PRMs still have first/global caps `100 / 4e6` s, while the latest
saved server run used `4e6 / 4e7` s. Preserve the user's chosen server caps when
removing its BP5 subsection. The full-resolution saved output-test-amg inputs
and outputs remain historical records and were not edited.

BP5 checkpoint/test-environment rejection checks remain in BP3 because they
prevent incompatible inputs, not because BP3 uses BP5 physics or timestep
selection. The separate BP5 benchmark is unchanged. Updating the server requires
rebuilding ASPECT and the matching BP3 plugin, removing `BP5 state startup` from
the controller list and deleting the corresponding PRM subsection. The two
predictor-owned CSVs stop being emitted; accepted-step dt output remains.
Local Release ASPECT/plugin/probe builds passed, all four maintained BP3 inputs
validate, and otherwise valid inputs reject both the retired BP5 selector and
its subsection. A new one-rank Cartesian GMG Stage-I initialization passes with
final bulk/fault residuals `9.165224e-08 / 0`. Source/symbol audits, shell syntax,
whitespace and supplied-output hash checks pass. See the
[cleanup report](../../benchmarks/reconstructed_fault/bp3/gmg-bp5-cleanup-evidence/README.md)
for exact scope and evidence. No full BP3 trajectory, restart or new server run
was performed. **Next bounded task:** transfer/rebuild this cleanup with the
working Intel 26.0 stack and update the server BP3 PRM as described above.

## Latest BP3 AMG timestep evidence — September 26

The new `bp3/output-test-amg/` run uses first cap `4e6` s, global maximum
`4e7` s and state-change bound `0.1`. This is executed evidence of a later
policy relaxation relative to the earlier `100 / 4e6 / 0.02` startup settings;
the maintained candidate PRMs are unchanged by this audit. Its timestep
selection audit directly identifies **BP5 state startup** as the limiter:
552.083566, 590.185962, 630.854254 and 674.325550 s. All four predictor
measures saturate `0.1`; convection, fault-law, first-step, growth and global
caps are larger, and termination reduction is false. Three physical intervals
are recorded as accepted; the fourth starts in the log but is not recorded
as accepted in the supplied files. The duplicated state-zero row is not a step.

The shallow initial Theta is 8,000 s with b/a = 1.5 and V near `1e-9` m/s;
the exact aging predictor explains the first 552.083566 s interval. See
[the timestep audit](../../benchmarks/reconstructed_fault/bp3/output_test_amg_timestep_analysis.md)
for evidence, formulas and limits. The active limiter is resolved without a
rerun. If larger startup steps are desired, the next bounded task is to assess
a proposed state-bound relaxation at matched physical times; the current
short successful solve does not itself qualify temporal accuracy.
Only documentation was changed; all source, PRMs and supplied outputs remain
intact. No simulations or Python scripts were used.

## Follow-up: server GMG diagnosis

**Resolution reported by the user (September 26):** rebuilding both deal.II
and ASPECT with `intel/26.0` instead of `intel/24.0` makes GMG run properly.
This supersedes the pending investigation/workaround recommendations below.
The report establishes a working toolchain remedy; a specific compiler defect
or faulty dependency instruction is not identified. New GMG build provenance
was not independently audited in this BP3 AMG timestep task. Preserve the
earlier Cartesian/Q1 experiments as historical evidence; do not repeat them.

**Newest server Q1 evidence (September 26):**
`server_gmg/output-gmg-q1.tar` records a successful OPTIMIZED/deal.II 9.6.0/
AVX512 one-rank Stage-I initialization: 4,096 cells, seven levels, four velocity
GMG setups, final printed bulk/fault residuals `9.165231e-08 / 0`, Stage-I
`verified`, and normal end-time termination. Saved parameters select local
smoothing block GMG and the zero-valued topography function that selects
MappingQ1 in the current source. The archive lacks the separate diagnostic
logs and binary hashes; no direct server mapping-type trace is claimed.
This completes the previously pending server workaround test. Keep the Q1 PRM
workaround for the affected flat-box build; velocity/pressure remain Q2/Q1.
The underlying Intel/deal.II defect is still unproven. A minimal source bypass
and its wider scope are documented, not applied, in
[the updated analysis](../../benchmarks/reconstructed_fault/server_gmg/mapping_cartesian_workaround.md).
**Next bounded task for an upstream repair:** locate the first invalid
Cartesian geometry/evaluation quantity in the existing one-cell probe before
choosing a dependency patch. No repeated Stage-I experiment is needed.
This update only inspected the supplied archive and edited documentation;
no simulation, build, Python script or production-code modification was made.

**Newest corrected Release context evidence (September 26):**
`server_gmg/output-coarse-context-release/coarse_diagonal_probe.txt` now contains
all three cases. `cartesian_active` and `cartesian_parent` both produce NaNs in
the production inverse, test raw diagonal and production action; `q1_parent`
passes. FEValues references remain finite in every case. This isolates the
failure to the server's matrix-free MappingCartesian path; neither a refined
hierarchy, coefficient projection, CG nor the fault model is needed to trigger
it. A particular Intel/deal.II source defect is still unproven. The assertion
correctly reports the failed comparisons; keep it.

A **PRM-only flat-box workaround** is prepared in `server_gmg/gmg_q1.prm`:
select initial-topography `function`, expression `0`, maximum topography `0`.
The existing mapping factory then selects MappingQ1 while zero displacement
preserves vertex coordinates. This is applied to Stage I, not to the standalone
probe that constructs its mappings explicitly. No production C++ or BP3/BP5
input was changed for this workaround. The new local one-rank Release Stage-I
initialization passes, with MappingQ1 confirmed, 28 eigenvalue estimates,
four consumers and final printed bulk/fault residuals `9.165224e-08 / 0`.
Evidence is under
[verification-q1-local](../../benchmarks/reconstructed_fault/server_gmg/verification-q1-local/).
The previously pending server `gmg_q1` Stage-I verification is now complete,
as recorded above; no core/deal.II rebuild was needed for this input change.
See [instructions and limits](../../benchmarks/reconstructed_fault/server_gmg/mapping_cartesian_workaround.md).

### Earlier context-selection correction (superseded by the new Release upload)

**Latest context-output audit (September 26):** The copied
`server_gmg/output-coarse-context-debug/` and `output-coarse-context-release/`
each contain only `case=q1_active refinements=0` and one successful result.
They repeat the baseline; the three mapping/hierarchy cases have **not** been
verified on the server. The earlier test selector depended on a runner-set
environment variable. This fixture weakness is corrected: `coarse_context.prm`
explicitly sets `Postprocess / Fault GMG coarse probe / Test mode = context`,
and the runner requires all three expected case headers/pass records through
`check_context_output.sh`. Rebuild the plugin and use the updated PRM/runner/
checker, then run the corrected Release context case to a new directory. No
core rebuild is needed for this correction. Only after the three cases pass
should the next original-context Stage-I measurement be interpreted.
The [analysis](../../benchmarks/reconstructed_fault/server_gmg/server_context_analysis.md)
records exact evidence, the PRM entry and next instructions. A direct local
launch without the old environment selector passed all three cases and matched
the earlier local context output exactly; validation rejected the supplied
baseline-only and truncated outputs. Evidence is in
[verification-context-selection-local](../../benchmarks/reconstructed_fault/server_gmg/verification-context-selection-local/).
Server artifacts remain unchanged. No production numerical changes or Python
scripts were involved in this correction.

**September 26 update:** Both supplied server coarse probes pass, including
Release/AVX512; the two production inverse diagonals match Debug exactly. See
[server_context_analysis.md](../../benchmarks/reconstructed_fault/server_gmg/server_context_analysis.md)
for the superseding analysis, configuration evidence and next commands.
The compiler is IntelLLVM/icpx 2024.0.0 with Intel MPI 2021.11. ASPECT already
has `-fno-finite-math-only -fp-model=precise -ffp-contract=off`; recommending
these again would repeat an existing setting. The supplied cache is
DebugRelease with unity build ON. No compiler bug or numerical fix is established.

The original passing reduced probe uses MappingQ1 on an active cell, whereas
the failing Stage-I configuration uses MappingCartesian on a level-zero parent
with six refined levels. The new `coarse_context` case isolates mapping and
parent hierarchy in three tests, without solving physical equations or replaying
the completed baseline. **Next bounded task:** run this case in server Release
against the same executable used for Stage I. If it passes, use one instrumented
Stage-I initialization to capture initialized/raw/constrained/inverse diagonals
from the original context. The observer in `matrix_free_operators.cc` is opt-in,
records to stderr and changes no diagonal or solver controls. Return the runner
logs, actual compile command and confirmation of executable identity/diagnostic
switch for the reported CG failure; general configuration files are now supplied.

Locally all three new context cases pass before and after the observer change;
their 96-line numerical probe outputs are byte-identical. Evidence is under
[verification-context-local](../../benchmarks/reconstructed_fault/server_gmg/verification-context-local/).
The core was rebuilt for this observer; its new hash is recorded there.
Server outputs/configuration/tar files were hash-checked unchanged. No local
Stage-I/BP3/BP5 rerun or Python script was used in this update.

### Earlier diagonal localization and recovery follow-up

The next task requested after recovery is a bounded debugging package for the
Intel-build GMG NaN, not first-event continuation. See
[server_gmg/README.md](../../benchmarks/reconstructed_fault/server_gmg/README.md)
for the small initialization-only model, shell runner, opt-in per-level C++
diagnostics, server build/backtrace instructions, and local verification.
No Python is used in that workflow. The copied server Debug/Release outputs now
localize the first observed NaN to **level-zero velocity inverse-diagonal DoF
16, before eigenvalue estimation**. Both runs have identical active/coarse
viscosities, one MPI rank and eight SIMD lanes. Debug's inverse diagonal agrees
with `15/(128 eta)` for the two free central Q2 velocity DoFs; its coupled
Stage-I verifier passes. Resolved PRMs differ only by plugin mode and output
directory. The first invalid operation and the cause of the Release-only
failure remain unknown; compiler/optimization, undefined behavior and build
consistency are hypotheses, not established fixes. See the
[server result analysis](../../benchmarks/reconstructed_fault/server_gmg/server_debug_release_analysis.md)
for exact evidence, source locations and missing provenance.

The new probe passed locally with GCC 12.4/OpenMPI, deal.II 9.6.2 and four SIMD
lanes; both free Q2 entries agree with the independent quadrature/analytic
reference. Build/run/provenance evidence is in
[verification-coarse-local](../../benchmarks/reconstructed_fault/server_gmg/verification-coarse-local/).
This validates the fixture, not the Intel server path. All 11 copied server
output files were hash-verified unchanged. No production source was modified
or core binary rebuilt in this analysis follow-up.

The **previous bounded debugging task (now completed on the server)** was the one-cell `coarse` probe, run with
the existing matched server Debug/Release binaries. It compares production
inverse diagonals and matrix-vector actions with a test-compiled raw diagonal,
FEValues quadrature and the analytic result. It solves no physical equations.
Do not repeat the completed startup trajectories or alter CG/timestep settings.
The later supplied cache/configuration is analyzed above; exact compile commands
and full failing-run stderr remain useful. The local `build-tmp` binary was
rebuilt for this follow-up, so its hash in the recovery inventory below is a
historical snapshot, not the hash of the new instrumented binary. Existing
startup outputs and physical/timestep settings are preserved.

## Recovery snapshot

Recovered 2026-09-25 from `/home/ein/repository/aspect`. This is a checkout and
evidence audit, not a new numerical qualification. No build, test simulation,
analysis that regenerates outputs, or server submission was run during recovery.

## Authority and checkout

The applicable instruction file is [AGENTS.md](../../AGENTS.md); no nested
AGENTS.md was found. Scientific/ownership/MPI authority remains
[current_design.md](current_design.md) and [specification.tex](specification.tex),
with [refactoring.md](refactoring.md) governing code quality. The
[recovery handoff](CODEX_RECOVERY_HANDOFF.md) is historical context, not proof of
implementation or an instruction to repeat completed experiments.

Branch: `pf-rsf`. HEAD: `3ce447a17e14ac21cb24abf8fbf679d77b4f3f0e`, September 24,
“Isolate BP3 execution diagnostics and share opt-in stress interpolation.”
At the start of this audit there were 42 modified tracked files, no staged diff,
and 2,505 untracked entries in default `git status --short` (entries can represent
directories; ignored outputs are additional). The tracked diff was 1,738
insertions / 289 deletions. HEAD alone does **not** reproduce the current model.

Recent history establishes these completed layers:

| Commit | Completed scope |
|---|---|
| `3ce447a17` (Sep 24) | Execution guards, shared opt-in stress-only LLS adapter, narrow A/B replay; substantial core changes remained uncommitted |
| `33228369d` (Sep 21) | Documentation update |
| `b56829272` (Sep 21) | BP5 loading initialization, refined normalization, startup/restart safeguards and benchmark infrastructure |
| `359ea223c` (Sep 17) | Modified BP3 research/long-run configuration consolidation |
| `3335d3d26` (Sep 15) | Work-measure replay and coupled-state timestep evidence |
| `fb4411915`, `86fff7395`, `dc1a96d3f`, `a24c36231` | K1–K4 bounded qualification; see historical reports for limits |

Uncommitted work includes normal-input filtering and derivatives; pressure
compatibility/convergence corrections; reflected-fault shear sense; diagnostic
history/restart hooks; solver-selector cleanup; tests; updated authority
documents; and restored BP3 inputs/runtime. In particular,
`source/reconstructed_fault/normal_filter_internal.h`,
`tests/fault_surface_reference.h`, and the entire restored `bp3/plugin/` runtime
are untracked. Preserve these with the tracked diff. Old experiment sources,
PRMs, binaries, archives and output directories were left intact.

## Current implementation and selected research model

The framework has reconstructed geometry, replicated fault Q1 properties,
manager-owned committed/current/trial slip rates, particle Maxwell history,
adaptive normalization, coupled bulk/surface residuals and exact condensation,
nonlinear acceptance/rollback, and accepted-state constitutive publication.
It is beyond the old geometry-only and Stage-F plans. The current research
fixture uses incompressible viscoelastic Q2/Q1 Stokes, frozen AT1 phase, mature
frictional C=0, true normal feedback, and fully frictional deep extension.
It is a modified BP3 research model, not exact standard SEAS BP3.

The canonical restored runtime is
[bp3/plugin/](../../benchmarks/reconstructed_fault/bp3/plugin/README.md), built
from `bp3.cc`, `mesh.cc`, `monitor.cc`, and `output.cc` (the BP5 state predictor
was retired in the cleanup above).
The parent historical `bp3.cc` is retained for other variants; it is not the
restored target's implementation. The restored monitor now installs surface
settings at `post_simulator_initialization`, after the surface system exists.
The constructor null dereference was reproduced and fixed; the early paragraph
in the cleanup report saying it was undiagnosed is superseded by its follow-up.

Verified current core behavior:

- Normal filtering solves `(M + Ls^2 K) z = b(sigma_raw)` with native work
  weights and physical arc derivatives, where
  `sigma_raw = sigma_background + p - n^T tau n`. It changes friction input
  and its consistent derivatives, not raw mechanics or retained stress.
  Raw remains the generic default; restored BP3 explicitly selects Helmholtz.
- Timestep-zero mechanics retains supplied Theta/H and zero initial perturbation
  Maxwell history. Its artificial interval is separate from physical aging.
  Saved startup rows confirm zero committed stress at state zero.
- Particle-to-FE publication sums incident-cell proposals with MPI ADD and
  divides by contribution count for continuous DoFs. It is not last-writer-wins
  ([initial_conditions.cc](../../source/simulator/initial_conditions.cc)).
- Pressure compatibility uses the assembly-roundoff bound capped by the mixed
  nonlinear target and checks compatibility even when skipping an unused
  direction at an already-converged base
  ([solver.cc](../../source/simulator/solver.cc)). This qualified correction is
  present in the dirty tree and recorded in both authority documents.
- Surface inversion always uses pivoted tridiagonal GTTRF/GTTRS. The ordinary
  `Stokes solver type` selects AMG/GMG; old environment backend selectors no
  longer select production behavior. AMG remains the restored PRM choice.
  This does not establish that the historical server GMG NaN is fixed.

No new specification/implementation conflict was established in these inspected
paths. This is not a full scientific code review.

## LLS routing: resolved discrepancy

**Restored BP3 uses native unlimited LLS for all five mapped composition fields:**
`tau_xx`, `tau_yy`, `tau_xy`, `theta_initial`, and `strengthening`, in continuous
Q2. Limiter and boundary extrapolation are false. The restored CMake target
does not compile the stress-only adapter, and publication calls the configured
particle interpolator with the selected property mask.

This is explicit in the current
[filter20 PRM](../../benchmarks/reconstructed_fault/bp3/bp3_150x50_filter20.prm),
generator, saved run PRMs, and the September 24
[restoration preflight](../../benchmarks/reconstructed_fault/bp3/restore_150x50_preflight.md)
(“Other requested choices”). Thus it is a documented restoration choice, not
an unnoticed routing adapter. It does not change the separate particle-to-fault
Q1 projection or turn committed fault Theta into a bulk composition history.

The earlier
[inclined interpolation comparison](../../benchmarks/reconstructed_fault/bp5/interpolation-inclined/README.md)
used [the stress-only adapter](../../benchmarks/reconstructed_fault/stress_only_interpolator.cc):
three Maxwell components to LLS, other properties to DWA. The legacy BP3 target
registers this optional adapter, while its maintained long-run PRM selects DWA.
Those are different configurations. The controlled A/B result therefore does
not isolate the effect of changing non-stress interpolation in restored BP3.
The three restored startups do exercise the all-field LLS configuration, but
are not a DWA-versus-all-field-LLS sensitivity study or long-time qualification.

## Active configuration

Maintained candidate:
[bp3_150x50_filter20.prm](../../benchmarks/reconstructed_fault/bp3/bp3_150x50_filter20.prm).
The [first-event PRM](../../benchmarks/reconstructed_fault/bp3/bp3_150x50_first_event.prm)
includes it and changes resume/output/termination settings only.

| Item | Current value / evidence |
|---|---|
| Domain / fault | 150 × 50 km; x = −60…90 km; 60° right-dipping thrust, (0,50 km) to (28.8675 km,0); positive V with selected shear sense −1 |
| Fault grid | 57.735026919 km; 2,889 Q1 vertices, all frictional; spacing 19.9818–20 m |
| Mesh | Saved `fixtures/bp3_150x50/target_cells.txt`; root 75 × 25, global 1/adaptive 8; runtime adaptation off |
| Realized inventory | 542,958 cells, 19,149,424 DoFs, 4,886,622 particles; confirmed in saved full-resolution log, not only the mesh estimate |
| Resolution / phase | Minimum edge 3.90625 m; ell = 20 m; frozen AT1 core 0.6; mature C=0 |
| Loading | Smooth bottom and rigid side Dirichlet velocities; relative far-side Vp = 1e−9 m/s; free perturbation top; no explicit traction list; zero gravity/surface pressure; pressure normalization `no` |
| Friction | a = .010→.025 over 15–18 km down dip; b=.015; Dc=.008 m; mu0=.6; V0=1e−6 m/s; Vmin=1e−20 m/s |
| Bulk / damping | G=32,038,120,320 Pa; viscosity=1e26 Pa s; radiation damping=4,624,440 Pa s/m |
| Background / initial state | 50 MPa normal; nominal uniform shear 26.5461223651 MPa; inverse initial Theta about 8,000 s shallow / 8e6 s deep; zero perturbation stress |
| Normalization | `cell intervals`, eight surface subdivisions; quadrature/tail tolerance 1e−10; prepared endpoint completion/profile retained |
| Normal input | Helmholtz 20 m provisionally selected; raw and 40 m are separate clean-start controls |
| Particles / FE | RK2, 3×3 per cell initially; native unlimited LLS, continuous Q2 compositions; strengthening spatially refreshed |
| Solver | AMG; linear 1e−9, nonlinear 1e−8; abort on nonlinear failure |
| Time controllers | Convection + reconstructed-fault law; CFL .5; first physical cap 100 s; global ceiling 4e6 s; BP5 state predictor removed by user request |
| Artificial initialization | Material `Initial time step = 4e6` s; not elapsed time or a state-aging update |
| Startup termination | Accepted step 10 or accepted-state 3,600 s wall guard; checkpoint-on-termination requested |
| Event continuation | Resume filter20; detailed diagnostic capture off; 82,800 s accepted-state wall guard; 1,500-year safety end; threshold crossing 1e−3 m/s followed by five states below threshold |

Startup diagnostics are detailed; the continuation keeps growth, station,
event/solver and scheduled outputs. Current compact cumulative-slip output uses
ten significant digits for slip, seventeen for time/coordinates; computational
state and checkpoints retain full precision.

## Completed experiments and available evidence

The following are historical completed tests/results, inspected rather than
rerun in this recovery. A report's narrower qualification is retained.

| Evidence | What is established / limitation |
|---|---|
| [Restored filter comparison, Sep 25](../../benchmarks/reconstructed_fault/bp3/filter-test/comparison/README.md), [summary](../../benchmarks/reconstructed_fault/bp3/filter-test/comparison/summary.json) | Raw/20/40 each accepted states 0–10, t≈1133.7390530913 s, 2 Newton updates/state, 493 Krylov iterations total per run, alpha=1, no lower-active nodes, fresh checks pass; Theta error ≤2.23e−16. All three logs report 80 MPI ranks. No first event. |
| [filter20 accepted states](../../benchmarks/reconstructed_fault/bp3/filter-test/output-150x50-filter20/accepted_steps.csv), [log](../../benchmarks/reconstructed_fault/bp3/filter-test/output-150x50-filter20/log.txt), [boundary constraints](../../benchmarks/reconstructed_fault/bp3/filter-test/output-150x50-filter20/velocity_constraints.csv) | Realized full mesh/particle inventory; zero reported side/bottom velocity-constraint error; summary net flux ≈4.8e−18. Final max V≈1.000264e−9 m/s; event table says not started/not complete. Logged wall times raw/20/40: 750/675/593 s; no all-rank peak RSS record identified. |
| [Restoration verification](../../benchmarks/reconstructed_fault/bp3/restore_150x50_report.md), [logs/hashes](../../benchmarks/reconstructed_fault/bp3/restore-150x50-verification/) | Sign/filter/profile/manager and Stage-I tests; reflected B, G and work identities on one/two ranks; generated mesh/completion. Its “startups not run” conclusion is superseded by Sep 25 outputs. |
| [Plugin cleanup](../../benchmarks/reconstructed_fault/bp3/plugin_cleanup_report.md), [evidence](../../benchmarks/reconstructed_fault/bp3/plugin-cleanup-evidence/) | 157 model comparisons, environment checks, four PRM validations, 20,236 assertions/16 cases, one/two-rank coupling checks. Constructor fix verified before mesh generation. Not a full restored restart equivalence test. |
| [Runtime cleanup, Sep 25](../../benchmarks/reconstructed_fault/bp3/runtime_cleanup_report.md), `/tmp/bp3-runtime-cleanup-KF1mxB/` | Later completed selector/output cleanup: 636 assertions/3 cases on one/two ranks; AMG/GMG coupled fixtures, two-rank GMG, intentional-failure rollback, condensed fixture and constructor checks. Logs still exist. Report also records 3 slip-history, 1 launcher and 5 plot tests. No post-cleanup full-resolution BP3 trajectory. |
| [Inclined LLS comparison](../../benchmarks/reconstructed_fault/bp5/interpolation-inclined/README.md), [metrics](../../benchmarks/reconstructed_fault/bp5/interpolation-inclined/comparison.json) | Six one/two-rank runs, four 0.1 s steps with RK2; B continuous-Q2 stress-only LLS reduces interior jump 0.293767→0.060060 Pa m; whole-domain jump slightly increases; DG is worse on that measure. No host-cell crossings, no free-RSF qualification. |
| [Execution cleanup](bp3/execution_cleanup.md), [replay metrics](../../benchmarks/reconstructed_fault/bp5/interpolation-inclined/cleanup_comparison.json) | A/B cleanup replays retained fields/histories with zero measured differences. Separate two-rank free-rate/state smoke failed its pressure-compatibility guard; not a passing test. Later successes do not demonstrate rerunning that exact failed fixture. |
| [Moment qualification](../../benchmarks/reconstructed_fault/bp5/moment-qualified-final/report.md) | A/B/C four-step horizontal prescribed-slip qualification and C two-rank replay after pressure-compatibility correction; 20,136 Stage-I assertions/14 cases. Does not select a production moment-transfer replacement. Supersedes earlier “partial” moment-study summaries. |
| [BP5 history-cycle audit](../../benchmarks/reconstructed_fault/bp5/normal-stress-cycle/stress_cycle_report.md), [half-clock filter study](../../benchmarks/reconstructed_fault/bp5/normal-filter/half/report.md) | Earlier loaded BP5 reached ~168 years and rapid slip; half-clock branches 5613–5622 finish near 5.3101110716e9 s, final Vmax≈.096–.097 m/s. Filter smooths input but raw bands persist and shallow rate roughness increases. Not a prescription to use BP5's 100/200 m lengths for restored BP3. |

The restored comparison recommends **20 m provisionally**: normal-input chord
roughness falls roughly 8–11×; bottom velocity-chord RMS/Vp is 5.9573e−7 raw,
2.1584e−7 at 20 m, 1.9408e−7 at 40 m. The ~1.0892%-Vp departure near 15 km
persists in all cases. These are startup findings, not an event-accuracy or
long-term stability result. Raw versus Helmholtz also changes projection;
there is no zero-length projected control in this restored comparison.

## Timestep policy after the September 25 report

**Historical recovery finding, superseded by the new AMG evidence above:**
at recovery, no later policy relaxation was evidenced in the checkout. HEAD/reflog ends
September 24. The later September 25 runtime cleanup report/source/PRMs change
solver selection and output, not timestep policy. Current raw/filter20/filter40
PRMs, their generator, and all three executed `original.prm`/`parameters.prm`
files retain first cap 100 s and state bound .02. The continuation inherits both.
File modification times help locate these later files but cannot establish an
unrecorded conversation decision.

The current predictor source is byte-identical to `bp5/startup_time_step.cc`
(SHA256 `c54c0bcb76234303363b25ba9df1b56d1ac647e1fa669c6399bb4f658fcde656`).
Its declaration default `.1` is an older default overridden by these PRMs,
**not evidence that .1 was adopted for BP3**. It bounds
`max (b/a)*abs(log(Theta_pred/Theta))` using committed V/Theta and the configured
constant-rate aging law; it is an accuracy heuristic, not a Newton tolerance.

Direct evidence:
[filter20 timestep selection](../../benchmarks/reconstructed_fault/bp3/filter-test/output-150x50-filter20/timestep_selection.csv)
and [state predictor](../../benchmarks/reconstructed_fault/bp3/filter-test/output-150x50-filter20/state_startup_predictor.csv).
At state zero the state proposal is 107.489084736883 s, fault-law proposal
2,666,352.925 s, global ceiling 4e6 s, and selected first interval 100 s.
Subsequent selections are predictor-limited. At state 10 the next proposal is
122.70173342449 s; the last **completed** interval was 121.077317003674 s.
The proposal row does not establish that an eleventh interval ran.

Inference limited to that saved state: raising only the first cap to 4e6 s would
leave the .02 predictor selecting about 107.49 s, not 4e6 s. Raising the state
bound is a separate, unqualified choice. The handoff's later user question about
100 s→4e6 s / .02→.1 remains a question, not an approved or tested policy.
Two Newton updates establish solver convergence, not temporal accuracy.

## Executables, provenance and checkpoint availability

The **recovery's full-resolution successful trajectories** were the three copied
`filter-test/output-150x50-*` directories. Their PRMs name
`plugin/build/libbp3_restore_150x50.release.so` relative to the server launch
directory; the executable path and binary hashes are not recorded in the
inspected copied artifacts. Logs verify Release ASPECT 3.1.0-pre, deal.II 9.6.0
(64-bit indices/AVX512), Trilinos 15.0.0, p4est 2.8.5, World Builder 1.0.0,
80 ranks. Do not substitute a local binary hash for the unknown executed pair.

The **later local cleanup verification** used `build-tmp`, as confirmed by
`/tmp/bp3-runtime-cleanup-KF1mxB/CMakeLists.txt` and its CMake cache. That cache
selects Release, `/opt/openmpi/5.0.6/bin/mpic++`, GCC 12.4 compiler utilities,
and `Aspect_DIR=/home/ein/repository/aspect/build-tmp`. The core cache has
`DEAL_II_DIR=/opt/dealii/9.6-local`, Voro++ and World Builder ON, and `-no-pie`
executable linking. These logs substantiate focused checks, not a long run.

Hashes measured during this audit (current bytes, not automatically run-time
attestations):

| Local artifact | SHA256 |
|---|---|
| `build-tmp/aspect-release` | `1b091f3b745fab911f62179fd659d3d028ba26101402d58254a4b581796def38` |
| `/tmp/bp3-runtime-cleanup-KF1mxB/build/bp3/libbp3_restore_150x50.release.so` | `cafe3a49172afdd1ad3a61fef917b0861aaecf9b7e7e0e072fa41a56fded9ac8` |
| `build-pf-cpdi/aspect-release` | `d986d6166db2a8e0f73e5b9735bb73887e6ba2f8d4967d9dbf90e28d3f2fa85e` |
| `benchmarks/reconstructed_fault/bp3/build/libbp3_restore_150x50.release.so` | `ef18ef25ebc6a1bf94278f598d09c8027f0a5c55e690a14b29e0513d6cd70588` |
| `benchmarks/reconstructed_fault/bp3/plugin.tar` | `a09a5556c550ff75c22d1c87fce2f4df6dbc39f3627f4d08eb06c75185c2e665` |

`build-pf-cpdi` still has the core hash in the earlier restoration verification;
its cached core build is Release, while the parent BP3 build cache says Debug
and names a `.release.so` output. Existing directories/library suffixes do not
prove a current matching build. The earlier plugin hashes `716f077b…` and
`b4162ab5…` in preserved verification records differ from the current library.
Use the recorded matched build context, not an arbitrary executable/plugin mix.

No resume/restart/checkpoint payload was found in any of the three copied
startup directories. The maintained PRMs target `bp3/restore-150x50-filter20`,
which does not exist here; copied evidence is under
`bp3/filter-test/output-150x50-filter20`. Thus the continuation PRM is prepared,
but cannot presently resume these local copies unchanged. Checkpoint-on-termination
being requested is not proof that a usable checkpoint was copied. Full restored
checkpoint/resume equivalence and any first-event continuation remain unverified.

Preserved recovery sources include
[execution-cleanup checkpoint](../../benchmarks/reconstructed_fault/checkpoints/bp3-execution-cleanup-4qkx63tj/),
`bp3/plugin-cleanup-evidence/before.tar.gz`, its pre-cleanup patch/status, and
`bp3/restore-150x50-verification/current_tracked.patch`. Historical raw outputs
may be archived: consult [CLEANUP.md](../../benchmarks/reconstructed_fault/CLEANUP.md).
The `.benchmark-cleanup-20260914-*`, `20260915-*`, `20260917-*`, `20260920-*`,
and `20260924-*` directories still exist locally. No archive was extracted,
repacked, removed or fully revalidated in this task. Temporary cleanup logs
still under `/tmp` are less durable than repository evidence.

## Unresolved questions and next bounded task

1. Recover the exact server executable/plugin identity and complete filter20
   accepted-state checkpoint, including its source/build provenance. Determine
   whether newer server outputs or an explicit timestep decision exist outside
   this checkout; their absence here cannot prove they never existed.
2. Resolve the outstanding first-step/state-bound question explicitly. There is
   no current temporal-accuracy qualification for .1 or larger. The all-field
   LLS choice is documented, but its non-stress effect is not separately isolated.
3. Full restored restart equivalence, long-loading raw/filtered stress behavior,
   the persistent 15-km feature, event completion and measured all-rank memory
   remain open. Keep the older failed smoke/server GMG evidence visible.

**Recovery's proposed bounded task (deferred during GMG debugging):** prepare a continuation-readiness and timestep
decision record from the existing filter20 evidence and the actual server
checkpoint/provenance. Verify model/filter/profile identity and path mapping;
identify the executable/plugin pair; distinguish inherited pending restart dt
from a fresh-start cap; and specify a small matched-time temporal-accuracy
check only if a policy change is requested. Produce reviewable inputs and gates
before authorizing that check or the first-event continuation. Do not repeat
the three startups, restore BP3 again, broaden the refactor, or launch a long
trajectory merely to rebuild context.

Recovery validation is limited to reading source/history/reports, inspecting
saved PRMs/logs/CSV summaries, hashing selected artifacts, checking document
links/whitespace, and verifying preservation of the pre-existing tracked diff.
No numerical tests were rerun because this task changes documentation only.
