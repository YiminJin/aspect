# R6b: nonlinear-bound presentation

## Pre-edit boundary and accepted reference

Accepted R6a: `1a3b57eda`, immutable `build-refactor-r6a/aspect-r6a-qualified`,
SHA256 `d373cb3308ecc00fb05a574975cf55f9d65facea003464b48153fc6aecf88f01`.
R6a's 258 comparisons/source/symbol checks were rechecked before its commit.
The repaired frozen AMG/GMG evidence remains applicable: this pass does not
touch linear output, the synchronous observer, or borrowed operator lifetimes.

| Operation/state | Owner and timing before/after | Extraction |
|---|---|---|
| Presence selector, lower-rate residual probe | Driver, current Newton iterate before convergence test; bulk fixed, unprescribed V replaced by Vmin for the probe | None; same all-rank evaluation/cache/MPI |
| Mass-normalized Fmin, bound counters/minima | Driver, original fault/vertex loop | None |
| Stream lifetime/open/header | Driver owns local ofstream inside bound_audit; rank-active setup after probe; destroyed before convergence test, including unwinding | Three narrow formatting functions borrow streams, retain nothing |
| CSV rows | Same rank-active loop site: current Newton V/dV/active set, with separate lower-rate probe density | Scalar formatting only; iteration/fault/vertex order unchanged |
| Summary | Same pcout object/site, after loop, all ranks call ConditionalOStream | Direct insertion, no flags changed or intermediate stringstream |
| Trial merit, detailed linear/nonlinear reports | Existing driver sites | Untouched |

No public header, persistent state, recorder framework, checkpoint, parameter,
validation or numerical ownership changes. File errors remain silent. No MPI
or callbacks in formatting functions. Keep original open/append/truncate,
precision17, schema and units. Disabled output still skips the whole probe.

## Planned verification

Separate Release build/plugins, independent driver/helper compilation and
2D/3D driver symbols. Same small automatic-completion BP3 off/on one/two ranks
for both binaries, plus blocked-file-open and accepted-update rollback. Require
nonempty exact bound rows and unchanged physical/history fields, solver decisions,
and same-selector work counters. Off/on may add audit work: compare physical
neutrality separately. Preserve prior artifacts and local edits. No Debug/3D
runtime, long/restart campaign, R6c/R7, or other diagnostic-family extraction.

## Reproduction and comparison rules

Configure `build-refactor-r6b` with the accepted R6a compiler/deal.II/Voro++
settings and `-fno-finite-math-only -ffp-contract=off`, Release. Exact commands,
environment, outcomes and unabridged logs are in `evidence/*.json`/`*.log`.
`run_logged.py` fixes thread counts to one and refuses to overwrite run logs.
The candidate plugins use the existing R6a plugin CMake project with the R6b
Aspect_DIR; the reference uses the qualified R6a plugins without rebuilding them.
`prepare_inputs.py` wraps the existing small automatic-completion BP3 and
accepted-update rollback PRMs, changing only plugins/output paths. `run_checks.py
reference` and `run_checks.py candidate` need OpenMPI local sockets.

`compile_independent.py` compiles both TUs without unity/PCH. `verify_source.py`
reverses the three substitutions to reconstruct the entire accepted driver and
compares the moved stream expressions; it also checks original unity groups and
source/header/local-edit preservation. `verify_symbols.py` excludes lambda
member symbols when counting the three driver's 2D/3D definitions. `compare.py`
compares all physical CSV columns, VTU arrays, exact statistics tokens and trace
values, plus byte-exact bound CSVs and detailed logs between versions. Within
off/on runs only, linear-summary fields are compared at the existing off-mode
precision because the switch itself increases printed precision and adds fields.
No tolerance or scientific output is changed. Timer durations and output paths
are excluded; deterministic counts are exact for matching selectors.

`qualify.py` runs these checks and freezes `aspect-r6b-qualified`, source and
executed-artifact manifests. Previous R6a artifacts are hash-checked with only
the intentional current CMake/inventory edits excluded. Its instruction document
and unrelated local tmp files remain unchanged.

## Completed verification

The pre-edit boundary table remains valid. Eight source/protection checks,
independent driver/helper builds and nine unique definitions (three driver
methods in 2D/3D plus three dimension-independent formatters) pass. Release
build and candidate plugins pass. All **18 runtime cases and 170 comparisons**
pass on one/two ranks. On BP3 runs emit 512 exact 11-column data rows across
seven timesteps, exercising append after iteration zero; rollback emits 16
active-bound rows. Silent blocked-open behavior and the complete accepted-update
rollback marker match. Both empty and "0" presence values enable the diagnostic.

Physical/history CSV/VTU fields, statistics, decisions, bound rows, summaries
and same-selector deterministic work counts match exactly. Within each binary,
off/on physical state is identical. The selector deliberately adds 16 lower-rate
probe evaluations: BP3 bulk-work/normal-filter call counts are 39 off and 55 on.
`off-on-work.json` retains all measured differences; elapsed timings are excluded.
The initial symbol checker counted lambdas as extra method definitions; the
matcher was corrected, with no source/runtime change.

Candidate: `build-refactor-r6b/aspect-r6b-qualified`, SHA256
`cdf7e0c58009d11dfe8c216caeb31ba6d8edc08b71360fec9fb63fcd95287596`.
Qualification records 865 source/header/unit entries, 40 executed artifacts,
and 25 build/runtime outcomes. Existing R6a artifacts remain intact except the
explicit rolling inventory/build-list edits. No public headers or other source
files changed. Local temporary edits are hash-preserved.

No numerical or MPI correctness defect demonstrated. No Debug/3D runtime,
long/restart campaign or injected mid-write failure; ordinary BP3 rows do not
exercise prescribed-node flags, while the existing rollback exercises active
bounds. Frozen evidence is reused because its observer/operator/output contract
is untouched. Prior scientific/cache gaps and historical failure reports remain.
R6b is uncommitted for review. Stop before R6c/R7. Proposed next bounded task:
surface stress-sample/weak-moment CSV formatting only, retaining all evaluations
and capture points in their present owners.

## Accepted R6b baseline

The user accepted the selected R6b pass and requested its commit before R6c.
The commit containing this entry records the accepted source; all 170 comparisons,
eight source checks and independent/linked symbols were rechecked before commit.
The immutable executable and manifests above remain the reference. R6c selects
benchmark setup boundary assessment under the existing callback/timing contract.
Unrelated local refactoring/tmp files remain outside this commit.
