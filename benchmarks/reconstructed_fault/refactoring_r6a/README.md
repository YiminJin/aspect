# R6a: material-history diagnostics

Reference: accepted post-R5 commit `3b4ae16dd`, branch `pf-rsf-refactor`.
Immutable executable: `build-refactor-r5b2/aspect-r5b2-qualified`, SHA256
`edf23c823e86fe579a231110f2e2167dd929fa531996501516759d8faa432cd2`.
Accepted 248-check evidence and all artifacts/protections were rechecked before
commit; no R5 simulations repeated. User R6 instructions and temporary files
remain unmodified/uncommitted. Scientific worktree remains untouched.

## Boundary and lifetime recorded before editing

| Responsibility | Retained owner/timing and selected change |
|---|---|
| Switch reads | M4 compute_history_candidates, every actual history preparation, after samples and before first candidate try/catch; two presence tests remain at those sites. Empty and 0 both enable. |
| Setup/filter/presentation | New source-private per-call FaultHistoryDiagnostics owns only the same source stream, cell-ID set and stress stream, in that order. Opens/headers/selection/rows move to history_diagnostics.cc. |
| Lifetime | Recorder occupies the existing HistoryCandidates stack storage; streams survive validation/publication, and close on scope unwinding. No stored borrowed samples, persistent material field or checkpoint data. |
| Capture | After candidate validation and insertion, inside original particle loop/try: actual old parent stress, sampled gradient/coordinates, coefficients and computed candidate. Diagnostic-only computations remain under open/selection guards; no resampling. |
| Admission | Stress row requires open stream and selected surrounding CellId. Source row requires open stream, continued active association and original inactive surface association. Numerical source admission stays in M4. |
| Error/MPI | File setup remains outside candidate try/catch; rows inside; stream failure is silent as before. Missing cell file gives empty filter and stress header only. Rank queries only on enabled setup; no added collective or exception policy. |
| History | M4 computes; manager/particles store; solver accepts. Sampling, source continuation, collective validation, projection, publication, rollback and timestep-zero bypass stay unchanged. |

Preserve rank/timestep filenames, truncation, precision17, row/header order, SI
units and historical `kappa` label (actually eta_ve). No shared-stream formatting
changes. The other stress-cycle reader in initial_conditions.cc stays untouched.

## Planned focused evidence

Separate Release build; independent old/new TUs and 2D/3D helper symbols; source
proof for unchanged surrounding lifecycle/numerical blocks and public headers.
Short existing automatic-completion BP3 on one/two ranks, both diagnostics off/on
for reference/candidate; require nonempty selected-cell and continued-source rows.
Use a small fixture-only input writer if needed, without production accessors.
Compare exact diagnostic/physical/history fields, solver decisions and work counts
across versions and off/on. Isolated missing-selection and stream-open-failure
cases; existing accepted-update rollback. Keep deliberate failure cases distinct.
No repaired frozen rerun: observer/operator lifetime and its output contract are
untouched. No R6b/R6c, switch migration/removal, numerical fix or R7.

## Completed changes and verification

Candidate: uncommitted R6a over `3b4ae16dd`, immutable
`build-refactor-r6a/aspect-r6a-qualified`, SHA256
`d373cb3308ecc00fb05a574975cf55f9d65facea003464b48153fc6aecf88f01`.

The pre-edit boundary/lifetime table above remains valid after extraction.
Only history.cc, two source-private diagnostic files and one focused CMake
no-unity/no-PCH entry change production. Public headers and existing unity groups
are unchanged. The recorder never stores borrowed values or samples fields.
Setup/row expressions and literal strings match after explicit input substitution;
reversing only diagnostic extraction reconstructs the old history source exactly.

Release and independent two-TU builds pass; all six helper methods have unique
2D/3D definitions. Fifteen source/protection checks pass. Twenty runtime cases
pass on one/two ranks, giving **258** matched and neutrality checks. Each ordinary
on-run has six nonempty steps: 16,875 stress rows and 146 source rows per update
(2 ranks split these as 8,433/8,442 and 40/106). Both 30/20-column headers and
every data byte match reference/candidate; core stress-transfer output is also
compared. Physical/history fields, solver traces and work counters match exactly
between binaries and within each binary off/on. One-rank on uses switch value
"0", two-rank on uses empty values, explicitly verifying presence semantics.

Missing input on one rank leaves stress headers only and continued-source rows
nonempty. Two-rank failed-open runs put directories at both streams' expected
output filenames before launch; both versions continue silently with the same
physical results. Existing accepted-update rollback runs off/on on both rank
counts reach the full marker with matching traces. The small fixture plugin only
writes locally owned cell IDs at pre_set_initial_state; it does not adjust any
physical setting, tolerance, state or sampling algorithm.

Initial comparison flags concerned only TableHandler padding widths induced by
different output-directory name lengths. `comparison-initial.json` retains them;
the final comparison checks all exact statistics tokens after path normalization.
No scientific column is discarded, tolerance relaxed or simulation rerun.
Qualification records 27 final build/runtime outcomes, 863 source/header/unit
entries and 44 input/plugin/harness artifacts. All previous qualified artifacts
and unrelated local inputs remain intact.

## Inventory, reproduction and limitations

The [refreshed inventory](switch_inventory.md) updates the existing R0 summary
and standing R6 guidance. It records 87 literal readers, including 23 production
selectors, plus dynamic guards and parameter/setter/observer alternatives.
Parsing, rank participation, output/consumer and compatibility notes distinguish
passive output from extra evaluation/collective or numerical behavior. The old
`kappa` column remains eta_ve in Pa s; obsolete solver environment names are not
revived. No selector migration/removal or changed default.

`evidence/*.json` records exact commands/environment/results and full `.log`
files. Build plugins separately against R5b2/R6a packages, then `prepare_inputs.py`
and `run_checks.py reference` / `run_checks.py candidate` (MPI socket access).
`compile_independent.py`, `verify_source.py`, `verify_symbols.py`, `compare.py`
and `qualify.py` reproduce structural/comparison/manifest checks. Runtime logs
refuse overwrite. The extraction provenance and pre-edit source/table are retained.
Python syntax and git diff checks pass. Repaired frozen observer/operator evidence
is reused, not rerun, because these paths are unchanged.

Not run: Debug or 3D simulation, long/restart campaign, injected mid-write or
late history-validation failure; no new unit tests that merely mirror formatting.
The existing rollback fixture and failed opens are the exercised error coverage.
Historical singular-fixture wording, cohesive restart nonconvergence and broader
scientific/cache gaps remain. No numerical or MPI correctness defect demonstrated.
R6a is ready for review and uncommitted. Stop before R6b/R6c/R7. Proposed next
bounded task: R6b nonlinear-bound CSV/report formatting, keeping lower-rate residual
audits, nonlinear decisions, collectives and capture timing in the driver.

## Accepted R6a baseline

The user accepted R6a and requested its commit before R6b. The commit containing
this entry records the accepted source; the immutable executable and qualification
above remain the reference. Rechecked all 258 comparisons and source/symbol
checks before committing. R6b selects only nonlinear-bound CSV/summary formatting.
The supplied R6 instruction document is included unchanged; local refactoring/tmp
files remain outside the commit.
