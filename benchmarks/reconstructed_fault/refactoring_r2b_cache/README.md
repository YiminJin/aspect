# R2b proposal 3: private normalization reuse decision

Reference: completed proposal 2 in this worktree, including the user's local
R1/limiter/R2a/R2b edits. The 25 recorded proposal-2 artifact hashes matched at
entry. `build-refactor-boundary/aspect-boundary-qualified` is the preserved
reference executable; no reference build, plugin, input or output is modified.
The candidate build is `build-refactor-r2b-cache`.

Only `normalization.cc` and the private portion of `phase_field_fault.h` change
production code. The existing 48-line input/comparison/collective block moves
verbatim into `prepare_normalization_reuse(previous_cache_valid) const`.
Its private result record is defined in the implementation file. It retains the
original borrowed partition reference, moves the captured vectors, and returns
the phase-block index, composition-independence flag, collective result and
timing markers. There is no new persistent cache,
public API, collective, comparison or numerical policy.

`verify_extraction.py` restores the original caller text mechanically and demands
byte equality with proposal 2. Thus automatic qualification, invalidation,
chemical projection, counters, integration and final publication keep their
ordering. It also checks that every other entry source/header/test file is
unchanged. Suspected correctness fixes belong to a separately selected task.

## Reproduction

Run from the repository root, with the toolchain used for the reference:
GCC 12.4.0, OpenMPI 5.0.6, deal.II 9.6.2, Voro++, Release and
`-fno-finite-math-only -ffp-contract=off`. Exact configure/build commands and
outcomes are recorded in `evidence/*.json`.

1. Configure/build `build-refactor-r2b-cache` with the recorded options.
2. Configure `plugin` into `plugin-build` using the candidate `Aspect_DIR`, then
   build it. This includes the unchanged existing lifecycle tests plus a
   read-only per-rank cache-counter observer. The same plugin is loaded in both
   executables; the extraction does not change class layout.
3. Run `python3 benchmarks/reconstructed_fault/refactoring_r2b_cache/stage_inputs.py`.
4. Run `bash benchmarks/reconstructed_fault/refactoring_r2b_cache/run_checks.sh reference`
   and then the same command with `candidate`.
5. Run `bash benchmarks/reconstructed_fault/refactoring_r2b_cache/run_restart.sh`
   to branch the same accepted automatic checkpoint for both executables.
6. Run `python3 benchmarks/reconstructed_fault/refactoring_r2b_cache/compare_results.py`
   and `python3 benchmarks/reconstructed_fault/refactoring_r2b_cache/verify_extraction.py`.

Input creation and logs refuse overwrites. Runs use one thread per MPI rank;
local MPI requires socket access. Existing parameters/tolerances are retained.
The existing 50-iteration initialization allowance is reused. BP3 remains the
small 1,875-cell, steps 0–6 functional fixture, not a production cycle.

## Coverage and acceptance

- Existing `[phase_field_fault_ih_cache]` unit cases on one/two ranks.
- Existing `VerifyFaultIhCache` assertions: unchanged hit with no FE requests,
  rank-zero-only phase mutation forcing a collective miss, phase/geometry
  restoration, failure without publication, and restart invalidator recovery.
  Run remote and cell backends on one/two ranks, plus composition-independent
  and no-composition cases on one rank.
- Existing I_h lifecycle fixture with only completed-value reuse disabled,
  exercising warm cell traversals on one/two ranks.
- Matched legacy and automatic-completion BP3 on one/two ranks. Automatic
  qualification remains before warm reuse; its ordering is additionally
  protected by the byte-exact caller check.

Require exact fields and solver decisions, per-rank cache counts, mechanical
work/correction CSVs, and cell-work lines after removing only elapsed geometry
time. The source equality check retains the original collective min and
short-circuit order. No tolerance is weakened. The existing restart invalidator assertions are reused, and a short matched
automatic restart from one to two ranks exercises the returned values in frozen
normalization restoration. The initial build caught the two missing return
fields; that failed compile is preserved separately in `build.log`.

The first comparison-script run incorrectly required a prior hit in the first
cold restart observation. Both executables correctly report 0 hits/1 integration
at step 5 and 1 hit/1 integration at step 6. The coverage assertion now requires
both paths over the case; all field/counter equalities remain exact. The initial
failed comparison is retained in `evidence/comparison.log`.

Final result: the Release build and all 30 matched runtime invocations passed;
405 field groups are exact and all 96 lifecycle/work checks pass. No runtime
baseline failure or missing cache dependency was identified. `source-verification.json`
protects the two-file production scope and 5,834 unchanged entry files. The
qualified executable and artifact manifest are retained for the next review.
