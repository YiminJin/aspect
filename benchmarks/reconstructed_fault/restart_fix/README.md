# Restart transient-state correction

The correction adds one transient initialization in
`ReconstructedFaultManager::rebuild_after_deserialization()`: one empty prescribed
rate map per reconstructed fault. Its short comment leaves reapplication of
actual prescribed rows with the caller. Serialization, physical parameters,
solver tolerances and history assertions are unchanged.

Reference: the completed R3a/investigation commit `8490431b5`, with preserved
executable `build-refactor-r3a/aspect-r3a-qualified`. The correction is committed
separately. For subsequent R3b comparisons use
`build-restart-fix/aspect-r3a-corrected-qualified`, SHA256
`2bd6dca29898f79eff45ca616e9db6f86e9d3c6611ee867b57d289c85798436b`.
`evidence/corrected-r3a-source-hashes.json` and `qualification.json` record the
source and executable/plugin identities. R3b has not been implemented.

## Verification

- Separate candidate compilation/link passed. Only manager and unit-test unity
  groups were rebuilt, without PCH, retaining Release optimization and all
  remaining R3a objects. The reference objects/executable were not overwritten.
- The new `[fault_slip_restart]` regression fails with SIGSEGV/139 when linked
  with the original manager object and passes with the correction. Its two
  archive sections cover absent and previously configured prescriptions, an
  absolute trial without a setter after load, accepted-Newton rollback,
  caller-reapplied masks/values, rejection of a violating trial before mutation,
  and final commit. Together with existing prescribed-rate, checkpoint and
  Stage-I tests, 20,173 assertions in 17 cases pass per rank on one/two ranks.
- The original one-rank cohesive checkpoint loads with all original restored
  history/V/geometry/bulk assertions passing. It no longer crashes. Fresh and
  restarted runs both retain step-two nonlinear nonconvergence (exit 1). All
  phase-field/linear/nonlinear iterations, residuals and line-search decisions
  match, including the pre-fix uninterrupted run. Stream formatting and
  stdout/stderr interleaving differ; numeric comparison uses exact doubles,
  with no tolerance. Accepted step-one checkpoint meshes, particle payloads,
  history/V/geometry/bulk fingerprints and fresh-run statistics match exactly.
  Fifteen cohesive checks pass. There is no accepted step-two state to compare.
- Four short legacy/automatic BP3 runs on one/two ranks and an automatic
  one-to-two-rank old-checkpoint continuation pass against the qualified R3a
  outputs: 405 exact field groups and 30 exact cache/work checks. Solver
  decisions and accepted fields are unchanged; timings are excluded.
- BP3's nonempty prescribed-row reapplication contract is checked in the unit
  regression. The short maintained BP3 fixture itself reattaches an empty map;
  the full 200-km reference's deep prescribed rows use the same setter. No full
  200-km runtime was performed. No 3D or long production simulation was run.
- All 5,822 other entry source/header/test files are unchanged. In particular,
  material lifecycle ordering, both R2 files, checkpoint serialization and
  solver acceptance responsibility remain unchanged. The prior cross-rank
  cohesive observer limitation and step-two nonconvergence remain separate.

## Reproduction

Run from the worktree root with the qualified R3a build/plugins and previous
one-rank cohesive/reference BP3 checkpoints available:

```sh
python3 benchmarks/reconstructed_fault/restart_fix/build.py
python3 benchmarks/reconstructed_fault/restart_fix/stage_inputs.py
bash benchmarks/reconstructed_fault/restart_fix/run_checks.sh
python3 benchmarks/reconstructed_fault/restart_fix/compare_cohesive.py
python3 benchmarks/reconstructed_fault/restart_fix/compare_bp3.py
python3 benchmarks/reconstructed_fault/restart_fix/verify.py
```

The final verifier also requires the captured entry source manifest and manager
snapshot under `evidence/`. Generated inputs, checkpoints, outputs and evidence
are ignored and retained locally. The runner refuses to overwrite logs; use a
fresh output/evidence directory for a repeated run. The suite shell continues
through expected failures; individual JSON exit codes and comparers determine
its verdict. The original cases and optional-observer diagnostic remain intact.
