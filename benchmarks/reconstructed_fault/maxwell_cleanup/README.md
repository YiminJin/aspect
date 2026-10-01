# Maxwell coefficient naming cleanup

`eta_ve` now names the viscoelastic viscosity formerly called `kappa` in
`MaxwellCoefficients`, material point responses, coupling caches and their C++
callers. The stable formula `-eta * expm1(-dt * G / eta)` and all uses are
unchanged. The design/specification identify the equivalent mathematical
notation; legacy diagnostic CSV column names remain `kappa` so saved analyses
and checkpoint workflows retain their existing formats.

At the user's request, `evaluate_frozen_maxwell_stress` is restored exactly to
its original implementation, including `return coefficients.beta * old_stress`
(with coefficient preparation inline). It supplies only the frozen history load;
`compute_maxwell_stress` also includes the current-strain contribution, which
Stokes assembly handles separately. No delegation or extra zero-strain arithmetic
remains. No scientific algorithm, state ownership, tolerance or parameter changed.
The R3b work and user's local documentation moves are preserved.

Verification against qualified R3b:

- Core Release build, rebuilt frozen-stress/temperature/surface test plugins,
  and maintained BP3 plugin pass. The final core rebuild follows restoration
  of the original frozen-stress method.
- Existing units pass 20,816 assertions in 26 cases per rank on one/two ranks.
- Existing frozen-stress and temperature fixtures pass on one/two ranks, with
  12 exact statistics/solver/assertion comparisons. Frozen-load tests include
  zero, constant and nonuniform retained FE stress.
- Four short legacy/automatic BP3 runs and the automatic one-to-two-rank
  old-checkpoint continuation pass: 405 field groups and 30 cache/work checks
  match R3b exactly. No numerical differences were observed; timing is excluded.
- Source verification permits only the rename, comments and updated assertion
  text across 25 C++ files. Another 1,309 entry C++ files are unchanged. The
  frozen-stress method is byte-identical to its pre-task version.
- No 3D or long/full BP3 run, or runtime of the updated historical/BP5 diagnostic
  callers, was performed. Earlier cohesive convergence limitations were not
  addressed. No tolerance or physical parameter changed.

Reference: `build-refactor-r3b/aspect-r3b-qualified`, SHA256
`fbd3a74e0238c0fcf5026c9ec8b554f5296302fe14f3d0de4d986096d2c109db`.
Candidate: `build-refactor-r3b/aspect-maxwell-qualified`, SHA256
`1cea4bfcc506594374e9b4556d7000fd059360d532070cbd1f190de32cdcc7b5`.
Both reference and candidate binaries are preserved.

Commands, entry snapshots, generated inputs and individual outcomes remain under
ignored `evidence/` and `inputs/`; the runner refuses to overwrite logs. The
source harness consists of `run_checks.sh`, `compare_frozen.py`,
`compare_bp3.py`, `verify_source.py` and `qualify.py`. Reproduction needs those
captured snapshots, qualified R3b outputs and the prior BP3 checkpoint. All MPI
invocations use the existing single-thread/BP3 environment recorded in the JSON
logs. The suite exit code alone is not the verdict: the verifier checks each case.

This cleanup is saved separately from R3b commit `4d2f14f53`.
Use the candidate executable above and `evidence/qualified-source-hashes.json`
as the reference for a subsequently selected R4 pass. The previously proposed next stage is
separately selected R4a solver-method relocation; it is not part of this cleanup.
