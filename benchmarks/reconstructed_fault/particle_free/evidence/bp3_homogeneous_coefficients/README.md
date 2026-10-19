# Homogeneous Maxwell coefficient repair evidence

Version control retains reusable checks and selected compact results. Raw logs,
per-run records, detailed manifests, old binaries and package archives remain
local. Historical source/artifact hashes below keep their original identities.

[Cause, repair and server procedure](../../../../../doc/reconstructed_fault/particle_free/BP3_DG_HOMOGENEOUS_COEFFICIENT_FIX.md).
Basis: reviewed b2dfdebe43437d6028b06d4f9f862cc80a00cc71 plus the preserved
r3 optimization source snapshot. The provided server log is preserved as
`server-log.txt`. It reports 48 ranks and a step-zero failure on 552084 cells;
these local checks do not qualify that server stack or a full event.
**strict_reference_match_passed=false** remains.

- `runs/coefficients-before-v3/` reproduces the exact `p.eta_ve==eta` assertion
  using the unmodified pre-repair executable, two ranks, a 4136-cell oblique
  fixture, mixed chemical labels and the server's 4e6 s initial interval.
  The first two input-construction failures are retained separately.
- `assessment.json` reduces five repaired native controls: optimized/reference
  two-rank adaptive advancement; serial step-one checkpoint/four-rank continuation;
  completed one-owner rejection followed by identical-dt retry. All succeed.
  Accepted timesteps are 4e6 and 7.64e6 s. Every accepted observation publishes
  once; rejection/retry fields match the uninterrupted control exactly.
- `operator-checks.json` records uncached source/allowance equality and frozen
  B/G/C actions with hanging constraints, endpoints, variable friction, pressure
  scaling and restricted surface rows. The original exact homogeneity guard,
  cache invalidation and publication allowances remain active.
- `logged/bp3-coefficients-units.*`: 127 native cases / 165094 assertions pass,
  including 1001 mixture fractions, four averaging choices, three intervals and
  independent heterogeneous controls. The old harmonic kappa spread is 32 Pa s;
  configured homogeneous G and Maxwell coefficients now agree exactly.
- `negative-admission.json`: genuinely heterogeneous moduli still fail the
  original material admission guard. The first direct MPI launch failed to
  open sockets in the sandbox; the bounded authorized rerun reaches the intended
  material rejection. Both logs remain.
- `parameter-validation.json`: all four unchanged server inputs validate with
  the rebuilt ordinary plugins; test macros are absent. Parser validation does
  not construct the full mesh. `source-state.json` and `entry-preservation.json`
  identify source changes and preservation of incoming work/logs/old archives.
- `bp3-homogeneous-coefficients.patch` and `patch-verification.json`: six-file
  repair against the r3 source snapshot. Applying it to copied original files
  gives the tested source bytes. r4 already contains the patch.

Raw outputs/checkpoints remain under
`build-particle-free/bp3-optimization/runs/coefficients-*`; core/plugin build
logs and the preserved failing core/plugin binaries are under
`build-particle-free/bp3-homogeneous-coefficients/`. Large fields and workstation
binaries are not shipped. `runs/` contains exact commands, input/artifact hashes,
logs and compact acceptance records. `build/` and `logged/` retain build/wrapper
evidence. Reported runtimes here are provenance, not a new performance comparison.

With the matching local build/plugins, reproduce the repaired operator control
using a fresh output label:

```sh
python3 benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py \
  bp3-coefficients-replay 240 python3 \
  benchmarks/reconstructed_fault/particle_free/run_bp3_optimization.py \
  coefficients-replay --adaptive --vary-bulk-composition --operators
```

`verify_native.py`, `analyze.py` and `validate_inputs.py` preserve the bounded
local orchestration/reduction. They refer to local build paths and refuse/reuse
specific output directories as documented by the helper. They are not server
runtime controllers. The server continues to use direct native
`ibrun ./plugin/aspect-release startup.prm`, followed by inspected continuation.
Existing first-event accuracy/resource limitations remain.
