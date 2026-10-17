# Pre-PF7 compact WENO candidate audit: local evidence

The [measured report](../../../../../doc/reconstructed_fault/particle_free/WENO_CLS_REPORT.md)
and [frozen formulation](../../../../../doc/reconstructed_fault/particle_free/WENO_CLS_FORMULATION.md)
describe the controlled Q2 adaptation, thresholds, checks and limitations.
The native runs improve sharp-pulse accuracy relative to the original WENO
candidates, but **do not pass temporal qualification** and introduce transverse
variation in x-only fixtures. Unlimited DGQ2 remains the experimental baseline.
Implementation verification passing does not imply scientific qualification.
No PF7, adoption, threshold tuning or production-default change is selected.

- `assessment.json`: all 30 runs, local independent KKT systems, extrema,
  initialization, fronts/all-row ranges, tensor errors, direct/total quadratic
  norm changes, temporal differences/screens, means/flux balances, transverse
  diagnostics, held limiting and measured costs.
- `verification.json`: reproduction/regression, preservation and replay gates.
- `provenance.json`: revision, source/executable/observer hashes, run labels,
  bounded execution settings and archive digest.
- `comparison.png`: final dt/4 native polynomials and middle-row physical means.
- `staircase-late-trace.json`: three final-stage mean/beta/weight/interface/flux
  records at a representative staircase cell, extracted from archived CSVs.
- `reference.tar.xz`: portable raw CSVs, records/inputs/logs/patches, source and
  mathematical-script snapshots, frozen equations, failed preliminary evidence,
  regression outputs and all earlier baseline CSVs used for reproduction.
  Executables/DSOs remain in the worktree's run directories; the archive records
  their hashes rather than including machine-specific binaries.

The archive's report snapshot precedes the final replay closeout and late-trace
commentary; `provenance.json` also records the final document hashes. The complete
mathematical assessment is identical. The local standalone replay passed all
1,307 payload hashes, exact assessment reproduction and the checks listed below.

From a checkout with Python 3.12+ and NumPy/SciPy installed, using a new directory:

```sh
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/evidence/weno_cls/replay.py --destination /tmp/weno-cls-replay-NEW
```

The wrapper verifies every archived payload hash, runs the **archived** independent
scripts, checks exact assessment reproduction, repeats 69 limiter and 94 PF1
checks, and compares 23 operator and 63 original baseline CSVs byte for byte.
It runs no native solver. The full assessment takes a few minutes; intermediate
output is saved in the destination's `assessment.log`. Floating-point software
versions can affect byte equality; recorded versions and numerical residuals
remain the relevant evidence on other platforms.

The original and preliminary native jobs, pinned binaries, build logs and source
manifest are retained under `build-particle-free/weno_cls/` and
`build-particle-free/runs/weno-*`. The failed strict cached/uncached final-field
equivalence target is explicitly included in `cache-equivalence.json`; the local
KKT verification passes. Earlier PF0–PF5 evidence is unchanged. These large local
artifacts are not automatically selected for a source commit.
