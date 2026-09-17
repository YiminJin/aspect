# Exploratory modified-BP3 long-run inputs

This is **not official BP3** or a spatially converged earthquake benchmark.
It retains the 200 x 100 km centered box, fully frictional continuous-Q1
fault, mature C=0, ell=400 m, work measure, paired endpoint treatment,
immutable effective background and plate loading. Only the special bulk/fault
refinement around 40 km is removed.

`prepare_long_fixture.py` is the reproducible generator. Its maintained
`ordinary_central_cells.txt` was recovered from the pre-junction-refinement
saved tree (`junction-matched-qualified-local4/coarse/target_cells.txt` in the
2026-09-15 recovery archive). It uses the maintained wide fixture for side
strips and the maintained research fixture for immutable physical inputs.
It does not depend on those investigation directories at execution time.
`manifest.json` records SHA-256 hashes; `preparation.json` records checks.

- 36,194 cells, versus 42,968 before removal.
- 1,156 vertices / 1,155 segments, versus 1,236 / 1,235 before removal.
- Fault spacing: 99.960336209–100.000000000013 m.
- All fault-crossed cells: square side 97.65625 m; ell/h=4.096.
- Cells grade to 25 km far from the fault.
- The endpoint mesh and all 28 nonzero completion integrals are unchanged.
- Prestress coefficients are restricted at retained physical vertices. Their
  denominator belongs to the immutable background; it is not replaced by
  current mesh-dependent Ih or recalibrated from the new solution.

The source/input hash checks reject incompatible tables. The actual exported
mesh and fault coordinates passed the checks in `long-run-preparation/`.
See [launch/output instructions](../../LONG_RUN.md) and the
[preparation report](../../../../../doc/reconstructed_fault/bp3/stage_K5_long_run_preparation.md).
