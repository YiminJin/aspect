# PF3 compact evidence

Selected scope and interpretation:
[PF3_REPORT.md](../../../../../doc/reconstructed_fault/particle_free/PF3_REPORT.md).
All data are from this worktree's uncommitted PF2/PF3 source at the recorded
HEAD; hashes identify actual files and binaries. No coupled trajectory or
transport qualification is implied.

- `assessment.json`:67/67 independent phase energy, derivative, quadrature,
  bounds/redistribution and approximation checks. Includes the nonmonotone coarse
  phase errors and the asymptotic refinement sequence.
- `failed-initial-assessment.json`: preserved failed raw-Hessian symmetry and
  preliminary all-mesh-monotonicity assumption, before arithmetic grouping and
  reference extension. Its early manufactured helper used m=10; final study
  reads the actual native m=128. Native physical inputs never changed.
- `fixed-profile-assessment.json`: final same-core FE/DG comparison against the
  retained source-proven fresh baseline. `provenance.json` additionally verifies
  byte-identical FE baseline and DG PF2 bulk/surface files.
- `provenance.json`:876 production hashes, all three complete run identities,
  observer/checker identities, dependency versions, source correspondence,
  incoming-file preservation and reference archive hash.
- `native-reference.tar.xz`: all compact CSVs for the final fixed incoming entry,
  including physical geometry, DG samples, Z, native CPDI weights, residuals and
  Hessians. The archive contains `run/*.csv`. It is sufficient to replay the
  independent checker without ASPECT or the original local run directory.
- `units.log`, `pf1-control.json`, `standalone.json`:628 assertions/8 cases,
  94 independent checks and standalone syntax compilation of all four affected
  production translation units without unity/PCH.
- `final-checks.json`: archive/checker replay, C++ style, Python/JSON syntax,
  documentation links, whitespace, source correspondence and preservation checks.

Full runs, expanded PRMs, immutable executable/plugin copies, raw logs, build logs
and preliminary/failed artifacts remain under `build-particle-free/`. Local run
labels are `pf3-phase-first`, `pf3-phase-final`, `pf3-fe-control` and
`pf3-dg-control`; PF2's baseline and all historical outputs remain intact.

To replay the portable reference, create an unused directory, extract the archive
there, then pass that directory (which now contains `run/`) to:

```sh
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/assess_phase_energy.py /tmp/pf3-entry --output /tmp/pf3-assessment.json
```

The checker requires NumPy/SciPy, refuses to overwrite its output, and runs no
native mechanics. Native reproduction commands and limits are in the report.
