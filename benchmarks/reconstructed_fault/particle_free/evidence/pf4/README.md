# PF4 compact evidence

Scope, equations, measurements and remaining dependencies:
[PF4_REPORT.md](../../../../../doc/reconstructed_fault/particle_free/PF4_REPORT.md).
This is a noncommitting prescribed DG transport experiment, without a limiter.
It does not advance native surface/phase histories or qualify coupled use.

- `assessment.json`:284/284 transport gates, independent operator/characteristic
  comparisons, separate quadrature/space/time errors, objective/non-objective
  rotation controls, mature direct/residual closure and signed-jump limitations.
- `transport-reference.tar.xz`: final four native mesh entries. It contains
  `pf4-final-{4,8,16,32}/record.json` and each entry's compact `run/*.csv`:
  actual cells/basis, DG tensor coefficients, fixed-time/error tables and the
  small independently checked operator/load data. No binary or particle dump.
- `provenance.json`:878 production hashes, seven complete run identities,
  compiled observer/checker/fixture identities, dependencies, incoming-file
  preservation, archive hash and byte-identical frozen controls.
- `fixed-profile-assessment.json`: same-final-core FE/DG comparison, source,
  authority, work and publication gates against the retained fresh baseline.
- `phase-assessment.json`:67 PF3 gates; byte-identical to the retained PF3 result.
- `units.log`, `pf1-control.json`, `standalone.json`:628 assertions/8 cases,
  94 independent checks, both DG translation units without unity/PCH.
- `final-checks.json`: portable-reference replay, source/artifact correspondence,
  preservation, C++ style, Python/JSON syntax, links and whitespace.

Full runs and immutable executable/plugin copies remain under
`build-particle-free/runs/`; build logs, incoming manifest and preliminary
assessments are under `build-particle-free/pf4/`. Earlier 4/8 probes and the
254-gate family remain intact. The final source adds spatially varying rotation
controls; no failed result or tolerance was replaced. Deliberate wrong-density
and omitted-commutator solutions remain documented negative controls.

To replay without ASPECT, extract into an unused directory and run from the
repository root (replace `/tmp/pf4-reference` with that directory):

```sh
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/assess_transport.py /tmp/pf4-reference/pf4-final-4 /tmp/pf4-reference/pf4-final-8 /tmp/pf4-reference/pf4-final-16 /tmp/pf4-reference/pf4-final-32 --output /tmp/pf4-assessment.json
```

The checker requires NumPy, refuses an existing output, and uses physical
coordinates/characteristics rather than production integration routines.
Sampled jump extrema and measured quadratic-norm loss are not limiter or full
energetic qualification; surface/localization histories are diagnostic data.
