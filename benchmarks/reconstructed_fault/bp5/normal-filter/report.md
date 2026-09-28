# BP5 normal-input filter — bounded implementation/qualification report

**Decision: implemented and locally verified; no scientific BP5 filter verdict
yet.** The experimental option remains off by default. R/F1/F2 are prepared,
but no BP5 trajectory was launched and no actual-checkpoint filter figure is
claimed. The recovered step-5612 archive matches the already recorded local
deserialization failure. Changing serialization, reconstructing history from
CSV, or substituting the small inclined model would not answer this experiment.
The next action is the supplied bounded comparison on a checkpoint-compatible
server build, not production adoption of filtering.

## What changed

The simulator-side surface system assembles full-fault M/K with the actual
Stokes-QP work measure and physical arc-length derivatives. Constant continued
endpoint bases have zero derivative. Natural filter endpoints are independent
of slip constraints and plotting windows; unsupported rows fail without shifts.
The original raw input bypasses filtering. Zero length is consistent Q1
projection; nonzero length solves the specified Helmholtz system.

Background enters the combined normal RHS once. The constitutive material
remains pointwise and exposes mu_V. Every base/trial uses its current raw
mechanical field to build b; only factors are cached. K_V remains tridiagonal
because the direct fixed-bulk normal derivative is zero. Its sigma*mu_V uses
the filtered input. G includes the nonlocal H inverse and spatially varying mu;
the old local sparse G is deliberately bypassed for filtered branches. Bulk
equations, B, FGMRES, tolerances, state timing, histories and pressure convention
are unchanged. No filtered stress is published to particles or bulk fields.

This is an explicitly authorized experimental change to the friction equation,
documented in `current_design.md` and `specification.tex`, not a repair of raw
stress. The current work formulation supports one straight mature 2-D fault;
this task does not add a new geometry capability.

## Executed checks

| Check | Result |
|---|---|
| Release core and diagnostic/BP5 plugins, `-j4` | Built |
| Operator tests, one/two ranks | 91 assertions per rank passed |
| 50 MPa constant reproduction | max error 7.451e-8 Pa (1.49e-15 relative) |
| Independent dense inverse | max difference 2.981e-8 Pa |
| Whole-fault mean | max error 5.663e-8 Pa |
| 1000-Pa generalized perturbations | max analytic-mode error 6.95e-8 Pa |
| Smooth/intermediate/short mode attenuation, L=200 m | 0.615454 / 0.0769231 / 0.0204082 |
| Free-equation, varying-mu coupled FD | second-order contraction; final relative errors 9.247e-7 projected, 1.662e-6 Helmholtz |
| Isolated spatially varying nonlocal G FD | maximum relative error 4.178e-14 |
| One/two-rank residual difference | ≤1.777e-13 relative |
| One/two-rank filtered coefficients | ≤3.074e-8 Pa |
| One/two-rank K/G actions | ≤3.0e-15 / 9.8e-15 relative |
| Raw before/after filtered trials | matched; no slip-rate publication |
| Existing raw surface/bulk coupling fixture | passed |
| Complete R/F1/F2 PRMs | `--validate` passed; not a restart qualification |
| Offline/evolution analysis pipeline | synthetic test passed; no BP5 inference |

The free-equation test uses a prescribed frozen FE phase profile only to prepare
its geometry; **fault velocities are not prescribed**. V varies along the fault,
so mu is not uniform. An oscillating pressure perturbation exercises nonlocal G.
Perturbation sizes are 0.1, 0.05, 0.025, 0.0125. Trial reevaluation and the saved
linearization agree after noncommitting probes. Test filtered compression is
positive, approximately 2.20033–2.20107 MPa for Helmholtz. No clipping was added.
This verifies equations/actions, not a nonlinear filtered BP5 trajectory.

Logs and action arrays are in `verification/`; exact numbers, tested source and
binary/plugin hashes are in `summary.json`. HEAD was `33228369d`, with existing
uncommitted work preserved; HEAD alone does not identify this tested build.

## Actual checkpoint and remaining work

Metadata confirms accepted step **5612**, time **5310111071.5634108 s**.
`resume.z` SHA256 is
`1dbde7a33ff690400b2b3bff6395809c385cfd09f5693ced09fd799c264243b7`.
The checkpoint was recovered from the local cleanup archive without modification.
Its hash matches `stress-cycle-qualified-20260923/dt/staging.json`; that run's
`run.log` records failure before mechanics with:

```
Cannot seem to deserialize the data previously stored!
class version St6vectorIN6dealii5PointILi2EdEESaIS2_EE
```

This is pre-existing evidence, not a new filter failure. The precise library/
archive-schema cause was not expanded into another investigation. The supplied
`report(4).md` was absent; current routines were checked against the retained
`moment-inclined/traction-audit/report.md` and source instead.

The server capture distinguishes the first resumed Newton base (checkpoint
histories, pending stress_dt) from accepted precommit stress. It does not pretend
to recover the lost precommit quadrature snapshot of accepted step 5612. All
offline representations share the same captured base and mu. Full M, K and
friction-weighted M_mu permit exact offline load comparisons without reconstructing
a mass matrix from row averages.

The complete PRMs retain physical settings from the resolved original input;
only paths, diagnostics/limits, filter selection and the additional replay cap
differ. The runner stages independent copies, hashes dependencies, records actual
constitutive intervals and cumulative elapsed time, requires fresh checks, and
preserves failures. Ten states per branch are targeted. One explicitly requested
common half-clock retry is supported, with at most twenty accepted states per
branch across attempts. There is no automatic tuning or continuation.

**Unmeasured:** actual-checkpoint 100/200-m length suitability and endpoint
overshoot; filtered nonlinear convergence and active-set changes; history/slip
evolution and friction attribution; actual BP5 filter time overhead. Thus neither
length is qualified, and none of the proposed scientific outcomes can yet be
selected. `README.md` supplies exact staging/run/analysis commands. The analysis
produces at most two figures and reports different sampling measures explicitly.

## Reproduce the local checks

From the source root:

```sh
cmake --build build-pf-cpdi --target aspect -j4
cmake --build build-pf-cpdi/tests --target phase_field_fault_surface_dynamic_pressure -j4
build-pf-cpdi/aspect-release --test '[fault_normal_filter]'
mpirun -np 2 build-pf-cpdi/aspect-release --test '[fault_normal_filter]'
ASPECT_TEST_NORMAL_FILTER=1 ASPECT_FAULT_EXPLICIT_G=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 DEAL_II_NUM_THREADS=1 build-pf-cpdi/aspect-release benchmarks/reconstructed_fault/bp5/normal_filter_test.prm
# Use a fresh output-directory override for the two-rank invocation.
ASPECT_TEST_NORMAL_FILTER=1 ASPECT_FAULT_EXPLICIT_G=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 DEAL_II_NUM_THREADS=1 mpirun -np 2 build-pf-cpdi/aspect-release benchmarks/reconstructed_fault/bp5/normal_filter_test.prm
python3 benchmarks/reconstructed_fault/bp5/normal-filter/test_analysis.py
```

No broad ASPECT suite, loading prefix, event continuation, stress-transfer
substitution, or tolerance adjustment was performed.
