# Bounded K2 Release performance pass

2026-09-10. No convergence campaign was resumed. Completed spatial results
and accepted partial temporal results remain unchanged. This pass measures
the accepted discrete formulation, adds diagnostic instrumentation, and reuses
geometric `I_h` point lookups. No equation, surface quadrature, initialization,
history, support, full `I_h`, tolerance, or iteration budget is changed.

## Scope and preserved comparison

All artifacts below are under
`benchmarks/reconstructed_fault/uniform_shear/nonuniform/performance/`.

- `accepted-aspect-release`, `pre-performance.patch`: executable/tracked diff
  before this pass; earlier accepted-source archives remain untouched.
- `instrumented-aspect-release`, `instrumented.patch`: complete uncached
  comparison implementation. The first instrumented short run is `baseline`.
- `baseline.prm`, `optimized.prm`, `accounted.prm`: the same accepted 32-cell
  fixture through initialization and one real 0.5-second step, with separate
  output directories. Only end time/output location differ from `space32`.
- `comparison.json`: positive convergence checks and byte-level comparisons.
- `fine-ih.log`, `fine-ih.resources.json`: representative saved fine-state
  kernel benchmark, with no mechanics or particle advection.

Release builds use `cmake --build build-pf-cpdi --target aspect.exe.release -j4`.
The benchmark plugin was rebuilt with `-j4`. Runs are sequential. Host process
inspection confirmed the one-rank application pinned to CPU 0, with no other
ASPECT convergence process running; the fine kernel used approximately one
full CPU. Reported times are single observations, not confidence intervals.

## Measured operation and optimization

In the first short replay, `I_h` preparation accounts for 11.31 of 33.97 seconds
inside ASPECT. A separate initialization-only breakdown finds 1,295 adaptive
batches: lookup construction takes 4.087 of 5.589 seconds of `I_h` preparation;
all FE evaluation including lookup takes 4.250 seconds. Surface nonlinear
integration is not the dominant measured operation.

`PhaseFieldFault` now retains the last preparation's batch-coordinate arrays
and deal.II `RemotePointEvaluation` maps. The invariant is **geometry only**:

- Every preparation restarts the adaptive kernel and evaluates current FE
  values, material mixtures, degradation, and cell-diameter reductions.
- Reuse requires coordinate equality, matching mapping/triangulation objects,
  and deal.II's valid-mesh flag. An MPI minimum makes a mismatch on any rank a
  collective rebuild, including ranks with no local requesting points.
- Unused trailing batches are discarded. Mesh-deformation configurations
  clear the sequence before use, since mapping motion need not signal a
  triangulation change. No cached value can replace a current phase field.
- Caches are reconstructible, uncommitted data; failure/rollback and restart
  do not publish them as histories. No new physical or numerical parameter is
  introduced, and no adaptive-panel sorting or acceptance formula changes.

The short second preparation reuses all 1,295 batches, containing 1,571,184
point requests (including batch duplicates and boundary probes). It performs
zero lookup rebuilds. The first preparation still constructs all lookups.

## Short-run timings and counts

The following algorithm-isolated comparison uses the same initial timing
instrumentation on both sides. Later instrumentation was made rank-local;
those final verification artifacts are recorded separately below.

| Work | Calls | Before (s) | Reuse (s) |
|---|---:|---:|---:|
| Entire ASPECT run | 1 | 33.97 | 29.37 |
| Entire executable, including startup/shutdown | 1 | 38.217 | 33.000 |
| `I_h` preparation | 2 | 11.310 | 7.345 |
| All surface property preparation, including `I_h` | 2 | 11.410 | 7.445 |
| Particle domains/CPDI | 3 | 2.403 | 2.399 |
| Parent association | 73,728 | 0.02272 | 0.02296 |
| Domain partition/quadrature | 45,696 | 0.04188 | 0.04282 |
| Particle cache validation | 33 | 0.01444 | 0.01544 |
| Particle cache builds, inclusive | 2 | 0.1440 | 0.1473 |
| Parent FE lookup construction | 28 | 0.3012 | 0.2978 |
| Parent FE sampling | 28 | 0.2694 | 0.2451 |
| Surface residual/K integration and reduction | 28 | 1.208 | 1.222 |
| Whole surface residual/K evaluation | 28 | 1.874 | 1.864 |
| `G` lookup construction | 13 | 0.1372 | 0.1362 |
| Repeated `G`, including sampling/reduction | 250 | 1.319 | 1.232 |

ASPECT time decreases by 13.54%; executable time decreases by 13.65%.
`I_h` preparation decreases by 35.06% across the cold-plus-warm pair.
Peak RSS increases from 492,556 to 600,516 KiB (about 105.4 MiB extra).
This is a time/memory tradeoff, not a free speedup.

Both particle cache builds admit 22,848 parents. They contain 70,227 domain
integration points initially and 351,417 after the first shear displacement.
Each build takes the straight path for all admitted parents: 22,848 straight
calls, zero general calls, and thus zero general-path candidate-segment tests.
Parent admission still scans 16 segments for each of 36,864 particles: 589,824
segment tests per build, 1,179,648 across the short run. The latter count is
derived from the unchanged exhaustive loop, not confused with the general
partition's separately instrumented pruning counts. No geometry was flattened.

The added accounting replay has 250 `B` actions (3.563 s), 13 `B`
linearizations (0.2047 s), and two history publications (0.04726 s).
The standalone bulk-slip residual entry point is not called by this fixture's
ordinary additive Stokes assembly. Standard assembly timers cover that path.

### Elapsed-time reconciliation

Inclusive nested timers must not be added indiscriminately. In `accounted`,
the disjoint regions listed in `comparison.json` sum to 22.62832 s.
The remaining 6.75168 s includes other setup/initial phase-field work, Krylov
preconditioner applications, linear algebra, constraints/reductions and
uninstrumented orchestration. It is not assigned to nonlinear surface
quadrature. Their sum is 29.380 s inside ASPECT. Startup/shutdown adds
3.61951 s, giving the measured 32.99951 s executable time. Local initialization
timers that enclose domain generation and all nested fault timers are excluded
from this disjoint sum. The remaining interval is not internally resolved by
this bounded pass.

## Accuracy and convergence

All 19 CSV exports of the optimized and accounting replays are byte-identical
to the uncached short replay and the accepted spatial run's initial/0.5-second
prefix: geometry, phase field, initial projections, bulk/particle data,
histories, slip rates and actual weak surface loads are unchanged.

Each replay has two genuinely converged states and 13 passing fresh linear
residual checks. `compare.py` explicitly requires both final nonlinear
criteria; neither exit zero nor an export message is treated as convergence.

| Time (s) | Bulk residual | Bulk target | Surface RMS (Pa) | Surface target (Pa) |
|---:|---:|---:|---:|---:|
| 0 | 6.73042e-12 | 1.58656e-5 | 7.12121e-8 | 9.51684e-6 |
| 0.5 | 4.83824e-12 | 9.99513e-8 | 6.16994e-9 | 3.39951e-6 |

Exact saved-data equality permits reuse of the accepted measurements:
omitted fraction 5.83027366e-5; maximum actual local slip-normalization error
5.23648706e-5 at both times. The separate 1e-4 requirements remain satisfied.
No initialization discrepancy is subtracted or reset. No new spatial/temporal
convergence conclusion follows from this performance check.

## Representative fine-state check

The saved 128x512 scalar Q1 phase field and actual reconstructed polyline are
loaded, preserving the four-root-cell/seven-refinement hierarchy. The nine
profiles are the three production Gauss profiles on segments 0, 32 and 63.
Each uses the full normal path, production degradation and adaptive kernel,
`ell=0.15625`, and unchanged quadrature/tail tolerances 1e-10.

| Quantity | Result |
|---|---:|
| Cold preparation of the nine profiles | 18.653252 s |
| Same preparation with lookup reuse | 0.306754 s |
| Kernel speedup | 60.81x |
| Adaptive batches per preparation | 3,505 |
| Stored point requests | 812,385 |
| Cold/warm integral differences | Exactly zero, bitwise |
| Difference from saved interpolated nodal `I_h` | At most 4.14389e-11 m |
| Whole benchmark wall time / user CPU | 23.07863 / 19.58188 s |
| System CPU / peak RSS | 1.94606 s / 305,508 KiB |

The saved-nodal comparison includes a profile-versus-Q1-projection difference;
it is not a substitute for the exact cold/warm comparison. Kernel setup,
loading, reporting and teardown account for 4.11863 s outside the two timed
integrations. The successful run has 262,217 passing assertions. This is not
a 60.81x whole-trajectory speedup claim or a complete fine-trajectory replay.

### Bounded attempts and cold-search uncertainty

Two preceding all-profile harness attempts hit the 300-second wall cap and
were stopped by the runner; neither is counted as a passed numerical test.
The first incorrectly created all fine cells as coarse trees. That harness
issue was corrected to match production. The second demonstrably reached
cold integration but still exceeded the cap, at approximately 1.36 GiB RSS.
The first stack-sampling attempt lacked symbols across PID namespaces; the
explicit-executable retry resolves the active stack in deal.II's unaccelerated
`find_active_cell_around_point` fallback and inverse mapping. These logs are
retained. A boundary-probe trigger is plausible but not proved by one sample;
no point-found tolerance, search semantics, or boundary rule was changed.

The successful nine-profile subset bounds cost without shortening any profile
or changing accuracy. Full-fine cache memory and full-trajectory speedup remain
unmeasured. A rough request-count extrapolation suggests order 1 GiB additional
cache memory for the full fine state; that is an estimate, not an RSS result.

## Focused verification and instrumentation safety

`unit-one.log`: 59,912 assertions in 11 cases pass.
`unit-two.log`: the same 11 cases pass on both ranks, with 59,912 assertions on
rank zero and 3,098 on rank one. Filters are:

```
[phase_field_fault_ih_cache],[phase_field_fault_ih_accuracy],[fault_domain_quadrature]
```

New checks compare cached and fresh lookups with changed FE values, shared-face
and missing points, a coordinate change confined to one requesting rank,
empty requesting ranks, and mesh refinement. Existing analytic/distributed
`I_h` accuracy and manufactured domain moment/nonlinear quadrature tests pass.
The opt-in fine benchmark uses `[.fault_ih_performance]` and never runs as part
of the ordinary default unit collection.

An instrumentation audit found that ASPECT's ordinary TimerOutput inserts MPI
collectives at section boundaries. The initial diagnostic placements therefore
could not safely be retained in local particle loops. All added timers were
moved to subsystem-owned MPI_COMM_SELF TimerOutput instances. No new Simulator
lifecycle hooks or generic SimulatorAccess proxies are introduced. Summaries
are opt-in via `ASPECT_FAULT_PERFORMANCE=1`, report rank-zero-local times/calls,
and do not synchronize other ranks or change exception unwinding. Geometry work
counters retain their explicit collective sums at the existing cache-build
boundary. Final instrumentation verification is recorded in the addendum below.

## Files and recommended next action

Production changes are limited to private lookup/timing state in the
PhaseFieldFault, particle manager, reconstructed-fault manager/surface-system,
and Stokes-assembler headers/sources, plus optional domain-quadrature work
counters in `reconstructed_fault/utilities.h/.cc`. The only new public numerical
utility signature detail is the optional `DomainQuadratureStatistics*` output;
the quadrature algorithm and returned points/weights are unchanged. Test changes
are in `phase_field_fault_test_access.h` and `unit_tests/phase_field_fault_ih.cc`.
Benchmark wrappers/scripts and this verification record preserve provenance.

Retain the exact geometric lookup reuse, with its measured memory cost. If a
further bounded performance investigation is approved, prioritize the cold
`I_h` fallback-search path while preserving its point-found/tolerance semantics.
There is no measured justification here for reducing nonlinear surface
quadrature, replacing it with particle-center weights, changing support, or
altering the solver. The larger convergence campaign remains paused, the
partial temporal anomaly trend remains unresolved, and K2.3 is not started.
