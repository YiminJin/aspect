# Bounded Release performance pass

The expensive K2.2 convergence runs remain stopped. Their completed results in
`../domain-convergence/` are not modified by this pass. `accepted-aspect-release`
and `pre-performance.patch` preserve the executable and tracked diff before
instrumentation; the earlier accepted-source archive also remains available.

The first measurement is `baseline.prm`: the unchanged 32-cell spatial fixture,
ending after initialization and one real 0.5-second step. Expected run cost is
under two minutes and 0.6 GiB, based on the completed 76-second full 32-cell run.
Release-only builds use `-j4`. Runs are sequential, with CPU affinity recorded;
two one-rank jobs must not compete on the same MPI-pinned core.

Set `ASPECT_FAULT_PERFORMANCE=1` to print globally summed domain work counters
at each cache rebuild and rank-zero-local timing summaries at subsystem
destruction. The added TimerOutput instances use MPI_COMM_SELF: local loops,
cache checks and exception unwinding never introduce profiling collectives.
They remain distinct from ASPECT's ordinary synchronized timing table. Calls
and inclusive wall times are reported; nested entries must not simply be summed:

- `Cache build total` contains `Parent association` and `Domain partition`.
- `Property preparation` contains `I_h preparation` and may build the cache.
- `Surface R/K total` contains validation, parent lookup/sampling, and integration.
- `Linearization total` contains one surface R/K assembly and G lookup setup.
- `G total` contains G FE sampling and integration/reduction.
- Existing setup/postprocessing timers can enclose these categories too.

`straight calls` includes the straight integrator applied to exact polygon
subpieces of the general polyline partition. `general calls` counts whole
non-collinear admitted domains. Segment tests/candidates refer to the general
partition's finite-projection pruning, not a changed parent-admission policy.
Reported points are the final cached points, not intermediate partition points.

No quadrature order, equation, support, history, tolerance, or iteration budget
is changed for measurement. The current order-three implementation is retained
as the accuracy baseline before any measured optimization.

## Saved fine-state scope

`run_kernel.py` selects the opt-in `[.fault_ih_performance]` unit benchmark.
It reads `phase_0.csv` and the actual polyline from `surface_0.csv` in
`ASPECT_FAULT_PERFORMANCE_STATE`. The accepted 128x512 hierarchy is recreated
as four coarse cells refined seven times. Only scalar Q1 data are needed.
The bounded check integrates the three production Gauss profiles on segments
0, 32 and 63, including their full normal paths. Cold/reused results must be
bitwise equal. Differences from the saved nodal Q1 interpolation are reported
separately, since a profile value is not its consistent mass projection.

The original all-profile harness and a hierarchy-corrected all-profile attempt
each hit their 300-second caps; both logs/resource records are retained.
The latter was sampled in deal.II's unaccelerated cell-search fallback during
cold lookup. The representative nine-profile check takes 23.08 seconds total;
no fine mechanical trajectory is relaunched. These are bounded kernel checks,
not additional K2 convergence evidence.
