# BP3 server computing statistics — 31 October 2026

The supplied production mature-DG logs now have timing and convergence figures.
This is read-only analysis of the existing BP3-derived trajectory, independent
of the newer cohesive-interface and opt-in objective-spin implementations.

- [Combined four-panel PNG](../../../build-particle-free/bp3-computing-statistics/computing_statistics.png) / [vector PDF](../../../build-particle-free/bp3-computing-statistics/computing_statistics.pdf).
- [Python script](../../../benchmarks/reconstructed_fault/particle_free/plot_bp3_computing_statistics.py), [summary and input hashes](../../../build-particle-free/bp3-computing-statistics/summary.json).

The revised figure uses the requested order: timers top-left, physical timestep
top-right, final accepted residuals bottom-left, and linear effort bottom-right.
The accepted histories are merged by step with no start/resume colours or labels.
For each timer section, the final invocation seconds are summed once and divided
by the sum of final invocation wall times (222,600 s, approximately61.8 h).
This gives61.1% condensed solves,13.6% checkpoint writing, and7.6% RHS assembly.
These are wall-time-weighted fractions, not an arithmetic mean of percentages.
The per-log accounting below and in sidecars remains for reproducibility; the
visualization presents a single run. Earlier two-figure exports are retained
as historical outputs; the script now generates only the combined figure.

Reproduce from the repository root (NumPy and Matplotlib required):

```sh
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/plot_bp3_computing_statistics.py
```

Optional positional log paths replace the two defaults, `output_start/log.txt`
and `output_resumed/log.txt` under `bp3_dg_first_event/`. `--output` selects the
output directory. CSV files preserve accepted-step statistics, all nonlinear
residuals including unaccepted attempts, all timer sections, and timing windows.
Identical log copies and overlapping accepted steps are rejected, so stdout
copies and a replay cannot silently count twice.

## Measurements and coverage

| Quantity | Start | Resumed |
|---|---:|---:|
| MPI ranks | 160 | 160 |
| Explicit accepted records | 816 (0–815) | 3422 (816–4237) |
| Last complete cumulative timer table | step 800 | step 4200 |
| Invocation wall time at that table, rounded | 52,600 s (14.6 h) | 170,000 s (47.2 h) |
| Condensed linear solve fraction | ~68.3% | ~58.8% |
| Checkpoint writing fraction | ~10.9% | ~14.4% |
| Stokes RHS assembly fraction | ~6.3% | ~8.0% |
| Median wall seconds / accepted step across timer windows | 61.96 | 49.56 |
| Median summed linear iterations / accepted step | 143 | 97 |
| Range of summed linear iterations / accepted step | 65–356 | 91–120 |
| Nonlinear residual evaluations / accepted step | 3–5 | 4–5 |
| Maximum final bulk normalized residual | 7.145331e−12 | 6.285993e−12 |
| Maximum final fault normalized residual | 9.940653e−9 | 9.996402e−9 |
| Logged line-search rejected candidates in accepted steps | 0 | 0 |

Fractions divide the printed section seconds by the printed **invocation** wall
time; their precision is limited by those rounded numbers. The 8 start and 34
resumed tables are cumulative: the figure combines only the last of each, without
summing repeated reports. Bars show independent timer sections. Nested timers,
uninstrumented work, omitted small sections and rounding prevent treating these
bars as a complete disjoint partition. No synthetic “other” slice is inferred.
The last tables omit the final 16 start and 37 resumed accepted records, plus
the unaccepted tails. They are not full-job completion timings.

Window costs difference consecutive integer-second “including restarts” clocks
**within each invocation**, then divide by the number of explicit accepted
records between those tables. The inherited restart offset cancels. The first
startup/resume window is omitted because its starting timestamp is unavailable;
no cross-restart difference or subtraction of low-precision section timers is
used. Each retained window here contains 100 accepted steps. The cost includes
checkpoint/output overhead. Differences between segments are not a controlled
speedup experiment: state, timestep and solver effort changed. MPI wall seconds
are not summed rank CPU seconds or scaling efficiency.

Both accepted residual components fall below the logged 1e−8 nonlinear
tolerance. The recorded linear fresh residual / target ratios are at most
0.999960 (start) and 0.999654 (resumed). Only final linear-solve residuals and
iteration counts are logged; an intra-Krylov residual trajectory cannot be
reconstructed. Nonlinear iteration indices include the final residual check,
so they are not counts of linear solves. At resumed step **4136**, five residual
evaluations replace four; the extra correction explains the sudden reduction
in final fault residual and the increase in linear work shown near the end.

The parser retains but excludes from accepted curves the failed start attempt
816 and the interrupted resumed attempt 4238. In particular, step4238 has small
nonlinear residuals but no accepted-state record. Acceptance cannot be inferred
from those residuals alone. See [BP3 review](BP3_REVIEW.md) for the observer stop,
scheduler time limit, incomplete accelerating event, and physical-accuracy limits.

## Verification and preservation

Entry root is `/home/ein/repository/aspect-pf-rsf-particle-free`, branch
`pf-rsf-particle-free`, revision `27178813cebd471b93a7a9308f41215b71b4a20b`.
Existing dirty source and manual work were retained; the remote executable is
not assumed to match this revision. No solver, build, checkpoint or server job
was run or changed. Prior cleanup availability losses remain unchanged.

Three focused checks pass:

```sh
OPENBLAS_NUM_THREADS=1 python3 -m unittest discover \
  -s benchmarks/reconstructed_fault/particle_free \
  -p test_plot_bp3_computing_statistics.py -v
```

These cover cumulative-table selection/record retention, an apparently
converged but unaccepted tail, nonzero inherited restart clocks, repeated
unaccepted attempts, and rejection of decreasing clocks. The combined figure was
visually inspected. Input hashes and compact preservation checks are in
[verification.json](../../../build-particle-free/bp3-computing-statistics/verification.json).

The next justified performance experiment, if selected, is a bounded fixed-input
profile of the condensed linear solve, including its preconditioner and operator
costs. Checkpoint cost also merits measurement at unchanged safety/retention
settings. The current plots do not justify loosening tolerances, reducing
checkpoint protection, or launching another long event replay.
