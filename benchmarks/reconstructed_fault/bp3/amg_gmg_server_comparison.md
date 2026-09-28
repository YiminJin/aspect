# BP3 server AMG / GMG comparison — September 26, 2026

The observed GMG slowdown is concentrated in the condensed linear solve:
**more outer iterations and more expensive block-preconditioner applications**.
The existing detailed profiles establish this without a new simulation or
reintroducing the removed GMG debugging code. They do not separate inner
velocity iteration counts from the cost of each multigrid cycle.

## Comparable portion of the supplied runs

Evidence: [AMG log](output-test-amg/log.txt), [GMG log](output-test-gmg/log.txt),
their `original.prm`, `parameters.prm` and `accepted_steps.csv`.

Both original and resolved parameter files differ only in output directory
and `block AMG` versus `block GMG`. Both logs report Release, 48 MPI ranks,
deal.II 9.7.0, Trilinos 16.2.1, p4est 2.8.7, 64-bit indices and AVX512/eight
doubles. Both reach 542,958 active cells, 19,149,424 total DoFs (4,505,676
velocity DoFs), and ten mesh levels. Binary hashes, job placement, CPU binding
and compiler/build flags for this pair are not recorded in these outputs.

Both still select `BP5 state startup` with limit 0.1, with first/global caps
`4e6 / 4e7` s. These are saved runs with the previous timestep configuration,
not evidence that the subsequently cleaned BP3 plugin still registers BP5.
The accepted clocks and nonlinear update counts match through state 2.
GMG records completion only through step 2; its log ends during step 3.
Therefore compare through completed step 2, not the final totals of the two
unequal-length uploads. Duplicate step-zero CSV rows are counted only once.

## Timing and iteration evidence

The ordinary ASPECT timing tables are cumulative. The per-step solve times
below are differences of those rounded cumulative entries.

| Step | Outer iterations AMG / GMG | Condensed solve AMG / GMG (s) | GMG setup inside solve (s) | Block preconditioner time/call AMG / GMG (s) |
|---|---:|---:|---:|---:|
| 0 | 80 / 84 | 76.3 / 91.0 | 1.52 | 0.499 / 0.607 |
| 1 | 43 / 55 | 39.7 / 55.0 | 1.46 | 0.415 / 0.514 |
| 2 | 60 / 66 | 57.0 / 89.0 | 2.54 | 0.504 / 0.839 |
| Total through step 2 | 183 / 205 | 173 / 235 | 5.52 | 0.481 / 0.657 |

Through that same accepted state:

- Total elapsed: AMG **340 s**, GMG **401 s** (approximately 18% slower).
- Condensed solve: **173 s versus 235 s** (approximately 36% slower).
- Ordinary Stokes-preconditioner construction: **11.2 s versus 11.3 s**.
- GMG's additional setup is **5.52 s**, already nested inside the 235 s solve
  timer. Do not add it again. Eliminating setup alone would leave about 229.5 s
  versus 173 s, so repeated hierarchy construction is not the main explanation.
- Outer iterations: **183 versus 205**, about 12% more for GMG.

The `Fault linear profile` entries give stronger localization than these broad
timers. Summing the seven actual solves through step 2 gives block-preconditioner
times of **88.066 s AMG versus 134.683 s GMG**. Mean time per application rises
about **36.5%**, from 0.481 to 0.657 s. At step 2 specifically, AMG needs 20
outer iterations per solve versus GMG's 22; the mean preconditioner application
is about **66% more expensive**. The same number of nonlinear updates is needed.

These profile values are **rank-zero, rank-local exclusive timings**, as
implemented by `FaultLinearTiming`; they are not an MPI maximum or a direct
measurement of V-cycle time. The preconditioner category contains the pressure
solve, approximate velocity inverse and associated vector operations. Summing
all profile categories as though they were the condensed timer would also mix
different timing scopes. The data nevertheless consistently localize the
dominant extra cost to repeated preconditioner application. Fine A and B/G
action times per call are broadly similar, while more iterations cause more
of these actions too. Small surface-inverse/factor times do not explain the gap.

## What this means for the GMG implementation

In [solver.cc](../../../source/simulator/solver.cc), reconstructed-fault GMG
replaces only the velocity-block preconditioner inside an **inner CG solve
using the assembled fine A matrix**. It does not make the entire coupled
solver matrix-free. The adapter copies between Trilinos and deal.II vectors
around each cycle. The pressure preconditioner, coupled operators and outer
FGMRES remain. The selected inner A tolerance is `1e-2` for both runs.

Consequently, slower block-preconditioner application can reflect more inner
CG iterations, a slower V-cycle, vector-copy cost, pressure work, or MPI
communication/waiting. The output does not count inner velocity iterations
or separately time these pieces. The current local-smoothing implementation
uses degree-four Chebyshev smoothing on nonzero levels and degree eight at
the coarse level; the logs alone do not prove these choices are inadequate.
The measured larger outer iteration count establishes less effective overall
preconditioning for these particular systems, not a failure to converge.

An AWK count of all 48 GMG `initial_mesh_*.csv` files finds **11,308–11,314
owned active cells per rank**, mean 11,311.625 and total 542,958. Gross active-cell
imbalance is therefore not the explanation. Active-cell balance does not prove
balance of multigrid levels, interface work, ghost communication or hardware
placement. Those remain plausible contributors, not verified causes.

## Why the earlier local result does not transfer directly

The earlier [frozen comparison](../../../doc/reconstructed_fault/bp3/stage_K5_gmg_prototype.md)
used a different 200 x 100 km model and selected step-2/Newton-4 operator on
**four ranks**, with **42,968 cells**, deal.II 9.6.2, Trilinos 14.2.0 and
four-wide SIMD. Its actual [comparison CSV](../performance/gmg/frozen-wide-verified-local4/frozen_gmg.csv)
has **17 iterations for both backends**, and GMG's preconditioner took
**1.454 s versus AMG's 3.165 s**. That is the opposite application-cost result
from the new server case. The historical experiment also explicitly used sparse
B/G, whereas the current server profiles record zero sparse B/G calls.

The restored server model, physical state, hierarchy, dependencies and execution
environment differ, so the earlier measurement was not a guarantee for this
case. Average active cells per rank are actually similar (about 10,742 locally
versus 11,312 here); simply claiming too few fine cells per rank would not be
supported. A larger MPI communicator and adaptive-level distribution may still
change cycle efficiency. No Intel 26.0 compiler-performance defect or specific
communication bottleneck is established by these logs.

## Next bounded task if optimizing GMG

Use one matched frozen linearization, preferably step 2 where the gap is
largest, to record inner velocity CG iterations and split velocity/pressure
preconditioner time, then V-cycle/vector-copy time and per-level MPI work if
needed. Existing `ASPECT_FAULT_LINEAR_PERFORMANCE` output is already present;
enabling it again would not add the missing breakdown. Reuse the existing
comparison observer rather than launching another full BP3 trajectory or
changing solver tolerances to hide the difference. No new experiment or code
change was performed in this audit.

Current conclusion: AMG is faster for this measured server workload. The
outputs explain the slowdown at the block-preconditioner level; deciding
between coarse-solver, smoother, inner-iteration or MPI optimizations requires
the narrower measurements above.
