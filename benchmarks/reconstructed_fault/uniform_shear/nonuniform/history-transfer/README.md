# Bounded frozen history-transfer audit

## Approved correction follow-up

The original INSERT audit below is preserved as the failure record; its source
and plugin are archived in `insert-baseline.tar.gz`. Current `history_transfer.cc`
tests the approved ADD/count alternative after DGQ1 encountered local compatibility
barriers. `average-comparison.json` passes traversal/MPI invariance and independent
weak integration. `timeline-verification.json` verifies actual first-assembly
history, prior commits and bulk/particle VTU semantics over 0/.5/1 s.
See `doc/reconstructed_fault/benchmarking/stage_K2_stress_transfer_timeline.md`
from the repository root for commands, measurements and remaining limits.
No K2.3 or convergence campaign belongs to this follow-up.

## Original INSERT audit record

K2.2 execution is accepted as complete; its reference remains provisional
and Gate K2 unmet. No further timestep or cancellation study is included.

Use a 8x32-cell periodic-x box, nine reference-cell particles per cell,
and retained xy stress 1500+20 sin(8 pi x)+10 y Pa. Temperature, composition,
particle locations, topology, and all other fields are frozen during the
postprocessor. Constant particle theta=200 supplies the nonzero constant
transfer control. No particle or published solution is mutated.

Call the actual configured cell-average interpolator at composition support
points. Replay the existing shared-node INSERT assignment in ordinary and
reversed locally-owned cell order, including the existing MPI compression.
The ordinary nonconstant result must match the actual published FE history.
The alternative order is a test-only replay of that assignment loop, not a
production traversal hook or new transfer algorithm.

Apply the actual physical constraints to each private vector, as mechanics
does. Evaluate the production reconstructed-fault Stokes assembler and
subtract its zero-history load, leaving precisely the frozen Maxwell weak
load. Compare with independent QGauss(4) integration of the realized FE
history (not the analytic stress), including homogeneous Newton constraint
elimination and MPI assembly. Export published/working QP values and global
load coefficients keyed by physical coordinates, not MPI-specific DoF IDs.
Compare identical physical particle tuples on one and two ranks.

Compile only this plugin (-j4). Each simulation is capped at 120 s; the
entire exploratory execution budget is 600 s. Expected cost is seconds per
small t=0 case, less than 1 GiB/rank. Stop on significant traversal/partition
dependence and propose the smallest correction; do not implement it or infer
that it caused the K2.2 plateau. Only if this check passes may a separately
documented bounded K2.3 pilot follow.

## Measured outcome: stop for review

Both rank counts pass forward-production matching, constant reproduction and
independent realized-FE weak integration. Reversing traversal changes the
constrained nonconstant history by 4.43 Pa RMS and the assembled weak load by
80.7%. Reverse-order one/two-rank weak loads differ by .583%; ordinary-order
MPI loads agree to roundoff here. No production correction or K2.3 run was
made. See comparison.json and
doc/reconstructed_fault/benchmarking/stage_K2_history_transfer_audit.md for
the explicit transfer-policy proposal and limitations. The K2.2 plateau is
not attributed to this test result.
