# Frozen history-transfer audit: stop for correction review

Historical failure record. The subsequently approved DG-first compatibility
check, continuous ADD/count correction and bounded timeline verification are
recorded in [the follow-up review](stage_K2_stress_transfer_timeline.md).
The original data and source snapshot remain preserved.

K2.2's approved execution campaign is accepted as complete. Its numerical
reference remains provisional and Gate K2 remains unmet. No further timestep
run or cancellation accounting was performed. This separate small frozen-data
audit demonstrates traversal-dependent constrained bulk-history loads. Stop
before K2.3; no production correction has been made, and no causal connection
to the K2.2 temporal plateau is established.

## Test boundary and inputs

Artifacts and standalone plugin are under
`benchmarks/reconstructed_fault/uniform_shear/nonuniform/history-transfer/`.
The test uses the existing Release executable and an 8x32-cell box,
x-periodic boundaries, continuous Q2 stress composition, nine reference-cell
particles per cell, and the existing cell-average particle interpolator.
Both rank counts have the same 2304 physical particle tuples (coordinates,
stress and constant property), checked directly from exports. Prescribed
old xy stress is 1500+20 sin(8 pi x)+10 y Pa. The retained constant particle
state 200 supplies a nonzero constant transfer control. No particle, published
solution, history or geometry is modified during the audit; the temporary
current linearization point is restored afterward.

The test calls the actual configured particle interpolator at composition
support points and replays the shared-node INSERT assignment loop in
`Simulator::interpolate_particle_properties`, once in normal and once in
reversed local-cell traversal. This is a **test-only replay** of the existing
write operations, not a new production traversal API. In both MPI runs, its
normal nonconstant result must first match the actual production-published
FE stress to 1e-10 Pa at DoFs; that anchor passes.

Each result is lifted privately using the actual physical constraints.
The actual reconstructed-fault Stokes assembler evaluates its frozen load;
subtracting the zero-history result removes the unchanged profile term.
An independent QGauss(4) integration of the **realized working FE history**
checks the sign and load, using beta=exp(-.02), the same homogeneous Newton
constraints and MPI assembly. No exact reproduction of the analytic stress
by cell averaging is required. Global weak coefficients are compared by
physical velocity-support coordinates and component, not rank-specific DoF
numbering. These are same-mesh coefficient norms, not mesh-convergence norms.

CSV `published` columns mean the unconstrained transfer result. Mode 3 is
verified against actual published history; mode 4 is the counterfactual
reverse-order result and is **never published**. `working` is the physically
constrained vector actually supplied to the production assembler.

## Results

| Comparison | Unconstrained FE difference RMS (Pa) | Working FE difference RMS (Pa) | Weak-load difference L2 | Relative weak-load difference |
|---|---:|---:|---:|---:|
| Forward vs reverse, one rank | 4.431133 | 4.431133 | 2.418184 | .807188 |
| Forward vs reverse, two ranks | 4.431545 | 4.431545 | 2.418263 | .807214 |
| One vs two ranks, forward | 0 | 0 | 7.96e-14 | 2.66e-14 |
| One vs two ranks, reverse | .02852722 | .02852722 | .01961151 | .00583239 |

The forward nonconstant load norm is 2.995814; the reverse norm is
3.362516/3.362573. Maximum traversal-induced working-field difference is
9.385553 Pa; maximum reverse-order MPI difference is .2147807 Pa.
These effects greatly exceed the independent weak-assembly discrepancy
(at most 3.68e-13 in global L2). They survive physical constraint application.

Constants reproduce in both orders/rank counts. Their constrained weak-load
norms are below 1.65e-14, and the one/two-rank difference is 1.49e-14.
Their relative error is intentionally not interpreted because the expected
load vanishes. The nonconstant published-to-working constraint correction
is itself nonzero: 1.931278 Pa RMS, maximum 10.28172 Pa. Thus the audit has
not conflated an unconstrained periodic trace with the history used by
mechanics. All forward-publication, constant and independent-integration
controls pass; **nonconstant traversal invariance does not**. MPI invariance
passes in the ordinary order for this partition but fails in the reversed
order. This is not a claim that every ordinary partition must differ.

## Smallest correction proposal — not implemented

The identified source is competing INSERT assignments to shared continuous
composition DoFs, not the particle cell-average calculation or the Maxwell
weak form. Reversing these writes changes which adjacent cell supplies a
shared coefficient; MPI insertion adds another ownership/order dependence.

Replace competing shared-DoF writes in the transfer with an explicitly
defined incident-cell average: accumulate each locally owned cell's
interpolated value and a contribution count using MPI ADD, then divide once
at owned DoFs. Each cell contributes exactly once. Unshared/DG coefficients
retain their single contribution. Keep the existing particle interpolator,
field mapping, subsequent private constraint lift, and history-publication
timing unchanged. Use the existing distributed vectors and compression;
no new framework or physical parameter is needed.

This proposal changes the shared-node **transfer policy** and needs approval;
it does not promise analytic polynomial reproduction or establish that an
arithmetic incident-cell average is an L2 projection. Verification after
approval should require the frozen nonconstant fields/weak loads to agree
under both traversal orders and MPI counts, retaining the independent
realized-FE weak-load and constant controls. Broader effects on generic
particle-advected continuous fields must be reviewed before implementation.
No source fix, new I_h work or K2.3 pilot is included here.

## Execution and recoverability

From the repository root, configure/build only the standalone plugin:

```sh
cmake -S benchmarks/reconstructed_fault/uniform_shear/nonuniform/history-transfer -B benchmarks/reconstructed_fault/uniform_shear/nonuniform/history-transfer/build -DAspect_DIR=/home/ein/repository/aspect/build-pf-cpdi -DCMAKE_BUILD_TYPE=Release
cmake --build benchmarks/reconstructed_fault/uniform_shear/nonuniform/history-transfer/build -j4
```

From that build directory:

```sh
timeout 120 /home/ein/repository/aspect/build-pf-cpdi/aspect-release ../frozen.prm > ../one.log 2>&1
mpirun -np 2 timeout 120 /home/ein/repository/aspect/build-pf-cpdi/aspect-release ../two.prm > ../two.log 2>&1
```

Then from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 timeout 120 python3 benchmarks/reconstructed_fault/uniform_shear/nonuniform/history-transfer/compare.py
```

Both completed simulation commands exit 0; ASPECT wall times are 1.386 s and
3.822 s, and comparison takes .153 s. All exploratory execution is comfortably
below the two-minute per-test / ten-minute aggregate bounds. Two initial
setup-only failures (missing included postprocessor registration and omitted
required particles postprocessor) are preserved in `setup-*-failure.log`;
neither reached a solve. The first two-rank launch was denied local sockets
by the sandbox (`mpi-launch-sandbox.log`); the permitted rerun succeeds.
No timed-out test or expensive harness repair occurred.

`one.log`, `two.log`, `comparison.json`, the per-rank particle/FE/load CSVs,
inputs and plugin source retain the evidence. The accepted executable hash
remains e2451b97ad5a76ba431967c4938bccb5b94cf910ff1182f9bd830e7666f1b453;
its recoverable source/binary snapshot remains
`nonuniform/performance/accepted-cartesian-baseline/`. Only benchmark test
files and progress/review documents were added or updated in this task.
Test source SHA256 is
`a6c8481ba2a1f03959f21d47869a3828ddc6db6de1a29e295699ff401eb77e7f`;
Release plugin SHA256 is
`de300a85f7fc02d5fe649cb192cfe4b7dfab4ce94f256d8a750b9d253903bd0e`.
