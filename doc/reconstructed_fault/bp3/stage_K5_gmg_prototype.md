# Bounded bulk-preconditioner GMG prototype

**Subsequent approved launcher promotion:** GMG is now the research-launcher
default, with both its velocity-cycle and mesh-hierarchy flags set
automatically. Explicit `--velocity-preconditioner amg` preserves the AMG
reference, and the frozen comparison driver explicitly selects that baseline.
All residual checks and equations remain unchanged. The initial raw-stress
comparison qualification below remains recorded; it is not relabeled passed.
Further setup/memory optimization is deferred. The initial recommendation at
the end of this report is historical and superseded only as to the launcher
default. No additional trajectory was needed for this configuration change.
Preparation-only regression `python3
benchmarks/reconstructed_fault/bp3/test_research_launcher.py` passed all three
selections (default, explicit GMG, explicit AMG), including inherited-flag
clearing and the default wide fixture. Syntax and diff-whitespace checks pass.

The 200 x 100-km fully frictional modified BP3 model is now the research
launcher default; the 100-km case remains an explicit reference option.

## Frozen experiment

Use real step 2, Newton linearization 4 of the wide saved-clock fixture.
This is after the first physical aging update, during the first strongly
nonuniform mechanical response. Reproduce only the necessary fresh prefix;
the existing checkpoint normalization issue is not reopened. The observer
stops before trial acceptance/history publication at the selected solve.

AMG and GMG receive the identical constrained condensed action, RHS, relative
linear tolerance, total iteration budget and four MPI ranks, and start from
zero directions. Keep full assembled A, explicit sparse B/G, pivoted K inverse,
outer FGMRES, true pressure, split histories and all physical inputs unchanged.
Each returned direction must pass an independent fresh C*x-b test; estimated
residual alone is insufficient. Residual replacements share the same budget.

Only the velocity-block **preconditioner** changes: replace its AMG cycle
inside the existing approximate A inverse by ASPECT's existing local-smoothing
GMG cycle, retaining the assembled fine A and existing pressure-block inverse.
This is narrower than a full GMG Stokes-solver switch. Use the existing
level viscosity projection, Chebyshev parameters, coarse smoother and edge
operators. No interface-aware multigrid hierarchy or altered fault operator
is introduced.

The mesh must be constructed with multigrid level ownership/ghost data;
`ASPECT_FAULT_GMG_HIERARCHY=1` enables that infrastructure for this assembled
prototype without selecting the matrix-free fine solver. The usual exact
active-cell and fault-coordinate guards remain mandatory. The first probe
without this flag stopped in deal.II's hierarchy guard before any GMG solve;
its artifacts are retained in `gmg/frozen-wide-local4/`.

The GMG entry point verifies identical velocity numbering and homogeneous
constraint rows. It omits auxiliary-field constraint callbacks on the separate
velocity DoFHandler and validates the resulting velocity constraints against
the actual coupled system. An unsupported velocity constraint fails clearly.
It evaluates coefficients but does not call matrix-free `assemble()`, which
would add an inappropriate physical RHS lift.

## Isolation, timing and decisions

The normal solver only exposes a synchronous observer after its fresh residual
check, while operator/preconditioner references are still valid. The benchmark
comparison lives in `tests/reconstructed_fault_frozen_gmg.cc`. Throwing the
intentional stop uses the existing nonlinear rollback path. No benchmark
comparison is activated in the default research fixture.

Sparse B/G versus quadrature-action comparisons move from production vmult
loops into the existing basis/random tests. Their numerical allowance remains
2e-11. Fresh residual tests for real solves are not removed.

Compare setup plus solve cost, outer iterations, preconditioner and operator
wall times, B/G/inverse timings, and peak rank RSS. AMG setup is measured at
the actual selected linearization; GMG setup includes initial hierarchy setup.
GMG memory is observed with both preconditioners resident, so the combined
peak is not a standalone GMG memory footprint. No unsupported memory-saving
claim follows from that comparison.

One four-rank probe has a 600-s cap and no automatic retry. A useful measured
improvement permits a short trajectory check; otherwise use the decomposition
to distinguish remaining coupling cost, bulk preconditioning cost and
interface-related iteration growth. Do not launch a longer trajectory merely
to compensate for a losing preconditioner.

## Qualified frozen comparison

`benchmarks/reconstructed_fault/performance/gmg/frozen-wide-verified-local4/`
contains the passed four-rank comparison (`frozen_gmg.csv`, `probe.log`,
`probe_execution.json`, source patch, binary/input hashes). Wall time for the
fresh prefix plus comparison was 142.364 s. The intentional nonzero process
exit follows the explicit comparison-pass marker and precedes step-2 commit;
it is not a failed physical trajectory or an inferred convergence success.

| Measured quantity | AMG | Velocity GMG |
|---|---:|---:|
| RHS norm | 1,225,889.2187472312 | same |
| Absolute linear target | 0.0012258892187472312 | same |
| Outer iterations / preconditioner applications | 17 / 17 | 17 / 17 |
| Estimated residual | 0.000440689594619463 | 0.000487082687551078 |
| Fresh residual | 0.000440689625725404 | 0.000487082850247206 |
| Setup (s) | 0.610405 | 0.412425 additional |
| Solve (s) | 4.984538 | 3.244416 |
| Preconditioner applications (s) | 3.165373 | 1.453620 |
| Operator applications (s) | 1.648472 | 1.620492 |
| Sparse B / G (s) | 0.034174 / 0.048417 | 0.033432 / 0.047051 |
| Surface inverse (s) | 0.008294 | 0.008790 |
| Peak rank RSS (KiB) | 1,556,012 | 1,568,440 with both resident |

AMG exactly reproduces the original returned direction. GMG differs by
7.542935e-10 in relative direction norm; both meet the independently evaluated
residual target. No tolerance was relaxed. Both bulk solution and working
linearization vectors are unchanged by the comparison.

The prototype still constructs the ordinary preconditioner, including its AMG
object, for the common pressure machinery/reference. Counting that setup for
**both** backends gives 5.594942 s versus 4.267246 s: a conservative **1.31x**
setup-plus-solve improvement. Solve-only speedup is 1.54x; preconditioner
applications are 2.18x faster. The extra observed peak is 12,428 KiB, not a
standalone GMG footprint. Timings are single bounded measurements, not scaling
statistics. B/G and the surface inverse are already small; they do not justify
another coupling optimization here. Equal outer iteration counts do not
indicate new fault-mode degradation in this selected system.

The first failed probe lacked multigrid hierarchy ownership. The second,
`frozen-wide-hierarchy-local4`, passed both linear residual checks but exposed
a test-only attempt to norm a ghosted snapshot. The corrected test compares
owned snapshots; all earlier evidence remains preserved. No physical or
acceptance change was used to repair either harness issue.

## Commands and focused checks

From the repository root:

```sh
cmake --build build-pf-cpdi --target aspect -j4
cmake --build benchmarks/reconstructed_fault/performance/build-gmg -j4
python3 benchmarks/reconstructed_fault/performance/gmg/run.py
python3 benchmarks/reconstructed_fault/bp3/run_research.py \
  --configuration wide --velocity-preconditioner gmg \
  --output benchmarks/reconstructed_fault/bp3/wide-gmg-seven-local4
python3 benchmarks/reconstructed_fault/performance/gmg/check_trajectory.py
```

Drivers refuse to overwrite completed output; select a new evidence directory
for a deliberately requested repeat. The frozen driver has a 600-s cap; the
seven-step replay retains its existing 2400-s cap and no automatic retry.

Sparse B/G basis/random tests passed on one and two ranks with the unchanged
2e-11 action allowance (`gmg/coupling-one.log`, `coupling-two.log`). Stage-I
unit tests passed 90 assertions in 11 cases (`stage-i-unit.log`); the focused
accepted-update rollback fixture also passed (`rollback.log`). These compare
reference actions in tests, not inside production Krylov calls. Production
fresh linear checks and nonlinear/rollback criteria remain enabled.

## Seven-step trajectory result and qualification limit

The authorized replay completed with no retry in `bp3/wide-gmg-seven-local4/`.
The unchanged saved clock reaches 4.190129311694 years after seven real steps.
Both runs have 1236 free nodes, zero prescribed/lower-active nodes, and pass
the first and subsequent split-history checks. Every final bulk/surface
nonlinear residual satisfies the existing criterion; all 71 returned linear
directions pass the fresh check. No physics, initialization, tolerance or
history-publication change was made.

| Seven-step quantity | Saved wide AMG | Velocity GMG |
|---|---:|---:|
| Wall time (s) | 721.060 | 615.865 |
| Peak child RSS (KiB) | 1,574,720 | 1,587,412 |
| Fresh linear checks | 71 | 71 |
| Total outer iterations | 1301 | 1266 |
| Iterations min / median / mean / max | 16 / 17 / 18.324 / 27 | 14 / 17 / 17.831 / 27 |
| Condensed-solve timer (s) | 376 | 263 |
| Ordinary preconditioner construction (s) | 43.2 | 44.1 |
| GMG setup inside condensed timer (s) | — | 27.6 |

GMG reduces observed end-to-end wall time by **14.6% (1.17x)** and the
condensed timer by about **30% (1.43x)**, already including repeated GMG
setup. Do not add its 27.6-s nested timer again. The old and new trajectory
timings are not simultaneous controlled repeats; the same-system frozen
comparison is the stronger isolated performance evidence. Peak child RSS
increases by about 12.4 MiB (0.81%). The conservative prototype still builds
AMG and the existing pressure hierarchy; standalone memory optimization was
not attempted.

`gmg_equivalence.json` deliberately records **overall strict equivalence NOT
PASSED**, rather than disguising a timestep-zero raw-stress discrepancy:

* Geometry, accepted time/dt, frozen phase/Ih, normal background, C,
  source association and integration weights are identical. Particle IDs and
  inert H are identical. Fault V/Theta/slip pass the unchanged 1e-8 comparison
  at every state; the largest relative difference is 1.2695e-9 in initial V.
  Exported shear-background and chi evaluations differ at roundoff level
  after step zero (maximum 1.49012e-8 Pa / 5.57e-16 relative and
  1.30104e-18 / 4.72e-16 relative, respectively). These non-bitwise fields
  remain separately recorded in `exact_failures`, not called identical.
  Initial background data are identical; no background update was added.
* At real steps 1–7, every compared field passes: maximum relative fault-state
  difference is 1.1564e-12; current raw-stress difference is 6.9971e-9
  (pressure, 3.2445e-6 Pa); particle-stress difference is 3.2823e-9
  (tau_xx, 1.0416e-5 Pa). These are per-field scales, not 50-MPa background
  scaling.
* At timestep zero, pressure differs by 0.00109835 Pa relative to a
  745.433-Pa pressure scale (1.4734e-6). Raw tau_xx/tau_yy/tau_xy/tau:N
  differences are respectively 0.00096256/0.00080907/0.00046284/0.00084903 Pa.
  All five exceed the strict 1e-8 field-comparison allowance. They remain
  **reported failures**, not new accepted field tolerances.

The only changed solve operation is preconditioning. Both initialization
solves meet the original nonlinear tests (bulk 5.28115e-10 AMG versus
5.09027e-10 GMG; surface 7.27603e-9 for both). A residual stopping criterion
does not imply the same relative error in every small derived initial stress
field; different acceptable inexact directions can leave these differences.
This is consistent with inexact-solve termination, not evidence of a changed
equation. Initial particle stress remains exactly zero; this difference is
in **evaluated current** stress, not a changed retained initial history. No
extra solve, tighter tolerance, fitted stress allowance or initialization
adjustment was introduced to remove it.

Final-binary Stage-I tests again pass 90 assertions / 11 cases
(`stage-i-final.log`). A separate opt-in GMG execution of the existing forced
failure fixture passes rollback after an accepted Newton update
(`rollback-gmg.log`, about 4 s); its intentional nonlinear-failure warning is
expected. Syntax checks and `git diff --check` pass. No full ASPECT suite,
cross-rank GMG qualification, long trajectory or restart requalification was
run.

## Decision and retained limitations

Keep the wide model as the research default and AMG as the default/reference
preconditioner. Retain velocity GMG as an **opt-in, bounded prototype**: its
frozen-system correctness, useful speedup, real-step field agreement and
rollback are demonstrated, but do not label the full strict initial-state
comparison passed or silently promote it. Review the initial raw-stress
equivalence qualification before making GMG the default. No further coupling
optimization or fault-aware hierarchy is justified by these measurements.

This is not full matrix-free Stokes integration. It is tested on the fixed
2-D Q2 incompressible four-rank research mesh; unsupported numbering or
velocity constraints fail explicitly. It borrows the existing hierarchy for
one linearization and currently rebuilds it per direction. It retains
duplicate AMG/pressure setup for clarity. Subsequent setup reuse or removal
of the unused AMG construction would be a separate measured cleanup, not a
reason to rerun this campaign now. The previously cancelled frozen-Ih restart
fix remains cancelled, and the existing restart limitation is unchanged.
