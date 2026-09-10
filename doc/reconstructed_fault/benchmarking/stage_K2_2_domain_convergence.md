**Subsequent completion:** the approved temporal cases have now been completed
using the accepted performance baseline. See
[the temporal completion assessment](stage_K2_2_temporal_completion.md).
All solver/allowance checks pass, but nonuniform temporal traction is not yet
a demonstrated resolved reference. The spatial evidence below is reused
unchanged; the stopped-run account is retained as history.

Both temporal runs were stopped at the user's request; neither remains running.
The domain-rule 32/64/128 spatial sequence is complete through 2 s and shows
contracting surface and raw-stress differences, with both 1e-4 allowances
satisfied. The .25 s run has accepted outputs through 1.75 s; the .125 s run
through 1 s. Their completed states pass the solver criteria, but partial
nonuniform temporal errors do not contract uniformly. Temporal convergence
therefore remains unresolved. No production equation, initialization/history
semantics, support, full I_h, tolerance or iteration budget changed. The
accepted solver/quadrature baseline and passing evidence are preserved.
K2.2 is incomplete, no Gate K2 claim is made, and K2.3 was not started.

# K2.2: accepted domain-rule convergence

## Baseline, scope and reuse

The accepted revision is documented in `stage_K2_global_accumulator_review.md`.
Its focused one-/two-rank consistency, coupling, lifecycle, rollback and
restart evidence is preserved, as are the through-1-s new-rule K2 outputs.
Executable SHA256 is
`0559a029c7f137d9d8297c39311d18cb2080f798a48d62512da0ce42c2cd4908`;
plugin SHA256 is
`40aac2f889e311f92cc6b69942b951fcafe4813dc5655460321f69f6e8d89880`.
The existing binaries match these hashes and require no rebuild. Source
snapshots and the working-tree patch are saved in the new artifact directory,
without resetting or committing unrelated changes.

No executable restart/checkpoint is present in the accepted new-rule 64/128
output directories. They therefore supply prefix verification, not a restart
state. All three new spatial trajectories run through 2 s with the domain
rule; old point-rule data are excluded from the convergence sequence.

The geometry, initial physical fields, 5% compact initial-Theta perturbation,
fixed converged Q1 phase field, prescribed 1000 Pa normal stress, full I_h,
support half-width and history publication remain unchanged. Independent
containment and actual local/global slip-normalization requirements are both
1e-4 for this approved fixture family; the original containment target was
1e-6. Same-support convergence does not establish full-profile equivalence.

## Planned execution and cost, recorded before the new runs

| Spatial mesh | Fault spacing (m) | dt (s) | Estimated wall time | Estimated RSS |
|---|---:|---:|---:|---:|
| 32x128 | 1/64 | .5 | 60--180 s | .5--1 GiB |
| 64x256 | 1/128 | .5 | 350--600 s | 1.5--2 GiB |
| 128x512 | 1/256 | .5 | 2400--3600 s | 4--6 GiB |

These use the measured accepted new-rule costs, not obsolete point-rule
timings. Runs are one-rank Release and sequential. New wrappers select the
existing planned input and change the output path only. Runner caps of
1200/1800/5400 s do not change solver iteration budgets.

If supported by spatial evidence, use .5/.25/.125 s on the resolved mesh,
with .5 already available. Initial fine-mesh estimates are 4000--6500 s and
7000--11000 s for .25/.125, respectively, with 4--6 GiB. Refresh them before
launching expensive time runs. Do not form the full parameter product.

## Diagnostics and review gate

Measure actual pre-publication weak traction, M, balance and the consistent
represented traction; do not substitute a normal-column average or published
parent stress. Use exact Q1/arclength norms and native FE raw-stress samples.
Report total and mean-removed fields, endpoints, center, initial projections,
V/Theta/C/slip, and differences from the initial error snapshot. That last
subtraction is accounting, not removal of the initial error's physical effect.

Stop for a genuine solver/invariant failure, an unexplained plateau, an
allowance failure, or a needed model/acceptance change. Mere remaining
discretization error is not authorization for another production correction.
The bulk-history transfer concern stays separate. Stop before K2.3.

Artifacts: `benchmarks/reconstructed_fault/uniform_shear/nonuniform/domain-convergence/`.

## Completed spatial runs

The 32/64 cases reach all five accepted times through 2 s. Every returned
linear direction passes its fresh residual check, and every accepted state
meets both final nonlinear criteria. The measured support and actual local/
global normalization allowances pass. Costs are 76.15 s / 562688 KiB (32)
and 407.60 s / 1330032 KiB (64). The fine run completes in 2571.12 s /
4488188 KiB, within the pre-recorded estimate.

For both 64 and 128, the accepted **new-rule** prefix through 1 s is reproduced
byte-for-byte in all seven exported datasets (time, surface, phase, particles,
bulk, actual weak loads, and geometry). `prefix64.json` and `prefix128.log`
record 21 checks each.
This is not a comparison against the old point-rule trajectory.

The benchmark-only convergence audit has three analytic tests for exact
weighted second moments, separating initial offsets from endpoint/evolution
differences, and the positive-state logarithmic slip-rate diagnostic; all pass.
The reused Q2 sampling/Q1 weak-projection diagnostic
tests also pass (two tests). No production source or binary was changed or
rebuilt; the previous focused solver/lifecycle evidence is reused.

## Spatial decision and refreshed temporal resources (before launch)

All three spatial trajectories now complete through 2 s, with 15 final
nonlinear acceptance checks and 75 fresh linear checks passing. Fine-grid
cost is 2571.12 s and 4488188 KiB peak RSS. All support and normalization
allowances pass. The actual weak-traction anomaly RMS difference at 2 s
contracts from 1.12168e-3 Pa (32--64) to 1.65367e-4 Pa (64--128), compared
with a fine-run anomaly RMS of 3.94281e-2 Pa. Both endpoint differences and
the central difference shrink. Native bulk-stress anomaly RMS differences
contract from 6.86696e-2 to 3.41007e-2 Pa; this slower trend is retained,
not smoothed or conflated with surface traction. Initial Theta projection
RMS errors are .0973196/.0212798/.00514082 s. There is no unexplained
spatial plateau in this sequence. Remaining spatial errors do not justify
a production correction.

This evidence supports using the planned 128x512 configuration for the three
timestep levels. The finest spatial trajectory is a same-support numerical
reference, not a full-profile or exact solution. The .5 s run is reused.

Refreshed prospective one-rank costs are 4000--6500 s for .25 s and
7500--11000 s for .125 s, each approximately 4.3--6 GiB RSS. Runner caps are
9000 and 14400 s; solver iteration budgets stay unchanged. There are 22
available logical CPUs, about 20 GiB available RAM and 77 GiB free storage
after the spatial analysis. To avoid unnecessary elapsed time, the two
independent one-rank Release jobs will run concurrently in separate output
directories (a scheduling change from the initial sequential estimate).
Budget up to 12 GiB combined RSS and roughly 2--3 hours elapsed, allowing
contention. This remains two prescribed time runs, not a parameter product.
Both use the same executable/plugin and the already planned inputs.

## Complete spatial evidence

Every row uses the domain rule and the same accepted solver revision.
The surface quantity q is M^{-1}Q from actual pre-publication weak traction,
not a column average or the published parent stress. Surface RMS norms use
arclength weights on the union of Q1 intervals. Bulk comparisons evaluate
native cell polynomials at fine-grid quadrature points and reconstruct the
unsmoothed Maxwell stress with the correct frozen FE history. Bulk anomalies
remove the along-fault mean separately at each transverse coordinate.
Prescribed frictional normal stress is 1000 Pa throughout; raw bulk pressure
is reported separately in the JSON and is not substituted for it.

### Initialization and allowances

| Mesh | Center phi | Independent full I_h (m) | Initial mean C (Pa) | Initial Theta projection RMS (s) | Realized Theta maximum (s) | Omitted fraction | Worst actual normalization error |
|---|---:|---:|---:|---:|---:|---:|---:|
| 32 | .5999103233 | 108.0980951 | 318.7476558 | .09731964 | 210.5235858 | 5.83027e-5 | 5.23649e-5 |
| 64 | .5997640169 | 108.1448338 | 317.5137627 | .02127977 | 210.1282189 | 5.91396e-5 | 6.00643e-5 |
| 128 | .5997110506 | 108.1266579 | 317.2451785 | .005140822 | 210.0317795 | 5.89728e-5 | 5.77115e-5 |

These are resolution-dependent initializations of the unchanged prescribed
physical fields, not identical discrete initial data. The initialized mean
C difference is 1.233893 Pa (32--64) and .2685842 Pa (64--128); at 2 s it
is 1.201317 and .2598681 Pa. The corresponding initial mean q differences
are 1.106076 and .2398020 Pa, becoming 1.207691 and .2613848 Pa at 2 s.
Subtracting initial differences yields useful accounting but does not erase
their influence on mechanics or histories. The integrated initial profile
is not assumed to change monotonically with mesh size.

The independent I_h comparisons, fixed-profile checks, local/global slip
normalization and weak-balance closure run at all accepted times and all
saved measurement columns (33/65/129). Maximum boundary velocity error is
4.75e-20 m/s. The independent exponential Theta-update check reports zero
floating-point difference for these trajectories. The independently
reconstructed strong surface RMS differs from production by at most
4.24e-22 Pa. Parent-particle raw traction extrema remain separately labeled
in each measurements/report.json.

### Actual weak-traction and raw-stress convergence

| t (s) | q anomaly RMS, 32--64 (Pa) | q anomaly RMS, 64--128 (Pa) | Raw bulk stress anomaly RMS, 32--64 (Pa) | Raw bulk stress anomaly RMS, 64--128 (Pa) |
|---:|---:|---:|---:|---:|
| 0 | .00386964 | .000940301 | .0182228 | .00464175 |
| .5 | .00153750 | .000358263 | .00367504 | .000972729 |
| 1 | .00118580 | .000248890 | .0323708 | .0160780 |
| 1.5 | .00108509 | .000187358 | .0534694 | .0266126 |
| 2 | .00112168 | .000165367 | .0686696 | .0341007 |

At 2 s raw bulk-stress total RMS errors are 1.209198/.2627261 Pa, with
sampled maximum absolute errors 1.895872/.5770763 Pa. At initialization
the total RMS errors are 1.256755/.2838259 Pa and maxima 3.816034/.9191465 Pa.
Thus raw stress retains a slower late-time nonuniform convergence trend than
the surface weak quantity. It is neither smoothed nor presented as eliminated.
The bulk-history transfer concern is separate and no transfer change was made.

| 2 s field | Total RMS, 32--64 | Total RMS, 64--128 | Mean-removed RMS, 32--64 | Mean-removed RMS, 64--128 |
|---|---:|---:|---:|---:|
| q (Pa) | 1.207692 | .2613848 | .00112168 | .000165367 |
| V (m/s) | 3.51802e-8 | 7.14854e-9 | 1.96839e-8 | 4.24929e-9 |
| Theta (s) | .1321714 | .02833580 | .05347278 | .01162508 |
| Retained C (Pa) | 1.201317 | .2598681 | .000558284 | .000123994 |
| Slip (m) | 1.19799e-6 | 2.56298e-7 | 5.94727e-8 | 1.30023e-8 |

The maximum absolute log10(V_coarse/V_fine) at 2 s is 3.78232e-4 and
8.34968e-5, respectively; absolute errors remain the main dimensional
diagnostic. The fine slip-rate anomaly RMS is 5.71054e-7 m/s, Theta anomaly
RMS is 1.773192 s, and accumulated slip anomaly RMS is 1.80344e-6 m.

### Endpoint, central and non-local response

| 2 s q anomaly difference (Pa) | Left endpoint | Center | Right endpoint | Outside-bump RMS | Inside-bump RMS |
|---|---:|---:|---:|---:|---:|
| 32--64 | .000968135 | .00124794 | .000729543 | .000992689 | .00123729 |
| 64--128 | -.0000759067 | .000621818 | -.000137171 | .0000511034 | .000228213 |

Endpoint topology is unchanged. The left/right fine q values at 2 s are
993.1339165/993.1338964 Pa, with center 993.2465002 Pa and mean 993.1867746 Pa.
The fine left/center/right V values are 1.22606107e-4/1.20882843e-4/
1.22606041e-4 m/s, and slip is .000698153620/.000692618870/.000698153555 m.
At s=.0625/.1875 m (the bump edges), q is 993.1836020/993.1836053 Pa and
V is .000122770219/.000122770230 m/s. The response outside the initial
perturbation is measured and contracts under refinement, not merely pictured.

Simple mirror differences are reported, not imposed as an invariant on
advected sheared histories. At 2 s the maximum q mirror difference decreases
from 3.20021e-4 to 8.14293e-5 to 2.01647e-5 Pa. Detailed initial-error changes,
endpoints and all common-time comparisons are in spatial-audit.json.

ParaView outputs remain in each case's solution/ and reconstructed_faults/
directories. All original sampled bulk/profile/history/weak-load CSV files
are retained. spatial-audit.png shows initial projection and final response;
spatial32-64.json, spatial64-128.json and spatial32-128.json include raw bulk
velocity, pressure, phase and stress comparisons without smoothing.

## Temporal initialization

The .25 and .125 s runs each reproduce the .5 s fine-grid timestep-zero
exports byte-for-byte for time metadata, surface fields, phase profile,
particles, bulk data, actual weak loads and geometry (14 comparisons in
temporal-initialization.log). Both initial mechanical solves meet the final
criteria. The artificial initialization interval remains 2 s. Thus this
time sequence has identical discrete initial data, unlike the spatial
sequence; histories are then advanced independently without resetting them
from another trajectory.

### Concurrent CPU-placement adjustment

After approximately 57 minutes of concurrent runtime, a read-only process
check showed both ASPECT main processes pinned to logical CPU 0, each using
about 48.5% CPU. Their command lines were verified as the two current
temporal fixtures. Thus the initial scheduling assumption (two independent
cores) was false; available CPU count alone had not established placement.
CPU topology reports CPU 0 on physical core 0 and CPU 1 on physical core 1.
The running .125 s process (PID 91642) and its two auxiliary threads were
moved to CPU 1 with `taskset -apc 1 91642`; the .25 s job remains on CPU 0.
affinity-adjustment.log records old/new affinity. No process was restarted,
no state was reset, and no executable, physics, numerical setting, solver
budget or runner timeout changed. Actual wall costs include the initial
contention. This is process scheduling, not a production correction.

## User-requested stop and partial temporal review

The two command lines were verified before sending SIGTERM to PIDs 91610
and 91642. Both processes were subsequently confirmed absent. The runners
record exit_status=-15, meaning the requested interruption, not a solver
failure. Nothing was deleted. Saved accepted outputs, full interrupted logs
and resource records remain in place. No simulations were restarted for
this review.

| Run | Last accepted t (s) | Accepted states including zero | Interruption point | Wall time (s) | Peak RSS (KiB) |
|---|---:|---:|---|---:|---:|
| dt=.25 | 1.75 | 8 | t=2, first Newton iteration | 6827.02 | 4502876 |
| dt=.125 | 1 | 9 | t=1.125, domain reconstruction | 6822.72 | 4501828 |

The incomplete t=2 and t=1.125 states are excluded from every comparison.
Only fully written accepted exports are used. All 17 accepted temporal
states pass both final nonlinear criteria; their 79 returned linear
directions pass the fresh residual checks, with fixed bulk/surface scales
within each solve. Including the spatial sequence gives 32 accepted states
and 154 checked directions, without double-counting the reused .5 s run.
Minimum accepted temporal V is 1.11883e-4 m/s, well above the unchanged
1e-12 m/s bound. No accepted temporal node is bound-active.

### Partial temporal quantities from actual weak loads

The saved domain mass matrix and weak-load vectors are used directly to
reconstruct q, evaluated C, friction, damping and the strong surface
residual at every accepted temporal state. The maximum difference between
this strong residual and production is 9.27e-23 Pa. The independent
exponential Theta-update check has maximum discrepancy 1.43e-14 s. Slip
is accumulated independently along each run's actual accepted dt/V sequence.
No histories are reset from a later ASPECT output or another run.

Below, the first pair is dt=.5 minus .25, the second .25 minus .125, on the
same 128x512 mesh and at t=1 s. All temporal initial datasets are identical.

| Field at t=1 s | Total RMS, .5--.25 | Total RMS, .25--.125 | Mean-removed RMS, .5--.25 | Mean-removed RMS, .25--.125 |
|---|---:|---:|---:|---:|
| Actual weak q (Pa) | 21.79301 | 9.593469 | 2.81087e-4 | 6.53755e-4 |
| V (m/s) | 9.68176e-5 | 3.59395e-5 | 1.80596e-7 | 9.29413e-8 |
| Theta (s) | 2.455821 | 1.062308 | .04509369 | .02246971 |
| Retained C (Pa) | .1958197 | .08585671 | 2.66630e-4 | 3.42427e-4 |
| Slip (m) | 2.17443e-5 | 9.60508e-6 | 2.86333e-8 | 3.69224e-8 |

| Common time | q anomaly RMS, .5--.25 (Pa) | q anomaly RMS, .25--.125 (Pa) |
|---:|---:|---:|
| .5 s | 5.18622e-4 | 6.68639e-5 |
| 1 s | 2.81087e-4 | 6.53755e-4 |

At 1 s the q anomaly differences at the left endpoint / center / right
endpoint are (-3.23173e-4, 5.35763e-4, -3.14199e-4) Pa for .5--.25 and
(-8.59680e-4, 1.04506e-3, -8.42156e-4) Pa for .25--.125. Thus contracting
total fields conceal non-contracting nonuniform quantities at this time.
There is no claim of uniform temporal convergence, a resolved finest
temporal reference, or a proven plateau mechanism. Identical initial fields
exclude different timestep-zero projections as the direct explanation, but
do not isolate temporal truncation, fixed-mesh effects or history transfer.
No production diagnosis/correction was undertaken in response to this
partial result. The bulk-history transfer concern remains separate.

### Allowances and remaining measurements

All accepted temporal phase-profile and segment-geometry exports are
byte-identical to initialization (temporal-fixed-profile.log and
temporal-fixed-geometry.log), and every saved plus/minus association
half-width remains .3088215939070757 m. The independent omitted
fraction is therefore the unchanged 5.89728e-5 at these saved profile
locations and accepted times, below the family allowance of 1e-4.

Actual local/global slip normalization was measured for all spatial states
and, for each temporal run, at zero and the first real accepted state. Those
early temporal measurements pass, with worst local error 5.77116e-5. The
saved early reports are time025-early-report.json and time0125-early-report.json.
Actual normalization for the later accepted temporal states has **not** yet
been measured; a fixed profile alone is not substituted for that check.
The raw temporal bulk-stress/velocity/pressure comparison is also unmeasured;
its saved bulk data remain available. No missing check is labeled passed.

The full three-level temporal sequence through 2 s is incomplete. Its
nonuniform convergence trend requires review even before a completion
claim. There is no authorization inferred here for another production
correction, additional timestep level, broader campaign or K2.3.

## Verification and artifacts

Production source and binaries were not edited or rebuilt during this
convergence task. The accepted one-/two-rank residual, coupling, lifecycle,
rollback, restart and geometric-quadrature tests are reused from the frozen
baseline. Newly added benchmark-analysis tests pass (3 tests); reused
Q2-sampling/Q1-projection analysis tests pass (2 tests). No complete ASPECT
test suite was run. The SIGTERM interruption is not a new rollback test:
no assertion is made about unpublished in-memory state after process exit.

Each trajectory uses `python3 benchmarks/reconstructed_fault/uniform_shear/convergence/run_case.py`
with its parameter wrapper under `nonuniform/domain-convergence/`:

| Wrapper | Options |
|---|---|
| space32.prm | --configuration Release --timeout 1200 |
| space64.prm | --configuration Release --timeout 1800 |
| space128.prm | --configuration Release --timeout 5400 |
| time025.prm | --configuration Release --timeout 9000 |
| time0125.prm | --configuration Release --timeout 14400 |

Analysis tests:

```sh
python3 -m unittest discover -s benchmarks/reconstructed_fault/uniform_shear/nonuniform -p test_convergence_audit.py -v
python3 -m unittest discover -s benchmarks/reconstructed_fault/uniform_shear/nonuniform -p test_diagnostics.py -v
```

Results: 3 tests OK and 2 tests OK, respectively. Logs are
audit-tests-clean.log and diagnostic-tests.log. An initial parallel comparison
raced an unfinished analysis CSV; it was rerun successfully after input
completion. Early temporal measurement initially included a not-yet-accepted
log step; saved completed-state log prefixes resolved this postprocessing
issue. Neither incident was a solver failure or caused a numerical change.

The evidence directory contains:

- accepted-baseline.patch, accepted-source.tar.gz and accepted-snapshot.sha256;
- spatial-audit.json/.png, spatial32-64.json, spatial64-128.json and spatial32-128.json;
- per-case measurements/report.json and sampled surface/weak/particle balance CSVs;
- prefix64.json, prefix128.log and temporal-initialization.log;
- temporal-partial-review.json: accepted-state criteria, weak balances,
  independent state/slip checks, and common-time surface differences;
- interrupted .log/.resources.json files, early temporal reports,
  affinity-adjustment.log and all accepted bulk/fault ParaView outputs.

Task-local changes are the five output-path-only parameter wrappers, the
convergence-analysis script and its tests, this report, the progress/README
records, and generated evidence. The older point-rule README is explicitly
marked historical. Existing production changes belong to the accepted
baseline and were preserved. No commit was created during this task.
