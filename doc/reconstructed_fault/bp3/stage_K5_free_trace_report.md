# K5: independent free-side trace at the 40-km junction

## Decision

The independent trace **reduces but does not remove** the actual BP3 notch.
At identical incoming history, the free-side endpoint chooses 0.928661 Vp
instead of the prescribed deep-side Vp. The 39.95-km deficit falls from
16.0631% to 13.1997%; the 39.90--39.95-km contrast falls 26.41%. Opposing
incident-element residuals decrease 32.14%, but remain present. The raw
normal-stress range increases slightly, rather than improving.

Thus forcing the two sides to share a prescribed rate contributes to the
notch in this frozen state, but is not its sole cause. This is **not** a
qualified cure or a reason to adopt a discontinuous production rate. Keep
the existing production representation. The remaining uncertainty is the
relative contribution of inherited nonuniform history and the resolved
bulk/surface response; this single late-state experiment cannot separate
them. No further simulation was run.

The subsequent explicit A/B request was completed by revalidating and reusing
these exact two solves, not repeating them. The added equal-trace recovery
check passes. Separating the undershoot **within the free interval** from the
free/deep jump gives a 47--49% reduction of the former, but leaves a clear
local minimum. The negative conclusion about adopting this as a cure is
unchanged. See the qualification below; zero additional mechanics solves
or history updates were required.

The user withdrew the particle-density task. No 4x4/higher particle experiment
was performed. This completes only the recommended actual-BP3 comparison.

## 1. Controlled experiment

The [addendum](stage_K5_free_trace_addendum.md) defines the explicitly selected
noncommitting exception to the default continuous-Q1 rate space. The saved
accepted step-9 checkpoint supplies the incoming state at 15.30625726 yr;
mechanics 10 uses the same 439775721.3758315-s interval and reaches the trial
time 29.24190894 yr. All incoming bulk/history fingerprints agree bitwise
with the previously verified lagged-state A control.

Node 795 at 40 km is now the free-side rate. The adjacent fully prescribed
deep element uses mechanical weights (1,0), giving exactly Vp through its
other prescribed endpoint. Its derivative with respect to the free trace is
exactly zero. The free element retains its original Q1 basis. This eliminates
the prescribed trace without moving geometry, adding physical measure or
duplicating sources. The same mechanical basis enters the bulk source, B,
surface residual, K, G, mass matrix and residual norm. Geometric coordinates
still interpolate Theta, material, I_h and background without alteration.

For this mature-fault work-measure case,

\[
 R_i=\sum_q J_q\chi_q\widetilde N_i
       [q_q-\mu(V_q,\Theta_{\rm old,q})\sigma_{n,q}-\eta_{\rm rad}V_q],
 \qquad V_q=\sum_j\widetilde N_jV_j.
\]

There is no cohesive resistance in this mature model. The pointwise shear
and normal tractions use the current constitutive stress with the same
frozen working FE old history as the original solve. The stress export is
not newly committed particle history. Theta is the unchanged geometric
Q1 interpolation of incoming nodal state; it is not updated in Newton.

Mesh, fault spacing, quadrature, support, completed I_h, background, boundary
corrections, pressure treatment, tolerances and iteration limits remain
unchanged. The selector rejects committing and candidate-Theta modes. All
trial fields are discarded; no history or slip is published.

## 2. Rates and predicted slip

| Quantity | Original shared trace | Independent free trace |
|---|---:|---:|
| V(39.90 km)/Vp | 0.957303069 | 0.954790569 |
| V(39.95 km)/Vp | 0.839368550 | 0.868003042 |
| V(40 km, free side)/Vp | 1, prescribed | 0.928660791 |
| V(40 km, deep side)/Vp | 1 | 1 |
| Neighbor contrast [V(39.90)-V(39.95)]/Vp | 0.117934519 | 0.086787528 |
| Last-free-element predicted total slip gradient | 0.00213375224 | 0.00125443212 |
| Last-free-element slip-gradient increment | 0.00141283623 | 0.000533516113 |

The deficit decreases 17.83% relative to its original value. The total
within-element predicted gradient decreases 41.21%, and its new increment
decreases 62.24%. These are old slip plus dt times the converged trial rate,
**not** committed slips. The unchanged incoming gradient is 0.000720916008.

The reduction is partly replaced by a discontinuity: predicted free-side
slip at 40 km is 0.891431214 m, versus 0.922804466 m on the prescribed side,
a -0.031373252-m jump. A within-element gradient alone would hide this jump.
There are 441 free nodes and no lower-active nodes; the original has 440
free nodes. All deep prescribed rates remain exactly Vp.

### Within-free undershoot, separate from the junction jump

Use the three free-side locations 39.90, 39.95 and 40-minus km, not the
prescribed 40-plus value, to quantify the local minimum. All entries below
are normalized by Vp:

| Diagnostic | A: shared trace | B: independent trace |
|---|---:|---:|
| Drop from 39.90 to 39.95 km | 0.117934519 | 0.086787528 |
| Rise from 39.95 to 40-minus km | 0.160631450 | 0.060657750 |
| Depth below the smaller of the two free-side neighbors | 0.117934519 | 0.060657750 |
| Depth below their linear chord at 39.95 km | 0.139282984 | 0.073722639 |
| Separate jump V(40-plus)-V(40-minus) | 0 | 0.071339209 |

The local-minimum depth decreases 48.57%, and the chord defect decreases
47.07%. Both remain substantial. In B the apparent 13.20% deficit relative
to Vp is the sum of the 6.07% free-side rise and the 7.13% junction jump;
those are different effects and must not be conflated. There is no claim
that all of 1-V(39.95)/Vp is artificial ringing. Conversely, permitting a
genuine free/deep difference does not make the remaining local minimum
disappear.

## 3. Actual weak balance and stress

At node 796 (39.95 km), the mechanical basis and support are unchanged. Its
incident-element residuals, in MPa m, are:

| Incident element | Original | Independent trace |
|---|---:|---:|
| 39.95--40.00 km | -2.98111491 | -2.02296968 |
| 39.90--39.95 km | +2.98111491 | +2.02296968 |

The complete unreplaced force split is retained below, also in MPa m.
The residual convention is driving minus friction minus damping:

| Case / tested row / incident element | Driving q | Friction | Damping | Residual |
|---|---:|---:|---:|---:|
| A / 39.95 / 39.95--40 | 650.210545135 | 653.191659940 | 1.01952e-7 | -2.981114907 |
| A / 39.95 / 39.90--39.95 | 672.719789588 | 669.738674579 | 1.02877e-7 | +2.981114907 |
| B / 39.95 / 39.95--40 | 650.733340665 | 652.756310241 | 1.01358e-7 | -2.022969677 |
| B / 39.95 / 39.90--39.95 | 672.371442072 | 670.348472290 | 1.05028e-7 | +2.022969677 |
| B / new 40-minus / 39.95--40 | 662.436722644 | 662.436722539 | 1.05430e-7 | 1.72e-14 |

The last row is the new endpoint equation, with measure 25.095315485 m.
Its deep-side contribution is exactly zero. Rounded displayed traction
terms need not reproduce its tiny residual; the CSV retains full precision.

The node balances opposing element contributions in both cases; reducing
their magnitude does not remove the undershoot. The changed trace's free
endpoint has no deep-element residual contribution and satisfies its own
one-sided weak balance. At 39.95 km the final row-normalized residual is
-2.74e-9 Pa; this does not mean each incident element separately balances.

Incoming Theta is already nonuniform: 8.000000e6 s at 40 km, 8.993659e6 s
at 39.95 km and 8.315647e6 s at 39.90 km. Those values and all retained
stress histories remain fixed, so improvement cannot erase their prior
physical influence. Attribution of the remainder to Theta alone would
require a different controlled test.

Raw current constitutive quantities over 37--43 km are compared at **35,430
identical active positive-chi bulk QPs**, with identical coordinates, chi
and work weights:

| Quantity | Original | Independent trace |
|---|---:|---:|
| Perturbation pressure range (MPa) | [-2.624986, 2.598243] | [-2.515180, 2.616012] |
| -tau:N range (MPa) | [-1.419445, 1.263749] | [-1.350090, 1.189703] |
| Total normal traction range (MPa) | [47.208131, 52.673123] | [47.188337, 52.836136] |
| Normal-traction peak-to-peak (MPa) | 5.464992 | 5.647799 |
| Total shear traction range (MPa) | [24.690232, 27.903733] | [24.696787, 27.622210] |
| Current perturbation tau_xx range (MPa) | [-0.830140, 1.762314] | [-0.766924, 1.760839] |
| Current perturbation tau_yy range (MPa) | [-1.821690, 1.104142] | [-1.820316, 0.936641] |
| Current perturbation tau_xy range (MPa) | [-1.225404, 1.333098] | [-1.166910, 1.320441] |

The pressure range falls 1.76% and the -tau:N range falls 5.35%, but these
cannot be interpreted separately: total normal-traction range grows 3.35%.
Its minimum moves from xd=39990.300 m to 40009.212 m, onto the deep side;
its maximum moves from 39975.980 m to 40000.394 m. No tensile sample appears
in this window. Common-QP RMS normal-traction change is 0.03619 MPa, with
maximum pointwise change 0.65658 MPa.

Native weak averages are not silently compared across changed test spaces.
`nodal_profile.csv` also applies the **original geometric Q1 observer** to
both raw stress fields. At 40 km that common-weight average is 49.997161
MPa originally and 49.998774 MPa in the trial, whereas the new one-sided
native average is 49.888038 MPa. The latter includes changed sampling
weights. At unchanged node 39.95 km, the average changes from 50.004567
to 49.992067 MPa. Small weak averages do not eliminate the raw dipole.

The secondary 13--20-km window uses 10,439 identical QPs. Maximum pointwise
normal-traction change is only 19.28 Pa, and its range remains
48.6633--51.4522 MPa. This local interface change does not explain or repair
the separate 15/18-km features.

## 4. Verification and limits

- Release executable and plugin built with -j4. Existing surface-solver and
  Stage-I tests: **635 assertions, 13 test cases, all passed**. The serial
  launcher emitted sandbox network warnings but completed these tests.
- Four-rank diagnostic: **157.718 s**, 267 Krylov iterations over seven
  returned directions; every fresh residual met its unchanged requested
  tolerance. Largest recorded child RSS: **1,509,104 KiB (1.439 GiB)**,
  not aggregate MPI memory.
- Final normalized bulk/surface residuals: 2.26011e-13 / 1.87689e-12.
  Absolute bulk norm: 8.19384e-4; surface RMS: 2.71318e-5 Pa.
- New K-column central-difference errors contract from 1.0118e-8 at
  epsilon=1e-3 to 1.0151e-10 at epsilon=1e-4. Neighboring/control columns
  and the pressure derivative also pass.
- B residual derivative error: 1.6143e-15. Full G velocity derivative:
  1.3831e-15; G pressure derivative: 6.9277e-11. Independent bulk-QP shear
  virtual-work error: 2.0906e-16. All 295 deep-element QPs have exactly
  zero free-trace weight; 298 free-element QPs retain the original basis.
- Sparse/reference B and G actions and pivoted/UMFPACK inverses are compared
  throughout the solve. Their existing assertions pass.
- Offline raw-QP reconstruction reproduces native weak q, sigma and R to
  8.59e-8, 1.63e-7 and 2.39e-9 Pa respectively. Exported V and frozen Theta
  reproduce the intended interpolation (Theta error 2.22e-16 relative).
  Tensor normal traction agrees with production sigma. Matched chi is
  identical. Empty rank-local control CSVs are valid: the last two elements
  belong to rank 2; the offline reader emits harmless empty-input warnings.
- Complete bulk/current-and-committed V, all particle and surface properties,
  IDs/positions and fault coordinates are restored. Original and copied
  checkpoint hashes are unchanged; no accepted-state output was written.
  Status 1 is the deliberate post-convergence rollback exception, not an
  unexamined solver failure.

The sandbox-blocked MPI launch is retained as `free-trace/independent-local4/`;
it never entered ASPECT. The actual solve is `free-trace/independent-local4-mpi/`.
No numerical failure was retried. No complete ASPECT suite, one/two-rank
equivalence study, general geometry, committing discontinuous-state lifecycle
or new trajectory is qualified by this diagnostic.

### Added equal-endpoint recovery qualification

`qualify_free_trace.py` verifies both saved execution records, fresh linear
checks, accepted clock, identical per-rank incoming fingerprints and
identical checkpoint SHA256 values. The original and copied incoming
checkpoints are rehashed. A's saved solve reproduces the accepted baseline
rates to 8.66e-15 relative. The opt-in source changes leave A's default
continuous-Q1 algebra unchanged; B is the already qualified four-rank solve
with that selector enabled. Neither solve needs to be repeated to answer
this identical frozen-state question. Their historical wall times are
140.530 and 157.718 s, each below the requested 600-s cap.

The added test uses all 206,549 exported production work QPs, identified
uniquely by cell ID and local QP across all four ranks. There are no duplicate
ownership entries; each is assigned to one segment, with positive measure.
All 295 QPs on segment 794 have mechanical weights (1,0); every other QP
retains its original geometric Q1 weights.

At the saved A velocity values, both junction traces equal Vp. The two
bases reconstruct chi*V with maximum absolute difference 5.69e-28 per second,
or 2.07e-16 relative to the maximum source. S and geometry are unchanged.
Thus identical bulk unknowns and frozen state give identical constitutive
point inputs when these traces coincide, without needing a new equilibrium
solve. Theta remains geometric Q1, with measured interpolation error
2.22e-16 relative; there is no state projection or update.

For weak rows, introduce an explicit virtual deep endpoint d in the offline
algebra, before prescribed-row replacement. On segment 794 its weight is xi;
the other endpoint 794 has weight 1-xi. The free endpoint 795 retains only
its free-side element. If r is this expanded row vector, the original
continuous rows are recovered by

\[
 r^{\rm old}_{795}=r_{795}+r_d,\qquad
 r^{\rm old}_{794}=r_{794}.
\]

The implemented elimination of the constant prescribed side instead gives

\[
 r^{\rm elim}_{794}=r_{794}+r_d,\qquad
 r^{\rm elim}_{795}=r_{795}.
\]

These identities hold for any common pointwise response. They were checked
using the actual saved production q, friction, damping, normal traction,
unreplaced residual and mass densities, and a random tied-direction K load.
Maximum continuous-row recovery difference is zero in the evaluated
arithmetic; maximum relative eliminated/native-row difference is 7.42e-16.
This is an algebraic quadrature/basis recovery test, not an independently
re-solved constitutive state. Together with source/input equality and the
existing production K/B/G finite differences and work check, it establishes
the requested coincident-trace consistency. It does not require the new
one-sided endpoint equation itself to vanish at A's solution.

The new matched-point CSV contains the full current constitutive tensor,
pressure, total sigma and q for A/B at 35,430 identical physical QPs near
the junction. Physical phase and chi are identical there. Native and common
geometric weak traction averages remain separately labeled in the original
analysis. No new production source, solver or physical parameter was edited
for this qualification.
The current affected production/plugin diff also matches B's launch-time
source snapshot byte for byte. The new qualification script and its Python
syntax check pass, as does `git diff --check`.

## 5. Recoverability and next decision

The core diff adds a guarded kinematic-coordinate selection in `manager.cc`,
uses the same mechanical weights in work-measure surface assembly/G, and
extends diagnostic exports. The manager header documents the distinction.
BP3 releases the selected node and verifies B/G/K/work and rollback. The
opt-in runner and offline analysis record provenance and common-QP comparisons.
Authority documents record this bounded exception, not its adoption as a
production method. Default behavior is unchanged. Earlier unrelated changes
are preserved. No commit was requested or made.

Base revision: `3335d3d26c298ff5aaeba77062b0a77c8d20f0b5` plus the recorded
working diff. Executable/plugin hashes and a launch-time source patch are in
the run directory. Entry points are `run_notch_probe.py --free-trace` and
`analyze_free_trace.py`.

Commands used (the run driver refuses to overwrite existing evidence):

```sh
cmake --build build-pf-cpdi --target aspect.exe.release -j4
cmake --build benchmarks/reconstructed_fault/bp3/build --target bp3.release -j4
build-pf-cpdi/aspect-release --test '[fault_surface_direct],Stage-I*'
python3 benchmarks/reconstructed_fault/bp3/run_notch_probe.py --free-trace --label independent-local4-mpi
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/bp3/analyze_free_trace.py
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/bp3/qualify_free_trace.py
python3 -m py_compile benchmarks/reconstructed_fault/bp3/run_notch_probe.py benchmarks/reconstructed_fault/bp3/analyze_free_trace.py
git diff --check
```

Build/unit logs were copied into the run directory. Offline interpolation,
weak-load and common-QP assertions, Python syntax checks and whitespace
checks all pass. No source changed after the qualified build; only offline
analysis and documentation were completed afterward.

- [Run and fresh-residual log](../../../benchmarks/reconstructed_fault/bp3/free-trace/independent-local4-mpi/run.log)
- [Execution/provenance checks](../../../benchmarks/reconstructed_fault/bp3/free-trace/independent-local4-mpi/execution.json)
- [Summary](../../../benchmarks/reconstructed_fault/bp3/free-trace/independent-local4-mpi/analysis/summary.json)
- [Comparison plot](../../../benchmarks/reconstructed_fault/bp3/free-trace/independent-local4-mpi/analysis/junction_comparison.png)
- [Incident-element budgets](../../../benchmarks/reconstructed_fault/bp3/free-trace/independent-local4-mpi/analysis/element_budgets.csv)
- [Raw extrema and locations](../../../benchmarks/reconstructed_fault/bp3/free-trace/independent-local4-mpi/analysis/raw_extrema.csv)
- [Added qualification summary](../../../benchmarks/reconstructed_fault/bp3/free-trace/independent-local4-mpi/qualification/summary.json)
- [Equal-trace weak-row recovery](../../../benchmarks/reconstructed_fault/bp3/free-trace/independent-local4-mpi/qualification/equal_trace_recovery.csv)
- [Within-free undershoot versus jump](../../../benchmarks/reconstructed_fault/bp3/free-trace/independent-local4-mpi/qualification/free_interval_undershoot.csv)
- [Matched full constitutive tensors](../../../benchmarks/reconstructed_fault/bp3/free-trace/independent-local4-mpi/qualification/matched_constitutive_tensors.csv)

**Recommended decision:** do not adopt the independent trace as a cure.
It identifies one material junction contribution, but trades part of the
continuous-space gradient for a slip jump without reducing the normal-stress
concentration. Preserve it as diagnostic evidence. If a corrective study is
authorized next, the manufactured evidence favors a matched bulk/fault
spatial-resolution check at frozen history, not particle-density changes or
a new committing discontinuous formulation on the strength of this result.
