# K5 local fault-grid sensitivity at the prescribed-slip junction

## Decision

**The sharp contact core follows the last unprescribed fault node, not the
fixed 39.9-km coordinate.** Bisecting the local fault grid moves contact to
39.95 km and reduces the final rate-depression width from **144.155 to
91.649 m**. The local reaction density grows 14.77%, but the integrated bound
reaction falls **42.55%**. The analogous contact-node Theta and accumulated
slip differ by only +0.98% and -1.48%. This is strong evidence of a
grid-dependent contact core, not a resolved fixed-width locked patch. It does
not establish that the entire surrounding response vanishes with refinement.

Raw stress is **not repaired**: the independently sampled bulk pressure
dipole grows about 3%, the selected junction tensile minimum changes from
-5.402 to -10.712 MPa, and the consistent-Q1 normal-stress variation increases.
The small projected stress variation must not be used to dismiss the raw
dipole. No production equation, tolerance, history representation or cutoff
was changed.

**Next recommendation:** before interpreting this as a physical locked region
or continuing the first-event trajectory, define one coupled local bulk/fault
resolution comparison, preserving their resolution ratio and an admissible
common clock. Its discriminants should be integrated bound reaction and raw
traction as well as the rate-depression width. This is a recommendation for
review, not an additional run or an instruction to smooth the junction.

## Executed comparison and verification

One fresh four-rank Release 50-m trajectory completed through
**2232176379.2516127 s**, initialization plus 13 real steps. Every requested
saved 100-m timestep passed the production restriction/replay guard, so the
existing qualified refined trajectory was reused; **no replacement 100-m
trajectory or second 50-m trajectory was run**.

- Exact initial bulk CellId/coordinate data agree: **42,880 cells**, with
  **385,920 particles**. Before mechanics, the fault guard found **1236
  vertices and zero coordinate error**, preserving all 1156 old vertices.
- The exported initial FE temperature, Maxwell history, Theta-initial,
  strengthening and phase arrays agree exactly at the exported precision.
  Final exported phase equals initial phase. This is an FE-array check, not
  a claim of bytewise identity of every particle property. All saved fault
  coordinates, frozen background tractions and fixed I_h pass the temporal
  invariance checks.
- All 14 accepted states satisfy the unchanged nonlinear criteria, not merely
  an exit-status check. The largest final relative bulk/surface residuals over
  those states are **4.9333e-13 / 3.5750e-9**. At the final state they are
  **4.1464e-14 / 6.7739e-11**.
- All **115 returned linear directions** pass their fresh residual target;
  worst fresh/target ratio **0.98529**, **2325 Krylov iterations** in total.
- Independent stable aging-law and slip-increment checks pass at every real
  step. The benchmark's largest Theta reference relative error is
  **2.22045e-16**, below the unchanged 1e-12 assertion. Supplied initial Theta
  is retained, initial particle Maxwell stress is zero, and prescribed Vp
  remains exactly 1e-9 m/s.
- MPI-summed physical weak residuals reproduce the exported unreplaced
  residual exactly. The independently recombined shear-minus-resistance
  densities agree within 1e-6 Pa. Intentional prescribed-row residuals and
  lower-bound reactions are not incorrectly counted as free-row failures.

Elapsed process time: **1397.53 s (23.29 min)**. Normal timers record 685 s in
the condensed solves, 89.4 s in Stokes assembly, 77.8 s in RHS assembly and
67.7 s building the Stokes preconditioner. Preparation scopes overlap and are
not added as disjoint costs. Peak RSS was not recorded by this runner; the
preflight memory estimate below is not a measured peak.

## Initial projections: visible, not corrected away

At all common vertices, the maximum absolute new-minus-old differences are:

| Initialized quantity | Maximum difference | Location |
|---|---:|---:|
| I_h | 30.7853 m (0.2403% of old value) | 36.0 km |
| Committed C0 | 3716.60 Pa | 37.9 km |
| Evaluated initialization cohesion | 3716.61 Pa | 37.9 km |
| Frozen shear background | 3716.61 Pa | 37.9 km |
| Supplied Theta0 | 0 | all common vertices |
| Solved initial V | 2.13001e-14 m/s | 36.0 km |

At **39.9 km**, I_h changes from 12960.2378739 to 12960.2454915 m;
committed C0 from 490016.455 to 489249.610 Pa; frozen shear background from
27046026.946 to 27045260.096 Pa. At **40 km**, the corresponding changes are
-0.00196904 m, -1420.5545 Pa and -1420.5530 Pa. At the **25-km control**,
initialized I_h, committed C0, frozen background and supplied Theta0 are
identical. These are normal reinitialization/projection differences and
remain in the evolved solution. No later history was transferred or reset.

## Contact location, physical width and integrated reaction

Width is the predeclared connected interval with
V < 0.5 V(39.7 km), evaluated on each native Q1 profile. It is a diagnostic,
not a new acceptance threshold. Both cases have zero width under this
definition through step 9. The late common-time results are:

| Step | Physical time (s) | Width: 100 / 50 m grid (m) | Minimum V/Vp: 100 / 50 | Integrated bound reaction: 100 / 50 (Pa m²) |
|---|---:|---:|---:|---:|
| 10 | 922804465.5975173 | 47.610 / 34.678 | 0.156728 / 0.152487 | 0 / 0 |
| 11 | 1762776093.4253557 | 76.573 / 54.513 | 0.0465702 / 0.0448697 | 0 / 0 |
| 12 | 2165471357.7276783 | 100.444 / 75.209 | 1e-11 / 1e-11 | 4.93900e10 / 2.97342e10 |
| 13 | 2232176379.2516127 | 144.155 / 91.649 | 1e-11 / 1e-11 | 1.51307e11 / 8.69308e10 |

The minima in these rows are at **39.9 / 39.95 km**, respectively. Both first
reach the bound at step 12; each has exactly one lower-active node. Their
basis supports are **200 / 100 m**, entirely inside the common 39--40.5-km
window. The reported integrated reaction is sum(-R_i) over these active
functions, not a sum of reaction densities.

At the final time, V(39.7 km)/Vp is 0.654370 / 0.589556, and the width
crossings are **39.788564--39.932719 / 39.873090--39.964739 km**. Width ratio
is **0.63577**, not exactly one half. The contact-to-junction distance halves;
the surrounding layer does not yet exhibit a demonstrated asymptotic law.

| Final coordinate/case | V/Vp | Committed Theta (s) | Slip (m) | Committed nodal C (MPa) | Lower-active |
|---|---:|---:|---:|---:|---|
| 39.8 km, 100 m | 0.284935 | 2.72136e7 | 1.174723 | 3.371662 | no |
| 39.8 km, 50 m | 0.454780 | 1.75853e7 | 1.143974 | 3.313705 | no |
| 39.85 km, 50 m | 0.393101 | 2.03722e7 | 1.002485 | 2.987570 | no |
| 39.9 km, 100 m | 1e-11 | 6.40276e8 | 0.342264 | 1.386585 | yes |
| 39.9 km, 50 m | 0.180188 | 4.11500e7 | 0.888677 | 2.620951 | no |
| 39.95 km, 50 m | 1e-11 | 6.46562e8 | 0.337208 | 1.492347 | yes |
| 40 km, 100 m | 1 | 8e6 | 2.232176 | 5.970110 | prescribed |
| 40 km, 50 m | 1 | 8e6 | 2.232176 | 5.868795 | prescribed |
| 25 km, 100 m | 0.220602 | 3.60028e7 | 0.711755 | 2.287797 | no |
| 25 km, 50 m | 0.220511 | 3.60174e7 | 0.711588 | 2.287375 | no |

The control changes are small: V -0.0412%, Theta +0.0406%, slip -0.0235%,
C -0.0184%. The analogous last-node history is much more stable than history
at the fixed old contact coordinate: contact follows the discretization.

## Which mechanical terms change?

Use the production sign R/m = driving - cohesion - friction - damping.
The following are **native weak-test averages**, not nodal constitutive
values. In particular, weak evaluated cohesion is distinct from committed
nodal C above. Units are MPa; damping is below 0.0014 Pa at these rows.

| Row | Driving | Cohesion | Friction | Physical R/m | Bound reaction density |
|---|---:|---:|---:|---:|---:|
| 100 m, 39.9 km | 28.326547 | 2.475052 | 26.808730 | -0.957235 | 0.957235 |
| 50 m, 39.9 km | 28.435217 | 2.496594 | 25.938623 | approximately 0 | 0 |
| 50 m, 39.95 km | 27.851904 | 2.386438 | 26.564084 | -1.098618 | 1.098618 |

At the **same 39.9-km coordinate**, driving increases 0.108670 MPa,
cohesion increases 0.021542 MPa, and friction decreases **0.870107 MPa**.
The friction change dominates disappearance of that row's reaction. These
are evolved solutions, not a frozen causal substitution of one history.

Comparing the **two last unprescribed rows** instead, driving decreases
**0.474643 MPa**, while cohesion and friction decrease 0.088615 and
0.244646 MPa. The loss of driving dominates the net **0.141382-MPa increase**
in local reaction. The reaction's integral still decreases because its
test weight nearly halves: 158066.525 to 79127.431 m².

As a footprint check, the original 39.9-km coarse Q1 test is represented
exactly in the nested fine space: fine row 39.9 plus half of each row at
39.85 and 39.95 km. Under this **common test**, fine-minus-coarse driving,
cohesion and friction changes are **+0.119657, +0.115149, -0.677745 MPa**.
Friction remains the dominant change; common-test R/m changes from
-0.957235 to -0.274982 MPa. That remaining negative value includes half the
fine contact row; it is not a failed fine free-row equation. No constitutive
history interpolation or additional mechanical solve is involved.

## Raw versus projected stress

Final-state results below use 39--40.5 km. Consistent-Q1 quantities are
separate from raw samples and independently sampled bulk pressure.

| Measure | 100 m fault | 50 m fault |
|---|---:|---:|
| Projected delta-p peak-to-peak (MPa) | 0.101796 | 0.265263 |
| Projected -delta-tau:N peak-to-peak (MPa) | 0.031269 | 0.156520 |
| Projected sigma_n peak-to-peak (MPa) | 0.132882 | 0.421783 |
| Selected raw constitutive sigma_n minimum (MPa) | -5.401867 | -10.711552 |
| Selected raw constitutive sigma_n maximum (MPa) | 106.084974 | 109.378260 |
| Independent bulk pressure minimum (MPa) | -51.868244 | -56.109112 |
| Independent bulk pressure maximum (MPa) | 54.207736 | 53.144524 |
| Independent bulk pressure peak-to-peak (MPa) | 106.075980 | 109.253636 |
| All-sample particle-history tensile weight (m²) | 6622.701 | 6887.598 |
| All-sample FE-history diagnostic tensile weight (m²) | 5827.979 | 6622.672 |

The independent pressure set is the same **1730 unique bulk vertices**
(6838 cell-local samples), within |normal distance| <=1500 m. These are VTU
precision values, not exact global surface-pressure extrema. The minimum is
at the same physical point (59033.203125, 65283.203125) m. Its dipole amplitude
increases **2.996%**. Fault refinement therefore does not reduce raw pressure;
nor does it merely improve its projection (projected sigma_n variation grows
by **3.174x**).

The fine raw tensile minimum has parent **250724**, cell
`0_11:30210333001`, segment **795**, xi=0.04652616,
surface xd=**39997.673692 m**, and parent position
(59040.862847, 65274.418576) m. It lies on the free-side last segment whose
basis spans free and prescribed nodes; its parent also touches both supports.
Its decomposition is
**50 - 52.697742 - 8.013809 = -10.711552 MPa**.
For the old minimum, xd=39983.561696 m and
**50 - 51.185246 - 4.216621 = -5.401867 MPa**. Thus the stronger tensile
minimum is not only a pressure change: the deviatoric normal contribution
at the extremizing sample also grows. These are different extremizing
particles, not a fixed-particle increment.

Tensile weights use **all production domain samples**, reduced through the
test functions centered in the reporting window, not just the selected
extreme sample list. The tensile support here lies within the window, so its
Q1 partition sums to the full tensile measure; the window's outer basis
tails do not change this total. The production weight rises about **4.00%**.
FE-based values are retained only as diagnostics; surface mechanics still
uses particle history. Projected and test-weight-averaged normal stress at
the contact node remain compressive. Tension at selected transverse samples
is not, by itself, an explanation of contact.

## Reproducible artifacts and changes

- [Compact six-panel profile figure](../../../benchmarks/reconstructed_fault/bp3/fault-grid-50-local4/fault_grid_profiles.png).
  Gray region has prescribed V. Nonzero physical residuals there are
  intentional; only lower-active unprescribed rows enter the bound total.
- [Every common accepted-time junction/control comparison](../../../benchmarks/reconstructed_fault/bp3/fault-grid-50-local4/junction_history.csv),
  including Theta used by mechanics separately from committed Theta.
- [Numerical summary and checks](../../../benchmarks/reconstructed_fault/bp3/fault-grid-50-local4/fault_grid_comparison.json),
  [analysis log](../../../benchmarks/reconstructed_fault/bp3/fault-grid-50-local4/analysis.log),
  [solver log](../../../benchmarks/reconstructed_fault/bp3/fault-grid-50-local4/run.log).
- [Grid, clock and input mapping](../../../benchmarks/reconstructed_fault/bp3/fault-grid-50-local4/grid_plan.json),
  [binary/plugin/source/input hashes](../../../benchmarks/reconstructed_fault/bp3/fault-grid-50-local4/provenance.json).
  Original baseline: `junction-matched-qualified-local4/refined/`, including
  the previously supplied `final_profiles.csv` and `node_history.csv` in its
  parent directory. Original outputs are untouched.

Commands actually used:

```sh
cmake --build benchmarks/reconstructed_fault/bp3/build --target bp3.release -j4
python3 benchmarks/reconstructed_fault/bp3/run_fault_grid.py prepare
python3 benchmarks/reconstructed_fault/bp3/run_fault_grid.py run
python3 benchmarks/reconstructed_fault/bp3/analyze_fault_grid.py > benchmarks/reconstructed_fault/bp3/fault-grid-50-local4/analysis.log 2>&1
```

The runner contains the exact four-rank launch and refuses to overwrite a
previous trajectory. Cheap width checks verify no depression for a constant
field and identical widths when the same Q1 triangle is represented on a
nested grid. Python compilation and `git diff --check` pass. No broad suite,
rollback rerun, or extra trajectory was needed for these benchmark-only edits.

Files authored/edited for this task: `run_fault_grid.py`,
`analyze_fault_grid.py`, the optional exact-target/expected-fault guards in
`matched_resolution.h`, this report, and the generated case/evidence directory.
The plugin was rebuilt only to enable these guards. No production files or
defaults were changed in this task; unrelated working-tree edits are retained.

**Limitation:** this establishes fault-grid sensitivity at fixed bulk
resolution and matched finite timesteps. It is neither a continuum limit
nor proof of a particular junction correction. The earlier temporal
uncertainty remains. Stop here: no additional bulk/fault/timestep refinement,
FE-history solve, changed cutoff or first-event continuation was performed.

## Predeclared comparison

Authority for this task: `benchmarks/reconstructed_fault/bp3/stage_K5_fault_grid_task.md`.
Use the qualified refined **42880-cell** bulk mesh and its saved 100-m fault
trajectory. Add midpoints to the 80 complete existing fault elements in
36--44 km: the actual interval is 36000--43998.413448 m. Preserve every old
vertex, including the exact 40-km boundary-condition junction, and both open
tips. There are 1236 vertices instead of 1156. Last unprescribed vertex moves
from 39.9 to 39.95 km; this need not be the last inactive/free-set vertex.

Supply the complete explicit vertex list through the existing fixed-geometry
input interface. Set its resampling cap to 101 m: some original nominal
100-m edges are roundoff-long, and ceil(length/100) would accidentally bisect
them again. Source inspection confirms the cap is used only for resampling.
The actual grid, not this cap, defines the experiment; all old edges outside
the requested patch remain unchanged. No production code/default changes.

The same initial physical profile, zero perturbation Maxwell stress, official
Theta prescription and normal background are used. Surface I_h, cohesion and
matching frozen shear background are reinitialized by the existing algorithm,
not interpolated from a late state. Their changes at common coordinates remain
part of the dynamics and are reported without retuning.

Use the old refined accepted times through **2232176379.2516127 s**, retaining
the actual CFL/RSF/generic restrictions through the existing replay-cap guard.
If these times cease to be admissible, preserve evidence and use the permitted
matched-clock fallback rather than overriding a restriction. No second 50-m
trajectory, further bulk refinement, time-halving campaign or FE-history solve
is part of the budget. Expected cost for one four-rank fresh run: approximately
18--25 minutes and 6--8 GiB summed rank memory. One explicitly core-bound job
avoids the prior paired CPU-placement contention.

The benchmark checks exact bulk leaves and expected reconstructed vertices
before mechanics. Ordinary per-accepted-state fault, weak-moment and selected
raw sample diagnostics are reused; no full-state or profiling campaign is added.

## Measures fixed before interpretation

At common times compare V/Vp, Theta, accumulated slip, committed cohesion,
mechanical weak cohesion/shear/friction, unreplaced R_i/m_i and active-node
reaction. Identify nodes by physical coordinate, not their changing indices.
The detailed junction window is **39--40.5 km**, with 25 km as unchanged control.

Report total bound reaction as **sum of -R_i over active basis functions in
the same junction window**, in Pa m² under the existing domain weak measure.
Verify active basis supports lie within that window; do not sum densities
without their positive test weights. Raw extrema and all-sample tensile
weights are distinct from consistent-Q1 stress projections.

The existing raw constitutive export selects normal-stress extrema, not
independent pressure extrema. Label its associated pressure range accordingly.
Also compare the saved bulk Q1 vertex pressures on the identical mesh in
39--40.5 km and |normal distance| <=1500 m. This separate fixed sample set
measures the raw bulk pressure dipole without surface projection; retain VTU
precision and do not call its extrema exact global surface-pressure extrema.

For a compact physical width diagnostic, use the connected depression around
the minimum between 39.7 and 40 km, bounded by crossings of
**V = 0.5 V(39.7 km)** in the native Q1 rate profile. Report the reference
rate and both crossing coordinates. This is a fixed diagnostic definition,
not a new pass threshold or continuum reference. Also report contact-node
locations and their Q1 basis support lengths; no smoothing/interpolation of
histories is used. The half-depth width and layer location discriminate a
last-node effect from a depression of stable physical extent.

Two fault grids and the known temporal uncertainty establish sensitivity,
not continuum convergence. Stop after the decisive comparison and one next
recommendation; do not continue the first-event trajectory.
