# K5 matched-time, localized bulk-resolution pair

Consolidated synthesis of these four follow-up tasks:
[history, spatial-resolution and timestep report](stage_K5_history_resolution_consolidated_report.md).
This individual report is retained as the chronological evidence record.

Completed follow-up: [fixed-mesh first-contact timestep subdivision](stage_K5_contact_subdivision.md).
It retains lower contact but exposes material transient/neighbour-rate
sensitivity; it does not establish temporal convergence.

## Decision

**Retain the particle-history formulation.** Local refinement substantially
reduces the particle/FE weak-history discrepancy and the oscillation in the
consistent surface stress representation, but does not remove the 39.9-km
lower contact. Neither representation is an independent accuracy reference.

Both four-rank runs completed initialization and 13 real steps at identical
accepted times through **2232176379.2516127 s**. The normal common-test
history-load gap contracts by **4.015x**, and the tangent gap by **4.854x**.
The final contact reaction changes by **-6.20%**, while both nodes remain at
`Vmin=1e-20 m/s`.

This is not full convergence: raw junction tension becomes stronger on the
refined mesh, and the first-contact reaction is considerably more sensitive
than the final reaction. The different common time grid also materially
changes the coarse reaction and neighbouring rate relative to the old run.

**One recommended next action:** before changing history representation or
continuing the long trajectory, perform a bounded timestep-subdivision check
from the accepted refined step-11 state through this same final time. It should
resolve the first-contact/split-aging interval at fixed spatial resolution.
No such additional solve, alternate-history solve, or further spatial level
was launched here.

## 1. Controlled comparison and initialization

The coarse case reproduces all 36106 saved CellIds and coordinates exactly.
The refined case has 42880 cells, with 48.828125-m junction centerline spacing
instead of 97.65625 m. Its 2179 requested parent splits plus 79 standard grading
splits leave the control and bottom mesh sets unchanged. Both retain the same
100-m fault discretization. The following remain unchanged: physical loading,
true-pressure treatment, 40-km prescribed-slip cutoff, particle history law,
transfer/interpolator, particles per cell, frozen phase, full I_h, support,
quadrature, active-set rules and solver tolerances.

The shared comparison clock is an additional cap below **both** actual
production CFL/RSF restrictions. Generic ASPECT timestep restrictions remain
active. Every accepted time/dt agrees between the two jobs to the checked
roundoff allowance; every selected dt is at most 95% of both published limits.
No constitutive state or bulk field is exchanged between jobs.

The new coarse initial C, I_h, background shear, Theta and V reproduce the
old coarse values exactly. Refined initialization is independent, not a
transfer of coarse nodal/particle state:

| Initial quantity | Maximum coarse/refined difference |
|---|---:|
| Committed C | 11302.158 Pa |
| Evaluated initialization C / matching background shear offset | 11102.047 Pa |
| I_h | 297.566 m |
| Theta | 0 |
| V | 1.03827e-13 m/s |

At 39.9 km, initial I_h changes from **12691.827 to 12960.238 m** (about
2.115%). Thus this tests combined bulk/particle/initial-projection resolution,
not an isolated transfer-method change. These initial differences remain in
the subsequent dynamics; none was subtracted out physically. Within each run,
the fault coordinates, background shear and I_h stay unchanged at every
accepted state, as checked against initialization.

## 2. Physical bound balance at the final common time

Numbers below are **physical weak rows divided by the positive test weight**
`m_i=integral N_i dnu`, not consistent-M-inverse nodal residual values.
The unreplaced physical residual is
`R/m_i = shear - cohesion - friction - damping`; the resisting lower-bound
reaction is `lambda=-R/m_i` when the node is active.

| 39.9-km node 756 | Matched coarse | Matched refined |
|---|---:|---:|
| V (m/s) | 1e-20 | 1e-20 |
| Test weight (m²) | 158130.714 | 158066.525 |
| Shear driving (MPa) | 28.380324 | 28.326547 |
| Cohesion (MPa) | 2.608635 | 2.475052 |
| Friction (MPa) | 26.792169 | 26.808730 |
| Damping (Pa) | 0.0009973 | 0.0009912 |
| R/m_i (MPa) | -1.020480 | -0.957235 |
| Lower reaction (MPa) | +1.020480 | +0.957235 |

The nonzero weak-average damping at a bound vertex includes Q1 contributions
from its neighbouring nonzero rates. It is not a claim that the vertex rate
itself exceeds the bound.

At each solution, the existing diagnostic substitutes only the **complete
retained stress tensor** by its constrained FE evaluation. Other bulk and
surface inputs remain frozen, using the production domain weights and point
response. The production solve itself still uses particle history.

| FE-minus-particle frozen change at node 756 | Coarse | Refined |
|---|---:|---:|
| Shear (kPa) | -38.608 | +7.480 |
| Normal traction (kPa) | -31.644 | -12.318 |
| Friction (kPa) | -15.877 | -6.016 |
| Physical residual (kPa) | -22.731 | +13.496 |
| Absolute residual change / actual reaction | 2.227% | 1.410% |
| V-tangent diagonal change | -0.1932% | -0.1380% |

The frozen FE alternative cannot reverse either final contact decision.
The residual change is assessed against physical term scales and the bound
margin, not against the tiny converged free-row residual.

Neighbouring final rates change from **2.96529e-10 to 2.84935e-10 m/s** at
39.8 km (-3.910%), and **6.36979e-10 to 6.54370e-10 m/s** at 39.7 km
(+2.730%). The 25-km control changes by only +0.1871%. Prescribed Vp remains
exact. Accumulated slip at 39.9 km is **0.377755 / 0.342264 m**; it is not
reset from output or from the other trajectory.

## 3. Common weak-history tests: a resolved transfer discrepancy contracts

These are the retained inputs to final step 13: histories committed after
step 12 at **2165471357.7276783 s**, multiplied by the actual final-step beta.
The benchmark samples the private mechanically constrained FE history and
the actual parent-domain particle history before Newton/history publication.
It does not refresh published fields or reevaluate a committed Maxwell update.

The existing compact smooth virtual velocities have identical physical support
at both resolutions (1000-m tangential and 800-m normal half-widths). Particle
loads integrate actual full domain polygons; FE loads use ordinary cell FE
quadrature. These are common smooth tests, not identical mesh-dependent Stokes
basis vectors. The sign is `-integral beta*tau_old:epsilon(w)`.

| Test at 39.9 km | Coarse particle | Coarse FE | Refined particle | Refined FE |
|---|---:|---:|---:|---:|
| Normal (Pa m) | -2.4031537430e10 | -2.3899021071e10 | -2.3959816658e10 | -2.3926808220e10 |
| Tangent (Pa m) | -8.5165028e6 | -3.8303373e6 | -6.0717640e6 | -5.1064380e6 |

The normal gap decreases from **1.32516358e8 to 3.30084381e7 Pa m**, a factor
**4.015**, or from 0.5514% to 0.1378% of the corresponding particle load.
The tangent gap decreases from **4.68616542e6 to 9.65325920e5 Pa m**, factor
**4.854**. The Q4-to-Q6 sums of absolute load changes are only
**562.6 / 20.8 Pa m** for coarse/refined normal tests and
**850.5 / 113.0 Pa m** for tangent tests. These are quadrature-change checks,
not rigorous error bounds; they are much smaller than the reported gaps.

At the unrefined 25-km control, the normal gap stays essentially unchanged:
**4.3080e6 / 4.3171e6 Pa m**. Its cancellation-small tangent gap
(500 / 538 Pa m) is below the roughly 6100-Pa-m quadrature change and is
**not resolved**. The unrefined bottom gaps also remain approximately unchanged
(-6.4638e7 normal and -6.5819e7 tangent Pa m). No accuracy conclusion is drawn
from relative errors of cancellation-small test loads.

## 4. Weighted surface profiles improve; raw tension is not cured

Over 37--43 km, using each run's consistent Q1 projection:

| Peak-to-peak field | Coarse | Refined |
|---|---:|---:|
| Delta p | 0.297138 MPa | 0.101796 MPa |
| -Delta tau:N | 0.137821 MPa | 0.062933 MPa |
| Total sigma_n | 0.434959 MPa | 0.132882 MPa |

The sigma_n oscillation contracts by **3.273x**. The spatial RMS difference
in projected sigma_n is 34.735 kPa in the junction window, versus 0.919 kPa
in the unrefined 24--26 km control window. This is consistent with a significant
local discretization contribution, but is not proof of asymptotic convergence.

Raw production samples tell a different—and compatible—part of the story:

| Junction quantity | Coarse | Refined |
|---|---:|---:|
| Minimum sigma_n | -0.940933 MPa | -5.401867 MPa |
| Down-dip location of minimum | 39961.330 m | 39983.562 m |
| Delta p at minimum | -42.604080 MPa | -51.185246 MPa |
| Delta tau:N at minimum | +8.336853 MPa | +4.216621 MPa |
| Maximum sigma_n | 100.125714 MPa | 106.084974 MPa |
| Total tensile integration weight, particle evaluation | 1059.644 m² | 6622.701 m² |
| Total tensile integration weight, frozen FE evaluation | 0 | 5827.979 m² |

Both minima are in segment 755, with mixed free/prescribed support. The coarse
minimum has xi=0.386702 and the refined minimum xi=0.164383. The negative
stress at a selected tiny-weight QP must not be interpreted as a zero-measure
defect: the all-sample tensile weights above account for the full production
quadrature, not just the selected extrema.

At the lower-active basis specifically, tensile weights increase from
**645.206 to 2579.335 m²** under particle evaluation; the refined FE alternative
also has **2125.676 m²** of tensile weight there. Consequently FE substitution
does **not** generally remove tension. Tension reduces the frictional resisting
term; it is not evidence that tension itself causes locking.

The unrefined bottom remains tensile, with minima -18.783285 / -18.782836 MPa.
No claim is made that raw stress extrema, the bottom, or the full stress field
are spatially converged.

## 5. Do not hide first-contact/time-discretization sensitivity

Both cases are free at step 11 and active at steps 12 and 13, but the reaction
at the first accepted contact is markedly resolution-sensitive:

| State | Coarse reaction | Refined reaction |
|---|---:|---:|
| Step 12, t=2165471357.7276783 s | 61.862 kPa | 312.463 kPa |
| Step 13, t=2232176379.2516127 s | 1020.480 kPa | 957.235 kPa |

At step 12, the frozen FE residual changes are -23.781 / +10.265 kPa. Even
there neither change reverses contact, although the coarse change is **38.44%**
of its smaller bound margin. The final 6.2% spatial reaction difference must
not be presented as representative of onset accuracy.

The mechanics at step 13 uses the Theta committed after step 12, not the newly
updated output Theta. At 39.9 km, step-11 Theta is 1.0703e8 / 1.7088e8 s,
and step-12 Theta is 5.0972e8 / 5.7357e8 s. The short final step follows this
large split aging update. The reference checker passes; this is temporal
discretization sensitivity, not the previously repaired cancellation bug.

At the same final physical time, the **old coarse** reaction was only
127.699 kPa, versus **1020.480 kPa** on the new common time grid. The old
39.8-km rate was 4.49297e-10 m/s, versus 2.96529e-10 m/s on the new coarse
grid. Initial fields are identical between those coarse runs. The temporal
schedule therefore materially affects the reaction and adjacent rate, even
though the projected junction sigma_n RMS temporal change (2.480 kPa) is much
smaller than the matched spatial change (34.735 kPa).

This motivates the single recommended fixed-mesh late-window timestep check;
it does not justify changing history representation, smoothing stress, shifting
the junction, or clamping normal stress. The physical existence of one contact
is robust in the tested pair, but onset time and reaction are not established
as converged.

## 6. Verification, cost and artifacts

| Check | Coarse | Refined |
|---|---:|---:|
| Accepted real steps | 13 | 13 |
| Returned linear directions | 113 | 115 |
| Total Krylov iterations | 2276 | 2337 |
| Worst fresh-linear residual / requested target | 0.98638 | 0.98529 |
| Final relative bulk residual | 3.8161e-14 | 4.1403e-14 |
| Final relative surface residual | 6.6668e-11 | 6.8202e-11 |
| Final absolute surface RMS residual | 9.0518e-5 Pa | 9.2687e-5 Pa |
| Maximum existing Theta-checker relative error | 3.3307e-16 | 2.2204e-16 |
| Dual particle weak R versus production R | exact match | exact match |
| Final free/lower-active counts | 399 / 1 | 399 / 1 |

Both process exits are zero, **in addition to**, not instead of, the numerical
and lifecycle checks. The final step accepts full Newton steps. No failed
physical solve, tolerance change, rollback bypass, or alternative FE-history
mechanical solve occurred. Previous failure/rollback evidence is reused, not
claimed as a newly rerun rollback test.

Elapsed paired execution was **3210.6 s (53.51 min)**, including initial CPU
contention described below. One memory snapshot summed to approximately
11.2 GiB of rank RSS; shared pages may be double-counted and peak memory was
not measured. No performance claim is made.

Commands:

```
cmake --build benchmarks/reconstructed_fault/bp3/build --target bp3.release -j4
python3 benchmarks/reconstructed_fault/bp3/run_matched_resolution.py prepare --tag qualified
python3 benchmarks/reconstructed_fault/bp3/run_matched_resolution.py run --tag qualified
python3 benchmarks/reconstructed_fault/bp3/analyze_matched_resolution.py --tag qualified
```

For a future concurrent rerun, arrange disjoint CPU allocation/binding **before
launch**: the helper currently uses plain `mpirun`, whose default placement
overlapped on this machine. The tested run was corrected live, as recorded in
`placement_note.json`; do not assume the command alone reproduces its runtime.

Main artifacts under `benchmarks/reconstructed_fault/bp3/junction-matched-qualified-local4/`:

- `comparison.json`, `final_profiles.csv`, `junction_profiles.png`, `node_history.csv`;
- each case's `accepted_steps.csv`, `fault_*.csv`, `history_surface_step*_rank*.csv`,
  `surface_comparison.csv`, `common_history_tests.csv`, and `analysis/step13/`;
- `clock/` (all paired proposed limits), exact target-cell lists, mesh exports,
  source/binary/input hashes, logs, and ordinary output/checkpoints.

The rolling restart slots retain the final three accepted states; with the
unchanged three-slot cycle and last slot 02, slot 03 corresponds to step 11.
No checkpoint extraction/restart replay was added to this task.

Only benchmark code and documentation changed. `matched_resolution.h` supplies
the saved-mesh plugin, paired cap and read-only diagnostics; `bp3.cc` connects
them only for this opt-in comparison. Python compilation and `git diff --check`
pass. No production algorithm or general test suite was changed/run.

## Execution preparation corrections

The first paired preflight is retained in `junction-matched-local4/`. Coarse
mesh reconstruction was exact. The refined guard stopped before mechanics:
ASPECT's existing grading/smoothing added 316 child leaves replacing 79 target
leaves, giving 42880 cells. Sixteen children were outside the initially assumed
3000-m normal grading envelope. One mesh-only export (13.344 s, no mechanics)
confirmed that every extra cell is a target descendant; their centers span
35.640--44.376 km down-dip and at most 4.823 km normal distance. These are bulk
grading cells, **not a change in mechanical association/support width**.

The actual grading closure was used as an explicit target for
`junction-matched-qualified-local4/`. Both targets then matched exactly,
with zero extra leaves. All 497 coarse control cells within 24--26 km and
1500 m normal distance, and all 696 bottom-control cells beyond 113 km in
that strip, remain unchanged. The original coarse initialization traction,
cohesion and Theta targets reproduce bit-for-bit. The qualified pair, not the
failed preflight, is the comparison trajectory.

The simultaneous MPI jobs initially bound to the same four OS CPU cores,
giving each rank about 45--54 percent CPU. At approximately 1880 s elapsed,
the existing processes were moved to disjoint physical cores (three performance
cores and one efficiency core per job), without restart or a numerical change.
`placement_note.json` records the exact PIDs and masks. Runtime includes this
contention and must not be interpreted as a solver-performance comparison.

## Predeclared comparison

The preceding preflight is preserved in `stage_K5_junction_resolution.md`.
This approved follow-up retains production particle-history mechanics and
compares fresh solutions at identical accepted physical times, ending at
`2232176379.2516127 s` (the old step-12 time). New step indices need not be 12.

The coarse mesh is reconstructed from the saved 36106 CellIds. The refined
target splits 2179 of these leaves within 36--44 km down-dip and the original
1500-m normal half-width, giving 42643 requested leaves. The existing mesh
infrastructure supplies required grading, with no requested coarsening or
global extra minimum-refinement pass. A pre-mechanics guard requires exact
coarse leaves, and permits only extra descendant leaves within 34--46 km and
3000 m normal distance in the refined case. Exported meshes must independently
confirm the control and bottom regions are unchanged.

Both independent jobs use four MPI ranks. At each committed mechanical state,
a benchmark-only clock calls the actual production convection timestep method
and the actual material split-RSF timestep method. Each job publishes only its
physical time and MPI-minimum timestep limit. The common additional cap is
`0.95*min(coarse_limit,refined_limit)`. The generic ASPECT manager still applies
the unchanged growth, first-step, end-time and other parameter restrictions.
No mechanical unknown or constitutive history passes between jobs. Atomic
file publication and a fresh exchange directory prevent partial/stale reads;
a peer failure or timeout stops the comparison, never forces a larger step.
This is a controlled comparison clock, not a production timestep modification.

The physical laws, loading, pressure, full I_h, support, surface quadrature,
fault grid, Vmin, solver tolerances, interpolation and history timing are
unchanged. Particle density **per cell** stays fixed. Initialization is fresh;
no coarse nodal or particle histories are transferred. Projected initial C,
I_h and background changes are reported, not subtracted from the dynamics.

Compare the 39.9-km physical weak residual/reaction and neighbouring rates,
37--43 km projected and raw traction/pressure profiles, and the 25-km control.
Use the existing dual full-tensor particle/constrained-FE response diagnostic
at the **same solution of each mesh**, retaining production particle history.
Record final common smooth-test weak history loads before Newton/publication,
using the existing independent Q4/Q6 integrations. No FE-history alternate
mechanical solve is part of this pair.

Compare the new coarse endpoint with the old coarse step 12 separately to
expose the changed temporal discretization. Two spatial levels can establish
sensitivity, not prove asymptotic convergence or select an exact history
representation. If the contact disappears at both resolutions under the new
clock, report that temporal sensitivity rather than manufacturing contact.

Estimated cost: 20--35 minutes elapsed for two concurrent four-rank jobs;
about 10--16 GiB combined memory expected. No retry, additional refinement,
or event continuation. A 60-step safety termination and a 600-s peer-arrival
timeout protect against uncontrolled work. All completed/failed evidence is
retained in `benchmarks/reconstructed_fault/bp3/junction-matched-local4/`.
