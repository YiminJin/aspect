# K5 consolidated history, spatial-resolution and timestep report

## Decision summary

**Retain the production particle-history formulation.** The particle/FE
retained-stress difference materially changes the 39.9-km bound margin and
neighbouring rates, but replacing it by constrained FE history does not release
the node or remove the junction pressure concentration. Neither representation
has been established as an independent accuracy reference.

The completed localized refinement reduces the common-test weak-history gap
by **4.015x normal / 4.854x tangential** and the consistent-Q1 normal-stress
oscillation by **3.273x**. However, raw junction tension becomes stronger;
weighted-profile improvement is not raw-stress convergence.

The subsequent fixed-mesh timestep subdivision retains lower contact and
changes the final reaction by only **-2.21%**, but changes the first common
endpoint reaction by **+129.35%** and the final neighbouring rate by
**-8.69%**. The split-history cycle is verified; its transient and adjacent
trajectory are not established as timestep-converged.

**Current next decision:** if quantitative junction rates/slip are required
before the long first-event run, perform one further halving of the same
fixed-mesh checkpoint window to test contraction. Do not substitute FE history,
smooth/clamp stress, change the 40-km cutoff, or alter solver criteria on the
basis of these results. That further comparison has not been run.

This document consolidates the following four task reports. It supersedes
their intermediate next-action recommendations, not their evidence or
recorded failures. Original reports and artifacts are preserved.

1. [Particle/FE history mechanics audit and conditional solve](stage_K5_history_mechanics_result.md).
2. [Stopped localized-refinement preflight](stage_K5_junction_resolution.md).
3. [Completed matched-time localized spatial pair](stage_K5_matched_junction_resolution.md).
4. [Fixed-mesh first-contact timestep subdivision](stage_K5_contact_subdivision.md).

The earlier [normal-stress consolidated report](stage_K5_normal_stress_consolidated_report.md)
remains the context for the junction-location diagnosis; that experiment was
not repeated by these four tasks. This consolidation runs no simulations and
changes no production or benchmark code.

## 1. Common definitions and comparison ledger

All production trajectories retain the stress-perturbation BP3 model,
50-MPa background normal traction, true normal-stress feedback, frozen phase
and fault geometry, full I_h, existing support/domain quadrature, 40-km
prescribed-slip junction, Vmin=1e-20 m/s, pressure treatment, physical loading,
and nonlinear/linear tolerances. The FE-history alternative is a guarded,
noncommitting diagnostic, not a promoted production method.

Three distinctions are essential:

- **Particle versus FE history:** surface mechanics normally consumes retained
  parent-particle stress. The alternative samples the complete retained stress
  tensor from the privately constrained FE history at the same parent
  coordinate. That working FE history is distinct from the published field;
  neither is refreshed merely for output.
- **Physical weak row versus nodal projection:** with positive test weight
  `m_i = integral N_i dnu`, report `R_i/m_i` and lower reaction
  `lambda_i = -R_i/m_i` at contact. These are not the consistent-M-inverse
  projected nodal residual. In the original case, the latter is +1.611 MPa
  while the physical row is negative; it cannot diagnose complementarity.
- **Raw versus represented stress:** raw constitutive samples and all-sample
  tensile weights are separate from consistent-Q1 projected traction profiles.
  A smaller projected oscillation or positive extrema under one evaluation
  does not establish elimination of raw tension.

The residual convention is

\[
 R_i=q_i-C_i-f_i-d_i,\qquad
 \sigma_n=50\,\mathrm{MPa}+\Delta p-\Delta\tau:N.
\]

The split cycle consumes committed preceding histories during mechanics and
updates them only after convergence. This follows
[current design, Stage J](../current_design.md#25-stage-j-constitutive-history-feedback)
and [the specification](../specification.tex). Initial histories are retained
under the approved timestep-zero semantics; they were not reset to later output.

| Label used here | State/trajectory | Legitimate comparison |
|---|---|---|
| Original coarse | 36106 cells; original controller trajectory; final step 12 at 2232176379.2516127 s | Frozen particle/FE evaluation and its one re-equilibrated diagnostic |
| Preflight | 47470 cells; stopped after converged t=0 mechanics, before accepted postprocessing | Mesh/controller preparation evidence only |
| Matched coarse / refined | 36106 / 42880 cells; 13 real steps at identical times; final step 13 at the same final time | Coupled spatial/particle/initial-projection sensitivity |
| Half-step refined | Branch from matched refined accepted step 11 at 1762776093.4253557 s; four steps instead of two; final step 15 at the same final time | Fixed-mesh late-window temporal sensitivity |

“Step 12” therefore does not identify the same physical state across reports.
The old coarse and matched coarse initial fields are identical, but their
later histories differ because their accepted time grids differ.

## 2. History representation: material effect, no demonstrated repair

### Frozen full-tensor comparison

The exact original step-12 Stokes FE polynomials were extracted by CellId and
local FE index: all 36106 cells, with 22 Stokes coefficients per cell. They
were not interpolated from VTU or matched by potentially different global
DoF numbers. A separate step-11 checkpoint copy performed normal step-12
advection and transfer, then privately supplied the saved u,p,V before Newton.
Retained Theta, cohesion and stress histories were unchanged.

The production unreplaced weak residual was reproduced to **9.7761e-17** of
the maximum weak shear-term scale. Both evaluations use the production domain
quadrature, Q1 weights, point response and full old stress tensor; both shear
and normal contractions change consistently.

At node 756, 39.9 km, terms divided by positive test weight are:

| Term | Particle history | Constrained FE history, frozen state |
|---|---:|---:|
| Shear driving (MPa) | 28.460551 | 28.420536 |
| Cohesion (MPa) | 2.596146 | 2.596146 |
| Friction (MPa) | 25.992105 | 25.977701 |
| Physical R/m (kPa) | -127.699 | -153.310 |
| Normal traction / test weight (MPa) | 50.220498 | 50.190932 |

The **20.06%** contact-margin change is material despite being only 0.0900%
of shear driving. The V tangent diagonal changes by **-0.13749%**; feasible
directional finite differences at h=1e-17 m/s agree with -K within **1.29e-6
relative** for both representations. Smaller perturbations reveal subtraction
noise, not evidence for changing tolerances. The frozen auxiliary history
does not introduce a new derivative into G.

Tensile samples contribute **-1.452 kPa** to the original signed friction
term at this basis, reducing resistance. **Tension is not demonstrated to
cause locking.** FE substitution reduces driving shear more than friction,
so its frozen contact reaction increases rather than disappears.

### One conditional re-equilibration

The one authorized disposable solve changed only surface old-stress sampling
to constrained FE history, consistently in residual/Jacobian shear and normal
terms. It retained the bulk history load and solved the original step-12
problem without committing history.

| Outcome | Original particle solution | FE-history diagnostic solution |
|---|---:|---:|
| 39.9-km V (m/s) | 1e-20 | 1e-20 |
| Bound reaction (kPa) | 127.699 | 183.696 |
| 39.8-km V (m/s) | 4.492974e-10 | 4.679628e-10 (+4.154%) |
| Junction pressure peak-to-peak (MPa) | 0.319999 | 0.321120 |

The reaction becomes **43.85% larger**, and the pressure concentration remains.
At the new solution, FE-mode diagnostics show no tensile integration weight;
the original particle evaluation still has **1059.643 m² at the junction and
5297.873 m² at the bottom**, unchanged from baseline. Apparent sign improvement
is therefore not a demonstrated mechanical repair. The later refined runs
also show tension under FE evaluation, so its absence here is not generic.

The solve passed all fresh-linear/nonlinear checks and complete in-memory
rollback verification: bulk state, committed/current V, particle and surface
properties, IDs/positions and fault coordinates. Exit 1 was the intentional
noncommitting stop. No history was published and no alternate solve was repeated.

### Common weak-history loads

Common smooth virtual velocities, with 1000-m tangential and 800-m normal
half-width, compare `-integral beta*tau_old:epsilon(w)` independently of
mesh-dependent Stokes basis functions. Particle P0 history is integrated over
full actual owned Voronoi polygons; FE history over actual cells. Q4/Q6 changes
are checked separately; no clipping, new force or domain modification occurs.

In the original frozen case, junction normal/tangent FE-minus-particle gaps
are **1.116709e8 / 3.912906e6 Pa m**, much larger than quadrature changes
**556.3 / 720.7 Pa m**. The normal gap is 0.5501% of the particle load.
The cancellation-small tangent load changes sign; that is not an equally
large change in total mechanics. Bottom gaps are separately large. The tiny
25-km tangential difference is below its quadrature change and remains unresolved.

## 3. Why the first refinement was not usable—and what replaced it

The 66.234-s preflight stopped intentionally before step 1:

1. The refined controller required a step **6.2942 s smaller (2.4041 ppm)**
   than the saved coarse first step. The actual initial rate and RSF formula
   reproduce that restriction. It was not overridden or treated as a solver
   failure.
2. An extra adaptive pass refined/coarsened cells outside the requested patch:
   3794 split parents, 1544 outside 36--44 km, and 24 coarsened leaves. The
   control's centerline was unchanged but its full history-transfer neighbourhood
   was not. This would confound an alleged localized comparison.

Its t=0 mechanics converged, but timestep selection stopped before accepted
postprocessing. It supplies no accepted trajectory or valid new checkpoint.

The approved replacement reconstructs the exact saved coarse tree and splits
2179 requested parents plus 79 required grading parents: **42880 refined
cells**. Both targets match their actual meshes exactly. The 497 control-strip
and 696 bottom-control coarse cells remain unchanged. Junction centerline
spacing changes from **97.65625 to 48.828125 m**; the 100-m fault grid and
particle density per cell stay fixed. Quadtree refinement halves both local
bulk directions, not only the normal direction.

Two independent four-rank jobs use a shared additional cap
`0.95*min(coarse controller limit, refined controller limit)`, retaining all
generic restrictions. Only time/limits are exchanged, never solution/history.
They complete 13 real steps at identical physical times.

A separate initial paired guard failure identified standard mesh grading
beyond an assumed normal envelope. A mesh-only diagnostic verified the
descendants; the observed grading closure became the exact refined target.
This expanded the **bulk grading region**, not mechanical support. Failed
preparations remain separate from the qualified trajectories.

## 4. Matched spatial comparison

### Initial projection changes remain part of the physics of each run

The matched coarse initialization reproduces old coarse C, I_h, background,
Theta and V exactly. Refined initialization is fresh from the same physical
data. Maximum coarse/refined differences are **11.302 kPa committed C**,
**11.102 kPa evaluated C/matching background shear**, **297.566 m I_h**, zero
Theta, and **1.03827e-13 m/s V**. At 39.9 km I_h changes from 12691.827 to
12960.238 m (+2.115%). These influence subsequent history; subtracting their
initial difference would not remove that influence.

Thus the pair measures coupled bulk/particle/initial-projection resolution,
not a pure isolated history-transfer error. Within each trajectory, fault
coordinates, background and I_h remain fixed.

### Weak-load and weighted-profile improvement

The final-step common tests use the retained histories committed at the
preceding common time, not post-solve or refreshed visualization fields.

| Junction measure | Matched coarse | Matched refined | Reduction |
|---|---:|---:|---:|
| Normal particle/FE weak-load gap (Pa m) | 1.325164e8 | 3.300844e7 | 4.015x |
| Tangential weak-load gap (Pa m) | 4.686165e6 | 9.653259e5 | 4.854x |
| Normal gap / particle load | 0.5514% | 0.1378% | — |
| Q1 sigma_n peak-to-peak, 37--43 km (MPa) | 0.434959 | 0.132882 | 3.273x |
| Q1 delta p peak-to-peak (MPa) | 0.297138 | 0.101796 | — |
| Q1 -delta tau:N peak-to-peak (MPa) | 0.137821 | 0.062933 | — |

Normal Q4/Q6 changes are only 562.6 / 20.8 Pa m; tangential changes are
850.5 / 113.0 Pa m. These are not rigorous error bounds but are much smaller
than the measured gaps. Unrefined control/bottom gaps remain approximately
unchanged. The 25-km tangential gap remains below quadrature uncertainty.

### Persistent contact; unresolved raw stress

| Quantity | Matched coarse | Matched refined |
|---|---:|---:|
| First-contact reaction, t=2165471357.7276783 s (kPa) | 61.862 | 312.463 |
| Final reaction (kPa) | 1020.480 | 957.235 |
| Final 39.8-km V (m/s) | 2.965288e-10 | 2.849348e-10 |
| Final 39.9-km accumulated slip (m) | 0.377755 | 0.342264 |
| Raw junction sigma_n minimum (MPa) | -0.940933 | -5.401867 |
| Location of minimum (m down-dip) | 39961.330 | 39983.562 |
| Particle-evaluation tensile weight (m²) | 1059.644 | 6622.701 |
| Frozen FE-evaluation tensile weight (m²) | 0 | 5827.979 |

Both have V=Vmin at 39.9 km. Frozen FE residual changes cannot reverse contact,
including at first contact. The final **-6.20%** spatial reaction change hides
the much stronger onset sensitivity. Raw minima are on segment 755 with
mixed free/prescribed support; all-sample tensile weights prevent mistaking
a tiny selected QP weight for a zero-measure phenomenon. The bottom is not
refined and remains tensile near -18.783 MPa.

The old coarse final reaction was only **127.699 kPa**, versus **1020.480 kPa**
on the new common schedule despite identical coarse initialization. The old
39.8-km rate was 4.492974e-10 versus 2.965288e-10 m/s. This time-grid effect
must not be mislabeled spatial error. Projected junction sigma_n is less
time-sensitive: old/new coarse RMS change is 2.480 kPa, versus 34.735 kPa
for the matched spatial pair.

## 5. Fixed-mesh timestep subdivision

### Identical branch state; controlled clock change

The refined accepted step-11 state at **1762776093.4253557 s** is copied from
`restart/03`. ASPECT checkpoints after advancing the clock to the pending
step, so the disposable archive changes **only pending time and dt** to
the first half-step. Inverse replacement reproduces every uncompressed byte;
all mesh/history files, old_dt, step number and plugin state remain identical.
Source hashes are unchanged. This is a local benchmark operation restricted
to the verified archive layout, not a new production restart API.

| Accepted physical time (s) | Original refined step / dt (s) | Half-step branch step / dt (s) |
|---:|---:|---:|
| 1964123725.5765171 | — | 12 / 201347632.15116143 |
| 2165471357.7276783 | 12 / 402695264.30232239 | 13 / 201347632.15116119 |
| 2198823868.4896455 | — | 14 / 33352510.761967182 |
| 2232176379.2516127 | 13 / 66705021.523934364 | 15 / 33352510.761967182 |

The first half-step is below the original cap; subsequent requested steps
equal production selection with CFL/RSF restrictions retained. There are
399 free / 1 lower-active RSF-region nodes throughout the new accepted states;
deep prescribed Vp stays exact. First accepted contact occurs at the new
midpoint, not an independently resolved continuous-time onset.

### Same committed Theta does not imply the same mechanical trajectory

| 39.9-km quantity | First common endpoint: original / half | Final: original / half |
|---|---:|---:|
| Reaction (MPa) | 0.312463 / 0.716640 | 0.957235 / 0.936078 |
| Theta used by mechanics (s) | 1.708753e8 / 3.722229e8 | 5.735705e8 / 6.069230e8 |
| Committed Theta (s) | 5.735705e8 / 5.735705e8 | 6.402756e8 / 6.402756e8 |
| Committed C (MPa) | 1.371222 / 1.384318 | 1.386585 / 1.414096 |

At Vmin, x=V*dt/Dc is about 5e-10, so exact aging is essentially
Theta_old+dt. Committed Theta and accumulated slip therefore agree at common
times at this node, while intermediate committed Theta enters the next
mechanical solve. The original first interval has dt/Theta_old=2.35666:
small V-based x does not imply a small relative aging increment. This is the
approved split ordering, not an extra history lag or a recurrence of the
cancellation-prone Theta checker.

At the first common endpoint, half-minus-original weak shear, cohesion and
friction changes are **-0.140, -3.164, +407.200 kPa**, accounting for the
**+404.177-kPa** reaction change. At the final time they are **+13.357,
+3.122, -10.922 kPa**, giving **-21.157 kPa**. The nonlinear Q1/domain response
also depends on neighbouring V/Theta; these comparisons do not isolate every
term's change to this node's aging alone.

The final half-step test-weight mean normal stress is **+50.132912 MPa**;
driving/cohesion/friction are **28.339904 / 2.478174 / 26.797808 MPa**.
Frozen FE substitution changes R by +12.876 kPa, **1.376%** of the reaction,
and cannot release contact. No additional FE-history solve was performed.

### Remaining temporal and stress uncertainty

| Final quantity | Original refined | Half steps | Change |
|---|---:|---:|---:|
| 39.8-km V (m/s) | 2.849348e-10 | 2.601753e-10 | -8.690% |
| 39.8-km Theta (s) | 27213612.46 | 29739191.95 | +9.281% |
| 39.8-km slip (m) | 1.17472277 | 1.15074863 | -23.974 mm |
| 39.7-km V (m/s) | 6.543704e-10 | 6.517528e-10 | -0.400% |
| 25-km control V (m/s) | 2.206020e-10 | 2.156919e-10 | -2.226% |

At the first common time the 39.8-km rate differs by -22.359%. Its smaller
final difference is **time evolution, not a refinement contraction factor**.
Two timestep levels do not establish a limit.

Final junction Q1 RMS differences are **1.797 kPa p**, **1.044 kPa -tau:N**,
and **1.071 kPa sigma_n**. The last is much smaller than the 34.735-kPa spatial
change; this subdivision does not explain away the spatial stress uncertainty.

The raw junction minimum changes from **-5.401867 to -5.380791 MPa** at
39983.562 / 39983.552 m, on the same mixed-support segment and parent 250740.
Its delta p is **-51.185246 / -51.303968 MPa** and -delta tau:N is
**-4.216621 / -4.076823 MPa**: the changes largely cancel. Particle tensile
weight is **6622.701 / 6357.792 m²**; FE tensile weight is **5827.979 /
6092.888 m²**. Raw selection and full-weight accumulation remain separate.
The global bottom minimum changes from **-18.783 to -20.485 MPa**; its
separate endpoint sensitivity is unresolved, not repaired by this test.

## 6. Verification, cost and implementation status

| Experiment | Verified outcome | Measured cost |
|---|---|---:|
| Exact export / frozen evaluation | Complete cell coverage; weak residual reproduction; independent Q4/Q6 tests; V finite differences | 2.078 / 26.082 s |
| One FE-history noncommitting solve | 5 Newton updates, 11 fresh-linear returns / 267 Krylov; worst fresh/target 0.85796; complete rollback and unchanged checkpoint | 97.115 s |
| Initial spatial preflight | t=0 mechanics converged; controller guard stopped before acceptance; no trajectory claim | 66.234 s |
| Matched coarse/refined pair | 13 real steps each; 113/115 fresh-linear returns, 2276/2337 Krylov; worst fresh/target 0.98638/0.98529 | 3210.602 s paired elapsed |
| Refined half-step branch | Four accepted steps; 39 fresh-linear returns / 772 Krylov; worst fresh/target 0.95468 | 375.920 s |

Final relative bulk/surface residuals are **1.6730e-13 / 2.1328e-14** for
the FE diagnostic, **3.8161e-14 / 6.6668e-11** for matched coarse,
**4.1403e-14 / 6.8202e-11** for matched refined, and
**1.0150e-13 / 3.0987e-10** for the half-step branch. All genuine positive
solves satisfy their unchanged criteria; exit status alone is not acceptance.

The matched Theta checker errors are at most 3.331e-16. The half-step branch's
independent exact-aging error is at most **2.210e-16**, with slip accumulation
also verified at every new state/node. Its geometry/background are bit-identical
to the branch point, and I_h agrees within 1e-12 relative after cache rebuild.
The production executable hash is unchanged. Original source checkpoints are
preserved throughout.

The paired runtime includes initially overlapping MPI core placement, corrected
live without numerical changes. A snapshot summed to about **11.2 GiB rank
RSS**, not a measured peak and potentially double-counting shared pages.
The half-step run used one explicitly bound four-rank job; peak memory was
not measured. Its logged cumulative restart time is not its 375.920-s cost.
These are not performance comparisons.

Preserved nonpassing preparation evidence includes incomplete diagnostic
exports/local-index mistakes, the stopped timestep/mesh preflight, a grading
guard failure followed by a 13.344-s mesh-only check, and the half-step runner's
1.505-s pre-load failure from an omitted target-mesh filename. The latter was
corrected in a fresh output directory. These are not accepted mechanical
states and do not establish production defects. The conditional FE mechanical
solve occurred once, not in a retry loop.

Implementation accumulated over the four tasks consists of:

- guarded opt-in dual-history/FE diagnostic responses in
  `source/reconstructed_fault/surface_system.cc`; the default method is unchanged;
- benchmark-local exact-state extraction, independent common weak tests,
  tangent checks and repaired complete rollback diagnostics;
- saved-mesh and matched/replay timestep-cap plugins, checkpoint-copy retiming,
  accepted-state CSV selection and reproducible runners/reducers.

Release builds used `-j4`; the focused checks above, Python compilation and
whitespace checks passed. No broad suite or new one-/two-rank invariance
campaign was run. Later runs reuse earlier rollback evidence rather than
claiming to repeat it. The FE alternative is **not** a validated production
replacement. No equations, default transfer, pressure treatment, support,
quadrature, Vmin, active-set rules or solver tolerances were changed.

## 7. Artifact and command index

All artifact paths below are relative to `benchmarks/reconstructed_fault/bp3/`.
Their provenance/execution files record exact inputs, hashes, environment,
commands and status. Runners refuse to overwrite existing output directories.

| Evidence | Artifact directory and principal files |
|---|---|
| Exact state | `history-mechanics-export-synchronized-local4/`: owned-cell Stokes coefficients |
| Frozen history | `history-mechanics-frozen-verified-local4/`: `surface_comparison.csv`, `comparison.json`, `common_history_tests.csv`, `history_tangent_fd.csv` |
| Conditional solve | `history-mechanics-solve-local4/`: `solve_comparison.json`, `re_equilibrated_profiles.csv/png`, dual samples, noncommitting history/surface CSVs, rollback log |
| Stopped preflight | `normal-stress-junction-refined-local4/`: mesh/initialization CSVs, `resolution-analysis/comparison.json`, guard log |
| Matched spatial pair | `junction-matched-qualified-local4/`: `comparison.json`, `final_profiles.csv`, `node_history.csv`, `junction_profiles.png`; each case's raw/projected CSVs and log; shared `clock/` |
| Timestep branch | `junction-contact-half-qualified-local4/`: `checkpoint_retime.json`, `sequence.csv`, `subdivision_comparison.json`, `node_history.csv`, `contact_history.png`, `final_junction_profiles.png`, raw/projected CSVs and log |

Useful completed-run entry points (do not rerun into existing directories):

```sh
python3 benchmarks/reconstructed_fault/bp3/run_history_mechanics.py export --tag synchronized-local4
python3 benchmarks/reconstructed_fault/bp3/run_history_mechanics.py frozen --tag verified-local4 --bulk-dir history-mechanics-export-synchronized-local4
python3 benchmarks/reconstructed_fault/bp3/run_history_mechanics.py solve
python3 benchmarks/reconstructed_fault/bp3/run_junction_refinement.py
python3 benchmarks/reconstructed_fault/bp3/run_matched_resolution.py prepare --tag qualified
python3 benchmarks/reconstructed_fault/bp3/run_matched_resolution.py run --tag qualified
python3 benchmarks/reconstructed_fault/bp3/run_contact_subdivision.py prepare --tag qualified
python3 benchmarks/reconstructed_fault/bp3/run_contact_subdivision.py run --tag qualified
```

The original reports retain full numerical tables, per-run analysis commands,
predeclared criteria and preparation details. This consolidation preserves
their limitations: **persistent tested contact, but no spatial/raw-stress or
temporal trajectory convergence claim, no demonstrated history-transfer repair,
and no authorization here for long continuation.**
