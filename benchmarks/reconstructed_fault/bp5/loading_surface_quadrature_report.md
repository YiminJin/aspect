# BP5: resolving the along-fault normalization RHS

The corrected initialization converged with true normal-stress feedback. In
27–29.8 km, native shear chord RMS decreased **97.16%** and velocity chord RMS
decreased **96.37%**. This identifies under-resolved tangential sampling of the
normalization projection as the dominant source of those shear/velocity teeth.
It does not remove the much smaller pressure/deviatoric-normal variations.
Equations, mesh, physical Q1 phase, width, support, friction, normal integration
tolerances and nonlinear tolerances were unchanged. No real timestep was run.

## Change and invariants

`Material model / Phase field fault / I h surface quadrature subdivisions`
selects equal panels within each existing fault element. Each panel uses the
existing three-point Gauss rule (`QIterated`). Default 1 retains the previous
input behavior. The successful case explicitly selects **8**: 24 profiles per
element, panel length approximately 12.5 m, versus 24.4140625 m bulk cells.

We still solve the same consistent projection

\[
 M_{ij}=\int N_iN_j\,ds,\qquad
 b_i=\sum_q w_qN_i(s_q)I(s_q),\qquad MI_h=b.
\]

Only the RHS quadrature is improved; composite Gauss remains exact for the Q1
mass matrix. No nodal smoothing, lumping, constant denominator, bulk quadrature
change or analytical replacement of the FE phase was introduced.

Profile IDs remain deterministic and MPI-distributed, ordered by fault,
element, panel and Gauss point. Surface mixture sampling and normal integration
use the new origins. The same outside-box virtual Q1 continuation was
reintegrated at those origins; completion validates both count and coordinates.
All 27,720 IDs occur exactly once across the four rank exports. Completion's
8/16-point integration check remains below the existing `1e-6` absolute guard.

The rule is fixed per run. Parameter parsing invalidates completed values;
cell traversal caches compare origins/normals, and remote lookup caches compare
requested coordinates. A completion file remains immutable during a run.
The successful initialization made two preparation calls but only one profile
integration/final projection: the second call used the completed-value cache.
No restart-cache behavior was redesigned or newly qualified here.

All consumers retain the shared
`PhaseFieldFault::evaluate_reconstructed_fault_localization()` path. Its
single Q1 `current_normalization_integrals` field supplies bulk and surface
responses, the frozen localization in B/G/K, source completion and stress
evaluation. The initialized previous-Ih snapshot matches it to `1e-13`
relative. Exported mechanical QP Ih equals that field's Q1 interpolation to
`1e-13`; `chi * Ih` agrees with the original run to `1e-13`. This checks that
the phase/degradation numerator was not changed along with the denominator.

## Independent quadrature convergence

`reintegrate_initial_profile.py` uses saved physical FE nodal values, not newly
sampled analytical phase values. Its reference integrates over actual cell and
fault-basis polygon partitions; order 10/16 checks are at roundoff. Independent
normal columns are split at bulk cell boundaries. The table is the maximum
relative error of the normalized RHS rows in 27–29.8 km, not a claim that the
Q1 projection exactly reproduces every normal column.

| Panels per element | Maximum relative RHS error |
|---:|---:|
| 1 (original rule) | 1.08710e-4 |
| 2 | 9.14920e-5 |
| 4 | 3.12252e-6 |
| 8 (selected) | 3.57549e-7 |
| 16 | 1.73570e-8 |

The actual production `M*Ih` error decreases from **1.09358e-4 to 9.01313e-7**.
The difference between production and the offline eight-panel rule is
6.69543e-7, retaining the previously observed normal-integration discrepancy
separately rather than claiming tangential quadrature is the entire error.
Neither the configured `1e-10` normal quadrature nor tail tolerance was changed.
No normal-integration diagnosis or correction was added to this task.

Normalized RHS peak-to-peak variation:

- Original production: 195.456 ppm.
- Corrected production: 5.41279 ppm.
- Independent resolved integral: 5.13345 ppm.

## One accepted mechanical initialization

Reference: `weakening30-dc010-ell100/loading-startup/startup`, step 0.
New run: `weakening30-dc010-ell100/surface-quadrature/panels8-initial`.

Mesh exports, fault coordinates, raw QP identities/coordinates/phase/JxW,
projected initial state and source-admission flags agree exactly. Physical
initialization is unchanged: the same projected-mixture state and native weak
prestress procedure are rerun with corrected work weights. The resulting
background coefficient changes are explicitly measured, not suppressed:
maximum 0.09082 Pa globally, 0.05264 Pa in the primary window. There is no
new prestress correction. State is retained, accumulated slip remains zero,
and committed initial Maxwell history remains zero.

Oscillation metric: departure from the neighboring-node chord, with the same
baseline native row-mass weights in both RMS measurements. Native shear means
the weak perturbation load divided by its native mass, **not** a plotted
consistent-Q1 coefficient. The secondary 30.2–32.8 km window is inside the
current transition. Raw curves are not smoothed.

| Window / metric | Original | Corrected |
|---|---:|---:|
| 27–29.8 km shear chord RMS [Pa] | 9.053775 | 0.256723 |
| 27–29.8 km V chord RMS [m/s] | 1.311954e-13 | 4.763015e-15 |
| 30.2–32.8 km shear chord RMS [Pa] | 12.62447 | 0.433092 |
| 30.2–32.8 km V chord RMS [m/s] | 4.193849e-14 | 1.575862e-15 |
| Primary pressure chord RMS [Pa] | 0.203718 | 0.200422 |
| Primary -tau:N chord RMS [Pa] | 0.096313 | 0.096749 |
| Primary p-tau:N chord RMS [Pa] | 0.128302 | 0.124937 |

Primary native row-mass chord RMS falls from 0.0128400 to 0.00199220 m;
the `integral(chi^2 N)` chord RMS falls from 1.77824e-4 to 3.78178e-5;
the second/first-moment ratio chord RMS falls from 9.02923e-7 to 2.44683e-7 /m.
Thus changes in the source moments accompany the mechanical improvement.

Broad response is retained. Over 1–110 km, weighted mean perturbation shear is
-39.45630 versus -39.44951 Pa; mean V is 9.999280635e-10 versus
9.999280274e-10 m/s. In the primary window, mean shear is -38.31902 versus
-38.20224 Pa, while mean mechanical normal perturbation is 3.5783105 versus
3.5783482 Pa. These comparisons use perturbation scales, not the 50 MPa
background as an error denominator.

Final relative bulk/surface residuals: **4.061279e-10 / 1.202142e-9**;
surface RMS **0.00228588 Pa**. One accepted Newton update, alpha 1,
41 total Krylov iterations (reference 42), **1156 free / 0 lower-active** nodes.
Fresh linear residuals were 0.1265332 versus target 0.2787666 and
8.571341e-10 versus target 1.278405e-9. Independent reintegration of the incoming
state friction load agrees within 9.07e-8 Pa; recovered incoming Maxwell history
is below 6.46e-11 Pa. No inference of convergence from exit status alone.

## Cost and the unsuccessful preparation

The first attempted 16-panel preparation did not reach mechanics: rank 2 was
killed by signal 9 at 216.47 s, during severe memory pressure (observed host RAM
near 29 GiB plus over 5 GiB swap). Its logs, parameters and execution record are
preserved under `surface-quadrature/initial`. It is **not a passing case**.
Eight panels were then selected using the already completed independent
convergence calculation, without weakening normal or mechanical tolerances.
This was the only completed mechanical solve in this task.

The eight-panel case took **197.12 s external wall time**, 192 s in ASPECT,
four ranks. Peak child RSS was **5,870,684 KiB (5.60 GiB)**; this is a maximum
child measure, not aggregate RSS. Host memory was observed around 23 GiB during
the run. The remote lookup cache retained 222,457,241 point records in aggregate
over 428 batches; this explains the substantial memory/cold-cost increase.

| Scope | New time |
|---|---:|
| First Ih preparation, before final completion/projection | 92.19 s |
| FE sampling/lookup | 81.16 s |
| Adaptive/request work | 3.10 s |
| Material/guards | 5.94 s |
| Adaptive MPI | 0.265 s |
| Final consistent projection | 0.00689 s |
| Two Ih calls including cached preparation | 94.0 s |
| Condensed solves | 38.7 s |

For context, the existing three-step baseline's entire Ih scope was 21.5 s
(five calls, one cold integration); the recent initialization-only
prescribed-normal control used 16.1 s for two calls and 117.24 s total. These
are reused timings, not a newly repeated controlled performance benchmark.
The cold cost is roughly 4.4–5.8 times larger. The final projection itself is
negligible; cached subsequent values still avoid reintegration. No new cache
optimization or backend switch was mixed into this experiment.

## Tests, commands and artifacts

Builds used `-j4`, target `aspect` in `build-pf-cpdi`,
`bp5_steady_initialization` in `bp5/build`, and `fault_mechanical_modes` in
`performance/build-gmg`. An initial test compilation needed a missing local
`dealii` namespace qualification; the corrected build passed.

```sh
python3 benchmarks/reconstructed_fault/bp5/reintegrate_initial_profile.py \
  --surface-subdivisions 2 4 8 16 \
  --output benchmarks/reconstructed_fault/bp5/weakening30-dc010-ell100/surface-quadrature-offline
build-pf-cpdi/aspect-release --test '[phase_field_fault_ih_accuracy]'
mpirun -np 2 build-pf-cpdi/aspect-release --test '[phase_field_fault_ih_accuracy]'
build-pf-cpdi/aspect-release --test '[phase_field_fault_cohesive],Stage-I*'
python3 benchmarks/reconstructed_fault/bp5/run_surface_quadrature.py prepare
python3 benchmarks/reconstructed_fault/bp5/run_surface_quadrature.py run
python3 benchmarks/reconstructed_fault/bp5/analyze_surface_quadrature.py
```

Normalization: **59,134 assertions / 4 cases** on one rank; on two ranks,
**59,134 and 2,340 assertions / 4 cases** respectively. Includes the new
composite-Q1 mass/RHS test, analytic boundary reproducer and independent
distributed FE integral tests. Cohesive/Stage-I: **20,746 assertions / 19 cases**.
Analysis checks passed; `git diff --check` passed for the touched core/spec files.
No full integration suite, new coupled finite-difference campaign, restart,
evolving-phase or real-step run was performed.

Results: `surface-quadrature/comparison/{summary.json,projection.csv,baseline.csv,corrected.csv}`;
plots `primary.png`, `transition.png`, `whole_fault.png` with identical axes.
`surface-quadrature-offline/` holds all four independent rule levels and input
hashes. `panels8-initial/launch.json` records executable/plugin/input/source
hashes and the complete parameter delta. `source-tested/` and `provenance/`
retain source snapshots, pre-change binaries, build/test logs and the core diff.
The initial sandbox MPI socket failure is separately retained; it did not enter
ASPECT. No old results were overwritten.

Source changes in this task: `phase_field_fault.h/.cc` (parameter and profile
rule/count), `unit_tests/phase_field_fault_ih.cc`, both authoritative specs;
offline script extension, new run/analysis scripts and this report. Existing
unrelated working-tree changes were retained. No commit or server-package
regeneration was requested.

**Decision:** retain the corrected quadrature capability and the eight-panel
qualified initialization. The evidence answers the shear/velocity-teeth
question without a new bulk refinement or a physics change. Residual normal
stress discretization, the sub-ppm production/offline integral discrepancy,
and remote-backend cold memory cost remain explicitly separate limitations.
