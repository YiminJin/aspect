Current P2 numerical task completed: [native solver accuracy, onset and saddle escape](../../../doc/reconstructed_fault/propagation/P2_native_solver_report.md).
Fresh58-step economical onset reaches .53125s in85min9s/1684.9MiB, with core
phase .60457720, zero healing, all seven unaligned field targets, complete-prefix
energy accounting and exact complete-step failure/retry. Default-false numerical
options retain governing energy, actual Hessian and final KKT. The resolved
saddle replay is noncommitting, not fresh continuation. One full154-step native
run is prepared at27–35h estimated and is not launched; full native/P3 remains
unqualified. [Field charts](native_endpoint_evidence/onset-economical-resumed-comparison.png) and [separate energy charts](native_solver_evidence/onset-economical-resumed-energy.png).

---

# Optional native endpoint P2 verification

The matching512-cell rich Q2 FE prototype at1/1024 and1/2048 completes .625s
and passes all seven unaligned temporal targets. The native default-off endpoint
algorithm passes fixed-state unloaded/growing/asymmetric operators, full2D phase
gradient and normalization adjoint. Fresh native flank growth passes matching
field/KKT/history checks through .52832s. A one-hour cost stop interrupts the
uncommitted core-growth step; no full native/P3 qualification is claimed.
The coarse prefix passes field targets but has a separately retained contact-path
energy error. Keep the option experimental and default off.
[Settings, algorithm, source purposes, commands and measured evidence](../../../doc/reconstructed_fault/propagation/P2_native_endpoint_report.md).
[Prototype field/energy charts](native_endpoint_evidence/temporal-comparison.png).
[Fresh accepted native charts](native_endpoint_evidence/onset-obstacle-search-comparison.png).

New harness: native_endpoint.py, assess_native_endpoint.py,
prepare_native_endpoint.py, native_endpoint_verification.cc,
run_native_endpoint.py, monitor_native_endpoint.py, check_native_endpoint.py,
assess_native_endpoint_run.py; existing CMake registers the focused observer.
check_native_endpoint_search.py and check_native_endpoint_path_quadrature.py
check search prediction and frozen-path time quadrature without changing physics.
Use OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 and MPLCONFIGDIR=/tmp/p2-native-mpl
for Python comparisons. Runs refuse existing outputs. Evidence and executable/
input/seed hashes: native_endpoint_evidence/. No new VTUs in this task.

---

# P2 matched initialization and ASPECT visualization

The latest bounded study uses one continuum seed/H definition on511/512/1024
normal meshes, exact bulk mechanics,dt=1/1024 and one positive diagnostic branch.
All controls reach .625s. Practical full-interval spatial targets pass for the
retained512 FE prototype; centroid/positional selection remains unqualified.
[Scientific report and commands](../../../doc/reconstructed_fault/propagation/P2_matched_initialization_report.md).

Open [accepted.pvd](matched_initialization_evidence/aspect-visualization/accepted.pvd)
in ParaView, click Apply, select `phase` or `localization` for coloring, and
use the time controls. Five actual ASPECT/deal.II postprocessed VTUs at
0/.53125/.5625/.59375/.625s total0.78MB (each<160kB). The fields come from accepted
shared512 exact-bulk prototype snapshots on the native16×512 mesh. Rendering is
output-only (`no Advection, no Stokes`), with no physical history import or native
coupled-solver qualification. Surface q/slip are extended uniformly for display.

New scripts: matched_initialization.py, assess_matched_initialization.py,
check_matched_modes.py, render_matched_snapshot.py, check_matched_visualization.py.
The snapshot_visualization subdirectory contains the registered ASPECT test
postprocessor and separate CMake project. Its build is
build-propagation-integration/snapshot-visualization. Re-rendering requires a
fresh output destination; successful VTUs are protected from overwriting.
Evidence/provenance: matched_initialization_evidence/assessment.json,
mode-comparison.json, reference.json, snapshot-native.json and centroid-trajectories.csv.
No new PNG figures, production edits, material sweep or P3.

---

# P2 translation-mode and reflected branch diagnostics (2026-10-10)

All 42 original admissible mode starts find stable reflected candidates;14
selected continuations reach .625s. The mode is nearly a band translation,
with continuum neutrality conditional on the previous-phase obstacle and
boundaries. Mesh placement/refinement change centroid motion and release time;
exact-bulk control also has the instability. Separate work errors decrease,
but physical asymmetry/native/P3 remain unqualified. Old saddles are preserved.
[Report, settings, source inventory and 512-cell-only figures](../../../doc/reconstructed_fault/propagation/P2_asymmetric_branch_report.md).
Evidence: asymmetric_branch_evidence/. Scripts: asymmetric_branch.py,
assess_asymmetric_branch.py, probe_asymmetric_release.py. No production change.
Read-only check: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/propagation/assess_asymmetric_branch.py --check`.

---

# P2 coupled continuation stability stop (2026-10-10)

The same three partitions were continued toward .625s. All encounter an
admissible antisymmetric saddle at .59765625s; failed trials are retained and
uncommitted. Common accepted fields through .59375s and separate residual
works converge, but the full interval remains unqualified. Production unchanged.
[Report, settings, physical/energy errors and mode](../../../doc/reconstructed_fault/propagation/P2_coupled_branch_extension_report.md).
Evidence: coupled_branch_extension_evidence/; original fresh prefixes preserved.
Read-only check: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/propagation/extend_coupled_branch.py check`.
Next question is stable asymmetric continuation/discrete pinning and justified
branch selection; no native integration, tuning or P3.

---

# P2 consistent-endpoint short branch (2026-10-10)

Three fresh unloaded coupled Q2/256 prototype paths complete0–.5625s
at dt=1/256,1/512,1/1024. Core phase grows smoothly .600057→.654367; reduced
admissible curvature is positive at tested states and smooth residual work
decreases ~4 per halving. The old replay was initially phase-inconsistent.
No production/model/parameter change, native adoption or P3 qualification.
[Report, settings, equations, source changes and commands](../../../doc/reconstructed_fault/propagation/P2_coupled_branch_report.md).
Primary evidence: coupled_branch_fresh_evidence/; earlier checkpoint controls
retained in coupled_branch_evidence/. Last pre-damage checkpoint fails phase
KKT, so primary runs use the requested unloaded fallback. Read-only check:
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/propagation/check_coupled_branch.py`.
Previous accepted and failed evidence below remains historical.

---

# Completed P2 evolving-profile consistency diagnosis (2026-10-10)

**Native gate remains blocked; no timestep or new native run is selected.** Two
permitted1/2048,1/4096 endpoint refinements preserve mesh, initialization, material,
loading, bounds and tolerances. Fixed-history phase/mechanical checks pass, but
completed-state phase KKT and profile redistribution do not show ordinary temporal
consistency. This is coupling lag, not failure of the incoming phase solver.
A common C2 prescribed-profile control gives first-order force/work reduction.
Boundary reactions identify external work; lifting extensions change work away
from equilibrium. Both original work measures and all failed evidence remain.

A derived consistent-endpoint saved-entry prototype removes KKT lag at the original
target but yields a .0529 phase increment and +159.949J/m incoming profile cost.
It is not energetically admissible under the current instantaneous fixed-history
phase→Maxwell path. No physical history is published by this replay. A mechanical
redistribution/finite-transition energy-path decision is required before native
adoption; no compensating loss, kinetics, constitutive change or P3 is introduced.

[Report, settings, scaling, external-work audit, candidate equations and evidence](../../../doc/reconstructed_fault/propagation/P2_profile_temporal_report.md).
Baseline de7beaa65846c7372711f7922abf1c5d9a745ee6/fault-propagation;
production/core/plugin unchanged. New files are benchmark diagnosis/replay scripts
and documents, with commands/source inventory in the report.

| Outcome | Measurement | Gate |
|---|---:|---|
| Solved versus endpoint phase KKT | ≤5.22e-12 versus5.79e-3 | Incoming solve passes; coupling remains unresolved |
| Last halving phase / traction | .00564886 /3.40121Pa | Original accuracy targets fail |
| Consistent replay KKT / profile cost | 8.91e-14 /+159.949J/m | Numerical pass; energy path fails |

Read-only command: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/propagation/check_profile_temporal.py`.
check_profile_temporal.py verifies unchanged production/old trajectories, numerical
checks and the failed native gate. Only diagnostic/prototype/documentation additions.
The next selected scope should derive the work-consistent redistribution/transition
path, rather than extending the native family or changing physical parameters.

---

# Completed bounded P2 timestep assessment (2026-10-09)

**No timestep is qualified; the native selection gate is not met.** The matching
16×256 Q2/Q1, six-point endpoint prototype completes40/80/160/320/640 steps to
.625s, with identical initialized phase/particle H and zero accepted FE/Z history.
The continuous .256 ramp is applied as interval means; displacement is .05m in
every run. Last1/512→1/1024 full phase/traction differences .00241105/2.82566Pa
exceed .001/1Pa targets. Slip/common-rate differences .005799%/.09062%; the finest
reference remains unresolved, and its self-comparison is not a pass. No new native
run is launched; the previous34.4min baseline is retained without conversion.

Largest phase increments nearly halve (.0170487→.00109694). D_relax decreases
.179911→.0229697J/m, with unresolved limit. Boundary-lifting absolute work error
falls .204356→.0232568J/m; bulk-rate absolute error fails to decrease during damage
and is .111051J/m at1/1024. Frozen controls and an exact general-beta identity
isolate the intermediate profile-induced force/work difference (identity error
≤1.79e-12J/m per step). No history, operator, bound or tolerance is changed.
Selected proper-incoming derivative/curvature/quadrature and replay checks pass;
these numerical checks do not qualify temporal/energetic accuracy. Spatial choice
remains16×256; production sources, binary, specifications and old failed artifacts
are preserved. Only benchmark diagnostics and documentation change.

[Decision report, settings, plots, costs and commands](../../../doc/reconstructed_fault/propagation/P2_time_resolution_report.md).
Evidence: benchmarks/reconstructed_fault/propagation/time_resolution_evidence/;
check_time_resolution.py verifies numerical/preservation checks and records the
failed accuracy gate. Further temporal refinement, evolving-profile interpretation
or a scheme revision requires a separately selected scope; no RSF/P3 is authorized.

---

# P2 Q2 normal-resolution decision

[Settings, convergence, selected native cost and source changes](../../../doc/reconstructed_fault/propagation/P2_Q2_normal_resolution_report.md).
normal_resolution_evidence/ retains the added128/256 cheap paths, fixed-profile
controls, independent stress/storage histories, plots, hashes and two initial-only
attempts. Working mesh16×256 Q2/Q1 meets the practical stress/energy and existing
phase/q/slip targets; six coupled bulk points and all physical settings remain
fixed. The native path completes40 steps atdt=1/64,T=.625; cost34.4min/1.95GiB.
Use run_normal_resolution.py only for fresh outputs (it refuses overwrite);
normal_resolution.py, assess_normal_resolution.py and check_normal_resolution.py
assess and verify retained evidence. Incoming Q3/three-point artifacts remain.
Time accuracy, phase-release/mechanical work closure, full stability and P3
remain open. Earlier checkpoints below are retained as historical evidence.

---

# P2 coupled quadrature accuracy and measured cost

[Report and recommended restricted configuration](../../../doc/reconstructed_fault/propagation/P2_bulk_quadrature_report.md).
quadrature_evidence/ contains fresh matched native controls, rule/cheap mesh
studies, profiles, resource records and task-only source/specification diffs.
Six normal points are selected with24/48-point independent checks. The same
factory serves every coupled bulk term; normalization/CPDI are unchanged.
run_quadrature.py refuses existing outputs; assess_quadrature.py and
check_quadrature.py read retained evidence. The phase/q/slip recommendation
is16x32 Q2; stress/energy/time/full stability/P3 remain unqualified. All old
three-point results and failed trajectories are preserved below.

---

# P2 plane-strain verification and spatial/stability checkpoint

[Readable report and settings](../../../doc/reconstructed_fault/propagation/P2_plane_strain_report.md).
Task-only inputs, source/specification diffs, profiles, plots and frozen integration
are in plane_strain_evidence/. Run check_plane_strain_algebra.py,
assess_plane_strain.py, then check_plane_strain.py. Native/core build identities
and unchanged historical results are pinned. The FE-only3D correction passes
operator/Maxwell/shear/default checks; Q3 enrichment exposes normal-quadrature
aliasing, not spatial convergence. No P3 or generic-projector/helper change.

---

# P2 native localization assessment (2026-10-09)

[Current model, numerical diagnosis, transverse profiles and transfer inventory](../../../doc/reconstructed_fault/propagation/P2_native_localization_report.md).
The original native failure and its qualification status are preserved below.
The bounded follow-up finishes affine/compatible-side and height2 controls at
unchanged physical parameters, bounds, consistent normalization and convergence
targets. It identifies a separate negative-curvature issue at the original affine
entry and substantial coarse native velocity-space error. No P3 qualification.

`localization_1d.py` supplies matching Q2 and rich direct one-dimensional Maxwell
endpoint references with Q1 phase/CPDI fracture measure. The analytic stationary
control and separate bulk/cohesive/fracture driving profiles use the current
total-energy formulation. `diagnose_phase_stall.py` audits saved native tangent/
trial data; `assess_localization.py` compares accepted native fields and direct
normal integrals. `check_localization_reference.py` and
`check_localization_continuum.py` verify derivatives, stationary Maxwell/work
identities and frozen quadrature. New evidence is in `localization_evidence/`.

```sh
python3 benchmarks/reconstructed_fault/propagation/check_localization.py
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/localization-mpl python3 benchmarks/reconstructed_fault/propagation/check_localization_reference.py
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/localization-mpl python3 benchmarks/reconstructed_fault/propagation/check_localization_continuum.py
```

All final trajectories use the compiler/executable hashes in `native-build.json`.
Inputs `affine-final-executable-native.prm`, `compatible-operator-check-native.prm`
and `wide-operator-check-native.prm` select distinct output directories; do not
overwrite evidence when repeating. The checker validates histories, preserved
incoming snapshots, final source/build identity and mature/default regressions.
Old manifests pin historical source/executables and are intentionally not repinned
to new code. Restrictions remain fresh serial fixed geometry/mesh, homogeneous
coefficients and no transport/birth. Velocity/slip rates are interval constants
selected by endpoint balance. G=1e6 Pa remains a dimensional synthetic value.

---

# Restricted native P2 FE strain (2026-10-08)

[Model, equations, verification and source inventory](../../../doc/reconstructed_fault/propagation/P2_native_FE_strain_report.md).
`native_fe.prm` selects the fresh serial accepted FE-strain model explicitly.
`native_fe_strain.cc` observes energy/phase/stress/slip and probes a nonuniform
Q2 field without publishing test history. `assess_native_fe.py` checks saved
recurrences, derivative/KKT targets and direct normalization; `check_native_fe.py`
checks provenance, restrictions and preserved evidence. Passing these checks
qualifies the stated representation, not the incomplete coupled trajectory.

The final loaded path accepts 19 steps to .296875 s and fails at .3125 above its
unchanged phase KKT target. No P3. Plots/evidence are in native_fe_evidence; all
prior failed paths are retained. Do not overwrite those outputs when repeating.
The fresh same-seed endpoint prototype is a one-mode comparator with different
side-boundary restrictions, not a full-Stokes reference.

---

# Current evolving endpoint comparison

[Report, scheme decision and native-entry equations](../../../doc/reconstructed_fault/propagation/P2_total_energy_endpoint_comparison.md).
Three missing endpoint trajectories and one focused finer reference reuse the
unchanged total-energy prototype and corrected intersection quadrature; matching
mean runs are preserved. Endpoint is selected, with decreasing independent phase
release and mechanical work discrepancies. Native entry stops at the accepted
bulk-state ownership decision explained in the report. Exact FE internal strain
is a viable uniform-case proposal; the rank obstruction does not rule it out.
Native implementation is not complete. No production/P3 claim.

```sh
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/propagation/total_energy_time_comparison.py --label aligned_evolving_endpoint256 --steps 256 --scheme endpoint --evolve --aligned-bulk-quadrature
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/total-energy-matplotlib python3 benchmarks/reconstructed_fault/propagation/assess_total_energy_endpoint.py
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/total-energy-matplotlib python3 benchmarks/reconstructed_fault/propagation/check_total_energy_endpoint.py
```

Runner labels are immutable. The comparison covers 64/128/256 steps and the
focused endpoint512 reference; reports state interval-rate versus endpoint
approximation meanings and retain absolute energy discrepancies.
[Assessment](total_energy_endpoint_evidence/assessment.json),
[manifest](total_energy_endpoint_evidence/manifest.json).
The checker independently validates seven saved paths, actual slip deposits,
common physical inputs and preservation of 378 incoming artifacts. It does not
treat solver passes as accuracy qualification. This remains widening of one
prescribed band in one free bulk mode; full Stokes/localization is necessary.

# Preserved loaded total-energy prototype comparison

[Report/settings/implementation decision](../../../doc/reconstructed_fault/propagation/P2_total_energy_time_comparison.md).
One unloaded seed/common ramp compares frozen endpoint/mean balance, then phase
feedback with a free bulk mode and the surface system. Intersection quadrature
repairs a diagnosed surface sampling artifact. Refinement decreases release and
field/force errors; onset is still underresolved. Mean mechanics is the total-
energy candidate; no production adoption or P3 qualification.

```sh
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/propagation/total_energy_time_comparison.py --label aligned_evolving_mean256 --steps 256 --evolve --aligned-bulk-quadrature
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/total-energy-matplotlib python3 benchmarks/reconstructed_fault/propagation/assess_total_energy_time.py
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/total-energy-matplotlib python3 benchmarks/reconstructed_fault/propagation/check_total_energy_time.py
```

Runners refuse existing labels. [Assessment](total_energy_time_evidence/assessment.json),
[sampling diagnosis](total_energy_time_evidence/sampling_diagnosis.json),
[manifest](total_energy_time_evidence/manifest.json). The final gate independently
rebuilds saved fields/forces/KKT/ledgers and verifies 240 incoming artifact copies.
Historical manifests pin earlier code/docs and are not rewritten; their physical
results and original controls remain hash-preserved/reproduced by this new gate.

# Propagation P1/P2 diagnostics

Current result: [isolated total-energy candidate](../../../doc/reconstructed_fault/propagation/P2_total_energy_verification.md)
and [algorithm](../../../doc/reconstructed_fault/propagation/P2_total_energy_algorithm.md).
Exact discrete phase derivatives, irreversibility and general Maxwell work pass
in the named controls. Mean mechanical balance is selected in this prototype;
endpoint controls retain work defects. Finite phase-release slack is explicit,
not a hidden viscous/H contribution. Production source/model remains unchanged;
no coupled native qualification or P3.

```sh
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/propagation/total_energy_step.py --label coupled_blocks
python3 benchmarks/reconstructed_fault/propagation/check_total_energy_step.py
```

The runner refuses label reuse. [Evidence](total_energy_evidence/coupled_blocks/summary.json).

Historical result: [P2 review checkpoint](../../../doc/reconstructed_fault/propagation/P2_implementation_checkpoint.md)
and [formulation comparison](../../../doc/reconstructed_fault/propagation/P2_formulation_comparison.md).
Frozen consistent normalization passes the .5% source-error target at .00625 m
on six accepted fields (worst .0250722%); .0125 fails. Coupled resolution and all
model options stay unchanged. Direct irreversibility is verified; profile energy
and stress representation still limit qualification. No new phase law, coupled
refinement, material tuning or P3. Stop for review.

```sh
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/propagation/review_normalization.py
python3 benchmarks/reconstructed_fault/propagation/check_review_checkpoint.py
```

The runner preserves existing output; the checker reads retained
[records](review_checkpoint_evidence/normalization.json) and
[manifest](review_checkpoint_evidence/manifest.json).

Historical result: [P2 accuracy and energy assessment](../../../doc/reconstructed_fault/propagation/P2_accuracy_energy_report.md).
Direct irreversibility remains successfully verified for the fixed-mesh fixture.
Frozen .025 m consistent normalization misses the provisional .5% target; native
bulk stress paths have no additional transfer drift, while FE representation and
evolving-profile energetics remain separate limitations. The derived bulk phase
variation is presented for review without changing the implementation. No finer
coupled runs or P3 after this failed accuracy/energy gate.
[Commands, settings and evidence](../../../doc/reconstructed_fault/propagation/P2_accuracy_energy_report.md#decision-changes-and-reproduction).


Historical result: [P2 direct irreversibility](../../../doc/reconstructed_fault/propagation/P2_irreversibility_report.md).
The physical admissible set now uses the accepted phase as lower bound; initial
relaxation remains box-only. Three fresh a=.256 paths all complete to t=1 with
zero healing and improving temporal agreement. Source normalization error and
profile work remain unresolved; material search and P3 stay paused.
[Run/replay commands and evidence](IRREVERSIBILITY.md). Earlier results below
remain historical, including their failed qualifications.



Historical result: [P2 coupling repair](../../../doc/reconstructed_fault/propagation/P2_coupling_report.md).
Surface-owned history and conjugate mechanical work pass fixed-phase checks.
The reference material stays unchanged; one a=0.256 ramp at three dt exposes
resolved tail healing. Previous high-load, preloaded and material-study evidence
remains historical and failed. No material optimization or P3 is resumed.

Historical result: [recoverable-energy P2 revision](../../../doc/reconstructed_fault/propagation/P2_recoverable_report.md).
The unloaded fixture and local state driver pass, but three coupled runs remain
unqualified because of traction-transfer and work-exchange discrepancies.

Current follow-up: [history semantics and timestep diagnosis](../../../doc/reconstructed_fault/propagation/P2_history_report.md).
Material tuning is paused. Native tests confirm the incremental-candidate/max-H
problem; no replacement law or stress-scaling qualification has been selected.

Latest selected follow-up: [1D reference and test-only box constraints](BOUNDS.md).
The restricted diagnostic passes; production remains unchanged and unqualified.

## Earlier bounded initialization and tail controls

Completed 2026-10-06. All four cases converge and evolve under prescribed H;
**all fail physical qualification**. No production algorithm changed and P2
was not started. See the [diagnosis report](../../../doc/reconstructed_fault/propagation/P1_tail_report.md)
and [numerical summary](evidence/summary.json). The existing native test plugin
is reused directly; no second phase-field implementation is introduced.

The original finite seed is compared with the same seed in a padded box and a
through-going straight band, all at h=1/32. The remaining straight-band tail
undershoot triggered one h=1/64 repeat. Each case performs initialization and
five scratch solves, without mechanics or transport. The activation mask and
prescribed monotone H are diagnostic controls. They do not qualify the
production mechanical H update.

From the repository root, the commands used were:

```sh
cmake -S benchmarks/reconstructed_fault/propagation -B build-propagation-p1-tail \
  -DAspect_DIR=/home/ein/repository/aspect-pf-rsf-refactor/build-refactor-post-r6-gcc12-unity \
  -DCMAKE_CXX_COMPILER=/opt/gcc/12.4.0/bin/g++ -DCMAKE_BUILD_TYPE=Release
cmake --build build-propagation-p1-tail -j2
python3 benchmarks/reconstructed_fault/propagation/run_case.py finite \
  --executable /home/ein/repository/aspect-pf-rsf-refactor/benchmarks/reconstructed_fault/refactoring_r7b/build/aspect-r7b-qualified \
  --build build-propagation-p1-tail \
  --evidence benchmarks/reconstructed_fault/propagation/evidence
```

The same run command selected `padded`, `straight`, then `straight_fine`.
The runner refuses to overwrite evidence. A deliberate reproduction needs a
fresh task-owned build/output and evidence path. ASPECT exits 1 at its physical
admissibility assertion; the independent numerical checker and runner exit 2.
This is a scientific failure, not an expected-failure qualification pass.
The original finite run's wrapper initially misread a wrapped assertion
message; its simulation and numerical check completed. The wrapper was fixed
without rerunning that simulation.

Read-only checks of the saved evidence:

```sh
python3 benchmarks/reconstructed_fault/propagation/check_evidence.py
MPLCONFIGDIR=/tmp/propagation-matplotlib python3 benchmarks/reconstructed_fault/propagation/plot_tail.py
```

`check_evidence.py` exits 0 only for internally consistent **failed** results
and an exact CSV replay of the earlier P1 refined experiment. Its
`physical_qualification` remains false. Per-case `.checks.json` files retain
the independent physical gates; expected-output matching cannot override them.
The [plot](evidence/initialization.svg) shows native initialization before the
tighter first hold. It is a diagnostic visualization, not physical validation.

Budget: at most four serial cases, 120 s / 2 GiB RSS / 10 MiB output each.
Actual runs took 2.8–6.9 s and 208–258 MiB RSS. No core rebuild, cluster run,
restart, physical rollback, KKT constraints, H-law change or propagation.
Exact commands, environment, binary identity, logs and compact CSV cuts are
in `evidence/`; [manifest](evidence/manifest.json) records source/input hashes.
