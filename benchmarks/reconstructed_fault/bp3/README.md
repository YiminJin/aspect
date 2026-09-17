# Reconstructed-fault BP3 workspace

This directory contains the maintained plugin and small benchmark drivers.
Completed experimental payloads were reorganized on 2026-09-15; see
[CLEANUP.md](CLEANUP.md) for the archive, checks and recovery instructions.
No physical or numerical settings were changed by that cleanup.

## Prepared long-run configuration

Use the separately named **`bp3_modified_long_run.prm`** and
[`run_long.py`](run_long.py) for the newly prepared adaptive exploratory run.
See [LONG_RUN.md](LONG_RUN.md) for exact commands, field definitions,
checkpoint/output schedules and storage estimates. Preparation is the
default; execution must be explicitly requested. No long run has been launched.
The next prepared server mode is **multiple events through 1500 years**,
not first-event termination. Source `environment.sh` for direct `mpirun`
(GMG default; optional `amg`). `cumulative_slip.csv` records every accepted
vertex state with local arclength and official down-dip coordinates, independently
of graphical output. See the resolution rationale and output aliases in
[LONG_RUN.md](LONG_RUN.md).
The maintained long-run model is now **300 x 100 km**, with maximum square
side 12.5 km, unchanged 97.65625-m fault-region spacing and the current
60-degree fault. The PRM is self-contained and native bulk output is quadratic.
It omits the special 40-km mesh/fault refinement and uses selected spatial
refresh of strengthening. The short four-rank restart and unchanged-mesh
cleanup comparisons passed bitwise. Old particle-layout
checkpoints are not converted. The seven-step research configuration below
is preserved as a separate reference, not silently made into a long run.

## Preserved seven-step research reference

**`bp3_modified_wide.prm`** (200 x 100 km) remains the seven-step modified-BP3
reference, not the current long-run input. Its launcher requires the separate
`bp3_research` library; see [reference build instructions](reference_200km/README.md).
`run_research.py` defaults to `--configuration wide`
and a new `wide-research-local4/` output directory. The verified 100-km
reference remains `--configuration frictional`, using
`bp3_modified_fully_frictional.prm`. Both retain the same
research configuration: continuous Q1 V/Theta/slip, mature C=0, the qualified
bulk-QP work measure, paired boundary corrections and unchanged plate loading.
`bp3_constrained_reference.prm` retains the prescribed-deep reference choice.
Both use explicit `Postprocess/BP3/Fault loading configuration`; neither relies
on a trace or alternative-state environment selector.

The bounded regression launcher is `python3 run_research.py` from this
directory (or its repository-relative path). It uses four ranks, the exact
saved seven-step clock, a 2400-s cap, no retries and a new output directory
`wide-research-local4/`. It refuses to overwrite evidence. Set
`--configuration constrained` only for an explicitly requested reference run;
this cleanup does not rerun that historical case. Check equivalence using
`check_research_cleanup.py`, followed by `analyze_fully_frictional.py --run
fully-frictional-cleanup-local4` for the physical/history checks.

The post-cleanup replay **passed** in 704.962 s. Compared V/Theta/slip,
current stress/strain and published particle stresses match the preserved run
exactly at all eight accepted states; 71 fresh-linear checks and 1360 Krylov
iterations are unchanged. All 1236 nodes remain free. See
`fully-frictional-cleanup-local4/cleanup_equivalence.json` for the detailed check.

The mesh, fault, fixed background, paired completion and replay clock now have
hash-preserving maintained copies in `fixtures/modified_bp3/`. Research
execution no longer depends on the original investigation directories.
See [restart and bounded continuation](fixtures/modified_bp3/continuation.md)
for the separately selected execution modes; no long-cycle run is implied.
The original version-4 restart qualification stopped at the cold-Ih bitwise invariant,
before the first resumed linear solve. The 0–4 prefix is exactly equivalent;
that original restart was **not qualified**, and its old continuation launcher
remains prepare-only. The new long-run mode above has its own verified
version-5 restart and restricted frozen-Ih restoration. See the historical
[restart report](../../../doc/reconstructed_fault/bp3/stage_K5_research_restart_report.md).
See the
[cleanup report](../../../doc/reconstructed_fault/bp3/stage_K5_cleanup_report.md)
and [retired investigations](investigations/README.md). Old probes require the
archived tree; the maintained plugin rejects their selectors. No global mature
or boundary-extension default was introduced.

### Centered 200-km-wide variant

`bp3_modified_wide.prm` extends the modified fully frictional fixture to
`[-50,150] x [0,100] km`. The box center and fault midpoint remain at
`(50,50) km`; no physical fault/history coordinates are translated. The
42,880 original cells, 1236 fault vertices and ell=400 m are unchanged.
Only 88 graded lateral cells are added. Use `run_research.py --configuration
wide --output <new-directory>` for the same seven-step clock, or add
`--prepare-only` to prepare without execution. The original configuration
remains available unchanged. See [wide fixture details](fixtures/modified_bp3_wide/README.md)
for the verified mesh and execution limits. The seven-step wide replay has
now passed in `wide-seven-local4/`, with the same accepted clock as the saved
narrow-box run. Over 0–40 km at 4.190129 yr, maximum differences are
0.007763 Vp and 0.681 mm slip; weak shear/normal evolution differs by
6.99%/30.78% RMS relative to the narrow evolving signals, not the background.
See the [width comparison report](../../../doc/reconstructed_fault/bp3/stage_K5_wide_box_comparison.md)
and `wide-seven-local4/width-comparison/comparison.png`. This is not a
domain-convergence claim.

### Research velocity-GMG default

The research launcher defaults to GMG for the tested fixed-mesh,
incompressible Q2 configuration. Select
`run_research.py --velocity-preconditioner amg --output <new-directory>`
to retain the AMG reference path. Explicit `--velocity-preconditioner gmg`
is also supported. GMG and its required mesh-hierarchy setup are enabled
automatically and recorded in launch provenance; no manual flags are needed.
It changes the velocity-block preconditioning cycle, not the assembled fine
operator, sparse B/G, surface inverse, pressure-block inverse, outer FGMRES,
fresh residual checks or physical model. The runner enables the required
multigrid mesh hierarchy. It does not load the frozen comparison plugin or
enable reference-action callbacks in an ordinary replay.

The bounded frozen-system test and its failure-preserving evidence are under
`../performance/gmg/`; see the
[GMG report](../../../doc/reconstructed_fault/bp3/stage_K5_gmg_prototype.md).
Restart qualification is separate from GMG; promoting this option did not
resolve the historical cold-Ih limitation. The later long-run correction and
new-layout qualification are documented above.

The four-rank frozen comparison passed; the seven-step GMG replay completed
in 615.9 s versus the saved AMG run's 721.1 s. All real-step field comparisons
pass the existing 1e-8 allowance. Initial raw stress differs by up to
0.00110 Pa and fails that strict per-field comparison, despite both solves
meeting their unchanged residual criteria. GMG has now been approved as the
research-launcher default with this qualification retained, not relabeled as
strict equivalence. Further setup/memory optimization is deferred. Detailed results and the
explicit failure list are in the report and
`wide-gmg-seven-local4/gmg_equivalence.json`.

### Preserved successful comparison

The authorized **fully frictional** seven-step corrected-work replay is
`fully-frictional-seven-local4/`. It removes all deep prescribed-V rows,
including the bottom, while retaining continuous Q1 V/Theta/slip and the
existing boundary corrections. At 4.190129 yr the 40-km local rate minimum
is absent and its raw normal-stress range falls from 170.784 to 7.475 kPa;
shallow loading/deficit and deep near-plate-rate creep remain. All convergence,
history and newly free bottom derivative checks pass. This is an explicit
modified-BP3 choice, not a new default or a long-term qualification. See the
[fully frictional report](../../../doc/reconstructed_fault/bp3/stage_K5_fully_frictional_report.md).
Its original driver `run_trace_replay.py --fully-frictional` and selector
`ASPECT_BP3_FULLY_FRICTIONAL_REPLAY` are historical: use the preserved source
snapshot to reproduce them. New replays use the explicit configuration above.

### Earlier independent-trace comparisons

The fresh seven-step committing independent-trace comparison is
`trace-replay-seven-local4/`. It retains separate free/deep V, Theta and slip,
with the entire deep segment prescribed. At 4.190129 yr the within-free notch
is 72.32% smaller, but the slip jump has grown to -1.76093 mm and the raw
junction normal-stress range is 9.64% larger (RMS variation 2.29% smaller).
All convergence, history and side-consistent quadrature checks pass. This is
not a qualified repair or a new production default. See the
[committing trace report](../../../doc/reconstructed_fault/bp3/stage_K5_trace_replay_report.md).
The maintained driver and analysis are `run_trace_replay.py` and
`analyze_trace_replay.py`; the bounded selector is `ASPECT_BP3_SPLIT_TRACE_REPLAY`.

The earliest-dip follow-up is `early-free-trace-matched/`. Saved steps identify
mechanics 2 as the first clear growth, with almost no local incoming Theta
peak or old-stress weak-row depression. One independent-trace solve removes
94.88% of newly generated chord defect, while raw normal stress changes little.
The short reconstructed prefix matches the original; the probe converges and
rolls back. A pre-solve clock-fixture rejection is preserved separately in
`early-free-trace/`. See the [early-notch report](../../../doc/reconstructed_fault/bp3/stage_K5_early_notch_report.md).
This identifies a dominant first-dip mechanism, not a committing trace method.

The latest bounded comparison is `free-trace/independent-local4-mpi/`: one
four-rank frozen-history mechanics solve gives the 40-km free side an
independent trace, with the deep side still exactly Vp. The new trace is
0.928661 Vp; the neighboring notch contrast decreases 26.41%, but does not
collapse, and the raw normal-stress range increases 3.35%. The predicted
slip has a junction jump. Derivative/work checks and full rollback pass;
this is not a production correction or an accepted trajectory. See the
[free-trace report](../../../doc/reconstructed_fault/bp3/stage_K5_free_trace_report.md).
The particle-density task was withdrawn and was not run.
The subsequent explicit A/B request reuses those exact solves. Added
`qualify_free_trace.py` checks equal-endpoint source/weak-row recovery and
separates the within-free undershoot from the free/deep jump: the former
decreases 47--49%, but remains, alongside a 7.13% Vp jump. No additional
mechanical solve or history publication was needed.

The follow-up `gradient-kink/` contains **independent manufactured** frozen-state
experiments, not new BP3 timesteps. A Fourier-resolved finite-width reference
separates bulk-resolution errors from the continuous-Q1 response at an
incompatible prescribed/free trace. Bulk refinement suppresses a compatible
kink's oscillations; an independent free-side trace removes the deliberately
manufactured junction notch with second-order convergence. Exact frozen-state
RSF checks reproduce both conclusions. This does not qualify a production
trace discontinuity or prove that all three BP3 features have one cause.
See the [gradient-kink report](../../../doc/reconstructed_fault/bp3/stage_K5_gradient_kink_report.md)
for equations, checks, measured errors and the proposed bounded next test.

The latest **mechanism diagnosis** is `notch-mechanism/`: four frozen-history,
noncommitting experiments plus saved-state force/gradient analysis. The 40-km
boundary impulse gives a negative adjacent-rate response; a 25-km interior
control reproduces the alternating pattern. A bulk-relaxed operator column
isolates a positive nearest-neighbour Maxwell/source coupling after bulk
relaxation. See the [notch-mechanism report](../../../doc/reconstructed_fault/bp3/stage_K5_notch_mechanism_report.md)
for the quantitative result, the distinction between the broad 15-km slowdown
and transient 18-km dip, and the remaining finite-width/space-resolution
uncertainty. No correction, accepted trajectory, or new production default
follows from these disposable constraints. All four solves roll back; no more
run is authorized by this record.

The latest bounded coupled-state timestep check is `coupled-substeps-50-local4/`.
From the common accepted step-9 state, two half-steps and four quarter-steps
reach 29.24190894 yr and commit candidate Theta exactly once per accepted step.
The notch deepens; successive differences contract by factors 0.655 (deficit),
0.766 (neighbor contrast) and 0.865 (slip-gradient increment). Temporal changes
remain material, so this is not a convergence claim or production-integrator
change. See the [substep report](../../../doc/reconstructed_fault/bp3/stage_K5_coupled_substeps_report.md)
for all six accepted states, publication checks, the preserved audit-order
failure, and comparison CSV/JSON. No further run is authorized by this record.

The preceding **noncommitting state** comparison is `within-step-50-local4/`. A and B
start from the identical accepted step-9 checkpoint and solve mechanics 10.
A reproduces the saved lagged-state rates within 8.66e-15 relative. Recomputing
candidate nodal Theta from immutable old Theta in B deepens the adjacent rate
contrast 21.30% and the predicted last-element slip-gradient increment 17.13%.
Both converge and roll back; the default committing split update is unchanged.
See the [within-step report](../../../doc/reconstructed_fault/bp3/stage_K5_within_step_report.md)
for the incident-element balance, nonsymmetric Jacobian verification, artifacts
and diagnostic-only frozen-I_h restoration. This does not qualify general
restart of the fresh-start-only revised-work fixture or a new state integrator.

The latest bounded temporal comparison is `work-replay-halfdt-50-local4/`:
one fresh four-rank run halves every real baseline duration, retains dt0=4e6 s,
and stops after 20 real steps at 29.24190894 yr in 1369.655 s. The accepted-clock
guard prevents a remainder step. All convergence/history checks pass and all
440 RSF nodes remain free. The 39.95-km notch deficit increases 20.3%, the
adjacent rate contrast 38.0%, and the last-element slip gradient 11.9%.
Temporal accuracy is the next question; two levels do not establish convergence.
See the [half-timestep report](../../../doc/reconstructed_fault/bp3/stage_K5_work_halfdt_report.md),
`comparison/` CSVs/plots, and `run_work_replay.py --half-timesteps` /
`analyze_work_halfdt.py`. No further run or model change is implied.

The reference committing comparison is `work-replay-50-local4/`, selected by
`Postprocess/BP3/Committing work-measure replay = true`. One fresh four-rank
run reached 29.24190894 yr in 967.3 s with the first history update checked
before continuation. See the [committing replay report](../../../doc/reconstructed_fault/bp3/stage_K5_work_replay_report.md)
and its `comparison/` CSVs/plots. Use **step 10** for the physical comparison:
step 11 is a separately preserved two-ULP end-time remainder, not an equivalent
final state. New prepared replays now select the tested `BP3 replay complete`
accepted-clock guard; the historical outputs are unchanged. Do not blindly
resume that old last checkpoint. No additional run is implied.

The subsequent [deep-mesh feature diagnosis](../../../doc/reconstructed_fault/bp3/stage_K5_deep_mesh_feature_report.md)
reuses this accepted step 10 and saved uniform sliding. Its one two-step causal
run is `deep-mesh-shift48-uniform-local4/`; offline overlays and full transverse
profiles are in `deep-mesh-feature-analysis/`. The main 44-km stress feature
follows the bulk edge to 48 km; the smaller 42.935-km wing feature stays beside
an unchanged transverse coarse-cell intrusion. No RSF/material correction follows
from this test alone.

The preceding bounded qualification is `work-measure-free-top-local4/`:
`analysis.json`, `run.log`, `provenance.json`, derivative/work checks and the
noncommitting surface/history CSV files are retained together. Its detailed
report is [K5 work measure](../../../doc/reconstructed_fault/bp3/stage_K5_work_measure_report.md).
It qualifies the fresh straight/frozen/mature free-top case. The subsequent
committing replay is separate evidence, not a general BP3 accuracy claim.
No new run is authorized by this directory organization.

| Location | Purpose |
|---|---|
| `bp3.cc`, `bp3_model.h`, diagnostic headers | Current plugin and benchmark-specific checks |
| `CMakeLists.txt`, `build/` | Current build definition and usable local plugin |
| `bp3_smoke.prm`, `bp3_pilot.prm`, `bp3_first_cycle_coarse.prm` | Existing fixtures; read their documented limitations before use |
| `run_*.py`, `analyze_*.py`, `compare_*.py`, tests | Maintained drivers and offline checks; historical inputs may require restoration |
| `first_cycle_coarse/` | Original supplied server output, including coherent `restart/03`; preserved intact |
| `evidence/server/` | Supplied `CMakeCache.txt` and `BP3.e3498041` with module/build context |
| `evidence/completed/` | Compact copies of archived logs, JSON summaries, plots and parameter wrappers, grouped by original case name |
| `stage_K5_*task.md` | User task specifications, retained at their original paths |

## Retained input directories are not complete output datasets

The following directories now hold the required inputs/metadata and completion
guards, rather than all historical VTU, particle, raw-QP and history exports:

- `mature-fault-50-local4/`: fixed captured prestress, timestep clock and fixture.
- `fault-grid-50-local4/`: the saved surface grid and parameter include.
- `junction-matched-qualified-local4/refined/`: target mesh and parameter include.
- `uniform-sliding-50-local4/`, `bottom-completion-50-local4/`,
  `bottom-source-complete-wedge-50-local4/`, `top-source-control-50-local4/`,
  `top-source-paired-50-local4/`: include chain, fixed completion tables,
  clocks, provenance, run logs and accepted-state summaries where available.

These exact paths are intentionally retained: current fixtures and immutable
provenance refer to them. Source files/runners have not been mass-renamed and
no compatibility symlink layer was introduced. Parameter includes, captured
prestress, saved mesh/fault coordinates and completion tables are unchanged.

Do not mistake a partially retained directory for a complete raw dataset.
Before rerunning an old analysis, restore the corresponding archive group.
Historical documents keep their original evidence paths. Compact copies in
`evidence/completed/` are for reading, not replacement launch locations;
their relative includes and recorded hashes retain their original meaning.

For future tasks, use a new clearly named case directory, save command/input
hashes, logs and compact results together, and preserve failed evidence.
Keep fixture dependencies explicit; do not silently turn a past run's output
into a new physical input.
