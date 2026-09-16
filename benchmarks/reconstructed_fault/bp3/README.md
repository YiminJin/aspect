# Reconstructed-fault BP3 workspace

This directory contains the maintained plugin and small benchmark drivers.
Completed experimental payloads were reorganized on 2026-09-15; see
[CLEANUP.md](CLEANUP.md) for the archive, checks and recovery instructions.
No physical or numerical settings were changed by that cleanup.

## Current starting point

The latest bounded coupled-state timestep check is `coupled-substeps-50-local4/`.
From the common accepted step-9 state, two half-steps and four quarter-steps
reach 29.24190894 yr and commit candidate Theta exactly once per accepted step.
The notch deepens; successive differences contract by factors 0.655 (deficit),
0.766 (neighbor contrast) and 0.865 (slip-gradient increment). Temporal changes
remain material, so this is not a convergence claim or production-integrator
change. See the [substep report](../../../doc/reconstructed_fault/bp3/stage_K5_coupled_substeps_report.md)
for all six accepted states, publication checks, the preserved audit-order
failure, and comparison CSV/JSON. No further run is authorized by this record.

The latest **noncommitting** comparison is `within-step-50-local4/`. A and B
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
