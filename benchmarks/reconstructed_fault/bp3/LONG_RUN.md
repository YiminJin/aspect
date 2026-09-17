# Prepared adaptive modified-BP3 run

The long simulation has **not** been launched. The maintained configuration is
now **300 x 100 km**, with square side lengths at most **12.5 km**. The physical
fault remains centered at (50,50) km and retains its **60-degree** dip; changing
to 30 degrees later also requires new fault/mesh/prestress/completion inputs.
Short four-rank Release tests qualified the new particle layout, native
quadratic output and fixed-mesh restart.
This is a coarse exploratory modified-BP3 model, not official BP3, a first-event
verification, or a recurrence-cycle/convergence result.

`bp3_modified_long_run.prm` is self-contained: no parameter includes or abandoned
replay/refinement selectors. The launcher also emits a fully resolved standalone
PRM. The leaf-tree filename is a normal `Mesh refinement/BP3 saved mesh/Target
cells file` parameter, not an environment-only input. Old 200-km evidence is
retained separately. See the [cleanup/300-km report](../../../doc/reconstructed_fault/bp3/stage_K5_long_run_cleanup.md).
The latest [multiple-event/output qualification](../../../doc/reconstructed_fault/bp3/stage_K5_multiple_event_output.md)
records the cleaned fault names, complete slip table, environment script and
bitwise fresh/restart comparisons.

## Launch

From the repository root, prepare a distinct **multiple-event** run:

```sh
python3 benchmarks/reconstructed_fault/bp3/run_long.py \
  --output benchmarks/reconstructed_fault/bp3/modified-long-multiple-events \
  --purpose recurrence --end-years 1500 --wall-hours 23
```

Preparation runs no ASPECT process. It writes `launch.sh`, resolved overrides,
input/binary/plugin hashes and a source diff. Execute the printed `sh .../launch.sh`
command yourself, with console redirection to a log. Alternatively add
`--execute` to the command above **when creating a new output directory** to
record `fresh.log` and execution timing automatically. Existing directories
are never overwritten.

`--purpose recurrence` (the default) does not stop after the first event; it runs to the
explicit end/wall bound. This selection is not evidence of recurrence
qualification. First-event mode stops only after onset at max(V)>=1e-3 m/s
and five consecutive accepted sub-threshold states. Ending at the physical
time/wall bound without an event is not event success.

For direct server execution, load your compatible Release-build/MPI modules,
then source the maintained environment file (bash or zsh):

```sh
source benchmarks/reconstructed_fault/bp3/environment.sh
mpirun -np 4 --bind-to core --map-by core \
  build-pf-cpdi/aspect-release /absolute/path/to/prepared/fresh.prm > run.log 2>&1
```

The environment script locates this checkout, clears inherited ASPECT probe
flags, selects sparse B/G, pivoted tridiagonal K, GMG plus its hierarchy, and
one thread per MPI rank. `source .../environment.sh amg` selects the reference
AMG backend instead. It does not load machine-specific modules or launch MPI.
`run_long.py` reads the **same** script; there is no second flag list to maintain.
Use the prepared `launch.sh` if you also want its executable/input hash checks;
bare mpirun intentionally does not provide those checks. Four-rank restart is
qualified; choosing another rank count for a fresh server run is not cross-rank
restart qualification. Adjust the executable/library build paths on the server
if they differ; do not use a plugin built against another ASPECT binary.

## Selected multi-event resolution and termination

Retain the tested 300 x 100-km, 60-degree configuration: near-fault square
side 97.65625 m, fault spacing approximately 99.9603 m, ell=400 m, graded
far field capped at 12.5 km. No special 40-km refinement or newly prescribed
deep segment is introduced. The scalar phase remains Q1 and velocity Q2.

Erickson et al., BSSA 113 (2023), pp. 505, 509, 511–512 and 517,
[local paper](../../../doc/reconstructed_fault/bp3/bssa-2022066.1.pdf), recommend
25-m BP3 spacing (approximately 16 process-zone and 100 nucleation-length
intervals), while Table 4 reports 100-m sbplib and 60-degree FDCycle cases
and larger spacings for some other cases. At our fault spacing, these nominal
400-m/2.5-km scales have approximately 4/25 intervals. Our ~100-m choice is a
tested multi-event **research** resolution, not the paper's fine reference.
The paper also shows improvements on 200-to-100-m refinement; it does not
establish convergence for our Q1/finite-width model or our finite boundaries.
Keeping the tested inputs is preferable to silently introducing an unqualified
25-m mesh, new completion tables and new projections before the server run.

Run to **1500 physical years**, the paper's interval, with `Stop after first
event = false`, ordinary adaptive timesteps and graceful accepted-state wall
stops/checkpoints. Repeated wall-limited jobs resume toward that physical end.
Do not stop at a preset event count: the 60-degree official case has several
characteristic event types, and a first-event result does not establish recurrence.
The `first_event.csv` observer still describes only the first event; identify
subsequent events from the every-step `accepted_steps.csv`, not that first-event
summary. Zero events at 1500 years is a scientific outcome, not an event success.
The fully frictional deep fault, mature law, ell and finite boundaries remain
explicit **modified-BP3** differences. No claim of official-cycle agreement is made.

Use `--velocity-preconditioner amg` to select the retained reference; default
GMG automatically enables the required hierarchy. Do **not** change the
ASPECT `Stokes solver type` to substitute an unrelated matrix-free solver.
The launcher retains assembled A, sparse B/G, pivoted tridiagonal surface
inverse, outer FGMRES, pressure convention and fresh residual checks.

The launcher removes inherited `ASPECT_*` probes and sets only the recorded
production selectors, exact-mesh/coordinate guards, GMG hierarchy and
single-thread library settings. It never enables comparison, frozen-solve,
profiling, alternative-state or saved-clock flags in long mode. Its generated
script also clears inherited probes and checks input/executable hashes.
The MPI and deal.II runtime libraries must be available as for the tested
Release binary. Local qualified paths are `build-pf-cpdi/aspect-release` and
`benchmarks/reconstructed_fault/bp3/build/libbp3.release.so`. Rebuilding requires
new qualification/hashes, not bypassing the launch check.

## Restart

Use a **new directory**, same binary/plugin/physical inputs and four ranks:

```sh
python3 benchmarks/reconstructed_fault/bp3/run_long.py \
  --resume-from benchmarks/reconstructed_fault/bp3/modified-long-multiple-events \
  --output benchmarks/reconstructed_fault/bp3/modified-long-multiple-events-resumed \
  --purpose recurrence --end-years 1500 --wall-hours 23
```

The default is the last-good checkpoint; `--checkpoint 1`, `2` or `3` selects
a retained slot explicitly. `restart/NN/bp3_accepted_state.txt` gives its actual
last accepted step/time, not an assumed preceding step. The launcher copies
the prior directory, restores the selected metadata prefix, and preserves
the original evidence. Reserve space for that copy. Unindexed payload files
beyond the checkpoint may remain but no restored time index points to them.
Old investigation/research particle-layout archives are **not converted**.
The cumulative-slip CSV is streamed down to the selected checkpoint's step
in the copied branch before appending resumed states. It is not copied into
each checkpoint: the existing checkpointed slip vector remains the authoritative
history. New-format runs must already have this table; old-output checkpoints
are not silently promoted into a complete every-step record.

The graceful wall bound is checked at accepted states, including initialization;
it is not a kill timeout. A long solve/output can overrun it. Leave margin to
the scheduler wall limit. Graceful stops write the final native state and an
ordinary checkpoint. A killed job resumes from its last complete checkpoint.

## Output and field meanings

**Reconstructed-fault visualization is synchronized with mesh and particle
visualization, not written at every timestep.** All three native writers use
the same accepted-state `heavy_pending` decision. The zero native time intervals
in the PRM let this shared BP3 schedule control output; they do not bypass it.
The every-step `cumulative_slip.csv` is intentionally independent of this schedule.

Open `solution.pvd`, `particles.pvd` and `reconstructed_faults.pvd` together in
ParaView: their physical timestamps agree. Filename suffixes need not agree:
fault VTUs use the actual timestep number, whereas bulk/particle files use
output counters. In the qualified fresh and restarted two-step tests, all three
wrote at t=0 and t=5332320.54099 s and skipped step 1; the second fault file is
`reconstructed_faults-00002.vtu`, alongside `solution-00001.pvtu` and
`particles-00001.pvtu`. This numbering difference does not indicate extra output.

- `accepted_steps.csv`: every accepted state, dt, extrema/counts, nonlinear and
  Krylov iterations, minimum alpha, normalized residual, dimensional surface
  RMS, independent Theta error, committed stress scale and max(dt V)/Dc.
- `stations.csv`: twelve official locations, every accepted step; accepted V,
  Theta, recorded cumulative slip, current work-projected total tractions.
- `cumulative_slip.csv`: **every vertex at every accepted timestep**, including
  zero initial slip. Columns: `step,time_s,fault,node,s_m,xd_m,slip_m`.
  `s_m` is cumulative arclength from vertex 0 in stored fault order (bottom to
  top here); `xd_m` is official down-dip distance from the surface (top to
  bottom). Use `xd_m/1000` versus `slip_m` for Fig.-8-style contours. These are
  accumulated accepted slip, not quadrature over sparsely sampled rates.
  The file is independent of all VTU/profile throttling. Figure 8 samples at
  one year aseismically and one second coseismically; select recorded states
  or explicitly label interpolation when plotting such times. The writer
  does not change timesteps to hit plotting times.
- `first_event.csv`: compact onset/peak/location/down-crossing/completion state.
- `profiles.csv`, `profiles/fault_STEP.csv`: stable node/fault IDs, physical
  coordinates/time, **recorded signed slip**, V, Theta and work-Q1 tractions.
- `heavy_outputs.csv`: actual heavy-output step/time and maximum per-node slip
  increment since the previous heavy output.
- `solution.pvd`, `particles.pvd`, `reconstructed_faults.pvd`: synchronized
  native series at the same accepted physical times. Individual file numbering
  differs (the fault writer uses accepted step numbers); use their PVD indices.

Heavy output uses 0.1 m or one year; light profiles use 0.01 m or 0.1 year,
plus onset/end/final milestones. Initial and gracefully final states are
always written. The time triggers can be disabled with zero in the BP3
subsection. The criteria do not force timestep/output interpolation.

Native bulk `tau_xx/tau_yy/tau_xy` are **retained FE history inputs**, not current
constitutive stress. Native particle `maxwell stress_0/1/2` is **newly committed
stress** (supplied zero stress at time zero); its `phase field fault state`
property is the initial-composition seed, not evolving surface Theta. Read
current Theta and slip from the fault writer. `pressure` is physical Delta p;
bulk stresses are perturbations, not the 50-MPa background. Profile `q_weak_Pa`
and `sigma_n_weak_Pa` are accepted **current work-weighted Q1 projections**,
not raw stress samples or simple bulk-column averages. No second Maxwell
update is performed for visualization.

Bulk VTU output uses `Interpolate output = true` and `Write higher order output
= true`: nine-node VTK Lagrange quadrilaterals for the Q2 FE mesh, verified from
the exported connectivity/types. Particle output remains points; the fault
remains its physical Q1 line representation. Use a VTK/ParaView reader that
supports higher-order cells.

Fault display names use underscores: `slip_rate`, `slip_state`,
`cohesive_traction`, `previous_I_h`, `composition_strengthening`,
`background_tractions`, `cumulative_slip`, plus `fault_id`. `previous_I_h`
retains its history meaning (equal to current Ih for this frozen fixture).
The background components remain [shear, normal]. Vertex/cell IDs are omitted
from VTU; stable node IDs remain in CSVs. `BP3 fixed shear correction` and
`mature fault reference geometry` are excluded using the postprocessor's
`Excluded properties` parameter, not removed from checkpointed model state.

Normal logs retain aligned linear fresh-residual/target, nonlinear bulk/fault
residual and line-search acceptance lines. Preparation begin/end messages and
test/action dumps are absent. Detailed nullspace/linear/nonlinear diagnostics
are opt-in (`ASPECT_FAULT_NONLINEAR_DIAGNOSTIC`); sparse matrix sizes are opt-in
performance data. Normal launch clears these flags. All fresh-residual,
compatibility, active-set and history checks still execute.

```sh
python3 benchmarks/reconstructed_fault/bp3/plot_recorded_slip.py RUN_DIRECTORY
python3 benchmarks/reconstructed_fault/bp3/plot_recorded_slip.py RUN_DIRECTORY --deficit
python3 benchmarks/reconstructed_fault/bp3/plot_recorded_slip.py RUN_DIRECTORY --event-time SECONDS
```

An event reference must be recorded exactly unless `--interpolate-reference`
is explicitly supplied; interpolated references are labeled, never described
as solved states. The script never reintegrates sparse V samples.

## Checkpoints, resources and limits

Periodic checkpoints use 1800 **wall seconds**, zero step-count interval,
and three retained slots (hard-coded by this branch). Output and checkpoint
schedules are independent. Heavy background-thread writing is disabled so
reference publication follows successful writer completion. Full particle,
raw-QP and DoF CSVs are off by default; enable `Audit full state every step`
only for a short investigation.

Observed early quadratic heavy output is 8.4–16.4 MB total (four ranks); light profiles
are 0.17–0.214 MB; a checkpoint is about 105 MB. A useful initial storage model
is 0.017*N_heavy + 0.000214*N_light + 0.32 GB for three checkpoints, plus logs,
station/summary metadata and restart-branch copies. A quiet 1500-year run at
the time-trigger cadence is roughly 29 GB before events and copies; this is
not a bound on a seismic trajectory. Allow at least 50–100 GB and monitor it.
Early compressed file sizes need not predict seismic-state compression.
Also budget roughly 0.1 MB per accepted step for the uncompressed 1156-vertex
cumulative-slip CSV (about 10 GB per 100,000 steps; actual decimal lengths vary).
It is streamed on restart rather than multiplied across checkpoint slots.

Four ranks with one thread each are qualified. Observed peak child/rank RSS
is 1.28–1.51 GiB, not aggregate job memory; 16 GiB job memory provides practical
headroom (32 GiB if available). No server scaling or cross-rank restart is
qualified. About 147 s covered initialization plus two adaptive steps on the 300-km mesh;
nonlinear work varied from 1 to 13 accepted Newton updates, so do not
extrapolate that runtime to a full earthquake cycle.

See the [detailed report](../../../doc/reconstructed_fault/bp3/stage_K5_long_run_preparation.md)
for numerical comparisons, exact commands and the known pre-existing generic
expected-output mismatch. The historical initial AMG/GMG raw-stress difference
remains a separate qualification; this task did not change solvers or tolerances.
