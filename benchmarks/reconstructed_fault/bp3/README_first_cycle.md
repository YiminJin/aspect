# Coarse BP3 first-event server fixture — prepared, held for review

No server job has been launched. The corrected three-step smoke and filesystem
restart test pass, but the approved boundary-integral correction changes the
old trajectory beyond numerical equivalence. `first_cycle_readiness.json`
therefore has `server_ready=false`, and the launcher refuses to run. Review
`doc/reconstructed_fault/bp3/stage_K5_first_cycle_preparation.md` before changing
that readiness decision. Do not undo the boundary correction to recover old
numbers or loosen a solver/physics tolerance.

## Configuration

`bp3_first_cycle_coarse.prm` inherits the physical settings used by the latest
verified coarse stress-perturbation trajectory through `bp3_smoke.prm`, with
its accepted Vmin=1e-20 override. It does not include `bp3_pilot.prm` or Airy
inputs. Box 100x100 km; dip 60 degrees; ell=400 m; crossed spacing 97.65625 m;
fault spacing 100 m (final shortened element unchanged); frozen phase/fault;
true normal traction 50 MPa + Delta p - Delta tau:N. Lateral rigid velocities,
zero top/bottom perturbation traction, no pressure normalization, official
friction/state, horizontal extensions, deep Vp, initialization interval 4e6 s,
timestep controller and all solver tolerances are unchanged.

Use the **corrected remote points** I_h backend. The actual BP3 cell/remote
comparison still exceeds 2e-10; the faster backend is not qualified here.
The exact completed-value cache remains enabled by normal production behavior.

Required performance selections:

```sh
export ASPECT_FAULT_SURFACE_SOLVER=tridiagonal
export ASPECT_FAULT_EXPLICIT_B=1
```

Leave `ASPECT_FAULT_EXPLICIT_G`, `ASPECT_FAULT_INTERFACE_MODES`,
`ASPECT_FAULT_COMPARE_COUPLING`, `ASPECT_FAULT_COMPARE_SURFACE_INVERSE`,
`ASPECT_FAULT_VERIFY_INTERFACE`, all `ASPECT_IH_*` debug/comparison flags,
`ASPECT_FAULT_PERFORMANCE`, `ASPECT_FAULT_LINEAR_PERFORMANCE`,
`ASPECT_FAULT_NONLINEAR_DIAGNOSTIC`, `ASPECT_FAULT_NORMAL_STRESS_DIAGNOSTIC`,
and other audit/profiling flags **unset**, not set to `0`. Many are presence-based.
The launcher clears inherited `ASPECT_*` flags and installs the two above plus
`ASPECT_SOURCE_DIR=BP3_SOURCE_DIR` for ASPECT's existing source/library-path override.
No few-mode preconditioner, reference callbacks, sparse G, or GMG is selected.

## Build and eventual manual submission

Build ASPECT Release with Voro/CPDI and the current plugin on Stampede3 using
one consistent MPI/deal.II toolchain. Do not copy the workstation binaries.
Keep the BP3 plugin in the source-relative `bp3/build` path used by the input:

```sh
cmake --build BUILD_DIRECTORY --target aspect.exe.release -j4
cmake -S benchmarks/reconstructed_fault/bp3 -B benchmarks/reconstructed_fault/bp3/build -DAspect_DIR=ABSOLUTE_BUILD_DIRECTORY
cmake --build benchmarks/reconstructed_fault/bp3/build --target bp3.release -j4
```

After review only, set `BP3_SOURCE_DIR`, `BP3_ASPECT_BINARY`, and a dedicated
`BP3_OUTPUT_DIR` under `$SCRATCH`. Submit `stampede3_first_cycle.sh` manually
with `sbatch -A YOUR_ALLOCATION` from scratch. The script is not submitted by
this task. It uses one SKX node, one MPI rank and a 12-hour wall-time chunk;
this deliberately matches the qualified trajectory/restart rank count rather
than claiming unmeasured MPI speedup. Runtime through an event is unknown.
Restart manually from the same output directory after a wall-time interruption;
there is no automatic retry/resubmission. Changing rank count is not covered by
this full BP3 restart test (small one-/two-rank coupling tests did pass).

The queue/launcher convention follows the [official Stampede3 guide](https://docs.tacc.utexas.edu/hpc/stampede3/):
use `ibrun` for this MPI executable and load its matching environment before
submission. Supply the allocation at submission; the script does not invent
an account or load potentially incompatible module versions.

Measured workstation smoke: 530.74 s, 4.30 GiB peak RSS. Restoring at step 1
and completing steps 2/3: 391.26 s, 4.28 GiB. These are not Stampede3 runtime
predictions. Each complete coarse checkpoint is approximately 98 MiB; three
rolling slots plus five milestone archives need about 0.8 GiB, excluding
VTU/profiles/logs. Reserve substantially more scratch for event outputs.
No first-event wall time or output-volume bound is claimed from three steps.

## Checkpoint and history meanings

Every accepted step checkpoints the mesh, FE solution histories, particles,
manager physical history and the benchmark slip/event/output state. Matrices,
factors, lookup caches and transient current I_h rebuild. The background data
are manager properties; the plugin reattaches their selector on resume and
does not recalibrate them. Older fresh-start-only BP3 checkpoints fail clearly.

Timestep zero retains supplied Theta0, H0 and zero particle stress change; the
4e6 s numerical Maxwell interval is not physical elapsed time or accumulated
slip. Subsequent slip uses dt*accepted V once per step. Station Theta and
particle stresses are accepted current histories; traction diagnostics are
from the accepted mechanical solve, which used preceding committed Theta and
particle stress. They are not recomputed from already updated histories.
Bulk `tau_xx/tau_yy/tau_xy` arrays are **published FE old-history perturbations**,
not total BP3 prestress or freshly committed particle stress. Bulk pressure is
Delta p. `component_3` is temperature and `component_9` is phase in this fixture.

ASPECT checkpoints store an accepted solution ready for the next timestep;
the serialized BP3 last-step/event data identify its accepted physical time.
`event_states/{before_onset,onset,peak,down_crossing,termination}` archive
complete checkpoint contents, not just plots. Peak is the highest sampled
accepted max(V); down-crossing is the start of the final uninterrupted
sub-threshold streak. Copy an archived checkpoint into the normal ASPECT
restart layout with its last-good marker when restoring a milestone.

## Output and termination

- `accepted_steps.csv`: append-only accepted step/time/dt, physical max V and
  its down-dip location, actual constitutive normal-stress extrema, free-rate
  extrema, free/lower-active counts, Newton updates, total Krylov iterations
  including residual replacements, and minimum accepted alpha. Newton updates
  plus one gives the number of base nonlinear evaluations in this solver.
- `stations.csv`: twelve official stations every accepted step; V, current
  Theta, accumulated slip, total shear and normal traction. Traction is the
  consistent Q1 representation of actual surface weak moments, not a bulk-column
  average. Both pressure-normal and shear backgrounds remain distinct in profiles.
- `fault_STEP.csv`, `history_STEP.csv`: periodic profiles/history checks, normally
  every 0.1 yr, every step at max V >=1e-5 m/s, and event milestones.
- `bulk_STEP.pvtu` and rank `.vtu` pieces: normally every year, every accepted
  coseismic state (max V >=1e-3), and milestones. No output schedule changes dt.
- `first_event.csv`: current event-state summary, including onset, peak/time/
  location, down-crossing, termination and consecutive-below count.
- `restart/`, `event_states/`: full rolling and milestone states.
- `aspect-JOBID.log`, Slurm stdout/stderr: solver/fresh checks and restart markers.

The observer begins interseismic, enters at max V >=1e-3, and terminates after
five consecutive accepted max V <1e-3 states. A value >=1e-3 resets the streak.
Expected messages are `BP3 FIRST EVENT ONSET`, `BP3 FIRST EVENT COMPLETE`, or
at the independent 1500-Julian-year safety end, `BP3 SAFETY END: no first seismic
event; event criterion NOT satisfied.` An event that started but did not finish
has a distinct safety-end message. Exit zero alone is not first-event success.

After an abrupt interruption, append-only files can retain records beyond the
restored checkpoint. Preserve those audit records, but follow the restart
branch in the per-job log and accepted-step indices; do not count duplicate
replayed rows as additional physical timesteps or five-state event evidence.
The serialized observer, not CSV row counting, controls termination.

## Completed bounded tests

`first_cycle_verified.prm` ran initialization and three real steps with full
audit exports. `first_cycle_resume.prm` continued a byte-identical copy of its
step-1 checkpoint; `first_cycle_checkpoint.prm` also describes that stopping
point but was not separately rerun. See `first_cycle_restart_comparison.json`
and `first_cycle_baseline_comparison.json`. The former passes; the latter
explicitly does not pass numerical old-trajectory equivalence.

`test_first_event.cc` covers onset, equality, re-entry, five-state completion
and peak retention without a seismic simulation. Milestone archival code is
implemented but has not been exercised by a real event, which is outside this
preparation task. The short test's audit-only every-step full exports are
disabled in `bp3_first_cycle_coarse.prm`.
