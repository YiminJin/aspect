# Direct server runs: BP5 normal-friction filter

For the single common half-timestep retry after a safety-clock mismatch, use
[`README-half-retry.md`](README-half-retry.md) and the three `*-half.prm` files.
They preserve the original outputs and require no Python preparation.

These are complete PRMs: no `include production_input.prm`, Python launcher,
or Python-generated timestep schedule is required. Run all commands from one
job directory containing these files. Use a new job directory; never overwrite
the original checkpoint or a previous experiment.

| Input | Friction normal input | Output | Limit |
| --- | --- | --- | --- |
| `R.prm` | Original pointwise, unfiltered | `R/output/` | 10 accepted steps |
| `F1.prm` | Helmholtz, 100 m | `F1/output/` | Same 10 actual intervals |
| `F2.prm` | Helmholtz, 200 m | `F2/output/` | Same 10 actual intervals |

Lengths are experimental, not qualified BP5 parameters. Keep the same MPI rank
count and executable/plugins for all branches. Ordinary advection and history
updates remain enabled. No production tolerance changes are made.

## Build and inputs

Use the checkpoint-compatible compiler, deal.II, Boost and ASPECT environment.
The local environment could not deserialize this server checkpoint. Do not
modify archives or reinitialize history to bypass that incompatibility.

From the updated ASPECT source root, with your own build-directory path:

```sh
cmake --build /path/to/aspect-build --target aspect -j4
cmake -S benchmarks/reconstructed_fault/bp5 -B /path/to/bp5-build -DAspect_DIR=/path/to/aspect-build
cmake --build /path/to/bp5-build --target bp5_steady_initialization bp5_normal_stress_diagnostic -j4
```

Copy both rebuilt release libraries into this job directory:

```sh
cp /path/to/bp5-build/libbp5_steady_initialization.release.so .
cp /path/to/bp5-build/libbp5_normal_stress_diagnostic.release.so .
```

Place the ORIGINAL matching `fault.txt`, `target_cells.txt` and `completion.txt`
in `fixture/`. In particular, use the eight-panel completion data that belongs
to the checkpoint, not newly regenerated data. These external inputs and all
checkpoint contents must remain identical across branches.

Prepare three independent copies of the complete accepted-step-5612 checkpoint
(`t=5310111071.5634108` seconds). Replace the path below with the original
`restart/01` directory. The copy must include mesh/particle files and benchmark
metadata, not just `resume.z`.

```sh
BP5_CHECKPOINT=/absolute/original/output/restart/01
cat "$BP5_CHECKPOINT/bp3_accepted_state.txt"
for branch in R F1 F2; do
  mkdir -p "$branch/output/restart"
  cp -a "$BP5_CHECKPOINT" "$branch/output/restart/01"
  cp checkpoint_selector.txt "$branch/output/restart/last_good_checkpoint.txt"
  cp -a "$BP5_CHECKPOINT/bp3_output_metadata/." "$branch/output/"
done
```

Run this setup only once in an empty job directory. Preserve source/input
hashes, module list and binary identity with the results, for example:

```sh
sha256sum *.prm *.so fixture/* R/output/restart/01/resume.z > inputs.sha256
sha256sum /absolute/aspect-build/aspect-release > executable.sha256
```

## Run directly

In bash, source the supplied environment in the batch job, not only on the
login node. Use a two-hour scheduler wall limit per branch (the diagnostic also
stops at accepted states after 7200 seconds). For a hard local limit:

```sh
source environment.sh
timeout --kill-after=15s 7200s mpirun -np 4 /absolute/aspect-build/aspect-release R.prm > R.log 2>&1
```

Use the actual server MPI allocation instead of `4`; substitute `ibrun` for
`mpirun -np 4` if required by the cluster. A timeout/failure is not a successful
comparison. Do not retry automatically or continue a failed branch.

After R completes successfully, inspect the offline normal-input comparison
before launching filtered branches. R exports `normal_initial_operator.csv`
and `initial_normal_*` for this purpose. The existing `../analyze.py` can be
used for analysis only; it does not launch or prepare any simulation. From its
location: `python3 analyze.py /absolute/job-directory`.

Confirm R has accepted steps 5613 through 5622 with passing fresh-linear and
nonlinear checks. It now writes **`R/output/normal_actual_intervals.txt`
automatically**, using the actual constitutive `dt`, not timestamp subtraction.
F1/F2 read that file directly, require exactly ten positive intervals, and stop
if another safety controller shortens the matched clock. No Python step is
needed to generate it. Do not run F1/F2 if R failed, even if a clock file exists.

Once the offline assessment supports the selected lengths, run independently:

```sh
timeout --kill-after=15s 7200s mpirun -np 4 /absolute/aspect-build/aspect-release F1.prm > F1.log 2>&1
timeout --kill-after=15s 7200s mpirun -np 4 /absolute/aspect-build/aspect-release F2.prm > F2.log 2>&1
```

Do not put these in an unattended chain that continues after a failure.
If 200 m clearly distorts broad features, stop before F2 and review the permitted
50-m offline alternative; do not run a length scan. The completed outputs retain
the existing analysis layout (`R/output`, `F1/output`, `F2/output`).

## Files changed in core ASPECT for the filter

- `include/aspect/material_model/phase_field_fault.h`
- `source/material_model/phase_field_fault.cc`
- `include/aspect/reconstructed_fault/surface_system.h`
- `source/reconstructed_fault/surface_system.cc`
- `source/reconstructed_fault/normal_filter_internal.h` (new)

Benchmark build inputs: `bp5/normal_stress_diagnostic.cc`,
`bp5/normal_filter_clock.cc` (new), `bp5/normal_stress_clock.h`,
`bp5/startup_time_step.cc`, `bp5/CMakeLists.txt` and `bp3/bp3.cc`.
The latter existing inputs are still needed to build; they were not all changed
by the filter task. Rebuild both plugins after updating core headers.

This list describes the filter increment, assuming the preceding normal-stress
diagnostic/restart-clock baseline is already present on the server. Use the same
complete source checkout for core and plugins; do not mix header versions.

No new ASPECT core changes were needed for this direct-launch packaging. The
only additional code change is benchmark output of the accepted interval file.
Actual BP5 restart comparisons remain to be run on the compatible server.

Packaging verification (2026-09-24): the updated diagnostic plugin built with
`-j4`; all three exact server PRMs passed `aspect-release --validate` with the
rebuilt plugins; `bash -n environment.sh` passed. Parameter validation does not
deserialize the checkpoint or qualify the ten-step trajectories. The automatic
clock export is compiled but has not yet been exercised by an actual BP5 restart.

### Accepted-work observer fix

If F1 stops with `Accepted work observer does not reproduce frozen mechanical
traction`, update `benchmarks/reconstructed_fault/bp3/work_replay.h` and rebuild
`bp5_steady_initialization`. The original observer compared its independently
recomputed raw mechanical normal load with `weak.normal_traction`, which is the
filtered friction input in F1/F2. It now compares against
`weak.raw_normal_traction` when present, retaining the old raw-mode path and the
unchanged `1e-5 Pa` assertion. No core ASPECT, equation, or tolerance change is
needed for this correction.

The small production-coupling regression passed on one/two ranks (6/5 seconds).
It verifies unchanged raw normal/shear loads and `normal_traction = M z` to the
existing `1e-5 Pa` threshold. Its Helmholtz raw-versus-filtered row difference is
281.713 Pa: comparing those different quantities would legitimately fail the
old assertion. This is not a rerun of the server BP5 trajectory.

Preserve the failed F1 output and restart the corrected F1 in a clean directory
from the original step-5612 checkpoint, not from the failed process's partially
published output. The completed R data/clock remain valid: this patch changes
only the observer check and has no effect on R's mechanics or timestep sequence.
