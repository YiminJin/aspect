# Single common half-timestep retry

Use the complete `R-half.prm`, `F1-half.prm`, and `F2-half.prm` in the SAME job
directory as the original tests. No Python launcher or schedule preparation is
required. All three read the original ten-line `R/output/normal_actual_intervals.txt`
and select `Halve recorded intervals = true` under `Time stepping / BP5 filter clock`.
**Do not manually halve that file or replace it with the retry output.**

| PRM | Friction normal input | New output directory |
| --- | --- | --- |
| `R-half.prm` | Raw | `half/R/output` |
| `F1-half.prm` | Helmholtz, 100 m | `half/F1/output` |
| `F2-half.prm` | Helmholtz, 200 m | `half/F2/output` |

Each starts from the original step-5612 checkpoint and runs ten accepted steps,
5613–5622, covering HALF the original R elapsed interval. This is not twenty
half-steps spanning the old end time. Compare the three NEW branches at common
times; their step numbers do not imply the same times as the original runs.
The saved pending first interval is reduced through the existing restart hook;
previous accepted time, old timestep and histories stay unchanged. All safety
caps, state-change limits and solver/clock tolerances remain active. Another
mismatch stops the experiment; do not halve again or force acceptance.

## Update the benchmark plugin

Copy these updated files into the server plugin's `bp5/` directory:

- `benchmarks/reconstructed_fault/bp5/normal_filter_clock.cc`
- `benchmarks/reconstructed_fault/bp5/normal_stress_clock.h`

Rebuild and copy the diagnostic library into the job directory:

```sh
cmake --build /path/to/plugin-build --target bp5_normal_stress_diagnostic -j4
cp /path/to/plugin-build/libbp5_normal_stress_diagnostic.release.so .
```

Retain the corrected `bp3/work_replay.h` and rebuilt
`libbp5_steady_initialization.release.so` from the preceding observer fix.
No new ASPECT core changes or executable rebuild are required. The new parameter
defaults to false, preserving the original full-clock PRMs.

## Prepare checkpoint copies once

Keep the original `fixture/`, libraries, `environment.sh`, `checkpoint_selector.txt`
and R interval file in the job directory. Copy the three new PRMs there too.
Use the original production checkpoint, NOT a checkpoint from either attempted
branch. Expected identity: step 5612, time 5310111071.5634108 seconds.

This block refuses an existing `half/` directory and never overwrites the old runs:

```sh
(
  set -eu
  BP5_CHECKPOINT=/absolute/original/output/restart/01
  test ! -e half
  test -f R/output/normal_actual_intervals.txt
  cat "$BP5_CHECKPOINT/bp3_accepted_state.txt"
  for branch in R F1 F2; do
    mkdir -p "half/$branch/output/restart"
    cp -a "$BP5_CHECKPOINT" "half/$branch/output/restart/01"
    cp checkpoint_selector.txt "half/$branch/output/restart/last_good_checkpoint.txt"
    cp -a "$BP5_CHECKPOINT/bp3_output_metadata/." "half/$branch/output/"
  done
  sha256sum R/output/normal_actual_intervals.txt *-half.prm *.so > half/inputs.sha256
)
```

Check the printed identity before launching. Do not regenerate mesh, fault or
completion inputs. Use the same executable, updated plugins and MPI rank count
for all new branches.

## Launch individually, in order

Source the environment inside the batch job. Substitute the actual executable
path and original MPI rank count. Use `ibrun` instead of `mpirun -np 4` if
required by the server allocation. Retain the two-hour wall limit per branch.

```sh
source environment.sh
timeout --kill-after=15s 7200s mpirun -np 4 /absolute/aspect-build/aspect-release R-half.prm > half/R.log 2>&1
```

After R-half completes successfully, run F1-half, then F2-half after checking F1.
Reuse the original offline length assessment. Do not chain launches so that a
failure is ignored:

```sh
timeout --kill-after=15s 7200s mpirun -np 4 /absolute/aspect-build/aspect-release F1-half.prm > half/F1.log 2>&1
timeout --kill-after=15s 7200s mpirun -np 4 /absolute/aspect-build/aspect-release F2-half.prm > half/F2.log 2>&1
```

All three read ORIGINAL R, not R-half. Successful runs write separate accepted
clocks. Check them without Python:

```sh
cmp half/R/output/normal_actual_intervals.txt half/F1/output/normal_actual_intervals.txt
cmp half/R/output/normal_actual_intervals.txt half/F2/output/normal_actual_intervals.txt
wc -l half/R/output/normal_actual_intervals.txt half/F1/output/normal_actual_intervals.txt half/F2/output/normal_actual_intervals.txt
```

Require ten lines per branch AND passing nonlinear/fresh-linear checks in the
logs and `accepted_steps.csv`; matching clock files alone do not prove success.
The existing analysis uses `half/` as its comparison root. No analysis script
is required to launch the runs.

On another mismatch the message prints step, expected dt, actual dt and signed
difference at full precision. Preserve the error and `timestep_selection.csv`.
This is the single permitted retry, not an automatic retry loop.

## Local verification

The updated diagnostic plugin and clock regression built with `-j4`.
`test_normal_stress_clock` passed the existing paired-half tests and the new
ten-interval retry tests: unchanged full-clock parsing, exact half duration,
malformed/incomplete clocks, and rejection of the reported 5614 mismatch and
further shortened/enlarged intervals. All three exact `*-half.prm` files passed
`aspect-release --validate` with the rebuilt libraries. No BP5 trajectory was
launched; checkpoint deserialization and runtime acceptance remain server checks.
