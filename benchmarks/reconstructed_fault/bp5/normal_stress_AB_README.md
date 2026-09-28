# Direct A/B restart inputs

Use `normal_stress_experiment_A.prm`, then `normal_stress_experiment_B.prm`.
Both contain the full parameters: **no includes, staging scripts, generated
clock files, Function expressions, or binary-checkpoint edits are required**.

Rebuild ASPECT and the two loaded libraries against the same headers:

```sh
cmake --build /path/to/aspect-build --target aspect -j4
cmake --build plugin/build --target bp5_steady_initialization bp5_normal_stress_diagnostic -j4
```

Run from the existing server job directory containing `plugin/build/` and
`fixture/`, with the same qualified environment and MPI rank count. As for any
ordinary restart, each branch needs its own **unedited** copy of the original
accepted-step-5612 checkpoint:

```
output-experiment-A/restart/01/<original checkpoint files>
output-experiment-A/restart/last_good_checkpoint.txt   (contains 1)
output-experiment-B/restart/01/<same original checkpoint files>
output-experiment-B/restart/last_good_checkpoint.txt   (contains 1)
```

Do not use the prior diagnostic's final `restart/02`, or A's new checkpoint for
B. Do not copy old diagnostic CSVs into either branch. Checkpoints and scientific
fixtures are ordinary input data; nothing regenerates them or modifies them in
place. Retain the server's separately accepted restart-normalization allowance
when merging these edits; quadrature and tail tolerances remain 1e-10.

```sh
ibrun /path/to/aspect-build/aspect-release normal_stress_experiment_A.prm > A.log 2>&1
# Only after A has completed four genuinely accepted steps:
ibrun /path/to/aspect-build/aspect-release normal_stress_experiment_B.prm > B.log 2>&1
```

For local execution substitute the same-rank `mpirun -np N` launcher. No BP5
trajectory is launched by building or validating these files.

## How B changes dt

```prm
subsection Time stepping
  set List of model names = convection time step, reconstructed fault time step, BP5 state startup, BP5 recorded half steps
  subsection BP5 recorded half steps
    set Reference trajectory file = output-experiment-A/normal_summary.csv
  end
end
```

The new benchmark-local controller requires exactly four contiguous accepted A
rows beginning after the configured checkpoint. It checks the restored pending
time and dt against A's first row, then requests half that dt through the new
core restart hook. The core changes only the pending time and dt; old dt, step
number, solution vectors and constitutive histories are not updated. No archive
bytes are rewritten. It rejects nonpositive, nonfinite, enlarged or rank-dependent
requests. The first half inherits the smaller-than-saved safety interval; later
steps use the minimum of all the unchanged safety controllers and this cap.

Before each mechanical solve B checks exact step/time/dt agreement. If any
controller shortens a scheduled step, B stops without that solve; it does not
override the reduction, change tolerances or take extra steps. At the end of the
eight-step sequence the normal diagnostic terminates the run. Both inputs keep
the existing 600-s accepted-state wall limit; wall-limited output is incomplete.

The second half is `A_endpoint - midpoint`, while the first is `A_dt/2`.
The difference from exact half is limited to one absolute-time ULP and recorded
in `output-experiment-B/normal_half_step_clock.csv`. At this late physical time
one ULP is 9.5367431640625e-7 s; report that adjustment rather than pretending
both represented half intervals are identical. All four common endpoints match
bitwise. The checks use full-precision values, not rounded log timestamps.

The precommit decomposition, native r=0 traces, particle/FE history comparison,
plots and matched-state analysis are described in `normal_stress_12solve_README.md`.
Use `analyze_normal_stress_experiment.py output-experiment-A --other output-experiment-B`.
The old preparation workflow is retained only to reproduce earlier experiments;
it is not needed for these direct PRMs and must not be combined with the new hook
(that would attempt to halve an already modified checkpoint).
