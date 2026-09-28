# Bounded 2-D BP5-friction diagnostic

Large generated outputs, including copied server bulk/particle results and
disposable checkpoints, were compressed after checksum verification. Compact
histories, plots, reports, fixtures and server packages remain available. See
[CLEANUP.md](CLEANUP.md) before running analyses that need archived raw data or
restarting from an older output directory.

This is the maintained dipping **modified BP3 research geometry with BP5
friction coefficients**, not a reproduction of the official three-dimensional
BP5 benchmark. The old BP3 cases and their failed 1% profile-width gate remain
unchanged.

## Current 30–33 km model

The explicitly selected `bp5_steady_initialization` plugin now provides the
approved alternative: uniform `Theta0=Dc/Vinit=1e8 s`, followed by a native
weak projection of the variable shear background using the prepared surface
material. It bypasses the old inverse-state routines and requires an empty
`Mature prestress file`. The old plugin/packages below preserve their original
initial data and must not be mistaken for this variant. See
[the steady-startup report](steady_startup_report.md) for qualification.

The subsequent [physical-timestep check](steady_large_step_report.md) finds that
4e6-s steps converge but do not resolve the early evolving changes. The selected
125000-s ceiling passes a bounded work-weighted step-doubling screen, retaining
explicit endpoint limitations. Use the new
[steady server package](server-30km-steady/README.md), not the inverse-state
packages below. Start fresh; no old eight-day checkpoint conversion is intended.

```sh
cmake -S benchmarks/reconstructed_fault/bp5 -B benchmarks/reconstructed_fault/bp5/build
cmake --build benchmarks/reconstructed_fault/bp5/build --target bp5_steady_initialization -j4
python3 benchmarks/reconstructed_fault/bp5/run_steady_startup.py prepare startup
python3 benchmarks/reconstructed_fault/bp5/run_steady_startup.py run startup
python3 benchmarks/reconstructed_fault/bp5/analyze_steady_startup.py startup
```

The separate `half` and `resume` cases use the same launcher/checker; `compare`
on the checker compares all three. These are bounded verification fixtures,
not server launch inputs for a full earthquake cycle.

The velocity-bound interpolation repair and regenerated 0–30 km weakening,
30–33 km transition are documented in [the startup report](startup_30km_report.md).
The original 4e6-s first physical step passes, but the following mechanical
solve stalls. Its `server-30km/` package remains immutable and launch-guarded;
it must not be mistaken for the corrected adaptive startup configuration.

The subsequent [fresh small-step check](small_startup_report.md) starts again
from the original initial state with 300-s and 150-s physical timesteps, keeping
the artificial initialization interval separate. Both reach 900 s, but the
mechanical timestep sensitivity remains measurable. The original large-step
failure is retained as a separate bound-contact regression; the old server
package remains blocked.

The [startup follow-up](startup_followup_report.md) adds one 75-s comparison
at 300 s, exercises the 0.02 predictor against the actual 4e6-s ceiling, and
checks an ordinary same-rank checkpoint/resume branch. Its inputs and results
are separate from the preserved large-step failure and server package.

The regenerated [server-30km-adaptive package](server-30km-adaptive/README.md)
uses that verified 0.02 adaptive restriction. Copy the whole directory; runtime
paths are relative to the job directory and require no `ASPECT_SOURCE_DIR`.
Build the included plugin against the matching server ASPECT and place both
runtime binaries as documented. Qualification covers bounded startup and
same-four-rank restart, not full-cycle or spatial accuracy. No long run has
been launched here.

The following sections retain the preceding 15–18 km investigation's inputs
and evidence; they are not the current 30–33 km server configuration.

The task is defined in [the short-test instructions](bp5_friction_dc010_ell100_short_test.md).
The bounded test stopped on a step-2 lower-bound invariant after passing all
four frozen probes and accepted states 0–1. See the [report](short_test_report.md).
No further trajectory is qualified by these results.
Inputs/results are isolated in `dc010-ell100/`. Its `source-before/` records
HEAD, the pre-existing dirty worktree, and the previously tested shared libraries.

## Subsequent initialization-only comparisons

The [clean-background comparison](clean_background_report.md) replaces both
stored shear and its correction with uniform effective nominal traction.
The separate `bp5_initialization` plugin additionally initializes state from the
projected material and balances the resulting Q1 friction loads once, before
mechanics. It does not change the subsequent aging law or add a prestress
correction. The original plugin and historical cases remain available.
See the [weak-initialization result](weak_initialization_report.md).

```sh
cmake -S benchmarks/reconstructed_fault/bp5 -B benchmarks/reconstructed_fault/bp5/build -DAspect_DIR="$PWD/build-pf-cpdi" -DCMAKE_BUILD_TYPE=Release
cmake --build benchmarks/reconstructed_fault/bp5/build --target bp5_initialization -j4
python3 benchmarks/reconstructed_fault/bp5/run_weak_initialization.py prepare
python3 benchmarks/reconstructed_fault/bp5/run_weak_initialization.py run
python3 benchmarks/reconstructed_fault/bp5/analyze_weak_initialization.py
```

This launcher stops at accepted step zero and refuses to overwrite its result.
It replaces the old nodal-inverse observer (which expects the original physical
initial function) with projected-material, native-weak-load, derivative and
retained-initial-state checks. Mechanical convergence, fresh linear residual,
endpoint, profile and ordinary output checks remain enabled.

## Configuration and decisions before execution

- Material configuration is authoritative: shallow/deep `a=0.004/0.04`,
  `b=0.03`, `Dc=0.1 m`, `ell=100 m`. A new inverse of the configured friction
  law supplies initial state; it does not change the production aging law.
- Keep the 300 by 100 km box, 60-degree fault, original physical background
  correction (identical file hash), all-frictional continuous Q1 fault,
  mature zero cohesion, frozen phase, true normal stress and paired endpoint
  completion/source treatment. The strengthening particle property is retained.
- Candidate: 114,984 cells, band side 24.4140625 m. Reference: 191,958 cells,
  band side 12.20703125 m in the 7–26 km patch and first/last 5 km, with the
  existing graded normal halo and 2:1 grading. Monitor patch edges at 5, 7,
  26 and 110.470 km. Maximum far-field side remains 12.5 km.
- Both mechanical modes use a cosine-squared taper centered at 16.5 km,
  half-width 6.25 km, amplitude `1e-12 m/s`, and nominal wavelengths 3125
  and 200 m. This changes neither the 15–18 km material transition nor its
  `a=b` crossing at 17.1667 km.
- Use block AMG, explicit B/G, pivoted tridiagonal surface inverse and the
  unchanged solver checks. Four MPI ranks, one thread per rank.
- Allow the explicitly approved approximately 2% candidate RMS-width error
  for this diagnostic only. Preserve the original 1% qualification failure.
- Simulations share a 3600-second budget, with an external timeout and no
  retries. Preparation/build time is separate. Probes are intentionally
  noncommitting and exit with the existing verification marker and exception;
  coupled runs must instead genuinely converge and exit normally.

## Reproduction

Run from the repository root. Existing result labels are never overwritten.

```sh
cmake --build build-pf-cpdi --target aspect -j4
cmake --build benchmarks/reconstructed_fault/bp3/build --target bp3 -j4
cmake --build benchmarks/reconstructed_fault/performance/build-gmg --target fault_mechanical_modes bp3_length_scale_mesh -j4
python3 benchmarks/reconstructed_fault/bp5/run_short.py setup
python3 benchmarks/reconstructed_fault/bp5/run_short.py prepare --label probe-candidate --probe
python3 benchmarks/reconstructed_fault/bp5/run_short.py prepare --label probe-reference --mesh reference --probe
python3 benchmarks/reconstructed_fault/bp5/run_short.py run --label probe-candidate
python3 benchmarks/reconstructed_fault/bp5/run_short.py run --label probe-reference
python3 benchmarks/reconstructed_fault/bp5/analyze_probes.py
```

Each launch manifest records exact input/binary hashes and MPI/environment
settings. `prepare` rejects an evolution request unless the probe gate passes.
The default six-step candidate uses the ordinary timestep controller with a
4e6-second ceiling, not a forced saved clock. Fine/restart comparisons must use
its realized physical times. No full-cycle run is configured or authorized.
