# Focused execution-path cleanup before returning to BP3

## Outcome and scope

The small moving-particle A/B comparison is preserved. Both fresh-start replays
complete initialization plus four real steps, with zero measured differences
from the saved one-rank fields, coefficients, next FE histories and committed
particle stress. Comparison tolerances, not byte equality, were the acceptance
rule (1e-8 relative field scale plus 1e-8 absolute).

The normal BP3 library now registers the tested **optional** stress-only LLS
router. DWA remains selected in the maintained BP3 PRM. No mesh, loading, state
initializer, pressure convention, friction law, solver tolerance, timestep rule
or production ASPECT numerical code was changed by this cleanup.

An additional existing two-rank free-rate/state coupled smoke **failed**, before
acceptance, at the fresh-linear pressure-compatibility guard. This is not counted
as a passing smoke and no solver adjustment or expected-output refresh was made.
The unchanged core binary was used; the ordinary BP3 plugin and routing adapter
are not loaded by that test. This failure must be kept visible when deciding
what coupled verification to require next, rather than interpreting the passing
prescribed-slip A/B test as qualification of free BP3 dynamics.

## Recoverable pre-cleanup checkpoint

Starting revision: `33228369da82011f509ee07937f27c17663c7f28`, **dirty**.

Checkpoint directory (repository relative):
`benchmarks/reconstructed_fault/checkpoints/bp3-execution-cleanup-4qkx63tj/`.

- `source-and-inputs.tar.gz`: 7,943 selected paths, 59,748,386 compressed bytes;
  tracked source/headers/tests, reconstructed-fault sources and documentation,
  relevant untracked C++/headers/scripts/PRMs, maintained fixture inputs, actual
  CMake caches, and compact A/B/C evidence. Executable/library hashes are recorded;
  binaries are not duplicated in this archive.
- `working-tree.patch`: binary-capable diff from HEAD, including prior changes.
- `index.patch`: initial staged changes (empty).
- `status.txt`: complete initial tracked/untracked status, including results not
  selected for the source archive.
- `manifest.json`: SHA256 and sizes of archived files and tested binaries.
- `HEAD`: exact original revision.

Restore into a **separate checkout/worktree** at that HEAD, apply the patch,
then extract the source/input archive there. Do not overlay the current worktree
or delete evidence. The archive also contains source that was untracked before
this task. The original A/B/C raw outputs remain in their original directories.
The checkpoint/results are local recovery artifacts, not added to the cleanup
commit. The archive has been read back and all member hashes checked.
Archive SHA256: `05ef9cb0a82e30bcd429f1c44e937bf2cee78e82e2f3bf58f46da68f4334c777`.

The following 19 pre-existing tracked changes were retained and are not silently
included in the cleanup commit:

```
benchmarks/reconstructed_fault/CLEANUP.md
benchmarks/reconstructed_fault/bp3/work_replay.h
benchmarks/reconstructed_fault/bp5/CMakeLists.txt
doc/reconstructed_fault/current_design.md
doc/reconstructed_fault/specification.tex
include/aspect/material_model/phase_field_fault.h
include/aspect/reconstructed_fault/surface_system.h
include/aspect/simulator.h
include/aspect/simulator/solver/reconstructed_fault_linear.h
include/aspect/simulator_signals.h
source/material_model/phase_field_fault.cc
source/particle/integrator/rk_2.cc
source/reconstructed_fault/surface_system.cc
source/simulator/assemblers/reconstructed_fault_stokes.cc
source/simulator/checkpoint_restart.cc
source/simulator/initial_conditions.cc
source/simulator/solver.cc
tests/phase_field_fault_surface_system.cc
unit_tests/reconstructed_fault.cc
```

In particular the cleanup commit alone is **not** a claim that HEAD contains all
the tested core corrections; the archive and outstanding diff are needed to
reproduce this working-tree build. No geometry/loading/physical-parameter changes
are included in the cleanup commit.

## Switch and hook inventory

The intended BP3 configuration means the current `bp3_modified_long_run.prm`
with the ordinary `bp3` library and `bp3/environment.sh [gmg|amg]`. It does not
mean an old investigation library or BP5 diagnostic restart input. The environment
script clears inherited `ASPECT_*` variables and enables only source directory,
explicit B/G, pivoted tridiagonal inverse, and optional GMG/hierarchy. A presence
test such as `getenv(...)` treats `0` as **enabled**, not disabled.

### Hooks that can change physical evolution or the discrete equations

| Selector / hook | Default; owner and effect | Intended BP3 |
|---|---|---|
| `ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION` | Absent. Core `rk_2.cc` substitutes zero advection dt, without changing constitutive dt. | Off; ordinary BP3 now rejects it, including value `0`. Retained for explicitly selected frozen fixtures. |
| `PhaseFieldFault::benchmark_retained_stress` | Empty callback. `MomentCycle` native-history mode supplies cell-trace retained stress in bulk, work-measure surface evaluation and particle update; storage/publication belongs to the benchmark. | Empty; no BP3 caller. Not checkpointed; unsupported by particle-domain surface rule. |
| `Postprocess/Moment cycle/History mode` | `production` in the comparison. Alternatives install native-history retention or change particle publication via the benchmark's test access. | Plugin not loaded. Moving A/B uses `production`. |
| `set_normal_stress_filter(mode,length)` | `raw`, length 0; bypass. BP5 plugin can select consistent projection or Helmholtz-filtered frictional normal traction with corresponding K/G. | No caller; raw. Requires explicit `BP5 normal diagnostic` postprocessor and parameter selection. It is an experimental equation change, not output smoothing. |
| `post_resume_time_step` | No slot, unchanged pending restart interval. BP5 clock/transfer plugins can only reduce the restored interval, with MPI agreement and safety checks. | No BP3 slot. Ordinary adaptive controller unchanged. |
| `ASPECT_BP3_DISTURBANCE_TEST` (compile definition) plus `ASPECT_DISTURBANCE_*` environment | Absent in normal target. Dedicated plugin changes incoming state and/or reference-state or reference-normal feedback and timestep. | Not compiled. Normal BP3 rejects inherited EPS/CONTROL/DT selectors. |
| `ASPECT_BP5_WEAK_INITIALIZATION`, `ASPECT_BP5_STEADY_INITIALIZATION`, `ASPECT_BP5_NORMAL_CONTROL` (compile definitions) | Separate libraries only; select weak-state/prestress construction or prescribed-normal initialization control. | Not compiled into `bp3`. Normal initialization unchanged. |
| `ASPECT_BP5_SHORT_TEST` | Absent; benchmark short-test output and step-2 checkpoint behavior, plus relaxed diagnostic coverage assumptions. | Rejected. Checkpoint disabling is now compiled only into BP5 initialization libraries, not ordinary BP3. |
| `ASPECT_IH_BOTTOM_COMPLETION_DIAGNOSTIC`, `ASPECT_BP3_UNIFORM_SLIDING`, `ASPECT_BP3_TOP_SOURCE_EXPERIMENT` | Absent. Legacy endpoint/denominator experiments; material fallback is used only without the supported completion selector. | Rejected by ordinary BP3. Supported paired completion remains enabled through maintained inputs/API. |
| `enable_bulk_work_measure`, `enable_bottom_source_continuation`, `enable_top_source_continuation` | Generic default off; narrow straight/frozen/mature benchmark capability. | Intentionally on in `BP3Benchmark::prepare`, reattached on restart. Qualified work/source corrections retained. |
| Background-traction property selector and completion-file setter | Generic default unset. BP3 attaches its frozen background/correction and exact fixture completion data. | Intentionally on; no new initialization or recapture on restart. |
| `set_prescribed_slip_rates` | Manager-owned prescribed rows. | BP3 explicitly supplies an empty mask: the maintained modified-BP3 fault is fully frictional. Old constrained reference is a separate configuration. |
| `post_constraints_creation`, `post_set_initial_state`, `post_advection_solver` BP3 callbacks | Plugin-only phase constraints, H initialization, and preparation. | Intentionally connected: fixed distance-profile phase, ordinary initial state/prestress, normal history lifecycle. |
| Initial composition `Spatially refreshed field names` | Empty default. Particle-property plugin can replace listed composition properties with spatial initial functions. | `strengthening` only, intentionally refreshed; Maxwell stress is not spatially reset. |

The old frozen-cohesion, independent-trace and alternate-state investigation
implementations are not selected by the maintained BP3 plugin. Historical
reference sources/artifacts are preserved; none was promoted to a global default.
Generic callbacks and the experimental filter were not removed: their callers,
tests and explicit opt-in defaults remain useful, and deleting them would exceed
this bounded cleanup.

### Numerical/backend switches (not new physical models)

| Selector | Default | Intended BP3 |
|---|---|---|
| `ASPECT_FAULT_EXPLICIT_B`, `ASPECT_FAULT_EXPLICIT_G` | Absent: reference actions. Presence selects assembled actions (filtered G retains its compatible nonlocal path). | Both on; retained unchanged. |
| `ASPECT_FAULT_SURFACE_SOLVER` | Absent or `tridiagonal`: pivoted tridiagonal; optional `umfpack`. | `tridiagonal`; indefinite free blocks supported. |
| `ASPECT_FAULT_VELOCITY_GMG`, `ASPECT_FAULT_GMG_HIERARCHY` | Absent: AMG/no special hierarchy. | Both on for environment's GMG selection; both absent for `amg`. Fine operator/FGMRES/fresh checks unchanged. |
| `ASPECT_FAULT_INTERFACE_MODES` | Absent: no few-mode preconditioner. Integer 1--4 enables experimental correction. | Rejected by ordinary BP3. |
| `ASPECT_DISABLE_IH_VALUE_CACHE` | Absent: exact-value reuse permitted. | Absent. |
| `ASPECT_IH_BASELINE_GUARDS` | Absent: qualified optimized guard path. | Absent. |
| `ASPECT_IH_COMPARE_CELL`, `ASPECT_IH_REFERENCE_FACTOR`, `ASPECT_IH_VERIFY_CELL_QUADRATURE`, `ASPECT_IH_COMPARE_SAMPLES` | Absent: no extra reference integration/sample exports. Reference factor may only tighten diagnostic accuracy. | Absent. |
| Material `I h integration backend` / `I h surface quadrature subdivisions` | `remote points` / 1. | Current BP3 PRM keeps remote/1; BP5's eight-panel selection is not inherited. No values or tolerances changed here. |
| `Fault constitutive mode`, `Evolve phase field`, `Use adiabatic pressure in fault friction` | Ordinary parsed material options. | `mature frictional`, false, false. True normal traction, fixed profile, C=0 are explicit BP3 choices, not global defaults. |
| Particle interpolation | Native configured scheme; LS limiter normally true. | DWA unchanged. New shared router is optional, requires unlimited LS explicitly, and changes only Maxwell components. |

### Observational paths

The following default to absent/unconnected and are not enabled by the BP3
environment: `ASPECT_STRESS_CYCLE_TRACE`, `ASPECT_FAULT_SOURCE_HISTORY_DIAGNOSTIC`,
`ASPECT_FAULT_HISTORY_AUDIT`, `ASPECT_FAULT_NORMAL_STRESS_DIAGNOSTIC`,
`ASPECT_FAULT_STRESS_SAMPLE_DIAGNOSTIC`, `ASPECT_FAULT_COMPATIBILITY_DIAGNOSTIC`,
`ASPECT_FAULT_NONLINEAR_DIAGNOSTIC`, `ASPECT_K1_FLOOR_AUDIT`,
`ASPECT_FAULT_COMPARE_SURFACE_INVERSE`, `ASPECT_FAULT_VERIFY_INTERFACE`,
`ASPECT_FAULT_PERFORMANCE`, `ASPECT_FAULT_LINEAR_PERFORMANCE`,
`ASPECT_BP3_LENGTH_COUPLED_DIAGNOSTIC`, `ASPECT_BP3_LENGTH_FULL_AUDIT_FROM`.
They export data, do independent comparisons/noncommitting evaluations, or time
work; they may be expensive or throw but do not publish alternative histories.

`set_normal_traction_diagnostic`, `normal_diagnostic_observer` and
`post_reconstructed_fault_linear_solver` are empty by default and not connected
by ordinary BP3. `post_reconstructed_fault_solver` is connected to record
iterations/active nodes after acceptance, not replace mechanics. Output/checkpoint
and termination callbacks remain, including lightweight cumulative slip. Fresh
linear residual, nullspace/compatibility and nonlinear acceptance checks remain
unconditional even when diagnostic prose is disabled.

## LLS availability: separate functional change

The numerical router is unchanged from the tested 2-D A/B/C code. Its implementation
now lives at `benchmarks/reconstructed_fault/stress_only_interpolator.cc` and is
compiled into the normal BP3 target as well as the small BP5 comparison library.
A three-line BP5 translation-unit shim preserves its existing build target.
Do not load both full benchmark plugins into one simulation: they also register
different simulator fixtures.

To explicitly select it in a **future separately approved BP3 configuration**:

```prm
subsection Particles
  set Interpolation scheme = stress only linear least squares
  subsection Interpolator
    subsection Linear least squares
      set Use linear least squares limiter = false
      set Use boundary extrapolation = false
    end
  end
end
```

Keep every other particle/FE setting unchanged. The router finds `maxwell stress`
through the property manager, applies native LS only to its three components,
and uses the configured native DWA for every other selected property. No production
ASPECT code/API changes were required. A BP3 `--validate` input with these overrides
passes using the rebuilt ordinary plugin. This is registration/parameter validation,
not a free-BP3 trajectory qualification. Standalone server packaging must include
the shared source one directory above `bp3/`, as referenced by its CMake file.

## Changes, verification, and deferrals

Removed from ordinary BP3 compilation: the BP5-only step-2 checkpoint-disabling
branch. Removed duplicate ownership of the LS implementation: one shared source,
with the old diagnostic translation-unit path retained as a build shim. Added a
normal-target-only fail-fast environment check; dedicated experiment targets keep
their explicit behaviors. No broad class reorganization or assertion deletion.

Retained: incident-cell ADD/count publication; constrained working history;
stable exact aging and accepted-state timing; source/work consistency; paired
boundary treatment; fixed-background perturbation formulation; I_h caches,
projection and endpoint completion; pivoted inverse; B/G actions; FGMRES and
pressure/fresh-residual safeguards. The existing dirty core fixes remain untouched.

Commands/results:

1. `cmake --build benchmarks/reconstructed_fault/bp5/build --target bp5_moment_cycle -j4`: pass.
2. `cmake --build benchmarks/reconstructed_fault/bp3/build --target bp3 -j4`: pass.
3. `cmake --build benchmarks/reconstructed_fault/bp3/build --target test_bp3_environment -j4`, then execute it: pass, clean environment accepted and `0`-valued forbidden selectors rejected. The first combined build invocation saw the old generated Make target list; a separate invocation built the newly generated test target successfully.
4. Source `bp5/interpolation-inclined/environment.sh`; run `cleanup-A.prm` and `cleanup-B.prm` with one rank and a 120-s cap each: pass, 0--0.4 s. The first shared-source build and the final compatibility-translation-unit build were checked in separate `cleanup-*` and `cleanup-final-*` directories; baseline outputs were not overwritten and no failed simulation was retried.
5. `OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/bp3/check_execution_cleanup.py`: pass. All measured differences zero; ten fresh-linear checks and all nonlinear criteria pass. Results: `bp5/interpolation-inclined/cleanup_comparison.json`.
   Final-build simulation wall times: A 21.958 s, B 22.532 s.
6. `aspect-release --validate /tmp/bp3-lls-validation.prm`: pass. The file includes the unchanged BP3 long-run PRM and overrides only library path and interpolation/LS options shown above.
7. `ctest --test-dir build-pf-cpdi/tests --output-on-failure --timeout 120 -R '^phase_field_fault_stage_i_rate_state$' -j1`: **FAIL**, 27.05 s, two ranks, timestep-zero first fresh-linear pressure-compatibility check (`solver.cc`, `abs(residual_null_component) <= compatibility_tolerance`). Printed fresh residual 0.9293325, linear target 1.780987. No accepted state. No tolerances or fixture changed and no retry. Current evidence is `screen-output.tmp` and `log.txt`; the older `screen-output` contains a stale passing result and must NOT be used for this attempt.

The failed smoke is preserved separately in the checkpoint directory's
`post-cleanup-smoke/`. The ASPECT Release executable still has SHA256
`f0aa968d7a299f31a354209f28b72b32ba609585b42d33ecd6117e813a84a66c`.
The final BP3 plugin SHA256 is
`e5ad90e418a5491f47ec655e5f21113ba4ce88c92267b5f34d585193947d5e84`;
the final A/B plugin SHA256 is
`0fb9722d05bad9bce50e3130e43e8977f4246a05db0facdf448a058557bfdb43`.
No new one-/two-rank interpolation campaign was necessary: previous MPI results
remain valid for the unchanged router, while this task reran A/B on one rank.
No BP3 long trajectory, geometry change, pressure repair, or new friction
experiment is authorized by this cleanup; all are deferred.
