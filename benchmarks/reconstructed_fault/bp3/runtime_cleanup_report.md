# Solver selectors and long-run output cleanup

2026-09-25. Changes are layered on the existing working tree (base commit
`3ce447a17`); unrelated experiments and saved outputs were preserved.

## Execution invariants

- Production surface inverses always use adjacent-pivoting LAPACK
  GTTRF/GTTRS. Indefinite free blocks, active-row zeros, singular/nonfinite
  checks and scaled backward-error checks are unchanged. UMFPACK comparison
  moved to `tests/fault_surface_reference.h`. Neither
  `ASPECT_FAULT_SURFACE_SOLVER` nor `ASPECT_FAULT_COMPARE_SURFACE_INVERSE`
  controls production code any more.
- `Solver parameters / Stokes solver parameters / Stokes solver type`
  selects the reconstructed-fault velocity preconditioner. `block AMG` is
  the default; `block GMG` uses the existing Q2/local-smoothing velocity cycle
  and automatically builds its mesh hierarchy. No
  `ASPECT_FAULT_VELOCITY_GMG` or `ASPECT_FAULT_GMG_HIERARCHY` selection remains.
  The assembled fine A/B/G, pressure preconditioner, surface inverse, outer
  FGMRES, fresh residual checks and history lifecycle are retained. Default
  fine-level material averaging remains none for both coupled backends.
  This cleanup does not repair or qualify the previously reported server GMG
  NaN; the current first-event PRM still selects AMG.
- The environment script clears obsolete selectors and has no backend
  selection role. Research/long-run launchers now write the backend into
  the generated PRM.
- `Postprocess / BP3 restored monitor / Write detailed diagnostics` defaults
  false. Startup/filter PRMs explicitly set true; first-event continuation
  sets false. False disables raw-sample capture and the per-step
  `restored_raw_*`, `restored_incoming_particles_*`, `restored_fault_*` CSVs.
  The growth summary, physical checks, stations, scheduled profiles,
  mesh/particle/fault visualization and cumulative slip remain available.
  The diagnostic switch is not part of checkpoint physical identity.
- Cumulative-slip CSV keeps every vertex/accepted state and the existing
  columns. Slip uses ten significant digits, timestamps and coordinates
  seventeen. Numerical state and checkpoints are not rounded. This plotting
  file must not be used to differentiate extremely small late-time slip
  increments at solver accuracy.

## Verification

Builds: ASPECT Release and the self-contained restored plugin, `-j4`.
The local shell initially selected a different GCC than the cached build;
the build was completed using GCC 12.4/OpenMPI 5.0.6 and an explicit non-PIE
executable link. No production source workaround for that toolchain problem.

Focused artifacts and exact PRMs/CMake configuration are in
`/tmp/bp3-runtime-cleanup-KF1mxB/`.

| Check | Result |
|---|---|
| `aspect-release --test '[fault_surface_direct],[fault_normal_filter]'` | 636 assertions / 3 cases passed |
| Same unit tests, `mpirun -np 2` | 636 assertions / 3 cases passed on each rank |
| Stage-I coupled fixture, PRM-selected AMG, stale GMG flags present | Genuine convergence, normalized bulk residual 5.089815e-8; surface 0; fresh linear checks pass |
| Same fixture, PRM-selected GMG without selector flags | Genuine convergence, normalized bulk residual 9.165224e-8; surface 0; fresh linear checks pass |
| GMG, two ranks | Genuine convergence, normalized bulk residual 9.165219e-8; surface 0 |
| Existing accepted-update rollback fixture, AMG and GMG | Intentional nonlinear failure; complete rollback verified, 3 s / 2 s |
| Existing condensed adiabatic fixture | Surface/bulk actions, UMFPACK comparison and recovery verified, 7 s |
| BP3 first-event and detailed-monitor constructor tests | Both reach Simulator::run; stop before mesh generation |
| `test_slip_history.py` | 3 tests passed (bash/zsh environment and restart-prefix preservation) |
| `test_research_launcher.py` | 1 test passed with three backend choices |
| `test_plot_cumulative_slip.py` | 5 tests passed, including compact slip and millisecond-separated times after centuries |
| `git diff --check` | Passed |

The first temporary condensed-test plugin omitted the inherited I_h
postprocessor registration and stopped during parameter parsing. Building
the existing combined `phase_field_fault_condensed_adiabatic.cc` fixture
fixed the harness; no numerical input or assertion was changed.

The saved filter20 cumulative-slip data contain 31,779 rows. Reformatting
only slip gives 2,416,161 bytes versus 2,619,706 bytes: **7.77% smaller**.
Largest observed relative rounding is 4.9824e-10. The original file was not
rewritten. Reader/checker allowances now explicitly account for that output
rounding, not a changed computational tolerance.

No full-resolution BP3 trajectory, server GMG run or full integration suite
was launched. Per-step quiet-output behavior was checked in source and via
resolved constructor parameters, not a new full BP3 trajectory.

## Files to rebuild

ASPECT:

- `source/reconstructed_fault/surface_direct_internal.h`
- `source/reconstructed_fault/normal_filter_internal.h`
- `source/reconstructed_fault/surface_system.cc`
- `source/simulator/core.cc`
- `source/simulator/assembly.cc`
- `source/simulator/helper_functions.cc`
- `source/simulator/parameters.cc`
- `source/simulator/solver.cc`
- `source/simulator/solver/reconstructed_fault_condensed_system.cc`

Current BP3 runtime: `plugin/monitor.cc`, `plugin/output.cc`.
`plugin.tar` was refreshed; the preceding archive is retained in the
temporary verification directory as `plugin-before.tar`.
Supporting tests, launch scripts, PRMs, generator and design documentation
were updated. Historical result directories were not rewritten or removed.
