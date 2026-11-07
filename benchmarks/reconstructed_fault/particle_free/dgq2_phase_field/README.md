# Native DGQ2 phase checks

Current closeout: the native checks and both positive coupled timesteps pass.
The user deferred reverse translation, the finer mesh and remaining tests.
The commands below are reproduction/resumption entry points, not an active queue.

The implementation belongs to `include/aspect/phase_field/`,
`source/phase_field/` and the native phase/endpoint owners. This directory contains
small drivers, independent manufactured data, noncommitting native observers and
comparison tools. Read
[`DGQ2_PHASE_FIELD_DESIGN.md`](../../../../doc/reconstructed_fault/particle_free/DGQ2_PHASE_FIELD_DESIGN.md)
and the [report](../../../../doc/reconstructed_fault/particle_free/DGQ2_PHASE_FIELD_REPORT.md)
for equations, supported configurations and evidence boundaries.

Run from the repository root. Do not overwrite retained outputs or infer a
binary's identity from HEAD. Build/test/heavy reduction jobs run sequentially,
with at most four workers/ranks. The configured compiler/runtime is GCC 12.4,
OpenMPI 5.0.6 and deal.II 9.6; the logged runner installs their runtime paths.

## Builds

The current separate core build is `build-particle-free/dgq2-phase/native`.
Preserve `candidate/aspect-release`, its matching checker/provider and
`candidate/source-identity.json` before replacing any of them.

```sh
cmake -S benchmarks/reconstructed_fault/particle_free/dgq2_phase_field \
  -B build-particle-free/dgq2-phase/operators \
  -DAspect_DIR="$PWD/build-particle-free/dgq2-phase/native"
cmake --build build-particle-free/dgq2-phase/operators --parallel 4
```

The standalone executable compiles the actual native transport, SIPG and
obstacle sources. It is not a copied Python implementation. The plugin exercises
native cohesive energy/normalization derivatives without publishing state.

## Bounded verification

Use a fresh evidence label instead of reusing the examples below. The logged
wrapper records the exact command, environment, status and elapsed time.

```sh
python3 benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py phase-transport-new 900 \
  python3 benchmarks/reconstructed_fault/particle_free/dgq2_phase_field/run_extended_checks.py transport-new transport
python3 benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py phase-refined-new 180 \
  python3 benchmarks/reconstructed_fault/particle_free/dgq2_phase_field/run_extended_checks.py refined-new refined
```

`run_equilibrium_checks.py` covers manufactured SIPG/obstacle MPI comparisons.
`run_native_growth.py` generates the supported fixed-fault cohesive fixture from
`growth.prm`; its default small check is distinct from mature moving transport.
`run_lifecycle.py` performs the short four-to-two-rank restart and forced-retry
schedule comparison. `check_checkpoint_identity.py` uses private copied
checkpoints to test cross-family rejection. Consult their command-line options
and the recorded command manifests before repeating a case.

`q1_control.py` is an independent retained-Q1 interpolation replay; its Python
runtime is not a performance comparison with C++. The current native Q1 control
is also compared bitwise with retained full coupled output.

## Selected four-case comparison

```sh
python3 benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py phase-coupled-new 60000 \
  python3 benchmarks/reconstructed_fault/particle_free/dgq2_phase_field/run_coupled.py \
  plus plus-half minus plus-fine
```

The four labels are fixed to `layered-dgphase-*` and existing directories are
protected. This command is the selected multi-hour matrix, not a default unit
test. It uses four ranks and runs the provider check and streaming reduction
between simulations. `--reduce-only` reuses completed outputs. All start fresh
with the explicit native DG phase selection, direct stress, corrected bulk spin,
matched exterior data and the old physical inputs. Growth stays disabled.

For the current two-case closeout, run `compare_coupled.py --available-only`.
After the deferred simulations/reductions are selected and completed, run
`compare_coupled.py` without that flag.
It reuses old Q1 reductions, quantifies different initialization, and evaluates
bulk norms on the union of both meshes and surface norms on a mesh-crossing
partition. Central and finite-end regions remain separate. Phase material
balance uses the actual SSPRK source/flux log, independently of endpoint versus
trapezoid diagnostics. `plot_coupled.py` also runs standalone to regenerate PNGs from the compact data.
`monitor_memory.py` only samples rank RSS; it changes no simulation state.

Evidence is under `../evidence/dgq2_phase_field/`. Failed checks are retained.
No production checkpoint conversion, arbitrary AMR, moving cohesive growth or
paper-scale earthquake qualification is implied.
