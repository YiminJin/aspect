# Local rotated-bottom experiment

This is the isolated implementation of [the requested test](../bp3_rotated_bottom_local_test.md).
See [REPORT.md](REPORT.md) for results and limits. No Python is used.

From the ASPECT repository root, the exact local build commands are:

```sh
cmake --build build-tmp -j2
cmake -S benchmarks/reconstructed_fault/bp3/rotated_bottom_local \
  -B benchmarks/reconstructed_fault/bp3/rotated_bottom_local/build \
  -DAspect_DIR="$PWD/build-tmp" -DCMAKE_BUILD_TYPE=Release
cmake --build benchmarks/reconstructed_fault/bp3/rotated_bottom_local/build -j2
```

The generated `fixture/` is retained locally but excluded from the source
commit. On a fresh checkout, generate it before running the tests:

```sh
bash benchmarks/reconstructed_fault/bp3/rotated_bottom_local/run.sh prepare
```

The generator uses the production stationary ell20 table, distributed deal.II
mesh grading, and the maintained leaf-tree replay. It regenerates the geometry
and virtual Cartesian-Q1 missing-normal-profile integrals at both endpoints.
The PRM uses four initial global tagging passes through `BP3 saved mesh`, rather
than adaptive error-driven passes. The mandatory exact leaf-tree check passes
on both one and two ranks. Near-fault spacing is 6.25 m; no physical or geometric
production dimensions are reused for the small box.

Run commands used (sequentially):

```sh
bash benchmarks/reconstructed_fault/bp3/rotated_bottom_local/run.sh uniform-A 1
bash benchmarks/reconstructed_fault/bp3/rotated_bottom_local/run.sh uniform-B 1
bash benchmarks/reconstructed_fault/bp3/rotated_bottom_local/run.sh uniform-B-mpi2 2
bash benchmarks/reconstructed_fault/bp3/rotated_bottom_local/run.sh A 2
bash benchmarks/reconstructed_fault/bp3/rotated_bottom_local/run.sh B 2
# The sole accuracy follow-up: halve dt, retain the same final physical time.
bash benchmarks/reconstructed_fault/bp3/rotated_bottom_local/run.sh A-half 2
bash benchmarks/reconstructed_fault/bp3/rotated_bottom_local/run.sh B-half 2
bash benchmarks/reconstructed_fault/bp3/rotated_bottom_local/analyze.sh
```

The launcher sources the maintained [environment.sh](../environment.sh), uses
OpenMPI `/opt/openmpi/5.0.6/bin/mpirun`, one thread per rank, and refuses to
overwrite outputs. Each simulation is bounded by 900 seconds, with an 840-second
graceful stop setting. `ASPECT_LOCAL_BINARY` can select a different executable;
the local default is `build-tmp/aspect-release`. A different executable/compiler
must have a matching rebuilt plugin. Do not edit a shell launcher while it runs.

The production-default build and parse-only check use:

```sh
cmake -S benchmarks/reconstructed_fault/bp3/plugin \
  -B benchmarks/reconstructed_fault/bp3/rotated_bottom_local/build-production \
  -DAspect_DIR="$PWD/build-tmp" -DCMAKE_BUILD_TYPE=Release
cmake --build benchmarks/reconstructed_fault/bp3/rotated_bottom_local/build-production -j2
bash benchmarks/reconstructed_fault/bp3/rotated_bottom_local/run.sh production-validate
g++ -std=c++17 -O2 benchmarks/reconstructed_fault/bp3/test_restored_model.cc \
  -o benchmarks/reconstructed_fault/bp3/rotated_bottom_local/build/test_restored_model
benchmarks/reconstructed_fault/bp3/rotated_bottom_local/build/test_restored_model
```

`production-validation.prm` only overrides the plugin/output paths. **Use it
with `--validate` only**, as the launcher does; it is production-sized.

## Parameter and implementation

The maintained plugin's single opt-in setting is:

```text
subsection Postprocess
  subsection BP3 restored monitor
    set Bottom velocity constraint = fault parallel
  end
end
```

Default `full` returns immediately from the new callback. B also removes bottom
from ordinary prescribed, zero, and tangential velocity boundary lists. The
callback rejects conflicts rather than overwriting existing constraints.
Left/right are unchanged, and the top has natural zero perturbation traction
in both cases. B keeps the full side-boundary constraints at bottom corners.

`ASPECT_BP3_LOCAL_BOTTOM_TEST` is a compile definition of this fixture target
only. It selects the 1 km fault height, uniform deep friction, local station
positions and `Local state disturbance` input. It is absent from the production
target. That input changes only the initial state, using the actual projected
fault height in the specified exponential factor; the steady background shear
is unchanged. The local plugin must not be used for a production checkpoint.

## Compact evidence

- `output-*/parameters.prm` and `parameters.json`: resolved configurations.
- `bottom_constraint_rows.csv`: callback coverage and physical/homogeneous mode.
- `local_metrics.csv`: trace errors, free variation, weak residual, flux,
  lumped traction-mass-weighted normal-stress summaries, V, state increments,
  and elapsed wall time.
- `local_fault_N.csv`: double-precision accepted traction decomposition, V and
  Theta. `normal_change` is relative to that case's step-zero mechanical
  equilibrium; `normal_rate` is the last accepted increment divided by dt.
- `bottom_N_rankR.csv`: uniquely owned boundary-face Gauss samples.
- `corner_metrics.csv`: four separate 200 m corner disks, ordered bottom-left,
  bottom-right, top-left, top-right. Pressure is accepted FE pressure; deviatoric
  norm is committed particle stress, zero at step zero under BP3 initialization.
- `accepted_steps.csv` and `log.txt`: iterations, state-law error, fresh linear
  residual checks, line-search rejection counts, timing and termination.
- `analysis/*-norms.csv`: **comparison norms** from exact integrals of the
  exported Q1 field squared over fixed arc-length intervals. These differ from
  the lumped, traction-mass-weighted online norms; do not mix their values.
- `analysis/final_endpoint.{png,pdf}`, `analysis/stress_history.{png,pdf}`:
  requested figures. The latter uses matching physical times and each case's
  own initial equilibrium.

Only initial/final volume visualization and compact step profiles are emitted;
full particle/QP CSV dumps are disabled. Build trees, initial/final VTUs, and
failed setup outputs are local evidence, not required source deliverables.
Generated `analysis/`, `logs/`, and `output-*/` files are also excluded from the
source commit. The report retains the measured results; its artifact links refer
to this local evidence or files reproduced by the commands above.
