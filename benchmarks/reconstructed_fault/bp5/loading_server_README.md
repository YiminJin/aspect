# Loading-driven modified BP5: exploratory first event

Copy this entire directory. All scientific inputs are relative to it; no
`ASPECT_SOURCE_DIR` discovery is needed. This is the 2-D modified-BP3 geometry
with BP5 friction, **not the official 3-D BP5 benchmark**. Start fresh; do not
resume an inverse-state, uniform-steady, seeded or other-ratio trajectory.

## Initial data and qualification

`Weakening initial state ratio = 0.8` selects
`Theta_i=(Dc/Vinit)*0.8^(1-f_i)` after the normal surface material projection.
Here `f_i` is the production strengthening fraction. Plateau state is 8e7/1e8 s;
the native weak Q1 friction loads set approximately 38.64537/26.54612 MPa shear
background with 50 MPa normal background. Maxwell stress is a perturbation,
initially zero. Artificial initialization retains the supplied state and zero
slip. Subsequent mechanics uses incoming state, followed by one exact aging
update per accepted step. Restart restores state/background rather than
reinitializing and checks the initial-data version and ratio.

The evidence directory records the three-step startup and its checkpoint-based
two-half-step comparison. Passing those provisional screens permits an
exploratory run, **not full-cycle temporal-convergence or event qualification**.
The scalar screening estimate of 1.54 years is not an event-time prediction.

Artificial initialization and the maximum first physical step are both 1e6 s.
The 1e7-s global ceiling is exploratory; convection, reconstructed-fault,
0.02 weighted-log-state predictor and 91% growth restrictions remain active.
`timestep_selection.csv` records their proposals and actual selected steps.

## Build and run

Use the exact base revision in `manifest.json` on a separate source checkout,
apply `source.patch` there after `git apply --check`, and unpack
`source-overlay.tar.gz`. Build ASPECT with its supported Release configuration
and `-j4`. Build the supplied benchmark against that same server build:

```sh
cmake -S plugin/bp5 -B plugin-build \
  -DAspect_DIR=/absolute/path/to/server/aspect-build -DCMAKE_BUILD_TYPE=Release
cmake --build plugin-build --target bp5_steady_initialization -j4
cp plugin-build/libbp5_steady_initialization.release.so ./
# Copy the matching server executable here as aspect-release.
source ./environment.sh
python3 verify_package.py
./aspect-release --validate first_event.prm
mpirun -np 4 ./aspect-release first_event.prm
# Alternatively, submit once from this directory:
sbatch -A YOUR_ACCOUNT first_event.slurm
```

The plugin target name is retained for the shared initializer implementation;
the explicit ratio and checkpoint identity select loading initial data. Do
not load another BP3/BP5 plugin simultaneously. The local `.so` in `provenance`
is evidence, not a portable server binary. Match compiler/MPI/deal.II libraries;
do not pass Intel Fortran flags to GNU Fortran. The batch template deliberately
does not guess the site's module versions. Load the matching stack before
submission. The environment clears old investigation flags and enables only
sparse B/G and pivoted-tridiagonal inversion, with one thread per MPI rank.
AMG, fresh residual checks and physical acceptance criteria remain unchanged.

## Output, termination, restart

The job requests four ranks, one node, 24 h. The 84600-s graceful stop leaves
a scheduler margin. The local three-step/end-time limits are removed.
First-event onset is an accepted max(V)>=1e-3 m/s; termination follows five
consecutive accepted states below it, resetting on a renewed up-crossing.
The independent 1500-year end without onset is **no event**, not event success.
Ordinary checkpoints occur every 3600 wall seconds and on termination.

`output-30km-loading/` contains accepted solver/station histories, cumulative
slip (subtract from Vp*t for slip deficit), predictor/selection records,
event metadata and sparse coordinated bulk/particle/fault visualization.
Fault profiles use 0.01-m slip or one-year intervals, heavy output 0.1 m or
one year plus event triggers, without forcing timesteps. Full-state dumps and
mechanical probe callbacks are disabled. Background and initial-state/mixture
provenance are retained in `steady_initialization.csv` (shared filename).

For this trajectory only:

```sh
source ./environment.sh
mpirun -np 4 ./aspect-release resume.prm
# or BP5_PARAMETERS=resume.prm sbatch -A YOUR_ACCOUNT first_event.slurm
```

Preserve checkpoint sidecars and their accepted output prefixes. Following a
crash, archive newer output and restore the selected checkpoint's metadata and
cumulative-slip prefix before resuming. Four-rank restart is tested;
cross-rank restart and long-cycle accuracy are separate qualifications.
