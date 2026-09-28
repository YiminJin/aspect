# Steady-initialized modified BP5 first-event job

Copy this **whole directory** to the server job directory. It is the 2-D
modified-BP3 geometry with BP5 friction, not the official 3-D BP5 benchmark.
Start a **fresh trajectory** with `first_event.prm`; never resume the old
eight-day inverse-state run. `resume.prm` is only for this new trajectory.

## Qualified configuration and limits

Uniform initial `Theta=Dc/Vinit=1e8 s`; projected surface composition and native
work integration construct spatially varying shear prestress once (about
38.980082 MPa weakening, 26.546122 MPa strengthening). Normal prestress is
50 MPa. The mature-prestress filename is empty: no old captured shear or
inverse-state initialization may overwrite this initial data. Restart restores
the evolved state/frozen background without recomputing initialization.

The 300x100-km domain, 0–30/30–33-km weakening/transition, fully frictional fault,
Dc=0.1 m, ell=100 m, frozen profile, work measure, endpoint corrections, split
exact aging, true normal traction, loading and solver tolerances are unchanged.
The mesh has 114,984 cells, 24.4140625-m near-fault spacing and 1156 fault nodes.
AMG, sparse B/G, pivoted tridiagonal surface inverse and outer FGMRES remain.

Time units are **seconds**. Artificial initialization remains **4e6 s**; this
does not advance physical time or age Theta. The qualified physical ceiling
is **@CEILING@ s**, not 4e6. Convection, reconstructed-fault and `BP5 state
startup` models remain active, with predictor bound **0.02**. Actual steps may
be smaller. The 4e6-s physical startup converged but failed the temporal screen;
see `evidence/steady_large_step_report.md` for background-independent errors.
Qualification is bounded startup, one-step/two-half-step comparison and
same-four-rank restart—not a guarantee of later event accuracy or spatial
convergence. The smaller ceiling increases possible long-run step count.
The final local comparison has 2.55/3.66/1.92/3.35% work-weighted errors relative
to evolving V/state/shear/normal changes. Endpoint maximum V and normal errors
are about 9% of global peak evolving changes; endpoint convergence remains
unqualified. The 125000-s fresh four-step startup was not repeated: this ceiling
was selected by the requested same-checkpoint comparison after the 4e6-s first
physical step, with smaller-step fresh/restart evidence retained separately.

## Build and run

Load the exact compiler/MPI/deal.II modules used for the server ASPECT build.
Use Release with `ASPECT_WITH_VORO=ON`. `manifest.json` records the source base,
patch and tested binary/plugin hashes. Reproduce source on a separate checkout:

```sh
git apply --check /path/to/job/source.patch
git apply /path/to/job/source.patch
tar -xzf /path/to/job/source-overlay.tar.gz
```

Build ASPECT using the server's normal configuration and `-j4`. In the copied
job directory, build the included plugin against that **same** build:

```sh
cmake -S plugin/bp5 -B plugin-build \
  -DAspect_DIR=/absolute/path/to/server/aspect-build -DCMAKE_BUILD_TYPE=Release
cmake --build plugin-build --target bp5_steady_initialization -j4
cp plugin-build/libbp5_steady_initialization.release.so ./
# Copy the matching server ASPECT executable here as aspect-release.
source ./environment.sh
python3 verify_package.py
./aspect-release --validate first_event.prm
mpirun -np 4 ./aspect-release first_event.prm
# Or submit on Stampede3; do not launch both:
sbatch -A YOUR_ACCOUNT first_event.slurm
```

No `ASPECT_SOURCE_DIR` lookup is needed. Inputs are relative to the job directory.
The plugin under `provenance/` is local provenance, **not a portable server
binary**. Do not load both old and new BP5 plugins. Keep compiler flags compatible
with the compiler actually selected; do not give Intel Fortran flags to gfortran.
The batch template requests one icx node, four ranks and 24 h, loads no guessed
module stack, sources `environment.sh` on the compute node and logs hashes.

The environment clears inherited `ASPECT_*` flags and enables only sparse B/G
and the tridiagonal inverse. Profiling, action comparisons, mechanical probes,
full-state audits and timestep-audit callbacks stay disabled. Normal fresh
linear/nonlinear residual checks, aging audit and lightweight accepted-state
diagnostics remain enabled. Do not remove physical assertions or enable
unsafe fast-math to work around mature-profile checks.

## Termination, output and checkpointing

After accepted max(V)>=1e-3 m/s first marks an event, five consecutive accepted
states below threshold end the run; another up-crossing resets the count.
The independent 1500-year safety end is **not** evidence of an event. The
84600-s graceful wall stop leaves margin in the 24-h allocation. Short-test
step/end limits are removed. No timestep is forced just for visualization.

Every accepted state retains solver diagnostics, station histories and cumulative
slip. Profiles use 0.01-m slip increments or one-year intervals; coordinated
bulk/particle/fault output uses 0.1 m or one year plus existing event triggers.
Expensive full-state auditing every step is off.

`output-30km-steady/` contains `accepted_steps.csv`, `state_startup_predictor.csv`,
`first_event.csv`, `stations.csv`, `cumulative_slip.csv`, profiles, visualization
and ordinary checkpoints. Checkpoints are requested every 3600 wall seconds
and on termination, only at safe accepted states. Preserve sidecars/metadata.

```sh
source ./environment.sh
mpirun -np 4 ./aspect-release resume.prm
# Or:
BP5_PARAMETERS=resume.prm sbatch -A YOUR_ACCOUNT first_event.slurm
```

After a crash, preserve outputs newer than the selected checkpoint and restore
its metadata/cumulative-slip prefix before resuming. Same-rank restart was
qualified; cross-rank restart and full-cycle performance were not tested here.
