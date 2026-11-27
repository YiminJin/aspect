# Historical 60° BP3-QD thrust on Stampede3

Start from zero slip and the 5,776 initial ages in `historical-vertices.txt`. Edit `production.prm` for the scientific inputs. The finite 150 × 50 km box solves friction along the entire 57.735 km fault. SEAS prescribes creep beyond 40 km. Here the plate boundary velocity is the integral of the transverse AT1 slip density. The top is traction-free. The native core supplies mature fault mechanics, aging and timestep selection. The extension supplies the historical spatial law, loading, mesh and observers.

## Build the matching software

Load the same Intel compiler, MPI, deal.II and dependency modules for the core build, extension build and job. Set `ASPECT_SOURCE` to the pf-rsf checkout containing core commit `807943f77`, `DEAL_II_DIR` to the matching MPI deal.II SDK, and `ASPECT_PREFIX` and `CORE_BUILD` to dedicated installation and build directories. Set `BP3_ZIP` to the absolute path of this archive and `TACC_ALLOCATION` to your allocation. Stampede3 supplies `SCRATCH`. Keep icpx and the finite-arithmetic compiler settings for both builds.

```bash
git -C "$ASPECT_SOURCE" merge-base --is-ancestor 807943f77 HEAD
cmake -S "$ASPECT_SOURCE" -B "$CORE_BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DUSE_MPI=ON -DCMAKE_CXX_COMPILER=icpx -DDEAL_II_DIR="$DEAL_II_DIR" \
  -DASPECT_ADDITIONAL_CXX_FLAGS="-fno-finite-math-only -fp-model=precise" \
  -DCMAKE_INSTALL_PREFIX="$ASPECT_PREFIX"
cmake --build "$CORE_BUILD" -j4 && cmake --install "$CORE_BUILD"
```

MPI support is inherited from deal.II. Verify that its installed configuration has DEAL_II_WITH_MPI enabled. The changed timestep policy requires the rebuilt core as well as the extension.

Preserve any previous package and output before unpacking. This creates a new working directory for the fresh run.

```bash
cd "$SCRATCH"
if test -e bp3; then mv -T --no-clobber bp3 "bp3.saved.$(date +%Y%m%dT%H%M%S)"; fi
test ! -e bp3 && unzip "$BP3_ZIP" -d "$SCRATCH"
cd "$SCRATCH/bp3/plugins"
cmake -S . -B . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_CXX_FLAGS="-fno-finite-math-only -fp-model=precise" \
  -DAspect_DIR="$ASPECT_PREFIX/lib/cmake/Aspect"
cmake --build . -j4
test "$(realpath aspect-release)" = "$(realpath "$ASPECT_PREFIX/bin/aspect-release")"
cd "$SCRATCH/bp3"
./plugins/aspect-release --validate production.prm
sbatch -A "$TACC_ALLOCATION" submit.sh start
```

The launcher runs on two ICX nodes with 80 MPI ranks per node and one thread per rank. It checkpoints and verifies step 1, resumes and verifies step 3, then continues in the same allocation. Startup checks use actual cell widths on the support and at each contact, accepted clocks, all material histories, native frames and three resolved slip-density cuts. The total mesh count is informational. The launch log records the binary revision, effective parameter files, hashes, paths and rank layout under `jobs/`. An exclusive lock prevents overlapping writers. Keep the original installation, inputs, jobs directory and output together.

## Time and output controls

The Maxwell initialization interval is 100 s. The first physical ceiling is 4e6 s, the subsequent ceiling 4e7 s and the growth cap 25%. For each identity with b > 0 and a ≤ b the slip budget in units of Dc is 0.5 a/b. All ages obey |Δ ln Θ| ≤ 0.2. The 0.99 margin applies to proposals. A solved trial uses the unscaled budgets. At 16 km a=b, so that station allows Dc/2. Deeper strengthening stations can slip freely while their state is resolved. ASPECT checks the bulk nonlinear residual at 1e−8 and solves the linear system at 1e−9.

Quiet bulk output uses the 100-year time cadence, a 1 m per-ID accumulated slip interval and ASPECT's step-count cadence. Full profiles use one year or 0.1 m. Both writers capture the accepted onset and completion. Peak and station statistics cover every accepted step. Onset is the first accepted peak ≥0.001 m/s. Completion is the fifth consecutive subsequent accepted peak below that threshold. The statistics record the two bracketing clocks for each milestone. A run also stops at 1,500 Julian years. During the bounded startup checks both writers emit every accepted step. Cadences do not change the physical timestep.

`shear_Pa` is mechanical shear positive up dip. The analysis tool subtracts ηrad V at the same material ID and clock to obtain frictional shear. It retains raw and friction-used normal compression in Pa.

```bash
cd "$SCRATCH/bp3"
python3 tools/tractions.py historical-output/fault-profile-0000001.txt > startup-tractions.csv
python3 tools/history.py historical-output/statistics accepted-history
# Optional full-field reconstruction requires NumPy 2+, SciPy and VTK.
OPENBLAS_NUM_THREADS=1 python3 tools/check_historical.py historical-output --through-step 3
```

## Resume this fresh run

A native wall-time criterion uses the remainder of the 48-hour allocation with 30 minutes reserved for checkpointing. Checkpoints also run every 1,800 wall seconds. Monitor actual solve and filesystem time and increase the reserve if needed. To stop at an accepted boundary, create `historical-output/terminate-aspect` and wait for the checkpoint. Preserve and move aside that file before resuming.

```bash
cd "$SCRATCH/bp3"
cat historical-output/startup-step-{1,3}.ok
sbatch -A "$TACC_ALLOCATION" submit.sh resume
```

A resume checks the run identity, hashes, native checkpoint and completed startup markers. The solver log and the completion marker identify the termination criterion. The historical donor's warm interseismic steps averaged 63.16 s on 160 ranks. Measure the current solver on the server before extrapolating a first-event date.
