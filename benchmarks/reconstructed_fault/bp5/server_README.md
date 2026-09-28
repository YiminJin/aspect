# Modified BP3 / BP5-friction: adaptive 30-km research case

Copy this **entire directory** to your server job directory. All runtime input
paths are relative to that directory. No source-checkout location, Git command
or `ASPECT_SOURCE_DIR` is needed to run it. This is modified 2-D BP3 using BP5
friction coefficients, not the official 3-D BP5 benchmark.

## Model and bounded qualification

The physical model is unchanged: 300 by 100 km box, 60-degree fully frictional
fault, weakening 0–30 km and horizontal 30–33 km transition; `a=0.004/0.04`,
`b=0.03`, `Dc=0.1 m`, frozen AT1 `ell=100 m`, mature C=0, split exact aging,
true normal stress and outer plate loading. The uniform effective background
is 26.546122365139291 MPa shear and 50 MPa normal; initial perturbation stress
is zero. Positive initial nodal state is solved once with projected material
and native work weights. Both endpoint source/normalization corrections remain.

The same mesh has 114,984 cells, 24.4140625-m near-fault spacing, maximum
12.5-km cells and 1,156 fault vertices. AMG, sparse B/G, pivoted tridiagonal
surface inverse, FGMRES and all solver/physical checks are unchanged.

**Artificial initialization remains 4e6 s, not elapsed physical time.** The
physical ceiling is also 4e6 s, but the selected `BP5 state startup` model
limits predicted weighted logarithmic state change to **0.02**, alongside
the existing convection/RSF restrictions. Its verified first physical step
is about **66.9642717091 s**, not 4e6 s.

See `evidence/startup_followup_report.md`: the 300/150/75-s differences contract
approximately twofold; four adaptive steps pass; same-four-rank restart gives
zero difference across 96 comparisons. These are short startup/restart tests,
**not** spatial or full-cycle accuracy qualification. The failed large-first-step
case remains documented separately. No long run was launched during packaging.

## Server build

Load the exact compiler/MPI/deal.II modules used for the server ASPECT build.
Use Release ASPECT with `ASPECT_WITH_VORO=ON` and the supplied source corrections.
`manifest.json` records the base revision and local binary/source hashes. Apply
both provenance artifacts to a **separate clean checkout** of that revision:

```sh
git apply --check /path/to/job/source.patch
git apply /path/to/job/source.patch
tar -xzf /path/to/job/source-overlay.tar.gz
```

Build ASPECT with the usual server configuration and `-j4`. Copy the server
executable into the job directory as `aspect-release`, or set `BP5_EXECUTABLE`
explicitly. Build the included standalone plugin against that **same** build:

```sh
# In the copied job directory:
cmake -S plugin/bp5 -B plugin-build \
  -DAspect_DIR=/absolute/path/to/server/aspect-build -DCMAKE_BUILD_TYPE=Release
cmake --build plugin-build --target bp5_initialization -j4
cp plugin-build/libbp5_initialization.release.so ./libbp5_initialization.release.so
```

The build path is needed only at compilation, not at job execution.
`provenance/qualified-local-plugin.release.so` is a local provenance artifact,
**not a portable server binary**. Runtime binaries intentionally are not
bundled at the job root: do not mix a server ASPECT with the local plugin.

## Environment and submission

```sh
cd /path/to/copied/job-directory
# Load the matching server modules first.
source ./environment.sh
python3 verify_package.py
./aspect-release --validate first_event.prm
mpirun -np 4 ./aspect-release first_event.prm
# OR on Stampede3; do not launch both:
sbatch -A YOUR_ACCOUNT first_event.slurm
```

The batch template changes to `SLURM_SUBMIT_DIR`, sources the environment on
the compute node, verifies inputs and logs binary hashes/modules. It requests
one icx node, four ranks and 24 hours. Supply the allocation and your actual
module setup; no module stack is guessed. Same-rank restart is qualified,
not cross-rank restart.

The environment script clears inherited `ASPECT_*` investigation flags, including
`ASPECT_SOURCE_DIR`, and enables only sparse B/G and the tridiagonal inverse,
with one thread per rank. Do not enable GMG, comparison callbacks, frozen probes
or short-test modes. Sourcing it starts no build or simulation.

## First-event termination, output and restart

An event starts when accepted max(V)>=1e-3 m/s. After onset, five consecutive
accepted states below that threshold terminate; a return to the threshold resets
the count. The 1500-year end time is a safety cap, not event success. A graceful
84600-s wall stop leaves margin before the 24-hour allocation; one unusually
long solve can still overrun.

Every accepted state writes diagnostics, stations and cumulative slip. Profiles
use 0.01-m slip increments or one year; coordinated bulk/particle/fault output
uses 0.1-m increments or one year, retaining established event/termination
triggers. Output does not force timesteps. Files under `output-30km/` include
`first_event.csv`, `accepted_steps.csv`, `state_startup_predictor.csv`,
`stations.csv`, `cumulative_slip.csv`, `profiles/` and ordinary visualization.

Ordinary checkpoints are requested every 3600 wall seconds at safe accepted
states and on termination. Preserve all checkpoint sidecars/output metadata.
Resume in the same job/output directory:

```sh
source ./environment.sh
mpirun -np 4 ./aspect-release resume.prm
# OR:
BP5_PARAMETERS=resume.prm sbatch -A YOUR_ACCOUNT first_event.slurm
```

If a crash left output after the last checkpoint, preserve it and restore the
selected checkpoint's `bp3_output_metadata` and cumulative-slip prefix first.
Never use failed in-memory state or reinitialize evolved histories. The unchanged
predictor appends its diagnostic stream; keep its existing CSV when resuming.
