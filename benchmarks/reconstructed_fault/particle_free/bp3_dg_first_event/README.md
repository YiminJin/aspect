# DG BP3 first-event server package

Use **unlimited domain-free DGQ2**, frozen prescribed phase, fixed 150×50 km
geometry/graded mesh and stationary history. This prepares the first-event
experiment requested by the user; it does not establish first-event accuracy.
Particle controls remain comparison evidence, with neither backend an accuracy
reference. The prior spatial differences, endpoint temporal screen failure,
49.704 Pa surface-state interpolation imbalance and
`strict_reference_match_passed=false` remain documented in `qualification/`.

## Install

Extract the archive in a staging directory, enter its top-level directory and
verify the source/fixture checksums. Copy into **fresh** destinations:

```sh
sha256sum -c SHA256SUMS
mkdir -p "$WORK/build/aspect/pf-rsf-dg/source" "$SCRATCH/bp3_dg"
cp -a aspect-source/. "$WORK/build/aspect/pf-rsf-dg/source/"
cp -a bp3_dg/. "$SCRATCH/bp3_dg/"
```

The archive contains ASPECT build sources, unit/regression tests, standard data,
bundled World Builder/Catch and the actual maintained BP3 plugin/observer sources.
The small `bp3_dg/plugin/CMakeLists.txt` builds those sources directly from the
supplied ASPECT tree. No workstation executable/library, old output/checkpoint,
external profile table or Python run controller is required. All runtime paths
in the four standalone inputs are relative to `$SCRATCH/bp3_dg`.

```text
$WORK/build/aspect/pf-rsf-dg/source/   supplied ASPECT and maintained BP3 sources
$WORK/build/aspect/pf-rsf-dg/build/    ASPECT Release build
$SCRATCH/bp3_dg/plugin/build/         plugin objects and CMake cache
$SCRATCH/bp3_dg/plugin/*.release.so   matching runtime libraries
$SCRATCH/bp3_dg/plugin/aspect-release matching Release executable symlink
$SCRATCH/bp3_dg/parameters.prm        fresh first-event input
$SCRATCH/bp3_dg/resume.prm            first-event continuation input
$SCRATCH/bp3_dg/startup.prm           bounded fresh startup (step 5)
$SCRATCH/bp3_dg/startup-resume.prm    bounded continuation (step 8)
$SCRATCH/bp3_dg/fault.txt             native ordered 60-degree fault
$SCRATCH/bp3_dg/environment.sh        environment variables only
$SCRATCH/bp3_dg/output/               fresh native output/checkpoints
```

## Environment and build

Load the site's compiler/MPI/deal.II modules used by the server dependency
installation. Use the **same stack** when compiling ASPECT/plugins and in the
allocated job. Required: CMake, a compatible C++17 compiler, deal.II >=9.6 with
MPI/p4est/Trilinos-AMG, **SUNDIALS** and Voro++. These installed dependencies and any licensed
compiler are not bundled. `ENVIRONMENT.md` records the local qualification stack;
server module names/paths must come from that installation. World Builder is
already bundled; no submodule download is needed. Verify
`DEAL_II_WITH_SUNDIALS=ON` in the deal.II configuration (the native reconstructed
fault implementation needs its SUNDIALS interfaces). The deal.II 9.7 parameter
printer spelling `ParameterHandler::PRM` is included.

Set the matching dependency paths/compiler in your shell; for an IntelLLVM stack:

```sh
export DEAL_II_DIR=/absolute/server/dealii/installation
export VORO_DIR=/absolute/server/voro/installation
export CXX=icpx
export BP3_FP_FLAGS="-fno-finite-math-only -fp-model=precise -ffp-contract=off"
cd "$SCRATCH/bp3_dg"
source ./environment.sh
```

For a GCC/Clang dependency stack, choose its matching compiler and use
`BP3_FP_FLAGS="-fno-finite-math-only -ffp-contract=off"`. Keep floating-point
finite checks active. `environment.sh` preserves loaded modules, PATH and library
paths, removes presence-based ASPECT development selectors and sets one thread
per MPI rank. It does not configure, build, submit, run or restart anything.

```sh
cmake -S "$WORK/build/aspect/pf-rsf-dg/source" \
      -B "$WORK/build/aspect/pf-rsf-dg/build" \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$CXX" \
      -DDEAL_II_DIR="$DEAL_II_DIR" \
      -DASPECT_WITH_VORO=ON -DVORO_DIR="$VORO_DIR" \
      -DASPECT_WITH_WORLD_BUILDER=ON \
      -DASPECT_ADDITIONAL_CXX_FLAGS="$BP3_FP_FLAGS"
cmake --build "$WORK/build/aspect/pf-rsf-dg/build" -j 4

cmake -S "$SCRATCH/bp3_dg/plugin" -B "$SCRATCH/bp3_dg/plugin/build" \
      -DAspect_DIR="$WORK/build/aspect/pf-rsf-dg/build" \
      -DBP3_ASPECT_SOURCE="$WORK/build/aspect/pf-rsf-dg/source" \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$CXX"
cmake --build "$SCRATCH/bp3_dg/plugin/build" -j 4
```

Confirm CMake found Voro++ and kept `ASPECT_WITH_VORO=ON`. The wrapper builds
ordinary BP3 without local strengthening/accelerated-loading definitions and
links the observer against that exact library. CMake places runtime libraries
and the executable symlink in `plugin/`, leaving objects/caches in `plugin/build`.
No manual executable copy or launch wrapper is needed.

## Validate and run directly

On the server stack, run the unit suite and parser checks. These commands do not
run the first-event mesh. Parser checks cover declarations/basic values, not all
runtime cross-parameter restrictions.

```sh
cd "$SCRATCH/bp3_dg"
source ./environment.sh
./plugin/aspect-release --test
./plugin/aspect-release --validate parameters.prm
./plugin/aspect-release --validate resume.prm
./plugin/aspect-release --validate startup.prm
./plugin/aspect-release --validate startup-resume.prm
mkdir -p evidence
module list > evidence/modules.txt 2>&1
cp "$WORK/build/aspect/pf-rsf-dg/build/CMakeCache.txt" evidence/core-CMakeCache.txt
cp ./plugin/build/CMakeCache.txt evidence/plugin-CMakeCache.txt
sha256sum ./plugin/aspect-release ./plugin/*.release.so \
          parameters.prm resume.prm startup.prm startup-resume.prm \
          fault.txt environment.sh > evidence/server-artifacts.sha256
```

Supply your normal allocation, node/rank layout and scheduler wall limit.
Start with the bounded startup below. In the
allocated job, load the same modules and source the environment:

```sh
cd "$SCRATCH/bp3_dg"
source ./environment.sh
ibrun ./plugin/aspect-release startup.prm
```

There is no Python/shell build controller, launcher, automatic retry loop,
resubmission script or job submission in this workflow. Native ASPECT handles
nonlinear cutback and accepted-state checkpointing. MPI ownership/restart was
locally qualified on 1/2/4 ranks; larger server rank counts and performance remain
server checks. No allocation size or site-specific scheduler command is guessed.

## Bounded server startup

This server startup is **prepared, not executed**. Local MPI tests qualify the
small mature-DG fixture; they do not qualify full-grid memory, conditioning or
first-event accuracy. Use a fresh `output/` for `startup.prm`. It uses the exact
first-event mesh, physical seed, material/loading and adaptive controls. Step zero
is initialization; the stop is accepted step 5, with a checkpoint after every
accepted step and a 7200 s graceful wall budget. A wall/event/physical-end stop
can occur earlier; inspect `accepted_steps.csv` and the checkpoint label rather
than assuming five steps completed.

After a successful graceful startup, preserve the logs, accounting, parameter
files, diagnostic indices and complete `output/restart/` outside the live output
directory. Confirm the label in the directory selected by
`output/restart/last_good_checkpoint.txt` agrees with the last accepted diagnostic
step/time. Do not manually select the newest-looking partial checkpoint. Verify
all ranks passed the once-only publication checks and inspect authoritative
stress, unfiltered weak loads, state/slip and solver history. Then run:

```sh
ibrun ./plugin/aspect-release startup-resume.prm
```

It resumes the same checkpoint, retains the 7200 s budget/checkpoint-every-step
settings and stops at accepted step **8** (three more steps if the first launch
reached 5). It retains the original 47336400000 s physical end cap. Preserve
its evidence and confirm continued time, step, history/state and scheduled
outputs. The wall clock restarts at each process launch. Neither startup input
introduces a fixed 50000 s timestep cap.

After these checks pass, continue the accepted trajectory into the **unqualified
first-event experiment** directly:

```sh
ibrun ./plugin/aspect-release resume.prm
```

That removes the startup step bound, restores checkpoint-every-10 and the
82800 s graceful budget while preserving physical inputs and checkpointed event
and history state. `parameters.prm` is the alternative **fresh** first-event
launch (`ibrun ./plugin/aspect-release parameters.prm`), using an empty output
directory; do not run it over startup outputs.

Both wall budgets are checked only at accepted states. Choose a scheduler limit
with margin **beyond** 7200 s (or 82800 s) for an additional nonlinear solve,
terminal native/dense output and checkpoint write. An individual solve can
exceed the remaining wall budget; this is a graceful request, not a hard timeout.
Use startup measurements to choose that margin; no fixed margin is yet qualified.
Every rank uses the collective OR of all physical/step/event/wall stop requests
before output and termination.

## Experiment settings

| Item | Value |
|---|---|
| Geometry |150×50 km Box, straight 60-degree mature fault; native anchors/completion|
| Mesh |fixed graded grid, finest/coarsest 3.90625/1000 m, requested surface spacing 10 m|
| Estimated size |552084 cells, 8833344 stationary diagnostic points; not a new server measurement|
| History |authoritative independent unlimited DGQ2, all nine modes/component, full 3D plane-strain convention|
| Bulk |homogeneous G=32038120320 Pa, eta=1e26 Pa s; prescribed physical seed peak .6, ell=20 m|
| Mature semantics |zero cohesive resistance/storage; independent auxiliary Z=G*s_mem retained|
| Friction |a=.010→.025 across 15–18 km down dip, b=.015, Dc=.008 m; native initial state|
| Loading |Vp=1e-9 m/s, normal background 50 MPa, shear 26.546122365 MPa, full bottom/sides|
| Quadrature/filter |Gauss6 mechanics, Gauss3 native source, Helmholtz 20 m; no limiter/filter tuning|
| Physical timestep |native convection/RSF controllers, first cap 4e6 s, max 4e7 s, 91% growth cap, max log-state change .2|
| Initialization interval |4e6 s, separate from the first physical timestep|
| Rejection |native nonlinear-failure cutback, factor .5; rejected trials never publish|
| Checkpoint |every 10 accepted steps and on graceful termination; same-mode DG restart|
| Termination |native first event complete, 1500-year end cap, or 82800 s graceful wall budget|

The user explicitly selected native adaptive first-event controls instead of the
pilot's fixed 50000 s cap. The eight-step/400000 s pilot caps are removed in this
separate input. This is a new long experiment, not temporal qualification from
the short strengthening control. The 82800 s wall stop is checked at accepted
states; allow scheduler margin for a solve and final checkpoint.

First-event completion is authoritative in `first_event.csv`: onset at
max V>=1e-3 m/s followed eventually by five consecutive accepted states below
1e-3 m/s. A normal exit with `complete=0` remains an incomplete event run.

## Output and manual continuation

`accepted_steps.csv`, `coupled-checks.csv` (rank-prefixed under MPI),
`particle_summary.csv`, `stations.csv`, `coupled-profile-N.csv` and the root
`surface-weak-N.csv` retain per-step scalar checks, state/rate, auxiliary memory,
raw/filtered projections and actual weak loads/mass. Background subtraction must
use each native mass's own M*1 background. Bulk stress exports are accepted DG
reconstruction; fixed particles provide independent validation samples only.
They do not feed mechanics/materials/phase/geometry and require no domains/CPDI.

Dense `authoritative-stress-N.csv`, `dg-coefficients-N.csv`, `physical-fe-N.csv`
and normal cuts use the existing **Heavy output** intervals: 100 physical years
or 1 m maximum nodal slip change, with initial, onset/down-crossing/completion
and graceful-stop snapshots forced. `field-outputs.csv` indexes these snapshots.
Every tensor component, including zz and both xy norm contributions, remains
available. Exported stress is unfiltered; filtering acts only in the unchanged
native normal-feedback law. Native lightweight profiles keep their 1-year/.1 m
cadence. These are output settings only; all publication/recurrence checks and
physical history updates run at every accepted step. The default-off scheduling
option preserves prior short-test exports and checkpoints when unused.

Reserve substantial scratch for dense CSVs and native checkpoints: a full-grid
snapshot can be several GB and a first event can require many snapshots. The
prior 35–100 GiB memory range is an estimate, not a DG server measurement.
Follow `RESOURCES.md` during both startup launches, especially checkpoint writing
and reading; the current DG serializer replicates global cell records on every
rank. Increasing ranks does not eliminate that replicated memory cost.
No full fine-mesh first-event cost or conditioning has been measured locally.

After a graceful stop with a matching terminal checkpoint, continue manually:

```sh
ibrun ./plugin/aspect-release resume.prm
```

Use this package's own DG checkpoint, unchanged physical inputs and source.
The native first-event and diagnostic scheduling states are checkpointed. Do not
convert particle/FE checkpoints or resume an older pilot checkpoint with the new
scheduled-output option. For an abrupt termination, preserve the entire failed
output directory. If its CSVs/profiles are newer than the last good snapshot,
the native consistency check intentionally rejects an in-place continuation.
Prepare a fresh output branch containing a copy of the complete `restart/`
directory, set that output directory in a copy of `resume.prm`, and keep the old
output as the parent evidence. The native histories/event state continue; the
new branch's diagnostic series starts at the resumed accepted time. No failed
output or comparison guard needs deleting or relaxing. For example, after
archiving scheduler logs/accounting and hashing the failed directory:

```sh
# Keep output/ untouched as parent evidence. Use a fresh branch name.
mkdir -p output-branch
cp -a output/restart output-branch/restart
cp startup-resume.prm startup-resume-branch.prm
# Edit only Output directory in startup-resume-branch.prm to ./output-branch.
ibrun ./plugin/aspect-release startup-resume-branch.prm
```

The marker must select a complete valid checkpoint. If termination interrupted
checkpoint rotation, preserve all three slots and the marker; do not fabricate
or advance the marker to an incomplete slot. If no valid checkpoint exists,
restart fresh in a new empty output directory and retain the failed branch.
After a branch's startup checks pass, make a copy of `resume.prm` with the same
`./output-branch` directory for first-event continuation. Document checkpoint
parent/label, native accepted time/step, source/input hashes and scheduler job ID
for every branch. Never merge diagnostic CSVs across branches as though they
formed one uninterrupted output series.

Phase evolution, moving geometry, fault birth, transport and automatic AMR are
outside this stationary-history first-event experiment. Production defaults,
physical laws and old failures are preserved. See `VERIFICATION.md` for exact
local package checks and the inherited accuracy limitations.
