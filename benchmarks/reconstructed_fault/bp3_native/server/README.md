# Native FE BP3: manual server build and first-event test

This is a **fresh stationary-history first-event test** on the fine full mesh.
It uses the section-8 candidate, not a particle restart. Spatial accuracy remains
provisional; the finer local profile check changed peak-scaled localization by
3.7%. The coarse full-geometry gate passed two positive steps, but this fine
mesh and a full first event have not been executed or qualified locally.

The source snapshot contains the current uncommitted FE changes on top of
`c33edaaeb145e5d3dce155bb815552ef80efc5f9`. `PROVENANCE.md`, the source diff and
`SHA256SUMS` identify the actual delivered files. Rebuild both ASPECT and the
plugin with the same server compiler/MPI/deal.II stack. No local binaries ship.

## Install the two directories

Extract the archive in a staging directory. From its top-level directory,
check its contents, then copy into **fresh** destinations:

```sh
sha256sum -c SHA256SUMS
mkdir -p "$WORK/build/aspect/native-fe/source" "$SCRATCH/bp3_native"
cp -a aspect-source/. "$WORK/build/aspect/native-fe/source/"
cp -a bp3_native/. "$SCRATCH/bp3_native/"
```

The resulting layout is:

```text
$WORK/build/aspect/native-fe/source/       ASPECT source, bundled World Builder
$WORK/build/aspect/native-fe/build/        your ASPECT build
$SCRATCH/bp3_native/parameters.prm          standalone fine-mesh fresh input
$SCRATCH/bp3_native/fault.txt               native ordered 60-degree fault
$SCRATCH/bp3_native/environment.sh          environment variables only
$SCRATCH/bp3_native/plugin/                 ordinary maintained BP3 C++/headers
$SCRATCH/bp3_native/plugin/build/           your matching plugin build
$SCRATCH/bp3_native/output/                 fresh run output
```

The input has no include chain, developer paths, external profile tables or
completion/mesh data files. All run paths are relative to `$SCRATCH/bp3_native`.
The packaged plugin C++/headers are byte-identical to the maintained sources.
Its CMake appends finite-check/precise floating-point compiler options only.
No local uniform-strengthening or accelerated-loading definition is enabled.

## Load dependencies and compile manually

Load the server compiler/MPI modules used by your installed deal.II. The module
names and installation paths are site-specific. The historical server handoff
used `intel/26.0`; this archive does not claim that module or its dependencies
are available on the current server. `environment.sh` preserves loaded modules,
PATH and library paths and does not load modules, compile or run anything.

Required: CMake, a compatible C++17 compiler, deal.II >=9.6 with MPI, p4est and
Trilinos/AMG, and Voro++. The local tests used deal.II 9.6.2, Trilinos 14.2,
p4est 2.8.7, OpenMPI 5.0.6 and Voro++ 0.4.6. Those installed dependencies and the
Intel toolchain are not bundled. World Builder and Catch sources are bundled.

For deal.II 9.7, the original archive needs the one-line
[compatibility patch](dealii-9.7-prm.patch): in
`unit_tests/phase_field_fault_maxwell.cc`, replace
`dealii::ParameterHandler::OutputStyle::Text` with
`dealii::ParameterHandler::PRM`. The deprecated alias was removed in 9.7.
Edit the source file, not the generated Unity file, and resume the existing
`cmake --build` command; no clean or reconfiguration is needed. This changes
only unit-test parameter printing. The original archive/checksums are preserved;
the patch is supplied separately. The corrected unit-test translation unit
passes local syntax compilation with deal.II 9.6.2; a 9.7 build remains a server
check.

Set these paths/compiler in your shell after loading the matching modules:

```sh
export DEAL_II_DIR=/absolute/path/to/the/server/dealii-installation
export VORO_DIR=/absolute/path/to/the/server/voro-installation
export CXX=icpx
export BP3_FP_FLAGS="-fno-finite-math-only -fp-model=precise -ffp-contract=off"
```

For a GCC or Clang dependency stack, use its matching `g++` or `clang++` and
`BP3_FP_FLAGS="-fno-finite-math-only -ffp-contract=off"` instead. These are manual
configuration choices; keep one compiler/MPI stack throughout both builds.

```sh
cd "$SCRATCH/bp3_native"
source ./environment.sh

cmake -S "$WORK/build/aspect/native-fe/source" \
      -B "$WORK/build/aspect/native-fe/build" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER="$CXX" \
      -DDEAL_II_DIR="$DEAL_II_DIR" \
      -DASPECT_WITH_VORO=ON -DVORO_DIR="$VORO_DIR" \
      -DASPECT_WITH_WORLD_BUILDER=ON \
      -DASPECT_ADDITIONAL_CXX_FLAGS="$BP3_FP_FLAGS"

cmake --build "$WORK/build/aspect/native-fe/build" -j 4

cmake -S ./plugin -B ./plugin/build \
      -DAspect_DIR="$WORK/build/aspect/native-fe/build" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER="$CXX" \
      -DCMAKE_CXX_FLAGS= \
      -DBP3_LOCAL_OSCILLATION_TEST=OFF

cmake --build ./plugin/build -j 4
```

Confirm ASPECT found Voro++ and kept `ASPECT_WITH_VORO=ON`; CMake otherwise
disables it when the dependency is missing. The plugin build automatically
creates `plugin/build/aspect-release` pointing to your `$WORK` Release executable,
so the requested launch path needs no wrapper or manual executable copy.

## Validate, record the build and submit manually

These commands parse inputs only; they do not construct or run the fine mesh:

```sh
cd "$SCRATCH/bp3_native"
source ./environment.sh
./plugin/build/aspect-release --validate parameters.prm
mkdir -p evidence
./plugin/build/aspect-release --output-json parameters.prm \
  | sed -n '/^{/,$p' > evidence/server-parameters.json
module list > evidence/modules.txt 2>&1
cp "$WORK/build/aspect/native-fe/build/CMakeCache.txt" evidence/core-CMakeCache.txt
cp ./plugin/build/CMakeCache.txt evidence/plugin-CMakeCache.txt
sha256sum ./plugin/build/aspect-release \
          ./plugin/build/libbp3_restore_150x50.release.so \
          parameters.prm fault.txt environment.sh > evidence/server-artifacts.sha256
```

The `sed` command removes ASPECT's leading shared-library announcement from
the JSON stream; it does not alter parameter values.

Supply your normal allocation, partition, node/rank layout and wall limit in
your own batch submission. In the allocated job, load the **same** modules and
run exactly:

```sh
cd "$SCRATCH/bp3_native"
source ./environment.sh
ibrun ./plugin/build/aspect-release parameters.prm
```

No job is submitted by the package. It contains no build/run controller,
automatic retry, continuation or resubmission script. ASPECT's source tree
retains its normal developer/test utilities; they are not part of this workflow.

## Resolved test settings and interpretation

| Setting | Delivered value |
|---|---|
| Geometry |150 × 50 km Box, origin x=-60 km, native 60-degree through-going fault|
| Fine/coarse cells |3.90625/1000 m; global level 9/minimum 1, native graded band/end buffers|
| Surface /mechanical quadrature |requested 10 m /q6 tensor-product rule|
| Size estimate |552084 cells, 8833344 initial particles, approximately 18–21 million total DoFs|
| Resource planning allowance |35–100 GiB, 2000–10000 s serial startup; estimates, not fine/server measurements|
| History |native accepted FE strain, homogeneous G=32038120320 Pa, eta=1e26 Pa s|
| Fixed inputs |frozen phase/geometry/mesh, ell 20 m, peak .6, Gc 1e5, cohesion 1e6|
| Friction |a=.010→.025 over 15–18 km down dip, b=.015, Dc=.008 m, native initial theta|
| Background/loading |50 MPa normal, 26.546122365 MPa shear, Vp=1e-9 m/s; zero initial bulk perturbation|
| Boundaries /normal feedback |full bottom and sides, natural top; Helmholtz 20 m|
| Physical timestep controls |convection and reconstructed-fault controllers; first cap 4e6 s, max 4e7 s, 91% growth cap, unweighted max abs log-state change .2|
| Initialization interval |4e6 s, separate from the first physical timestep|
| Solvers |block AMG, nonlinear 1e-8, linear 1e-9, at most 30 Newton updates; abort on failure|
| Particle lifecycle |stationary samples, 4×4 per cell, load balancing=none; no insertion/deletion|
| Output |compact accepted/station/particle summaries and fault profiles; bulk/particle visualization off; native memory statistics on|
| Stop |first event complete, 1500-year physical end cap, or 82800 s graceful wall budget|
| Recovery |resume=false; all checkpoint writers off; no automatic cutback|

The timestep caps and .2 state bound come from the latest intended local BP3
input, not the uniform-strengthening fixture. The coarse full-geometry gate
selected 1772.48/2166.15 s, with the .2 state limiter active. The fine mesh may
select different accepted intervals; do not force the local 200000 s schedule.

**Stationary-history restriction:** authoritative FE memory is fixed Eulerian,
with derived samples at unchanged particle positions. Moving particles, births,
deletions, remeshing and FE restart/conversion are unsupported. A graceful wall
stop, solver failure or scheduler termination before event completion cannot be
continued from a checkpoint with this model. Existing outputs remain evidence.
The 82800 s stop is checked at accepted states, not a strict within-step deadline;
allow scheduler margin for a solve and final output. No allocation size is guessed.

Native `first_event.csv` is authoritative for event completion: onset at
max V>=1e-3 m/s, then five consecutive accepted states below 1e-3 m/s. A normal
exit or reaching the wall/end cap with `complete=0` is an incomplete first-event
test. Inspect `accepted_steps.csv`, `particle_summary.csv`,
`first_update_maxwell.csv`, `bp3_resolved_settings.json`, `parameters.prm`,
`statistics`, fault profiles and the scheduler log. The ordinary plugin retains
native Maxwell/state/publication audits. MPI consistency and a completed event
would not by themselves establish spatial accuracy or physical validation.

Local package verification is recorded in `VERIFICATION.md`; server compilation,
fine-mesh startup and event evolution remain to be performed manually.
