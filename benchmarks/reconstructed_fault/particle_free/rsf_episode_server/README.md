# Native RSF episode continuation — server handoff

The estimator repair is qualified in the reported local scope. This package
continues the SAME physical case and accepted histories. The first acceleration
episode is **not complete**. Read `manifest.json` for accepted step/time/ranks.
No remote job was launched. BP3 is protected: use new source/build/run directories.
The retained [event PNG](analysis/event-window.png) and
[GIF](analysis/event-window.gif) show the incomplete local episode. The full
repair report and equations are in `doc/reconstructed_fault/particle_free/`
inside `source.tar.gz`.

## Build and environment

Local qualification used deal.II9.6.2, Boost1.84, p4est2.8.7.0, GCC12.4 and
OpenMPI5.0.6; see `environment.json`. Native checkpoint portability across library
versions is NOT assumed. Prefer the same checkpoint/archive-compatible versions;
verify the read-only restart before advancing. Build ASPECT and the plugin with
the same compiler, MPI and deal.II. deal.II must support MPI/Trilinos/p4est/zlib,
LAPACK and **SUNDIALS**; the local build also enables NetCDF and WorldBuilder.
The package contains source, not locally linked executables/shared libraries.

For example, after unpacking the handoff into a fresh directory under
`$SCRATCH/rsf_episode`, choose your server's deal.II and compiler:

```sh
export DEAL_II_DIR=/path/to/compatible/dealii
export CXX=/path/to/the/compiler/used/for/dealii
export RSF_SOURCE="$WORK/build/aspect/rsf-episode/source"
export RSF_BUILD="$WORK/build/aspect/rsf-episode/build"
mkdir -p "$RSF_SOURCE" "$RSF_BUILD"
tar -xzf source.tar.gz -C "$RSF_SOURCE"
cmake -S "$RSF_SOURCE" -B "$RSF_BUILD" -DDEAL_II_DIR="$DEAL_II_DIR" \
  -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_BUILD_TYPE=Release \
  -DASPECT_WITH_NETCDF=ON -DASPECT_WITH_WORLD_BUILDER=ON -DASPECT_WITH_LIBDAP=OFF
cmake --build "$RSF_BUILD" -j 4
cmake -S "$RSF_SOURCE/benchmarks/reconstructed_fault/particle_free/cohesive_rsf_native" \
  -B plugin/build -DAspect_DIR="$RSF_BUILD"
cmake --build plugin/build -j 4
ln -s "$RSF_BUILD/aspect-release" plugin/aspect-release
sha256sum -c SHA256SUMS
```

Use your allocation's four MPI ranks. Configure the scheduler/account/partition
in `submit.slurm` yourself; no partition availability is assumed. All runs below
are direct native launches from the handoff root. Keep thread counts1 unless
separately measured. `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and
`DEAL_II_NUM_THREADS` should all be1. `ibrun` is the requested launcher; another
MPI launcher must use the same allocated rank count and MPI stack as the build.

## Read-only restart and bounded startup

Preserve `initial-checkpoint/` as an immutable fallback. It contains full native
bulk DGQ2/FE histories, surface histories, geometry, identities, clocks and numerical
estimator metadata. No physical state is reconstructed from the plotted CSVs.
A four-rank local restart and exact restored CellId record checks were exercised.

```sh
mkdir -p zero-output
cp -a initial-checkpoint/restart zero-output/
export ASPECT_COHESIVE_WRITE_HISTORY_RECORDS=1
ibrun ./plugin/aspect-release zero.prm
python3 verify_startup.py checkpoint-reference zero-output --output zero-verification.json
```

This must restore the reported accepted endpoint without advancing time. The
native owner independently requires exact CellId/tensor/native-memory restoration.
The checker demands exact persistent records/IDs/geometry and reports reconstructed
stress differences. A nonzero derived-stress difference needs review, not automatic
acceptance; the checker does not waive native tolerances or repair basis mismatches. Stop on an archive,
configuration, history, source, contact or energy failure. Do not convert an
incompatible checkpoint or silently initialize fresh histories.

Then branch the SAME original checkpoint for a split and an uninterrupted control:

```sh
mkdir -p startup-output control-output
cp -a initial-checkpoint/restart startup-output/
cp -a initial-checkpoint/restart control-output/
ibrun ./plugin/aspect-release startup.prm
ibrun ./plugin/aspect-release startup-resume.prm
ibrun ./plugin/aspect-release control.prm
python3 verify_startup.py control-output startup-output --output startup-verification.json
unset ASPECT_COHESIVE_WRITE_HISTORY_RECORDS
```

Startup stops after4 additional accepted steps; startup-resume stops after8.
Control computes the same8 steps uninterrupted. A duplicate initial index entry
on restart is a read-only export, not a second publication; each row must still
have publications equal to its accepted step. Keep the CSV/JSON/log evidence.
The startup wall budget is15min per invocation. If setup exhausts that budget,
report it and increase ONLY the computational budget, not physical/numerical checks.

## Continue the present episode

After startup verification, promote that accepted checkpoint once into a new
output directory and use the native resume file on each subsequent allocation:

```sh
mkdir -p episode-output
cp -a startup-output/restart episode-output/
ibrun ./plugin/aspect-release parameters.prm
# Subsequent wall-limited allocations use the existing episode-output:
ibrun ./plugin/aspect-release resume.prm
```

The native standard wall criterion uses the collective termination manager and
checks accepted states. Its1.5h wall budget needs scheduler margin for another
solve, terminal output and checkpoint; the2h scheduler example is a starting
budget, not a guarantee for an unmeasured server. Checkpoint every accepted step.
The376s physical horizon and20000-step bound are observation/capacity guards,
NOT evidence of a finished pulse. Continue by checkpoint while the episode remains
active; diagnose a physical/numerical guard before changing any bound. Preserve
failed trials and the last valid checkpoint. An abrupt scheduler stop is not
arrest. Branch from a complete last-good slot into a new output directory; never
edit partial checkpoint contents. Preserve the original output and marker first.

The selected experimental estimator uses continuous-history error5e-4 and the
retained slip/age/damage increments. Raw/resolved rate changes remain diagnostics.
Minimum dt, material law, full pressure coupling, R, lifting width, quadratures,
limiter settings and acceptance tolerances are unchanged. A low-damage point is
never exempted. On a new floor failure inspect both `cohesive-error-estimate.csv`
and `cohesive-rate-failure.csv` before proposing a correction.

## Scientific endpoint and review

Use t_ref=374.5s explicitly for this retained trajectory. The first1e-3m/s rate
crossing was bracketed at [374.461102325,374.466166456]s. The observation criterion
for a completed first episode is: observed maximum >=1e-3m/s, current global rate
below10% of that maximum continuously for max(.005s,5Dc/V_peak). This criterion
is a diagnostic, not a phase mask or a physical speed validation. A shallow local
maximum alone is insufficient. Analyze after each checkpointed segment:

```sh
python3 analysis/event_window.py --run episode-output \
  --prefix analysis/event-gauss.csv.gz --output review --reference 374.5 --gif
```

CSV preserves native Gauss histories/material IDs; spatial interpolation in the
plots is visual only. Read the actual frame times; animation spacing is not
physical time spacing. Distinguish slip activity, partial damage, a=1 fronts and
new persistent geometry. The installed tied reserve does not count as fracture.
No waves or physical rupture-speed claim: bulk mechanics is quasi-static with
existing radiation damping. Retain old temporal/spatial/energy failures.

## Focused follow-ups prepared, not executed

`temporal.prm` branches the original packaged checkpoint into `temporal-output`
with quarter defect tolerance and half increment limits. Compare the completed
event over common times AND accumulated mature absolute slip/Dc. Earlier shared
histories retain their old temporal uncertainty. Do not reset them.

`fresh-coarse.prm` and `fresh-fine.prm` start freshly from the SAME continuum seed,
preload, histories and loading. Fine doubles both bulk directions128×96→256×192
and seed surface intervals96→192, halves future surface spacing, and preserves
all physical lengths/parameters and numerical lifting width. These are future
spatial inputs, not authorization to call a coarse-checkpoint conversion a fine
initialization. Run them only after the present episode supports the comparison.
Report first-threshold/peak timing separately from progress-aligned fields and
fronts. For fresh cases, `--threshold-reference` selects each run’s first accepted
1e-3 m/s crossing and reports its absolute bracket, avoiding an assumed374.5s onset. Two meshes do not establish a convergence order.

Collect actual cell/DoF counts, native peak_rank_memory_bytes, step times/iterations,
setup/history/surface and checkpoint timers, output sizes and scheduler MaxRSS.
Local baseline has12288cells/185667reportedDoFs, about498MiB/rank on4ranks, and
~8s warm accepted steps. Fine setup/memory/cost is UNMEASURED. Baseline dense
exports cost about2.5MB/step; fine can be roughly4times larger. Reserve sufficient
scratch (tens of GB for thousands of steps). Full checkpoints remain authoritative;
do not remove diagnostic particles or redesign storage here. No particle population
is used by this selected cohesive pilot. The optional test-only full-history CSV
should be unset for long runs after startup.
