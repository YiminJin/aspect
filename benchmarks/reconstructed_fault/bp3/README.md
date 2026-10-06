# Maintained BP3 entry point

Use [production/bp3_fresh.prm](production/bp3_fresh.prm), its
[settings/provenance](production/README.md), and the single [plugin package](plugin/README.md).
The current candidate uses regular 4×4 / 12–24 particles, a live stationary
profile, generated refinement and automatic completion. Mesh A's retained
endpoint buffer passes admission on the actual 60° production mesh; full
normalization/mechanics startup and Intel server execution remain unqualified.

The incoming [Stampede3 handoff](production/stampede3/README.md), server packages
and `bp3_fresh_revised.prm` are preserved as local user work. The handoff remains
pinned to its recorded revision; it is not silently repackaged as post-R7.
The revised PRM has user-selected timestep/output settings and is not substituted
for the production template or declared newly qualified by this cleanup.

## Build, validate and launch

From the repository root, with the matching ASPECT/deal.II/compiler environment:

```sh
export ASPECT_SOURCE_DIR="$PWD"
cmake -S benchmarks/reconstructed_fault/bp3/plugin \
  -B benchmarks/reconstructed_fault/bp3/build-maintained-buffered \
  -DAspect_DIR="$PWD/build-refactor-post-r6-gcc12-unity" \
  -DBP3_LOCAL_OSCILLATION_TEST=OFF
cmake --build benchmarks/reconstructed_fault/bp3/build-maintained-buffered -j2
build-refactor-post-r6-gcc12-unity/aspect --validate \
  benchmarks/reconstructed_fault/bp3/production/bp3_fresh.prm
# Production execution is a separately selected scientific/resource task:
# mpirun -np N build-refactor-post-r6-gcc12-unity/aspect \
#   benchmarks/reconstructed_fault/bp3/production/bp3_fresh.prm
```

Preserve a qualified library before rebuilding its directory. The PRM names the
library path above and `production/fault.txt`; no historical profile, target mesh
or completion table is a runtime dependency. The top-level `bp3/CMakeLists.txt`
builds the same maintained target, with legacy/research/tools options OFF by
default. The closeout used a fresh isolated build, preserving existing binaries.
Its optional restored-model tool currently has the include-setup limitation
listed in the [benchmark index](../README.md); this does not block the default
plugin target.

Keep the physical/timestep/output values in the selected PRM. The production
README distinguishes inherited caps and provisional state-bound/cutback choices;
the user's revised PRM is a separate input. A zero native output interval does
not bypass BP3's heavy-output scheduler. See [runtime outputs](plugin/README.md)
for mesh/particle/fault/profile scheduling and checkpoint conventions.

## Checks, restart and analysis

- [Birth/completion](../bp3_birth_completion/README.md) and
  [local tests](../bp3_local_tests/README.md): particle ID reuse, mesh buffer,
  transport, oscillation and refinement checks.
- [Geometry](../bp3_geometry/README.md), [runtime](../bp3_runtime/README.md),
  [packaging](../bp3_packaging/README.md): endpoint/filter conventions, sole
  geometry input, output cadence and restart identity. Use their recorded
  inputs/observers; do not substitute production-sized runs for smoke tests.
- `bash benchmarks/reconstructed_fault/bp3/branch_output.sh PARENT ID NEW_DIRECTORY`
  prepares a new branch from a complete native checkpoint and its output
  metadata; it preserves the parent and copies required native payloads.
- `plot_fault_evolution.py`, `plot_cumulative_slip.py` and
  `analyze_long_run_fault.py` are retained CLI analysis tools; use `--help` for
  their run-directory arguments. Historical analysis scripts may require
  archived comparison payloads and their recorded deployment layout.

## Legacy dependencies retained deliberately

`bp3.cc`/`bp3_model.h` remain included by registered disturbance and length-scale
tests. `reference_200km/` and exact `fixtures/modified_bp3*` meshes are needed by
the repaired frozen-GMG/replay fixtures. Unit tests read
`fixtures/bp3_150x50/profile.txt`; it is not interchangeable with the convex R1
profile used by the runtime comparator. Old PRMs, R1–R6 dependencies and replay
clocks remain at their original paths. `bp3_150x50_first_event.prm` resumes an
old checkpoint and is not a fresh-run template.

The prior long narrative is recoverable from baseline `f277a53ae:benchmarks/reconstructed_fault/bp3/README.md`;
its incoming local additions are preserved in
`.benchmark-cleanup-20261006-closeout/incoming.tar.gz`. See the
[benchmark recovery instructions](../README.md#historical-material-and-recovery).
Archived reports retain their historical conclusions and qualification limits.
