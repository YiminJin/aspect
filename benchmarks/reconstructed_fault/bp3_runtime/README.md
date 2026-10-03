# BP3 runtime-cleanup section 3

**Implemented for review; not yet qualified for resolved graded runs.**
The maintained plugin no longer reads `completion.txt`, `profile.txt` or
`target_cells.txt`. No production mechanics, normalization, source association,
filter, particle/history or solver implementation changed. Section 4 production
packaging and later refactoring were not started.

Accepted work was committed first:

- `bc2b5eb64`: native prescribed BP3 geometry and shared initialization.
- `82a1daaa8`: endpoint association roundoff correction.
- `48f23492d`: symmetric filter traces and accepted qualification evidence.

The user instruction files and local `refactoring/tmp/` remain untouched and
untracked. Historical fixtures and qualified libraries remain in place.

## Changes

- Require core `automatic prescribed` completion, remove the maintained plugin's
  file selector and explicit top/bottom continuation calls, and require two
  qualified endpoint supports. Generic core legacy support is unchanged.
- Prepare a transient symmetric loading primitive from the live profile factory
  and `energetic_degradation`, with h=1/g−1. Material profiles/laws must agree.
  Gauss8 versus split-panel integration is checked to 1e−13 of the estimated
  integral, apportioned by interval length; cubic interpolation is checked at
  quarter/mid/three-quarter points to 2e−14 of that integral. Queries interpolate
  only and clamp exactly outside support. The discrete mechanical Ih is intact.
- Replace saved leaves with `BP3 fault support`: the requested conservative
  cell distance, fine band and exterior dyadic gradation. ASPECT's initial-global
  loop calls the criterion before particle initialization and allows it to
  unflag exterior cells. Use **Initial global refinement = finest level,
  Minimum refinement level = coarsest level, Initial adaptive refinement = 0**.
  For the old production size controls, this means 9/1/0; no production PRM is
  supplied in this pass. Native smoothing flags match the old preparation tool.
  Runtime checks cover square cells, coverage, global totals and resolution.
- Model identity now includes generated primitive data (v3); the derived cache
  rebuilds on restart. Checkpoint payload/history ownership is unchanged. Old
  plugin checkpoints require their original plugin; new-plugin restart is
  tested, not cross-version checkpoint migration.

## Verification

Release plugin and observer builds pass with GCC 12.4/deal.II 9.6.2. Commands
are recorded below; logs, exact run commands/environment and actual input copies
are in `evidence/`. Qualified executable and old/new library hashes are in
[summary.json](results/summary.json). The core executable is unchanged.

- **1,888/1,888** unchanged physical and exact replay checks pass in
  [comparison.json](results/comparison.json). Matched cases use the same MPI
  count: 60° on two ranks; 45° forward/reverse serial; actual crossing births,
  serial retry, two-rank retry and restart. Fields, membership and solver
  decisions agree within the original `5e-10*column_scale + 1e-22` comparator.
  Retry/restart bulk, particle, history, weak rows and accepted lifecycle hashes
  are bitwise identical within the new implementation. Both two-rank events
  change RNG state and the second inherits nonzero stress. Newborn H and outside
  Hc checks pass before the audit. Fresh residual gates pass unchanged.
- At 2,001 matched signed distances, generated versus legacy C differs by at
  most **2.89e−14**; phase differs by **6.37e−14**. Independent live Gauss12 checks
  give maximum C error **6.67e−16**. Symmetry, monotonicity, exact clamping and
  generic endpoint support checks pass. Completed Ih (steps 0/2) and all sampled source associations, phase, Ih and
  localization values (steps 0–2) match exactly. No analytic loading integral
  replaces mechanical Ih.
- `model-only.prm` runs accepted steps 0–1 with only the maintained plugin,
  executable and its fault data. No observer library or old data path is in its
  runtime dependency graph. Heterogeneous profiles and legacy completion are
  explicitly rejected.
- The affordable uniform 60° mesh remains exactly 1,875 cells/65,560 total DoFs,
  h=2,000 m, with unchanged resolution and field numbering. The independent
  generated-mesh check reports:

| Geometry / ranks | Cells | Cells intersecting profile | h range (m) | Area (m²) |
|---|---:|---:|---:|---:|
| 60° / 2 | 10,634 | 6,386 | 3.90625–125 | 2,000,000 |
| 45° / 1 | 13,178 | 7,936 | 3.90625–125 | 2,000,000 |
| 45° reversed / 2 | 13,178 | 7,936 | 3.90625–125 | 2,000,000 |

The 60° case has 99,621 Stokes DoFs (376,262 including all other fields).
All generated leaves satisfy the original gradation rule. Replaying the exact
60° leaves through the accepted old plugin gives identical mesh rows and the
same rejection below. This is a diagnostic legacy input only, not a new runtime
mesh dependency. Full 542,958-cell production inventory was not run.

## Blocking qualification finding

Automatic completion computes its support padding from the **largest projected
cell width anywhere in the mesh**, then demands a uniform endpoint boundary
lattice over the entire enlarged footprint. Here R=39.5292 m and the prescribed
fine band is 47.3417 m. With hmax=125 m, the 60° required boundary half-width is
242.813 m; the preserved fine-band half-width is only 54.6654 m. Both candidate
and accepted legacy plugin reject the same mesh with `Nonuniform or misaligned
boundary ghost-Q1 lattice`. The 45° forward/reversed meshes also fail this
unchanged admission check. No boundary rule, tolerance or grading was relaxed.

**Next bounded task:** assess a conservative enclosure using the actual local
Q1 profile support, prove it covers all nonzero contributions, and qualify its
boundary-lattice/source/overlap checks. This is a separate numerical correction,
not silently included in fixture removal. Until then, resolved coupled smoke,
graded-interface particle transport and server readiness remain unqualified.

Startup experiments also established why initial-adaptive passes are rejected:
the fixed-mesh birth observer misclassifies retained particles; the maintained
audit detects regenerated IDs with changed initial H; skipping intermediate
initial conditions instead triggers native replenishment on an empty cloud.
These failed configurations remain in evidence. Native initial-global tagging
prepares the complete geometry-only mesh without those intermediate populations;
no history audit or particle algorithm was changed.

The first 45° attempt omitted its observer registration and failed parsing;
linking the existing observer fixed the test harness. The first 60° comparison
accidentally compared one-rank arrays against a two-rank baseline; the matched
two-rank run passes all original checks. Neither was a numerical failure.
All local attempts total about 228 seconds of simulation, excluding compilation.
No Debug/3D runtime, long-loading, production inventory or graded transport claim.

## Reproduction

```sh
export PATH=/opt/gcc/12.4.0/bin:/opt/openmpi/5.0.6/bin:$PATH
cmake -S benchmarks/reconstructed_fault/bp3/plugin \
  -B benchmarks/reconstructed_fault/bp3_runtime/build \
  -DAspect_DIR="$PWD/build-refactor-r6b" -DCMAKE_BUILD_TYPE=Release
cmake --build benchmarks/reconstructed_fault/bp3_runtime/build -j 4
cmake -S benchmarks/reconstructed_fault/bp3_runtime/plugin \
  -B benchmarks/reconstructed_fault/bp3_runtime/build/observer \
  -DAspect_DIR="$PWD/build-refactor-r6b" -DCMAKE_BUILD_TYPE=Release
cmake --build benchmarks/reconstructed_fault/bp3_runtime/build/observer -j 3
python3 benchmarks/reconstructed_fault/bp3_runtime/run_cases.py model-only 1
```

The runner refuses existing log labels; use a new isolated output and label for
reruns. `prepare_restart.py` clones a completed checkpoint without overwriting
outputs. `compare.py` and `summarize.py` regenerate compact evidence from retained
local outputs. Preparation scripts produce test inputs only; runtime needs no
preparation script. Failed setup input snapshots are preserved alongside logs.
