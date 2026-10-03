# Section 4: maintained BP3 inputs and outputs

Section 3 was committed first as **`e2ff248fb`**. This pass is implemented and
verified for review; its changes are uncommitted. It does not fix or bypass the
Section-3 graded-boundary completion failure or start Section 5/server execution.

## Delivered

- [Fresh 150×50 km candidate](../bp3/production/bp3_fresh.prm), its sole data file
  [fault.txt](../bp3/production/fault.txt), and [settings/provenance](../bp3/production/README.md).
  Resume is false, regular seeding is 4×4, limits are 12–24, effective native
  H/Maxwell limiting is derived by name, completion is automatic and profile/mesh
  preparation is live. Production ell=20 and a=0.010,0.025 are retained.
- Updated maintained entry points and the single plugin package's responsibility
  guide. Historical PRMs/fixtures and all prior evidence are preserved in place;
  table-based inputs remain tied to their original plugin. Removed the unused
  `Mature prestress file` declaration, state and empty-only checks. The BP5
  checkpoint rejection guard remains a necessary compatibility check.
- Resolved configuration report, printed at initialization and written to
  `bp3_resolved_settings.json`. Model identity v4 covers actual geometry,
  material/profile, loading, mesh policy, composition/discretization, native
  population controls, effective limiter and filter. It uses conservative exact
  parameter strings. Paths and output schedules are excluded; timestep/solver
  controls remain caller-adjustable and explicitly reported. Older identity
  versions require their original plugin; no physical archive conversion occurs.
- Optional native bulk/particle visualization on the existing heavy schedule.
  The historical default stays true; the new candidate explicitly selects false.
  Checkpoints, accepted summaries and fault profiles remain. Existing schedule
  references, accepted-history v6 serialization and physical/RNG owners are intact.
- Compact `particle_summary.csv`: accepted global counts, surviving births since
  backup, incoming-population removals/outflow, H range and signed stress-component
  extrema. MPI migration cancels in global totals. Step-zero event counters are
  zero, with the realized population reported. Two transient diagnostic scalars
  refresh at the existing backup, adding no checkpoint state. Metadata copies and
  prefix checks include the new CSV; no cumulative per-step slip table is added.

## Settings requiring review

No latest server input was identified. The candidate labels inherited historical
150×50 km settings and **provisional** choices: unweighted log-state bound 0.1
(not silently 0.2) and nonlinear-failure cutback 0.5 (historical input aborted).
This does not claim either choice was used in the last server run. Maximum dt
4e6 s, first dt100 s, and nonlinear/linear tolerances 1e−8/1e−9 are inherited.
The native default relative-increase setting resolves to 91.0; it is printed,
not replaced with a guessed server value. No additional repeat-on-cutback model
is selected. See [exact explicit changes](results/production_changes.json) and
[default-expanded values](results/production_resolved.json).

## Verification

**211/211 checks pass** in [checks.json](results/checks.json). Evidence includes
exact commands/environment/input copies, Release plugin/observer build logs and
[artifact hashes/run inventory](results/summary.json). Local runs total about
102 seconds, excluding compilation. Compiler GCC12.4, OpenMPI5.0.6, deal.II9.6.2;
core executable remains the accepted filter-derivative binary.

- Production PRM validation and default expansion pass without constructing the
  production mesh. All current source translation units compile for registered
  dimensions; runtime tests are 2D only.
- Six matched one/two-rank cases reproduce Section-3 accepted fields, bulk and
  particle dumps, histories, weak/work samples, solver decisions and lifecycle
  records **bit for bit**. No tolerance was loosened. Cases include actual
  nonzero-history births, serial rejected/direct retry, and two-rank restart.
- Independently reconstructed particle IDs and values match every compact count,
  birth/loss and H/stress extreme. The two-rank crossing case has counts
  30000→30122→30124, with 122 births/0 losses then 123 births/121 losses.
- Quiet visualization/cadence produces identical accepted physical/lifecycle
  results and valid checkpoints. No native bulk/particle payloads, detailed
  per-step dumps or cumulative-slip table appear; initial/final profiles remain.
- Restart with changed profile cadence and visualization selection matches the
  uninterrupted final state and summaries. The actual `branch_output.sh` helper
  successfully branches an older checkpoint from a parent containing newer
  output; its summary prefix and accepted history resume correctly.
- Five two-rank restart checks explicitly reject changed Dc, mesh level,
  population bound, native interpolation/limiter selection and filter length.
  Output-path/cadence changes remain admissible. Exact spelling comparison may
  conservatively reject equivalent parameter spellings; this is documented.
- Core source/headers/tests, historical runtime evidence/fixtures and all reported
  unrelated user files are unchanged. No scientific-worktree files were modified.

The initial build found a remaining obsolete prestress assertion; removal fixed
it. Test-harness import/syntax issues were corrected before their affected runs;
no simulation failure was hidden or physical setting adjusted for a pass.

## Remaining boundary

**Not ready for production/server execution.** The generated resolved mesh still
conflicts with automatic completion's global maximum-cell-width support padding.
The accepted-plugin reproduction and 60°/45° evidence remain in
[Section 3](../bp3_runtime/README.md). No new full production inventory, resolved
coupled trajectory, graded-interface transport, Debug/3D runtime or long/event
campaign is claimed here. Estimates based on the old production mesh are labeled.

Recommended next bounded task: separately design and qualify a conservative
local-support completion enclosure, preserving source/overlap admission, before
attempting the remaining resolved-mesh qualification. Section 4 alone does not
select that numerical correction or Section 5.

## Reproduce

The production README contains the exact plugin build and parse-only commands.
Build the observer with:

```sh
cmake -S benchmarks/reconstructed_fault/bp3_packaging/plugin \
  -B benchmarks/reconstructed_fault/bp3_packaging/build/observer \
  -DAspect_DIR="$PWD/build-refactor-r6b" -DCMAKE_BUILD_TYPE=Release
cmake --build benchmarks/reconstructed_fault/bp3_packaging/build/observer -j 3
python3 benchmarks/reconstructed_fault/bp3_packaging/run_cases.py model-only 1
python3 benchmarks/reconstructed_fault/bp3_packaging/compare.py
python3 benchmarks/reconstructed_fault/bp3_packaging/summarize.py
```

Reruns require new isolated output paths/log labels; existing evidence is not
overwritten. Inputs are standalone except the deliberate production parse/JSON
wrappers. Preparation scripts are test conveniences and not model dependencies.
