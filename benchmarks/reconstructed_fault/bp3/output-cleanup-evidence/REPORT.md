# Maintained BP3 output cleanup — 2026-09-26

Scope: restored `bp3/plugin`, production continuation PRM, readers and restart
branching. No core solver, timestep policy, particle interpolation, stress/state
update, physical production coefficient, filter, boundary loading or prestress
was changed. All pre-existing source and supplied results were preserved.
`before.tar.gz` and `before.release.so` preserve the starting runtime;
`supplied-preservation.txt` verifies the saved server AMG/GMG inputs and summaries.
`runtime-inputs-readers.patch` contains the isolated runtime/PRM/reader changes
against that starting state; its dry run passes (`patch-check.log`). Documentation
and focused tests are in the checkout, with test sources/PRMs also collected in
`validation-sources.tar.gz`. Do not apply this patch over the already edited tree.

## Environment consumer audit

Paths below are relative to the repository, unless prefixed `bp3/` (the parent
benchmark directory). A guard rejects **presence**, including value `0`.

| Variable | Executable consumer and purpose | Disposition in maintained plugin |
|---|---|---|
| `ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION` | `source/particle/integrator/rk_2.cc`: replaces transport dt with zero | Retain guard; live core changes transport |
| `ASPECT_IH_BOTTOM_COMPLETION_DIAGNOSTIC` | `source/material_model/phase_field_fault.cc`: legacy completion file fallback | Retain small guard for live legacy completion path |
| `ASPECT_BP3_UNIFORM_SLIDING` | Same material source: validates legacy completion mode | Retain with completion guard; current explicit file normally takes precedence |
| `ASPECT_FAULT_INTERFACE_MODES` | `source/simulator/reconstructed_fault_interface_preconditioner.h`: optional few-mode interface preconditioner correction | Retain guard; not friction or normal filtering; not reactivated |
| `ASPECT_BP3_TOP_SOURCE_EXPERIMENT` | `bp3/reference_200km/uniform_sliding.h`, legacy reference sources | Remove guard; not compiled/loaded by maintained package |
| `ASPECT_BP5_SHORT_TEST` | Legacy `bp3/work_replay.h` and dedicated BP5/test sources | Remove guard; no maintained executable consumer |
| `ASPECT_DISTURBANCE_EPS` | `bp3/disturbance_diagnostic.h`: imposed disturbance amplitude | Remove guard; dedicated diagnostic plugin only |
| `ASPECT_DISTURBANCE_CONTROL` | Same header: selects disturbance channel | Remove guard; dedicated plugin only |
| `ASPECT_DISTURBANCE_DT` | Same header: diagnostic timestep | Remove guard; dedicated plugin only |
| `ASPECT_BP3_LENGTH_COUPLED_DIAGNOSTIC` | Former maintained output selector; still legacy `bp3/bp3.cc`, launched by `length_coupled.py` building old `libbp3` | Remove maintained read; explicit diagnostics now suffice. Dedicated legacy launcher retained |
| `ASPECT_BP3_LENGTH_FULL_AUDIT_FROM` | Former maintained output selector; still legacy `bp3/bp3.cc` | Remove maintained read; use explicit full-state audit. No maintained launcher sets it |
| `ASPECT_FAULT_LINEAR_PERFORMANCE` | Core reconstructed-fault linear profiling | Allowed, unchanged |

No blanket ASPECT environment filter and no C++ environment unsetting were
added. The maintained library must be loaded alone, not alongside legacy BP3/BP5.

## Implementation and intentionally retained machinery

`work_audit.cc` now owns invariant/native/Maxwell auditing. Native weak traction,
stable-ID initial H, geometry and completed Ih checks still run every accepted
state. The first real step always samples particles and reports Maxwell
publication. Subsequent remote particle point evaluation and common-FE/parent-P0
replay run only for explicit diagnostics. No discarded background-vector solve
remains. Monitor profile loading/filter initialization and callback order remain
mandatory; incoming-Theta copies, detailed traction reconstructions and samples
are gated. Mesh verification remains mandatory; its dump is optional.

The replicated H baseline, all version-5 fields (including retired-writer
placeholders), BP5 checkpoint incompatibility check, and native-output completion
postprocessor remain. Their removal would weaken history auditing or restart
compatibility. Omega uses the live material Dc accessor; diagnostic windows use
the current weakening length/endpoints. Mature prestress is empty-only deprecated;
its empty PRM entries are removed. All restored PRMs have an empty prescribed
traction list, so only their unused zero-traction registration was removed.
Historical inputs using the parent registration remain intact.

Canonical scheduled profiles retain all vertices and full precision. No duplicate
every-step slip table is written. Committed slip still integrates/publishes at
every accepted physical step, with no initial increment. Production continuation
and defaults use 0.1 m / 31557600 s for profiles and heavy output; startup
comparison PRMs retain their explicitly dense diagnostics. Exported legacy slip
tables carry a saved-only provenance sidecar; readers do not infer velocity
across their missing states. Direct profile readers use instantaneous V.

Version-5 load retains physical history and last-output references but applies
parsed PRM intervals. Root filesystem failures propagate to every rank before
further collectives; schedules advance only after payload/index close succeeds.
Growth headers depend on file existence/size. Fresh nonempty histories and
newer/conflicting restart profiles fail. `branch_output.sh` restores a selected
metadata prefix and only indexed profile payloads, preserving the parent.
Native payload directories are copied once at branch creation, never at each
checkpoint. Old checkpoints lacking a growth snapshot start a new growth table.

## Executed validation

Release: local GCC 12.4 / OpenMPI 5.0.6 / deal.II 9.6.2, matching `build-tmp`.
Build logs: `build-final-headers.log`, `checks-build-io-final.log`.
`production-parse.log` parses the actual updated first-event input, overriding
only the local library path. Parsing is not a production startup qualification.

The specified existing small full restored fixture could not be located. The
new test-only `prepare_fixture.cc` uses the same chart/endpoint completion and
creates 1,875 square cells, 16,875 particles and 32 fault vertices. It uses 2 km
spacing, ell=4 km, Gc=2e7 (preserving degradation convexity), and uniform direct
effect 0.025 to avoid unresolved initial-Theta interpolation. Production inputs
are untouched. Both baseline/new versions use these identical fixture physics,
AMG, unlimited LLS, six 100-second physical steps, plus state zero.

The first exploratory coarse inputs failed the existing convexity and positive
Theta assertions before mechanics. Their logs/outputs remain in `before.log`,
`before-model.log`, `before-convex.log` and associated failed output directories.
No assertion was relaxed to make this fixture pass. It is a functional regression,
not scientific resolution qualification of the production model.

`comparison.txt` is produced by `compare.sh`:

- Baseline `before-uniform` versus `after`: byte-identical accepted times/counts,
  solver summaries, V/Theta/slip/tractions at every profile, stations, growth,
  event decisions and first-step audit.
- `diagnostics`: both diagnostic switches true, same exact physical observations.
  Ordinary output has no plugin raw/full-state/mesh dumps; first Maxwell and
  native/inert/Theta checks remain active.
- `thin`: time-triggered profiles 0/3/6; same exact physical sequence.
- `production-cadence`: ordinary 0.1 m/year settings, initial/final profiles only.
- `slip-trigger`: time trigger disabled, 2.5e-7 m slip trigger, profiles 0/3/6.
- `restart-v5`: load baseline plugin's version-5 checkpoint at state 4 with the
  new library; states 5/6 and saved histories exactly match uninterrupted output.
  Its newly created growth table has a header.
- `restart-thin`: branch state 4 between scheduled writes, restore index/payloads
  0/3, change interval to 100 s; append 5/6, retain exact physical history and
  no duplicate profile rows.

`test_output_schedule`, `test_first_event` and `test_execution_environment` pass.
These cover signed-slip spatial maxima, forced/time triggers, archive references,
onset/re-entry/down-crossing/five-state event completion and guard semantics.
There is no physical seismic event in the short stable fixture: event threshold
behavior is checked synthetically, not claimed as first-event qualification.
`test_output_files` passes with one and two MPI ranks: new/empty-file headers,
valid prefixes, future state and missing-payload rejection, collective write
failure and unchanged schedule. Two-rank execution needed local MPI sockets
outside the sandbox. `reader-tests.log` covers scheduled gaps/instantaneous V,
sparse legacy export, missing files, branch payloads and legacy launcher checks.
Python was used only to run these existing reader/test modules; fixture
generation, simulation orchestration and comparisons use C++/shell.

## Replicated audit map and measured output

`after/audit_map_size.csv`: 16,875 global entries on one rank; map object 48 B,
measured glibc heap allocation delta for a copy 1,079,328 B; serialization with
the actual ASPECT archive type 202,562 B. The allocator delta is implementation
specific, not process RSS; it excludes unrelated simulation memory. The map is
replicated, so each rank stores the global baseline, not its local particle
subset. The measurement observer is test-only and not compiled into production.
No full-server map memory measurement or new distributed-history scheme is claimed.

`sizes.csv` and `sizes.sh` count actual logical file bytes by category, including
three retained checkpoint slots. Profile counts below exclude the index file.
All rows use seven accepted states (zero plus six physical steps), 32 vertices,
and identical heavy outputs at states zero/six. Native completion diagnostics
from the core are grouped with required audits and deliberately retained.

| Configuration | Profiles | Profile files + index bytes | Dense slip bytes | Optional diagnostics bytes | Summary/audit bytes | Native bytes | Checkpoint bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 7 | 36,367 | 13,529 | 85,143 | 103,317 | 1,182,167 | 16,980,233 |
| New, same dense profile clock | 7 | 36,367 | 0 | 0 | 103,317 | 1,182,167 | 16,983,826 |
| New, 250 s profile clock | 3 | 15,165 | 0 | 0 | 103,317 | 1,182,167 | 16,983,235 |
| New, production profile clock | 2 | 9,866 | 0 | 0 | 103,317 | 1,182,167 | 16,983,324 |
| New, full diagnostics | 7 | 36,367 | 0 | 48,448,980 | 103,317 | 1,182,167 | 16,984,077 |

The 105-byte map-measurement report is listed separately as test instrumentation.
Logs/input copies are separately counted in `sizes.csv`. Checkpoints grow slightly
because they now preserve growth-summary prefixes and parent provenance, not
profile payloads. The tiny fixture's total storage is dominated by checkpoints
and native data, so these results do not imply a fixed full-server reduction.
Dense duplicate slip cost scales with vertices × accepted states; canonical
profile cost scales with vertices × saved states.

## Limits and next bounded task

No full first-event trajectory, server rebuild, solver performance change, or
scientific model redesign was undertaken. Existing server results were not
rerun. Before a production continuation, build this package with the matching
Intel 26 server stack and select a preserved filter20 checkpoint branch, keeping
its physical settings and desired timestep caps. A short startup/restart check
there is the next bounded deployment task; it is separate from diagnosing the
previously reported AMG/GMG performance difference.
