# R6c: benchmark setup boundary assessment

## Decision and accepted baseline

**Retain the existing legacy completion integration. No production or plugin
source moves in this pass.** R6c explicitly requires retention when existing
extension points cannot preserve behavior. Its boundary assessment is complete;
no migration, new signal/accessor, parameter/default change or R7 is selected.

Accepted R6b: `00ad5ce1c`, immutable `build-refactor-r6b/aspect-r6b-qualified`,
SHA256 `cdf7e0c58009d11dfe8c216caeb31ba6d8edc08b71360fec9fb63fcd95287596`.
Before its commit, all 170 comparisons, eight source/protection checks and
independent/linked symbol checks passed again. Those 18 one/two-rank runs remain
numerical evidence for the identical source, not a newly executed R6c campaign.

Authority: [R6 §5](../../../doc/reconstructed_fault/refactoring/codex_R6_instructions.md),
[current design](../../../doc/reconstructed_fault/current_design.md) sections
on source continuation and normalization, and specification.tex's
Normalization-profile material mixture contract. No specification conflict was
found in the selected boundary. Legacy denominator-only input remains distinct
from paired source continuation and automatic prescribed completion.

## Current setup ownership inventory

The [rolling switch inventory](../refactoring_r6a/switch_inventory.md) retains
exact selector/default/consumer classification. This table adds setup boundaries;
BP3 use alone does not make an underlying mechanism benchmark-owned.

| Choice or operation | Present owner / dependency | R6c disposition |
|---|---|---|
| Fixed mesh, profile, initial fields, far-field velocity, phase constraints | Maintained `bp3/plugin/mesh.cc`, `bp3.cc`, `bottom_constraint.cc`, `monitor.cc`; existing mesh/initial-composition/boundary-velocity plugins and constraint signals | Already outside production; retain input files, collective validation and callback order |
| Initial Theta/V and nominal fixed background; later selector reattachment | `BP3Benchmark::prepare` and `initialize_mature_prestress` in maintained plugin; manager properties and material APIs | Already benchmark policy; initial writes stay separate from later/restart attachment |
| Prescribed slip rates and continuation selection | Reference `bp3/reference_200km/bp3.cc::prepare`; maintained plugin explicitly installs an empty prescribed map; both use existing manager/material setters | Keep distinct loading policies; M3 owns maps/source geometry, M4 owns normalization. Do not reintroduce reference prescribed rows into maintained fully frictional BP3 |
| Raw/Helmholtz choice and diagnostic windows | `plugin/monitor.cc::initialize` loads profile, then `post_simulator_initialization` attaches existing surface options | Already plugin-owned selection; filter application/factors remain M5 numerical machinery |
| Output schedules, cumulative slip, restart metadata, acceptance observations | `plugin/output.cc`, `work_audit.cc`, monitor and existing signals | Already plugin-owned output, with existing accepted-state timing; solver acceptance remains simulator-owned |
| Retained stress research | BP5 `moment_cycle.cc` installs `benchmark_retained_stress` at `post_set_initial_state`; plugin owns captured native data | Keep synchronous M4 history/M5 bulk-work and assembler reads; particle backend rejection remains. Not a passive output callback or generally qualified restart facility |
| Legacy completion environment selection/admission | M4 `normalization.cc::apply_boundary_normalization_completion`; automatic exclusion in `boundary_completion.cc` | Selected assessment below; retain core integration and exact compatibility behavior |
| Boundary contacts, automatic Q1 exterior integration, profile ownership/projection | M3 manager/contact utilities and M4 normalization/completion | General supported mechanisms remain production, including existing MPI and cache lifecycle |

## Selected boundary: legacy completion setup and application

`set_boundary_normalization_completion_file(path)` already exposes the supported
explicit immutable input. It does **not** represent all legacy environment
semantics. The legacy selectors are read by production even without a BP3 plugin.
Maintained BP3's execution-environment guard rejects their presence, including
empty or `"0"`; the historical reference plugin supports those experiments.
Moving the reader into maintained BP3 would therefore reverse its current role.

| Existing site / data and dependencies | Timing, restart and failure contract | Existing callback candidate / conclusion |
|---|---|---|
| Explicit setter: nonempty path, mature/frozen material; same-path repeat accepted, changed path rejected; first attachment invalidates the value cache | Benchmark preparation attaches before constitutive preparation and reattaches on resume; path/file are external input, not serialized history | `post_advection_solver` already supplies SimulatorAccess and prepared fault geometry. Keep the existing plugin attachment; no new move needed |
| Legacy path choice: explicit member path wins; otherwise getenv returns a filename, including empty/`"0"` | Only on legacy completion application on a cache miss; member-empty case rejects resume and requires presence of uniform sliding; dim2/mature/frozen/no-compare-cell checked before file read | Moving getenv to constructor or preparation changes read/failure timing. Passing it to the setter bypasses the member-empty legacy guard, changes empty-path behavior and adds setter invalidation/immutability behavior |
| File read/distribution, profile count/origins/outside data; actual owned profiles and in-box integrals | After integration and minimum-phi validation, before consistent Q1 projection. Read/distribute, local row validation, then MPI min of geometry agreement. Missing/malformed/count/origin data throw in current order | No existing setup/postprocess callback receives these private profile/integral arrays at this point. Earlier setup cannot validate actual profiles; later output has projected values and would miss the original data/capture point |
| Addition and `ih_bottom_completion_rankR.csv`, precision17, nine columns | Per owned profile: capture inside, add outside, write completed value. Truncate on each actual application; failbit/badbit exceptions stay enabled. Output can fail before/during additions; no new transactional claim | Postprocessing would change both capture and error timing. Do not expose private arrays or create a callback for relocation |
| Collective reuse and automatic completion | Automatic qualification/exclusivity precedes the collective hit decision; legacy cache hit reuses completed values and does not call application or re-read the file | Setup callbacks are neither this cache-hit boundary nor this collective sequence. Keep admission, invalidation and reuse ordering untouched |

The initial/setup alternatives have different contracts:

- `post_simulator_initialization` occurs at constructor completion. BP3 uses it
  to register properties, attach a later initial-history slot and configure the
  surface after construction. It does not supply integrated profiles.
- `pre_set_initial_state` actually passes the triangulation; fresh-only
  `post_set_initial_state` runs after initial fields/pressure, with particle and
  manager H slots preceding BP3's slot. It is not emitted on restart. Sharp
  reconstruction for the selected solver scheme follows phase evolution at
  timestep zero, before temperature/composition solves.
- Temperature `post_advection_solver` invokes BP3 preparation after that
  reconstruction and before coupled mechanics. Its connector is registered
  before the monitor's incoming-state observer. Existing restart preparation
  reattaches selectors without reinitializing histories. It precedes, rather
  than replaces, the material's actual integration/application site.
- `post_resume_load_user_data` handles restored cell data/cache invalidation;
  `post_resume_time_step` may only reduce the restored pending interval. Neither
  is a completion-input publication hook. Postprocessing is later than the
  required application, and is not an alternative owner of normalization.

No values, streams or callback references are moved, so their existing lifetime,
exception propagation, MPI participation and checkpoint contracts remain intact.
No new assertion or collective is introduced to enforce environment agreement.
As before, callers must supply consistent distributed inputs/settings.

## Future decision, not an implementation proposal for this pass

The smallest separate question is a **compatibility decision**: must the legacy
fresh-only denominator experiment remain usable without an extra plugin and
with the same read/failure timing? If yes, retain this integration. If explicit
migration is selected later, specify the adapter's loading requirement, legacy
restart rejection, empty-path handling, selector precedence, immutable-file
contract and migration tests first. The current setter alone does not preserve
that contract. Do not broaden it or add profile-access APIs merely to move code.
No interface redesign is required for the retained implementation.

For any future selected change, existing relevant checks are the legacy and
move-only comparisons in `refactoring_boundary`, its automatic-completion
rejection/derivative cases and restart branch, `refactoring_r2b_cache` hit/miss
and MPI cases, the maintained BP3 restart/prescribed-input regressions, and
`bp3/test_execution_environment.cc`. Additional focused coverage would need to
exercise legacy-selector empty/zero paths, restart rejection, explicit-input
precedence and file failures at the original cache boundary; current automatic
BP3 success alone does not establish those legacy behaviors.

## Verification and limits

This pass changes guidance/inventory/review only. `verify_preservation.py` checks
that production, public headers, unit/tests, benchmark code/inputs, CMake,
authoritative specifications, the R6 instructions and local tmp files retain
the captured hashes, except the explicitly updated rolling inventory. It also
checks all 865 qualified source entries and 40 executed R6b artifacts (inventory
excepted). The exact baseline and result counts are in `evidence/*.json`.

The existing `test_execution_environment.cc` was compiled with GCC 12.4.0 and
run unchanged: `BP3 clean/contaminated environment checks passed.` It covers the
current clean/contaminated guard and zero-valued selectors; no new parser or
production test hook was added. Commands and output are recorded in evidence.
Markdown links and git whitespace checks pass. No new numerical/MPI simulation,
new TU build, frozen probe, restart campaign or broader compatibility matrix was
run for a documentation-only assessment. Earlier scientific/cache/restart
limitations remain; no numerical/MPI defect was demonstrated or repaired.

R6c assessment is complete for review, with the production integration retained.
Recommended next bounded task: R7 core-integration inventory/proposal only,
using this retained-boundary decision and the accepted R6 source as reference.
Do not begin R7 or a legacy-switch migration without the user's selection.

## Accepted R6 closure

The user accepted this assessment and requested its commit as R6 closure. The
commit containing this entry records accepted R6c. R6a (`1a3b57eda`) and R6b
(`00ad5ce1c`) remain the source-extraction commits; the qualified R6b executable
above is the unchanged post-R6 numerical baseline. The production legacy
completion integration is intentionally retained. No R7 or compatibility
migration is authorized. Unrelated local tmp files remain outside the commit.
