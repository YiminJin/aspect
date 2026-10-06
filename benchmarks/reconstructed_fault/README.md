# Reconstructed-fault benchmarks and checks

Use [BP3's maintained entry point](bp3/README.md) for the current frozen-fault
model and [uniform shear](uniform_shear/README.md) for independent constitutive
verification. Run commands from the repository root unless an entry says
otherwise. Current source is the local post-R7 branch with the user's imported
CMake checks reverted in `f277a53ae`; R7 evidence remains qualified at its recorded
revision. R8 is deferred. Cleanup is not new production/scientific qualification.

| Retained entry point | Purpose and status |
|---|---|
| [bp3/plugin](bp3/plugin/README.md), [production PRM](bp3/production/README.md) | Maintained runtime and single `fault.txt`; buffered 60° boundary admission qualified, full production normalization/mechanics and Intel server run unqualified |
| [uniform_shear](uniform_shear/README.md) | Independent reference, pilot and spatial/time comparisons; `python3 -m unittest discover -s benchmarks/reconstructed_fault/uniform_shear -p test_reference.py -v` |
| [bp3_birth_completion](bp3_birth_completion/README.md), [bp3_local_tests](bp3_local_tests/README.md) | PD-reuse birth identification, buffered completion, velocity oscillation/refinement and transport/retry; preserve their recorded scientific limitations |
| [bp3_geometry](bp3_geometry/README.md), [bp3_runtime](bp3_runtime/README.md), [bp3_packaging](bp3_packaging/README.md) | Endpoint/filter derivative, single-file geometry, removal of runtime fixtures, output/restart identity; scripts/inputs remain at existing paths |
| [particle_lifecycle](particle_lifecycle/README.md), [frozen_particle_H](frozen_particle_H/README.md) | RNG/particle retry/restart, frozen newborn H and evolving-history guard |
| [particle_replenishment](particle_replenishment/README.md) | Retained independent transfer/reference tables, including the historical failed RNG gate; not a current-source qualification runner |
| [frozen_gmg_repair](frozen_gmg_repair/README.md) | Real AMG/GMG condensed-system/state-preservation check; exact historical mesh/fault/replay dependencies retained |
| [refactoring_r7b](refactoring_r7b/README.md), [post_r6_cleanup](post_r6_cleanup/README.md) | Final local numerical/build qualification; source-specific histories, not scripts to run indiscriminately on another revision |
| [bp5](bp5/README.md), normal-filter, performance/server_gmg, nonuniform shear | Retained research/manual deployment tools; unresolved findings and external inputs prevent automatic retirement |

The [dependency inventory](maintenance/dependencies.json) records shared source,
registered tests, optional CMake targets, parameter includes, Python imports,
environment-selected fixtures and checkpoint assumptions. The current
[switch inventory](refactoring_r6a/switch_inventory.md) stays at its canonical
path. Earlier R1–R6/Maxwell/restart helpers remain historical dependencies; they
may require their recorded executable and restoration of old output groups.

## Historical material and recovery

[Classification](maintenance/classification.tsv) covers every one of the 3,397
tracked files in cleanup baseline `f277a53aeadbe99ee093d5eb5421680dad2055d4`.
1,184 completed logs, expanded PRM copies and exports are untracked, with exact
local copies retained for existing comparison paths. Compact qualification
summaries, numerical inputs, deliberate references and all R7 evidence remain
tracked. No checkpoint, executable, current output directory or old archive is
removed. This reduces the tracked tree, not the Git history or local disk usage.

| Tracked benchmark tree | Files | Uncompressed blob bytes |
|---|---:|---:|
| Actual post-R7 baseline `f277a53ae` | 3,397 | 78,793,540 |
| Closeout staged tree (pending review) | 2,221 | 52,531,009 |

The final row includes the new inventory/recovery tools and the exact previously
ignored R1 profile reference. Incoming untracked server packages and ignored
local outputs/builds are not counted as tracked files.

Recover retired tracked files into a new directory without overwriting a run:

```sh
python3 benchmarks/reconstructed_fault/maintenance/recover.py \
  benchmarks/reconstructed_fault/bp3_runtime/results/staggered \
  --destination /tmp/bp3-closeout-recovery
# Inspect the preview, then repeat with --write.
```

The exact baseline commit supplies old code/configuration/committed evidence.
The ignored local archive `.benchmark-cleanup-20261006-closeout/` also contains
`retired-tracked.tar.gz` (verified 1,200 members, including 16 replay inputs
ultimately kept tracked), `incoming.tar.gz` and its checksum manifest, and the
verification logs. These are recoverable local copies, not off-machine backups.
The incoming archive preserves the full pre-shortening BP3 README and all local
edits; Git alone does not contain those edits or qualified binaries. To inspect
that archive, extract into a **new empty directory**, not over this worktree.

For raw outputs previously archived on October 2, follow the existing
[CLEANUP.md](CLEANUP.md) and `.benchmark-cleanup-20261002-refactor/restore.py`.
Do not mix partial checkpoint metadata with another run. Restore complete groups
and use their recorded source/plugin/configuration and MPI count. Old reports
remain immutable historical records; their log/result links may require this
recovery even where a local copy happens to remain available.

## Closeout checks and limitations

[Verification](maintenance/verification.json) records fresh configure/build,
parser/smoke, independent reference, dependency/syntax, recovery and preservation
checks. The maintained BP3 target is built from scratch with local diagnostics
OFF. The optional aggregate `BP3_BUILD_TOOLS=ON` target exposes a pre-existing
failure: `test_bp3_restored_model` lacks ASPECT include setup after the geometry
header gained an ASPECT dependency. Its target/source are retained; no interface
repair is bundled here. `test_bp3_environment` builds and passes independently.

Five historical PRM includes remain unavailable at this checkout's documented
working directory (listed in the audit): old `first_cycle_coarse/original.prm`,
BP5 `dc010-ell100/fixtures/candidate/case.prm`, deployment `production_input.prm`,
`cell_actions_one.prm` and `bound_tip_smoke.prm`. These were absent before cleanup;
no dependent historical run is claimed to pass. Incoming revised/server PRMs and
packages, BP5/normal-stress research, large immutable meshes and chained stage
helpers are deliberately retained. Intel, full Debug/3D, production first event,
known cohesive nonconvergence and asymmetric cache-entry limits remain open.
