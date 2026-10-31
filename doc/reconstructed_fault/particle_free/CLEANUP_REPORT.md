# Cleanup and closeout — 30 October 2026

The cleanup was executed within the worktree-only deletion scope, with the legacy animation-data loss documented below. **38.22 GiB of filesystem space recovered.** No constitutive, solver, parameter or scientific conclusion changed; no simulation/server job was launched or terminated. No symlink target or other worktree was deleted. Useful pending implementation changes and this handoff are committed separately on the existing branch.

Worktree: `/home/ein/repository/aspect-pf-rsf-particle-free`; branch `pf-rsf-particle-free`; entry HEAD `f2f730529289eff2c5e8fbba09a224c9bd9a7247`. The older reviewed BP3 baseline is `b2dfdebe43437d6028b06d4f9f862cc80a00cc71`; history was not reset to it. Entry tracked/staged patches and the complete tracked/untracked status are retained in the [audit directory](../../../benchmarks/reconstructed_fault/maintenance/disk_cleanup_2026_10_30/).

| Allocated usage | Before (GiB) | After (GiB) |
|---|---:|---:|
| Entire worktree | 81.06 | 42.84 |
| `build-particle-free` | 74.15 | 35.87 |
| Filesystem available | 0.97 | 39.20 |

The worktree reduction is 38.23 GiB; filesystem recovery differs by only 2.10 MiB at measurement. There are no sizeable deleted-but-open files. Counts use allocated bytes, do not follow symlinks, and count hard links once. Later Git/audit metadata adds a few MiB. Logical path sizes are not claimed as recovered storage.

## What was removed or reduced

Sixteen explicit file batches record exact paths, pre-deletion sizes/inodes/hashes, reasons and retained replacements. The [compressed audit](../../../benchmarks/reconstructed_fault/maintenance/disk_cleanup_2026_10_30/audit.json.gz) carries the complete manifests; loose manifests and initial/deep inventories remain locally. No blanket clean/reset was used.

- 1.61 GiB of object/PCH/dependency intermediates; live executables, plugins, CMake configuration and generated headers remain.
- 6.49 GiB of obsolete rolling generations, followed by 0.24 GiB of old labelled local generations. The active RSF set was excluded. Another 0.97 GiB of superseded P5c operational waypoints was removed; scientific initial/admission/pre-rapid/final anchors remain.
- 6.76 GiB of unused linear field-response caches. They are derived operators, not accepted Maxwell histories. Initial/final/current propagation and fitted-reference dependencies remain.
- 5.14 GiB of superseded **local** BP3 quadrature audits; original server results are untouched. Necessary final qualification arrays were retained losslessly compressed.
- 2.93 GiB actually freed from repeated P5c mean/phase-force exports (8.19 GiB of logical paths included hard links), plus 2.11 GiB of repeated completed native bulk exports. Selected original fields and all compact plotted/history data remain.
- Duplicate native-control/fracture bundles released 3.14 GiB. Every necessary member is an existing exact file, a retained field-archive record, or the explicitly recovered unique old `native.prm`; obsolete generations are separately recorded.
- Thirty-eight field archives were checked against actual reference-archive members. Duplicate records were removed; unique records were independently verified in `.unique.tar.zst` replacements. Original SHA/member mappings survive in the recovery index. This is information-preserving deduplication, not a new backup of obsolete runs.
- Inactive runtime artifacts remain as exact compressed bytes, with modes and aliases indexed. Necessary local BP3 fields are also lossless. These representations cost less than their removed uncompressed copies.
- 0.26 GiB of expanded package copies was removed only after verifying every removed file against the retained complete package, including dirty historical source versions.

No unique source was lost: 40,681 source/document/configuration paths were checked; 4,157 removed duplicates or short generated labels are exactly recoverable. Before maintenance annotations, the tracked diff was byte-identical to entry. All 328 critical files and every RSF package checksum are unchanged. Five isolated single-file recoveries pass, covering compressed artifacts, unique archives, external reference archives, source-package members and literal metadata. Runtime dependency checks find no missing libraries. Existing zero-advance native restart evidence remains valid; no historical matrix was rerun.

## Retained substantial data

Every remaining direct child of `build-particle-free` over 500 MiB is listed here. Sizes are allocated and depend on first hard-link ownership; do not add per-family independently measured totals.

| Group | GiB | Contents and reason |
|---|---:|---|
| `archive/` | 10.53 | Unique dense scientific records; deduplicated field archives and compressed historical runtime artifacts. Supports retained stress/energy, FE/DG/particle, orientation, quadrature and failed-accuracy comparisons. Dirty historical runtime variants remain byte-recoverable. Duplicate records already in reference bundles were removed, not copied again. |
| `runs/` | 3.20 | PF0–PF6 and P2/P3 native controls: exact inputs, sparse tensor/physical states, remaining checkpoint fixtures and logs. Retains the all-mode, constrained derivative, source/projection, publication and negative-control evidence; historical executables are compressed separately. This is the reduced record, not a set of runnable full build trees. |
| `bp3-optimization/` | 2.93 | Current BP3 plugins, four final reference/optimized timing pairs, frozen operators, rejection/restart/wall-stop checkpoints and compressed full fields. Required BP3 implementation/regression reference. Original server data are separate. Superseded quadrature audits were deleted; necessary final arrays remain losslessly recoverable. |
| `p5c-campaign/` | 2.74 | Initial, intact-admission, pre-rapid and final-hold tensor-history anchors; selected phase/force frames and compact histories. Costly fixed/radial early-growth and focused replay results; preserves temporal/branch limitations. Shutdown/0.155 s/redundant replay waypoints were pruned. |
| `bulk-continuation/` | 2.02 | Stable/unstable hold and temporal/spatial/domain incoming states, relaxed-stability matrices and 22 plot arrays. Preserves the unstable-root diagnosis, remedy and still-failed accuracy/domain findings. Duplicate 2.43 GiB control archive was removed; its unique old input was rescued. |
| `coulomb-gradient/` | 1.12 | Joint-microforce loading/hold/reversal records and profiles at the tested physical slip lengths. Evidence of short-wave regularization and the incompatible accumulated-slip hardening; the retired model is retained as a negative constitutive control. |
| `p5c-axial/` | 0.90 | Fixed/radial 96/192-normal comparison anchors, selected initial/pre-growth/final field snapshots. Supports the measured refinement change and unresolved spatial interpretation, with each mesh’s own initialization retained. |
| `cohesive-rsf-instability/` | 0.78 | All surface event histories, selected old bulk frames, full current segment and current/failing checkpoints. Active step748 continuation plus step660 floor failure and b=a control. Current three-slot restart and full accepted local segment remain intact. |
| `mature-pilot/` | 0.69 | Small mature-DG/FE/particle, MPI/cutback and fixed-profile mechanical/timestep control data. Qualified local precursor to BP3; neither backend is treated as an accuracy reference. This is separate from first-event server observations. |
| `cohesive-interface-turn/` | 0.68 | Fitted/reference and selected turning operators, exact geometry and pressure/contact profiles. Independent curved-mechanics comparison and unresolved sided/ahead normal-reaction evidence. Unused field-response caches were pruned. |
| `cohesive-interface-propagation/` | 0.67 | Initial/final/current accepted prototype kernels, accepted journals, material ledgers and mixed tables. Reproducible integrated turning/mature-sliding reference and its essential restart dependencies; intermediate field-response caches are rebuildable. |
| `bulk-source-history/` | 0.66 | Distorted/rotated controls with native non-DG gradient histories, source comparisons and retained states. Unique evidence for the demonstrated non-affine representation limitation and its correction. |
| `cohesive-rsf-native/` | 0.65 | Working native ASPECT build/plugin, pilot/general-history/lifecycle fixtures and retained endpoint fields. Current native implementation and general tensor-memory/restart regression. Active portable workflow does not require historical prototype files. |
| `bulk-stability/` | 0.60 | Mechanically relaxed tangents, unstable predictor and stable continuation states, rejection/restart controls. Preserves the actual unstable-root evidence and the limited-scope safeguard verification; it is not a certificate for later non-associated laws. |
| `bulk-cohesive/` | 0.52 | Original fresh-seed controls, native history/phase snapshots and parameter/source identities. Retains initialization, energy-screen, finite-tip and branch findings underlying later physical-separation choices. |

The archive group was inspected member by member. The remaining substantial raw records are unique to retained numerical/constitutive comparisons. It also holds compressed historical executable variants whose exact dirty implementation should not be guessed from HEAD. These are evidence artifacts, not active build trees. Everything else under `build-particle-free` is below 500 MiB; the [full inventory](../../../benchmarks/reconstructed_fault/maintenance/disk_cleanup_2026_10_30/final-build-usage.tsv) records it.

Outside the authorized deletion scope, `benchmarks/` remains about6.6 GiB: the largest evidence groups are `pf6_mature_pilot` (1.05 GiB complete reference), `bulk_parallel` (0.81 GiB native/quadrature comparisons), and `cohesive_interface_evolution` (0.72 GiB trajectories/physical fields). The largest single file is `evidence/pf6_mature_pilot/reference.tar.xz` (1.05 GiB). These were reported, not deleted. Detailed outside usage is in `final-outside-detail.tsv`. Original `first_event`, `server_3610007`, `server_3613912`, raw BP3 logs and all supplied packets remain.

## Resume and recovery locations

| Purpose | Authoritative location |
|---|---|
| Current BP3 executable | `build-particle-free/native/aspect-release` (SHA prefix `d62541e3fc2a`) |
| Current BP3 ordinary/test plugins | `build-particle-free/bp3-optimization/server-validation/bp3_dg/plugin/`, `bp3-optimization/plugins/`, `bp3-optimization/wall/injected/` |
| Current cohesive native executable/plugin | `build-particle-free/cohesive-rsf-native/native/aspect-release` (`7277eebc6636`); `checks-native/libcohesive_rsf_native_checks.release.so` (`7bfbeaa35e7a`) |
| RSF continuation | `build-particle-free/cohesive-rsf-instability/runs/episode-local/output/restart/`: marker2, slot02 step748 at374.58003334330067s; preceding complete slot01 step747 at374.57999502565559s. Slot03 is also retained; four-rank ownership unchanged. |
| Complete RSF server package | `benchmarks/reconstructed_fault/particle_free/rsf_episode_server.tar.gz`, sibling checksum and extracted `rsf_episode_server/`. Hash prefix `2b1498acc446`; all three restart generations/source/launch/validation records retained. |
| Separate BP3 first-event source package | `build-particle-free/bp3-optimization/package/bp3-dg-stationary-optimized-20261017-r3.tar.gz`; apply the documented Maxwell/observer repairs for its historical scope. Current source and launch/repair inputs are under `benchmarks/reconstructed_fault/bp3_native/server/`, `particle_free/bp3_dg_first_event/`, and `evidence/server_3613912/`. |
| Scientific evidence | `benchmarks/reconstructed_fault/particle_free/evidence/`; current [RSF index](../../../benchmarks/reconstructed_fault/particle_free/evidence/rsf_episode/INDEX.md) and [research handoff](RESEARCH_HANDOFF.md) |

The local BP3 packets contain labels for server809/815, **not the complete server restart payload**. Its server location/required files are documented in the replay/repair README. No local cleanup can establish remote completion or manufacture missing rank files. All locally supplied restart-related records and local qualification checkpoints remain.

Older statements that every raw field/cache is extracted are superseded by this availability report. Locate retained files without changing any run:

```sh
python3 benchmarks/reconstructed_fault/maintenance/disk_cleanup_2026_10_30/recover_retained.py ORIGINAL_PATH
# Optional recovery to a NEW file; refuses to overwrite different contents:
python3 benchmarks/reconstructed_fault/maintenance/disk_cleanup_2026_10_30/recover_retained.py ORIGINAL_PATH --destination /tmp/recovered-file
```

The compact recovery index covers16,128 original paths. Source packages/reference bundles are local artifacts, not implied Git payload. Compiler intermediates require rebuilding when next needed. Historical runtime/plugin paths require indexed re-extraction; active ones remain usable. Removed linear response fields require regeneration from their preserved input/native/operator metadata before rerunning those old kernels. Intentionally pruned intermediate visualizations/waypoint checkpoints are not recoverable; use retained plotted data, scientific anchors and the explicit batch reasons. `PRUNED_RESTART.md` prevents treating an obsolete waypoint as a valid restart. No retained accepted history was projected into another representation.

## Legacy animation-data loss

The final figure-dependency audit found an error in batch09: native bulk frames
were pruned before matching the actual45-frame selection of the historical
`cohesive_rsf_native/native-propagation.gif`. Four missing CSV files were recovered
from byte-identical retained aliases. **78 of its90 original rank-frame CSVs are
still absent (39 of45 original frames).** The original GIF, all Gauss surface
histories, quantitative time/energy/front data, selected bulk snapshots and
initial/final/restart states remain. Earlier bulk-field animations using the same
pruned export families may also require replay; their PNG/GIF renderings remain.
The current RSF event-window plot/GIF uses complete retained surface data, and the
current accepted segment, checkpoints and all original BP3 server data are unaffected.

Exact re-rendering of the older native bulk-stress animation is therefore not
supported by the retained CSVs alone. This does not alter the recorded numerical
results, but it fails the requested preservation of every raw animation frame.
`legacy-animation-availability.json` records the affected paths and exact recoveries.
No lossy image reconstruction was substituted for scientific data. Recover from
an external original backup if one exists; otherwise a separately authorized
historical pilot replay would be needed. Cleanup did not launch that replay.

## Future retention policy

Keep exact inputs, source identity, commands, summaries, plotted raw tables, unresolved failures, and one current plus one useful complete fallback per active trajectory. Retain expensive paper-relevant states deliberately. Full-field exports should be event/snapshot selected; audit fields should be opt-in. Separate rebuildable caches from accepted histories. Close each experiment with a small evidence index, then remove superseded builds/operational copies. Never classify by age/extension/ignore status alone. Keep BP3 original results protected until scientific review; retain the one authoritative RSF handoff package independently.
