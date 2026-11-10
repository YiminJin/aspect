# Worktree storage cleanup — 9 November 2026

**Recovered 19.37 GiB of filesystem space.**
`build-particle-free` is reduced from 71.43
to 52.03 GiB. No scientific datum, checkpoint,
source version or runtime executable was discarded. No simulation was run,
terminated or resumed; no solver/physics/parameter change or commit was made.

Worktree `/home/ein/repository/aspect-pf-rsf-particle-free`, branch
`pf-rsf-particle-free`, HEAD `5f1fc6f3c656dfb7e1ea03e2a6507258929dbca2`.
The incoming synthetic-benchmark and specimen-assessment work, including
untracked inputs/reports/scripts, is preserved. No other worktree, shared
installation, source branch or symlink target was altered.

## Actual recovery

| Allocated usage | Before, GiB | After, GiB |
|---|---:|---:|
| Whole worktree | 79.26 | 59.87 |
| `build-particle-free` | 71.43 | 52.03 |
| Filesystem available | 1.90 | 21.27 |

Small differences between allocated-tree reduction and filesystem recovery are
within about .02 GiB of audit/other filesystem activity. Sizes count allocated
blocks and hard links once; moving data into another directory is not counted
as recovery. Final documentation adds only a few KiB.

- Removed **1006 regenerable object, dependency and precompiled-header files**,
  freeing 5.57 GiB. The current `bulk-tip/native` compiler cache is retained.
  Comparison build executables, plugins, CMake settings, generated headers,
  libraries and recoverable source identities remain.
- Reorganized **8624 completed dense observer exports** into **92 bounded,
  indexed archives**: 20.52 GiB of raw payload becomes 6.71 GiB. Every member
  was decompressed and SHA-256 checked before its original raw copy was removed.
  No field frames were sampled away or numerically recompressed. All numerical
  payloads remain exactly recoverable.
- Kept the current `synthetic-preflight03` output expanded. All checkpoint sets,
  original BP3/server families, inputs, logs, summaries, plots, scientific
  failures and protected runtime identities remain at their original locations.
  Hard-linked fields and symlink targets were excluded from field archival.

The main reduction is in `transported-objective`: about 25.6 GiB becomes
5.87 GiB in its original directory, with its large completed fields retained
in compressed form under `archive/`. Archives are grouped by experiment family,
roughly 256 MiB raw payload per shard, rather than one large opaque bundle.

## Retained large groups

Every current direct child over 500 MiB is listed below. The figures come from
one allocated-usage traversal; shared hard-link ownership can affect per-group
attribution. Previous catalogs were inspected, not treated as disposable caches.
No duplicate dense-field archive was found. Two small compressed-runtime aliases
(total 22.55 MiB) remain at the paths used by earlier recovery indexes, preserving
those catalogs and metadata without churn.

| Group | GiB | Why retained |
|---|---:|---|
| `archive/` | 17.245 | Verified compressed scientific exports, source-era/runtime recovery records and previous unique field archives. Includes the new 6.71 GiB catalog; contents remain available without rerunning experiments. |
| `transported-objective/` | 5.867 | Closed mature campaigns and Stage C controls: four completed A/B/D/C compressed result sets, complete restarts, protected executables/providers, small profiles/energy tables and exact inputs. Large raw rank fields are now archived. |
| `runs/` | 3.125 | Retained PF/P2/P3 verification and negative controls: smaller physical/tensor/velocity tables, frozen states, source/input/command records and plugin references. These support independent operator and representation comparisons. |
| `bp3-optimization/` | 2.933 | Protected BP3 executable/plugin controls, accepted/rejected/restart evidence, local audit records and server package. No original BP3/server output was pruned. |
| `p5c-campaign/` | 2.693 | Initial/admission/pre-rapid/final scientific history anchors, corresponding phase fields and summaries. Complete checkpoint dependencies remain; no additional generation pruning. |
| `bulk-tip/` | 2.427 | Current native build and its incremental compiler cache, protected BE/midpoint/Bernstein controls, failing-parent and restart evidence, current synthetic preflight fully expanded, plus small summaries and inputs. |
| `bulk-continuation/` | 2.019 | Retained stable/unstable, temporal/spatial/domain controls, relaxed-stability references, physical history anchors and analysis arrays; unresolved failures stay interpretable. |
| `coulomb-gradient/` | 1.115 | Negative constitutive/regularization controls with loading/hold/reversal, physical slip-length profiles and accepted histories. |
| `p5c-axial/` | 0.896 | Fixed/radial and normal-resolution comparison anchors, phase observations and each mesh initialization. |
| `cohesive-rsf-instability/` | 0.780 | Protected step748 continuation, predecessor/current restart generations and full retained surface event/failed-control evidence; distinct from the current bulk-tip formulation. |
| `mature-pilot/` | 0.689 | FE/DG/particle precursors, cutback/recovery and mechanical comparison inputs, selected fields and histories. |
| `cohesive-interface-turn/` | 0.683 | Retained fitted/reference curved mechanics, required operators, geometry/pressure/contact profiles and unresolved direction/tip evidence. |
| `cohesive-interface-propagation/` | 0.671 | Initial/final/current prototype kernels, accepted journals and material ledgers needed for historical surface-cohesive interpretation. |
| `bulk-source-history/` | 0.661 | Unique native/non-affine source-gradient and tensor-history comparisons, including the demonstrated projection limitation. |
| `cohesive-rsf-native/` | 0.650 | Preserved working historical executable/plugins and native lifecycle/pilot fixtures; compiler intermediates removed. |
| `bulk-stability/` | 0.599 | Relaxed tangent, unstable predictor and stable continuation/rejection controls; negative stability evidence is retained. |
| `bulk-cohesive/` | 0.524 | Initial energetic controls and native physical-history/phase evidence underlying the retained bulk formulation. |

## Recovery and active entry points

The current core remains `build-particle-free/bulk-tip/native/aspect-release`
(SHA-256 `5d8f9ab1adc94662f7b96a9b63267657a18bc0674dfb51693e763f5877c18e3e`).
The selected synthetic checker remains
`build-particle-free/bulk-tip/synthetic-checks/libbulk_tip_checks.release.so`.
The latest preflight input, eight accepted intervals, fields and restart slots
remain in `bulk-tip/runs/synthetic-preflight03/`. The synthetic specimen selection
and scientific limits in [its report](SYNTHETIC_PROPAGATION_BENCHMARK.md) are
unchanged; cleanup authorizes no growth run.

Historical BP3 and cohesive-native executables remain at
`build-particle-free/native/aspect-release` and
`build-particle-free/cohesive-rsf-native/native/aspect-release`.
The step748 RSF continuation and supplied server package are unchanged. Their
preservation does not equate that historical surface-cohesive law with bulk tip.

Newly archived run directories contain `ARCHIVED_FIELDS_20261109.md`.
Before running a raw-field analysis, preview exactly the prefix needed:

```sh
python3 benchmarks/reconstructed_fault/maintenance/disk_cleanup_2026_11_09/restore.py \
  build-particle-free/transported-objective/stage-c/material-runs/layered-bulk-plus
```

Add `--write` to restore under the original path, or use
`--destination /tmp/my-recovery --write` for an isolated copy. Changed existing
files are never overwritten. Individual file paths can be restored too. Runs
with older archival notices also require their older catalog for those older
files. Do not restore every archive merely to reproduce one comparison.

Only discarded compiler intermediates require rebuilding. Ordinary current
incremental builds retain their cache. Do not rebuild a frozen historical
executable against today's source to replace its preserved identity.

## Verification and audit

The [maintenance directory](../../../benchmarks/reconstructed_fault/maintenance/disk_cleanup_2026_11_09/README.md)
contains entry Git state/patches, full inventory, exact deletion reasons/paths,
archive/member hashes and recovery mappings. `verification.json` confirms:

- **197926 retained regular files** have identical hashes, modes and modification
  times; **302 symlinks** retain identical targets.
- All original archives and all new archive checksums pass.
- Three isolated single-file restorations (bulk fracture, bulk tip and mature
  material transport) reproduce original bytes, modes and times.
- Four current/historical executable/plugin dependency checks report no missing
  libraries. No runtime or historical numerical matrix was rerun.

After that exhaustive check, this report and maintenance banners were added to
current documentation; those deliberate annotations are recorded separately.
Original data losses remain **39/45 pilot frames, early temporary logs and the
missing old `bulk_continuation.zip`**. This cleanup introduces no additional
scientific loss and does not pretend to restore those unavailable records.

## Retention policy

Keep one active incremental core build. Preserve comparison executables and
matching plugins/source manifests, rather than all of their compiler caches.
Keep current checkpoints and near-term working outputs expanded. On closing an
experiment, retain compact histories/figures/input/provenance directly and
losslessly archive large diagnostic fields with a tested selective-restoration
index. Reuse existing archives; do not create duplicate full-run backups. No
change to numerical output frequency is made in this storage-only task.
