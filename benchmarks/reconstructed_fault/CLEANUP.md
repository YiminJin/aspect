# Refactoring-worktree cleanup — 2026-10-02

This cleanup applies to **`/home/ein/repository/aspect-pf-rsf-refactor`**, at
`8b93d69fd`, not the separate scientific-test worktree referenced in older notes.
It reclaimed approximately **19.4 GiB**, increasing free space from **9.3 GiB
to 29 GiB**. No tracked source, fixture, scientific evidence or local user edit
was deleted. No scientific algorithm or qualification result changed.

| Material | Action |
|---|---|
| 2,656 old compiler objects, dependency files and precompiled headers (14.59 GiB) | Removed; rebuildable |
| 8,787 older raw output/checkpoint files (6.58 GiB) | Losslessly archived into 14 verified chunks (1.75 GiB) |
| `build-refactor-r6b/` | Entire current build preserved |
| All qualified executables and shared libraries in older builds | Preserved at their original paths |
| `particle_replenishment/`, `refactoring_r6b/`, `frozen_gmg_repair/` | Entire benchmark families preserved and directly usable |
| Sources, scripts, PRMs, logs, compact summaries, provenance and user documents | Preserved |

The benchmark tree decreased from about **11 GiB to 4.2 GiB**; combined
`build-refactor-*` directories decreased from about **24 GiB to 8.5 GiB**.
The local archive occupies about **1.8 GiB**, including its compressed inventory.
It is recoverable local storage, **not an off-machine backup**.

## Organization and recovery

All maintained source/input paths remain unchanged. Historical raw payloads
are centralized at the worktree root:

```
.benchmark-cleanup-20261002-refactor/
```

This archive covers older `output*` directories in refactoring stages R1–R6a,
`refactoring_boundary`, `restart_fix`, `restart_investigation`, and
`maxwell_cleanup`. Exact members and hashes are in `archives.json`; all original
and retained paths are in `plan.json.zst`. `family_summary.json` lists per-family
sizes. Earlier archives in the separate `aspect` worktree were not touched.

Preview restoration from this worktree root:

```sh
python3 .benchmark-cleanup-20261002-refactor/restore.py \
  benchmarks/reconstructed_fault/refactoring_r5b2/output-candidate-frozen
```

Add `--restore` to restore the selected original paths. Prefixes may name a
single file, run directory, or benchmark family. Python and `zstd` are required.
Restore a **complete run directory** before historical field comparisons,
visualization or checkpoint restart. Unarchived logs/metadata alone do not prove
that a checkpoint remains usable. Current R6b, replenishment and repaired-GMG
outputs need no restoration.

Restore checks compressed-payload and per-file SHA256, checks available space,
skips identical existing files, and refuses to overwrite differing files. A
1,611,610-byte particle CSV was restored and verified; repeat restoration and
refusal to overwrite a deliberately different temporary copy were both tested.
The temporary copy was then removed. See `restoration_test.json`.

## Build use and verification

Older build trees keep their configurations, generated makefiles, test inputs,
executables and shared libraries, but their next build must regenerate removed
objects/precompiled headers. Local `ARTIFACTS_AFTER_CLEANUP.md` notes mark those
trees. To reproduce a historical build, use its recorded source revision and
matching toolchain; building against the current checkout does not recreate an
older qualified baseline. The current R6b build remains intact for incremental
work.

Before removal, every archived member was decompressed and SHA256-checked.
After cleanup, **56,756 retained files were rehashed**, including source,
scientific evidence, qualified binaries and the active build; **488,096 other
retained build files passed size/mtime checks**, with their initial SHA256 hashes
also recorded. These checks preceded this cleanup-note update. Full results are
in `verified.json`; scripts and archive checksums remain with the archive.

The current qualified R6b executable retains SHA256
`cdf7e0c58009d11dfe8c216caeb31ba6d8edc08b71360fec9fb63fcd95287596`.
Its dependencies and the three matched BP3/replenishment plugin dependencies
resolve. The retained replenishment evidence checker passes unchanged, including
its explicit **failed server-qualification gate for RNG restoration**. No new
simulation or build was needed for artifact cleanup.

---

# Latest cleanup — reconstructed-fault artifacts, 2026-09-24 UTC

Lossless archival reclaimed **11.62 GiB (12.47 GB)** of actual disk space.
The active benchmark tree decreased from approximately **23 GiB to 4.6 GiB**.
No scientific evidence was permanently discarded and no source, fixture,
parameter, authoritative specification or numerical result was rewritten.

| Quantity | Result |
|---|---:|
| Archived generated files | 1,381 |
| Uncompressed archived bytes | 19,115,365,151 |
| Compressed payload bytes | 6,642,728,400 |
| Payload bytes reclaimed | 12,472,636,751 |
| Retained benchmark files SHA256-verified | 16,434 |
| Documents / pre-existing modified source files separately verified | 207 |

Archived material comprises older raw QP/particle CSVs, visualization and binary
exports, disposable checkpoint payloads, Python bytecode and large historical
cumulative-slip tables. It includes restored/copied outputs in BP3
`first_long_run/` and `length-scale-study/`, BP5's old loading/startup studies,
three large stress-cycle staging copies, and normal-stress A/B output.
Directory skeletons, reports, logs, summaries and small histories remain.
Large historical `cumulative_slip.csv` files now require restoration; this
supersedes earlier notes below saying every such file remains extracted.

Kept directly accessible:

- BP5 `moment-consistency/`, including failed controls and exact tested snapshots;
  `clean-stress-cycle/`; and `normal-stress-cycle/`.
- Maintained fixtures, server packages, meshes, source/plugins/scripts, all PRMs,
  reports, figures, provenance and tracked files.
- Existing build trees, to avoid an unnecessary rebuild for the next task.
- The protected BP3 accepted-step-11 mechanical-discrimination source checkpoint
  and `state-disturbance/reference32/restart/`.
- All existing archives. These were neither repacked nor deleted.

Documentation is only about 7 MiB. Its original paths remain intact to preserve
links and historical authority. A new [documentation index](../../doc/reconstructed_fault/README.md)
identifies the authoritative pair, historical evidence, and latest experiment.

## Recovery of the 2026-09-24 archive

The local archive (not an off-machine backup) is:

```
/home/ein/repository/aspect/.benchmark-cleanup-20260924-gTxCPW
```

It contains 27 `part-*.tar.zst` chunks, per-file SHA256 `manifest.json`, the exact
cleanup/restoration scripts, `completed.jsonl`, `verified.json`, and a successful
`restoration_test.json`. Each member was decompressed and hash-checked before
the unchanged original was removed. All retained benchmark files were checked
before this cleanup note was updated.

Preview restoration from repository root:

```sh
python3 .benchmark-cleanup-20260924-gTxCPW/restore.py bp5/output-normal-diagnostic
```

Add `--restore` to recover matching files at their original paths. A single file
or narrower directory can be specified. Python and `zstd` are required. Equal
existing files are left alone; differing files are never overwritten.

Restore complete output/checkpoint groups before visualization or restart.
Some older data were already archived by earlier cleanups: consult those
manifests as well. Surviving checkpoint metadata does not mean its entire
payload is still extracted. Existing analysis scripts expect their original
paths and may need restoration first; they were not modified to hide missing data.

Verification restored a 3,314,806-byte CSV with its original SHA256, repeated
restoration idempotently, and confirmed refusal to overwrite a deliberately
different disposable target. No ASPECT simulation was launched. The current
moment-consistency qualification remains partial; cleanup does not change it.

# Earlier cleanup — BP5, 2026-09-20

The new BP5 outputs, including both copied server results, were losslessly
archived with per-file verification. This reclaimed **12.73 GiB** and reduced
the active BP5 tree from approximately 21 GiB to 1.1 GiB. Source, fixtures,
server packages, compact histories/reports and the latest step-1 source
checkpoint remain in place. Earlier BP3 archives were not touched.

See [bp5/CLEANUP.md](bp5/CLEANUP.md) for retained evidence, checkpoint caveats,
integrity checks and restoration commands. The separate local archive is
`.benchmark-cleanup-20260920-bp5-nKnB1v` at the repository root.

# Generated-artifact cleanup — 2026-09-17

The latest cleanup includes the copied BP3 long-run output. It losslessly
compressed large generated artifacts, verified every archived file, then removed
those exact files from their original paths. No source, numerical configuration,
or tracked file was removed. Existing working-tree changes were preserved.

| Quantity | Result |
|---|---:|
| Active benchmark tree | approximately 39 GiB → 3.9 GiB |
| Archived regular files | 9,723 |
| Original archived bytes | 37,024,931,489 |
| Compressed payload bytes | 15,693,902,482 |
| Payload disk saving | 21,331,029,007 bytes (19.9 GiB) |
| Retained files verified unchanged | 10,622 |

Unlike the earlier relocations below, this cleanup reduces total disk usage.
The archive metadata adds a few megabytes to the compressed payload size.

## What remains accessible

- Maintained sources, scripts, parameter files, meshes, fixture inputs, reports,
  plots, provenance snapshots, and existing build directories.
- Native `accepted_steps.csv` and `cumulative_slip.csv` histories, including the
  copied server run, plus compact comparison tables and JSON summaries.
- The latest mechanical-discrimination, bulk-refinement, mode and velocity
  decomposition evidence under `bp3/first_long_run/`.
- The accepted-step-11 source checkpoint under
  `bp3/first_long_run/mechanical-discrimination-roundoff-clock/restart/`, and
  the fine evolving reference checkpoints under
  `bp3/first_long_run/state-disturbance/reference32/restart/`.
- Existing `fixtures/`, `reference_200km/`, `evidence/`, `investigations/`, and
  `server_info/` contents in BP3.

Archived payload includes raw visualization, large particle/QP/history tables,
binary diagnostics, large logs, and older/disposable checkpoints. In particular,
6,491,664,073 bytes were archived from `bp3/first_long_run/output/`; approximately
1010 MiB remains there, including native slip history and compact evidence.
Historical analyses requiring raw fields or checkpoints need restoration first.
Retained checkpoint metadata alone does **not** mean its raw checkpoint remains
present: consult the manifest before using an older checkpoint.

## Recovery of the 2026-09-17 archive

The archive is local and git-ignored, **not an off-machine backup**:

```
/home/ein/repository/aspect/.benchmark-cleanup-20260917-reconstructed-alLfUb
```

It contains 47 `part-*.tar.zst` chunks, a per-file SHA256 `manifest.json`, an
append-only `completed.jsonl` archive-checksum journal, `verified.json`, and the
exact selection/execution script. The manifest records source HEAD
`359ea223cd0fa088ba7c8b03338cc7fe19d4acdc`; retained working files were additionally
verified by their own hashes, not assumed equal to that commit.

Preview restoration of the copied server output, from the repository root:

```sh
python3 .benchmark-cleanup-20260917-reconstructed-alLfUb/restore.py \
  bp3/first_long_run/output
```

Add `--restore` to restore the matching files at their original paths. A narrower
directory or exact file path can be supplied; prefixes are relative to
`benchmarks/reconstructed_fault/`. Restoration requires Python and `zstd`, checks
free space and hashes, and refuses to overwrite differing files. Equal existing
files are left alone. Archives remain available after extraction.

Restore a complete output family before using its ParaView collection, and all
files of a checkpoint before restarting. The old paths in scientific reports
remain valid after restoration. Do not delete the archive if its evidence is
still needed. The separate September 14 and 15 archives were not changed.

## Verification

- Every archived member was decompressed and SHA256-checked before its original
  was removed; all 10,622 retained files were SHA256-checked before documentation
  updates and the derived-summary rerun.
- An 8,003,967-byte history table was restored into a temporary directory with
  its original SHA256. Repeating restoration was idempotent; a deliberately
  differing target was correctly rejected without being overwritten.
- `summarize_disturbance.py` completed successfully using retained evidence,
  without restoring raw QP exports or launching ASPECT. Its history, accepted
  timestep, fresh-linear and active-set assertions passed, and it regenerated
  the summary and plots. Output is in the archive's
  `retained-disturbance-check.log`.
- No simulations or builds were run. This is artifact cleanup, not new numerical
  qualification or invalidation of any archived result.

## Earlier cleanup — 2026-09-14

For the subsequent BP3-only organization on 2026-09-15, see
[bp3/CLEANUP.md](bp3/CLEANUP.md). It uses a separate recoverable archive,
retains the current work-measure qualification and required fixture inputs,
and does not modify the older archive documented below.

The active benchmark tree was reduced from approximately 48 GB and 13,868
regular files to 5.1 GB and 3,462 regular files (before adding this note).
No tracked benchmark file was deleted or changed by this cleanup.

The following untracked generated directories were moved out of this tree:

| Kind | Directories | Files/symlinks |
|---|---:|---:|
| Historical simulation output | 182 | 9,749 |
| Old generated CMake build trees | 11 | 593 |
| Python bytecode caches | 15 | 76 |

Simulation-output directories were identified by generated `parameters.prm`
files. Directories containing tracked files, source, scripts, Markdown reports,
or non-generated parameter fixtures were excluded. Standalone run logs,
comparison reports, input fixtures, scripts and source snapshots remain in place.

## Kept active

- All tracked benchmark files (417 files verified byte-for-byte).
- BP3 `first_cycle*` output/checkpoint directories, including the supplied
  server run and the verified local restart comparison.
- BP3 `theta_junction_audit`, current `build`, and `tmp` source snapshot.
- Source code, executable scripts, input fixtures and reports outside the
  generated build trees.

The BP3 `first_cycle_coarse/restart/03` checksums were verified after cleanup.
No benchmark was run, and no physics, tolerances, source implementation or
fixture parameters were changed.

## Recovery

Historical scientific evidence was archived rather than permanently erased.
The local, git-ignored archive is:

```
/home/ein/repository/aspect/.benchmark-cleanup-20260914-ietM4m
```

`manifest.json` records every original directory, artifact count, byte count,
and the tracked-file verification hashes. `payload/` preserves the original
repository-relative paths. The archive occupies approximately 43 GB; this
cleanup reduces active-tree clutter, **not total disk usage**. It is not a
committed or remote backup: retain it if the historical evidence is needed.

To restore one listed directory, for example:

```sh
python3 /home/ein/repository/aspect/.benchmark-cleanup-20260914-ietM4m/restore.py \
  benchmarks/reconstructed_fault/performance/k3_cell
```

To restore everything:

```sh
python3 /home/ein/repository/aspect/.benchmark-cleanup-20260914-ietM4m/restore.py --all
```

Restoration refuses to overwrite existing paths. Historical reports still
name their original output paths; restore the corresponding directories
before rerunning analyses that consume those files. Old plugins can instead
be rebuilt from retained source. Scientific results have not been classified
as invalid merely because they were archived.
