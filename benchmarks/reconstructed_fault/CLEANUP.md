# Latest cleanup — BP5, 2026-09-20

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
