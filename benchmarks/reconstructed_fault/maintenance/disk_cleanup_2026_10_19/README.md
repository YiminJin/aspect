# Particle-free worktree storage cleanup — 19 October 2026

Scope: `aspect-pf-rsf-particle-free`, branch `pf-rsf-particle-free`, base
`b2dfdebe43437d6028b06d4f9f862cc80a00cc71` plus all preserved incoming work.
The user requested disk cleanup; no scientific stage, source refactoring,
server job or numerical rerun was selected.

**24.7 GiB reclaimed.** Filesystem space available increased from **4.3 to
29.0 GiB**. Worktree usage fell from approximately **52 to 27 GiB**.
The current native/plugin builds, current BP3 optimization/diagnostic/repair
runs, all checkpoints and every source/benchmark input remain extracted.
No unique scientific data was discarded.

## Storage organization

| Location | Contents |
|---|---|
| `build-particle-free/native/` | Unchanged working ASPECT build, binary, objects and precompiled header |
| `build-particle-free/bp3-optimization/`, `bp3-work-*/` | Current BP3 builds, verification runs and checkpoints; unchanged |
| `build-particle-free/archive/executables/SHA256/` | Immutable snapshot anchors; 358 old executable paths share 43 verified byte-identical groups |
| `build-particle-free/archive/disk-cleanup-20261019/` | Eight verified lossless archives of historical dense CSV exports |
| Original server-package directories | Compressed r2/r3/r4 archives/manifests retained; four exact expanded copies replaced by restoration notes |

Executable sharing freed **13.48 GiB**. Historical executable paths and bytes
remain usable, with execute permissions retained. These copies now share
read-only inodes; build into a new directory or copy to a new file before
modifying a frozen executable. Live CMake targets were excluded. Original
permissions/timestamps are recorded in the baseline inventory.

**2128** dense historical CSV files (**14.39 GiB**) are preserved in **4.24 GiB**
of Zstandard-compressed tar archives. Each source payload was verified against
the archive before its uncompressed copy was removed. Compact acceptance,
surface, weak-load and profile CSVs remain extracted. Archive membership excludes
restart/checkpoint directories and all current BP3 runs. Each affected raw-field
directory contains `ARCHIVED_FIELDS.md`.

Four expanded package copies were removed only after matching every file to
their retained source archive; this freed **1.05 GiB**. The older first-event
distribution contains additional plugin build artifacts and was preserved in
full. `package-copies-retained.json` records that decision. Compressed source
packages, checksums and manifests retain their original paths and bytes.

## Restore historical fields or an expanded package

Run from the repository root. Preview a file/directory prefix:

```sh
python3 benchmarks/reconstructed_fault/maintenance/disk_cleanup_2026_10_19/restore.py \
  build-particle-free/runs/pf6a-publication-native512
```

Add `--write` to restore at the original paths:

```sh
python3 benchmarks/reconstructed_fault/maintenance/disk_cleanup_2026_10_19/restore.py \
  build-particle-free/runs/pf6a-publication-native512 --write
```

To recover an expanded package:

```sh
python3 benchmarks/reconstructed_fault/maintenance/disk_cleanup_2026_10_19/restore.py \
  build-particle-free/bp3-homogeneous-coefficients/package/distribution --write
```

A single exact filename also works. `--destination /absolute/empty/directory`
restores the original repository-relative layout there. The default destination
is this worktree. The tool verifies archive and file hashes, restores original
file modes/timestamps, accepts identical existing files and **refuses to overwrite
different content**. Restore the relevant prefix before raw-field analysis or
rebuilding from an archived package copy. Allow space for the previewed size.
Archives remain local to this worktree; retain the archive directory and cleanup
manifests together for recovery.

## Verification and provenance

Version control retains the tools, this guide, compact check results and
`artifact-index.json`. Compressed payloads, the initial Git-state snapshots and
detailed per-file inventories (`before.json.gz`, `field-archives.json`,
`package-copies.json`, `shared-executables.json`) remain local. The compact index
records their hashes and archive locations. Recovery needs those retained local
manifests and archives; cloning the source alone does not recreate them.

- `before.json.gz`: full initial metadata/SHA256 inventory of **117525** paths,
  including all incoming source/evidence and build files.
- `incoming.diff.gz`, `incoming-status.txt.gz`: pre-cleanup Git worktree state.
- `plan.json`: exact scope and exclusions before mutation.
- `shared-executables.json`: every retained executable alias and immutable anchor.
- `field-archives.json`: original names, modes, times, sizes and hashes of all
  compressed fields, plus archive hashes and sizes.
- `package-copies.json`: exact removed copies and their retained source archives.
- `verification.json`: **81983** retained regular files and **149** symlinks
  checked after cleanup; source bytes, checkpoints and live builds unchanged.
- `restore-checks.json`: actual field and package extraction tests, metadata
  verification, identical-file reuse and differing-file overwrite rejection.
- `ARCHIVE_SHA256SUMS`: independently rechecked compressed archive hashes;
  run `sha256sum -c benchmarks/reconstructed_fault/maintenance/disk_cleanup_2026_10_19/ARCHIVE_SHA256SUMS`
  from the repository root to verify the retained containers.
- `final-checks.json`: final source/index separation, live binary isolation and
  preservation checks after the two documented index edits.
- `cleanup.py`, `restore.py`: exact cleanup and recovery procedures.

Mutating cleanup phases are recorded and refuse a second application. The first expanded
package comparison stopped on additional build files before deleting anything;
the final policy retains that entire package. No cleanup touched other worktrees,
external builds or server state. No physical laws, tolerances or production
defaults changed. Source audit/status index edits were made only after the full
preservation verification.
