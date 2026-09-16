# Generated-artifact cleanup — 2026-09-14

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
