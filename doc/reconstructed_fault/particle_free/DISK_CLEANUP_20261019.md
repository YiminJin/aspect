# Worktree storage cleanup — 19 October 2026

The user selected reorganization/disk cleanup. **24.7 GiB** was reclaimed;
available filesystem space rose from **4.3 to 29.0 GiB**, with worktree size
approximately **52 → 27 GiB**. The current native/plugin builds, source changes,
BP3 server packets and repair evidence, current BP3 runs and all checkpoints are
unchanged. Other worktrees/builds/server jobs were not touched. No scientific
scope or numerical qualification changed; strict reference matching stays false.

[Manifests, verification and restoration instructions](../../../benchmarks/reconstructed_fault/maintenance/disk_cleanup_2026_10_19/README.md).

- 358 frozen executable paths now share 43 byte-identical, read-only snapshot
  groups. Their original paths remain executable; live build targets were
  excluded. This saves 13.48 GiB without discarding historical binaries.
- 2128 historical dense CSV exports are losslessly archived under
  `build-particle-free/archive/disk-cleanup-20261019/`: 14.39 GiB becomes 4.24 GiB.
  Every payload was hash-verified before removing its expanded copy. Compact
  acceptance/surface/weak-load/profile evidence remains extracted. Per-directory
  `ARCHIVED_FIELDS.md` notes point to recovery instructions.
- Four expanded source-package copies exactly matching retained compressed
  archives were removed, saving 1.05 GiB. A package containing additional build
  artifacts was retained in full. All compressed packages/manifests remain.

After cleanup, all retained regular files and symlinks were checked against the
initial inventory. Actual extraction tests restore a historical field and source
file with original bytes/modes/times; existing identical files are reused and
different content is rejected without overwrite. Archive/source checks replace
numerical reruns for this storage-only task.

Before an old raw-field comparison, preview and restore its original prefix:

```sh
python3 benchmarks/reconstructed_fault/maintenance/disk_cleanup_2026_10_19/restore.py \
  build-particle-free/runs/pf6a-publication-native512
python3 benchmarks/reconstructed_fault/maintenance/disk_cleanup_2026_10_19/restore.py \
  build-particle-free/runs/pf6a-publication-native512 --write
```

The next scientific action remains the prepared bounded server verification of
the observer repair from checkpoint 815 through 818. Its source, inputs, evidence
and local verification runs remain immediately available; cleanup does not
authorize a server launch or advance that task.
