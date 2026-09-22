# BP3 artifact organization — 2026-09-15

For the latest **2026-09-17 compressed cleanup**, including the copied long-run
output, see [the parent cleanup record](../CLEANUP.md). Its archive and restore
utility are separate from those below. The retention statements in this older
record describe September 15, not the current availability of raw output.

The active BP3 tree was reduced from approximately **14 GB to 498 MB**.
The original 283 loose root files became 90 before adding this README/cleanup
documentation. Maintained source, scripts and task instructions remain at
their established paths. No tracked source file or numerical fixture was
rewritten, no simulation was run, and no scientific evidence was deleted.

## What moved

**6,092 files** (13,975,179,495 bytes) were moved to a recoverable local archive:
historical full simulation exports, checkpoints other than the supplied server
run, raw domain/QP/history tables, failed/interrupted attempts, old source
snapshots, standalone logs and obsolete experimental parameter wrappers.
The archived byte count is recorded authoritatively in `manifest.json`.

Compact logs, JSON summaries, plots, notes and wrappers under 2 MiB were also
copied, checksum-verified, into `evidence/completed/` under their original
relative names. This makes past conclusions browsable without restoring
large raw datasets. The archive retains the originals as well.

The supplied server `CMakeCache.txt` and `BP3.e3498041` were relocated to
`evidence/server/`, unchanged. The current `build/`, complete latest
`work-measure-free-top-local4/`, and supplied `first_cycle_coarse/` including
`restart/03` were retained. Existing `evidence/` snapshots were retained.

The mature/boundary fixture dependency chain was checked recursively before
moving anything: **62 required files** remain in place, including parameter
includes, immutable prestress, completion tables, fault coordinates, target
mesh, clocks and launch-readiness data. Completion logs remain beside these
fixtures so an old case is not misidentified as an unexecuted run.

## Recovery

Archive location (git-ignored, local only):

```
/home/ein/repository/aspect/.benchmark-cleanup-20260915-bp3-0edlufa2
```

`manifest.json` records original BP3-relative paths, sizes and SHA256 hashes;
`payload/` holds the original files. `verified.json` records the completed
archive/retained-input verification. `cleanup_script.py` preserves the exact
selection and move procedure for audit, not for automatic re-execution.

Preview one group:

```sh
python3 benchmarks/reconstructed_fault/bp3/restore_archived.py \
  .benchmark-cleanup-20260915-bp3-0edlufa2/manifest.json mature-fault-50-local4
```

Add `--restore` to restore that group's raw data. Omit the group to preview
all archived files, or to restore all with `--restore`. A file path can be
used instead of a directory. Restoration verifies hashes and refuses to
overwrite any differing existing file; equal retained files are left alone.
Files are moved back, not duplicated. Restore prerequisite groups named by
an analysis as well as its main output directory.

This cleanup reduces **active-tree clutter, not total disk consumption**.
The archive is not a committed or off-machine backup. Do not delete it if
historical raw evidence or checkpoints may still be needed. The older
2026-09-14 archive documented in the parent `CLEANUP.md` is separate and was
not changed.

## Verification

- SHA256 verified all 6,092 archived files after relocation.
- SHA256 verified 859 retained non-build files, including the complete supplied
  server data and current work-measure result; source/fixture content unchanged.
- Verified all 62 retained runtime/input dependencies still exist.
- Reran `analyze_work_measure.py` successfully against the untouched latest
  result: derivative, work, fresh-linear, nonlinear, prescribed-rate and
  rollback checks still pass. No ASPECT execution.
- Tested a real single-file restore of `cache-unit.log`, repeated restoration
  (idempotent, zero additional files), then moved that file back into the
  archive to preserve the cleaned layout.
- `git diff --check` passed. No build or mechanical tests were needed because
  this task changed organization/documentation only.

Historical analysis scripts remain available, but their archived raw inputs
must be restored before use. The cleanup does not claim those analyses were
rerun or that archived failed attempts became valid results.
