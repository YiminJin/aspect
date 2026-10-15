# Disk cleanup record — 2026-10-15

Scope: this `fault-propagation` worktree only, with all incoming edits retained.
The user selected disk cleanup, not another scientific stage. See the
[cleanup report](../../CLEANUP.md) for amounts and restoration commands.

| Record | Purpose |
|---|---|
| `incoming-status.txt.gz`, `incoming.diff.gz` | Full initial Git status and local tracked diff, including pre-existing manual edits |
| `before.json.gz` | Baseline revision, branch, free space and SHA256/mode/time inventory for 109801 paths; incoming modified-file hashes |
| `deletions.json` | Exact 388 regenerable files removed, their hashes and reasons |
| `archived-files.json`, `archive.json` | Eight losslessly archived CSVs and compressed archive checksum |
| `relocations.json` | Four physical moves with original compatibility paths |
| `verification.json` | Retained-file/link/source preservation and space measurements |
| `checks.json` | Actual extraction, differing-file protection, dependency and syntax checks |
| `cleanup.py` | Exact one-shot cleanup procedure; refuses a second execution |

The incoming diff/status and full `before.json.gz` inventory are local
preservation files, not Git artifacts. The committed record carries the script,
exact archive/deletion/relocation manifests and actual verification results.
Their local-only hashes are preserved by the [worktree checkpoint](../../bp3_native/CHECKPOINT.md).
Likewise, the archive payload remains in this worktree's ignored build directory.

The archive lives at
`build-propagation-integration/archive/cleanup-20261015/P2-bulk-fields.tar.zst`.
It preserves the large native bulk snapshots from the completed normal-resolution
case and its failed probe. Restore before a raw-bulk comparison; compact reports,
surface/phase histories and all checkpoints remain extracted. This local archive
does not replace an external backup.

All scientific sources, specifications, inputs, binaries and other evidence were
hash-checked unchanged. After that check, only the top-level benchmark/document
indexes, `CLEANUP.md` and `CURRENT_STATUS.md` were intentionally edited, and small
cleanup/index/restore notes were added. Existing scientific notes remain below
the new recovery entry. No numerical results, algorithms or defaults changed.
