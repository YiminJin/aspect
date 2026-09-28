# BP5 generated-output cleanup — 2026-09-20

Only new BP5 generated artifacts were cleaned. No ASPECT source, benchmark
source, configuration, fixture, package or scientific result was changed.
Both copied server results (`output-30km/` and `output-30km-steady/`) are included.
Older BP3 outputs and earlier cleanup archives were not touched.

| Measure | Result |
|---|---:|
| Archived regular files | 1,377 |
| Original payload | 20,842,164,775 bytes |
| Compressed payload | 7,170,510,481 bytes |
| Reclaimed space | 13,671,654,294 bytes (12.73 GiB) |
| Retained files verified unchanged | 1,503 |
| Active BP5 tree | approximately 21 GiB → 1.1 GiB |

## Retained in place

- Maintained sources/scripts, build tree and plugins, parameter files, fixture
  meshes/fault/completion inputs, source snapshots, provenance and reports.
- All server packages and existing supplied `.tar.gz` files, unchanged.
- Accepted-step and cumulative-slip histories from **both** copied server runs,
  plots, compact state/weak-force tables, fault visualization, comparison JSONs,
  logs and configuration/provenance sufficient to understand the conclusions.
- The complete step-1 checkpoint directory
  `weakening30-dc010-ell100/steady-large-step/startup/restart/01/`.

Archived material consists of large bulk/particle/QP/history exports,
bulk/particle visualization, other generated checkpoints and Python bytecode.
The copied old server run contributed 1,044,774,155 archived bytes and the
copied steady server run 79,245,739 bytes. Their compact histories remain usable.

Directory skeletons and some checkpoint metadata remain. **A remaining restart
directory is not evidence that its checkpoint payload is still present.**
Restore a complete checkpoint group before using it. The retained `01/` is
complete, but its parent `last_good_checkpoint.txt` and other checkpoints were
archived: do not directly resume the partially retained startup output directory.
Use the retained checkpoint as a source for a correctly prepared new branch,
or restore the complete restart directory first.

## Recovery

The local, git-ignored archive is:

```
/home/ein/repository/aspect/.benchmark-cleanup-20260920-bp5-nKnB1v
```

It contains 27 `.tar.zst` chunks, per-file hashes in `manifest.json`, archive
hashes in `completed.jsonl`, `verified.json`, and the selection/restore scripts.
This is **not an off-machine backup**. Keep it while raw evidence is needed.

Preview restoring the copied old server result, from the repository root:

```sh
python3 .benchmark-cleanup-20260920-bp5-nKnB1v/restore.py bp5/output-30km
```

Add `--restore` to recover the files at their original paths. Substitute
`bp5/output-30km-steady` for the copied steady result, or any exact archived
file/directory prefix relative to `benchmarks/reconstructed_fault/`.
For example:

```sh
python3 .benchmark-cleanup-20260920-bp5-nKnB1v/restore.py \
  bp5/weakening30-dc010-ell100/steady-startup/startup --restore
```

Restoration checks free space and SHA-256, leaves identical files alone, and
refuses to overwrite differing files. Restore a whole bulk/particle output
family before opening its ParaView collection; analyses needing raw QP or
full-state data likewise need restoration first.

## Checks

Every compressed member was decompressed and hashed before deleting its exact,
unchanged working copy. All retained files were hashed again after cleanup.
Recovery of a 16,719,380-byte friction-audit CSV into a temporary directory
passed; a repeated restore changed zero files; a deliberately conflicting
destination was correctly rejected.

The current steady server package still passes its integrity verifier. The
selected 125000-versus-two-62500-s comparison was recomputed from retained
state/weak tables and matches the recorded V, slip, state, shear and normal
RMS ratios to 1e-12 absolute. No ASPECT simulation, build or numerical change.
