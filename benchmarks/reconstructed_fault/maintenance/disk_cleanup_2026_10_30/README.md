# 2026-10-30 cleanup audit

[Cleanup report](../../../../doc/reconstructed_fault/particle_free/CLEANUP_REPORT.md) ·
[Research handoff](../../../../doc/reconstructed_fault/particle_free/RESEARCH_HANDOFF.md).

Cleanup is restricted to generated build-particle-free artifacts. Original BP3
server results, current native binaries/plugins, unique source and the complete
RSF server package/checkpoint are protected. No job was launched or terminated.

`audit.json.gz` preserves entry identity, complete deletion batches, replacements,
counts/sizes and verification. Loose initial inventories, source fingerprints,
entry patches/status and batch manifests also remain locally. The compressed
bundle is metadata, not a backup of intentionally discarded fields/checkpoints.
`retained-over-500MiB.csv` explains every substantial remaining build family.
`recovery-map.json.gz` maps16128 original paths to exact retained data.

Use `python3 recover_retained.py ORIGINAL_PATH` from this directory (or the full
repo-relative script path from root) to locate an artifact. Add
`--destination /tmp/new-file` for verified recovery without overwrite. It covers
unique-member zstd tar archives, reference tar.xz bundles, historical executables,
expanded source-package copies, local BP3 compressed fields and short checkpoint
labels. Large artifact payloads remain local; a Git clone alone is not a copy of
those source/evidence packages. Five representative isolated recoveries passed.

Older archive manifests/recovery notes are historical records. Some original
archive names were replaced by `.unique.tar.zst` and exact reference members;
use this index instead of assuming the older restore tool can find old names.
Intentionally removed redundant visualizations/operational checkpoints and object
files have no recovery copy. See each recorded reason and retained replacement.
No full backup of deleted generated data was made.

The cleanup executable validates scope, symlinks, inode/size, protected text,
pre-deletion content hashes and recorded reasons before unlinking. Source copies
are removable only with verified archive identities; short generated checkpoint
labels are preserved literally. Compiler/header/data semantics remain unchanged.

The final frame audit identified a preservation mistake in batch09: some older
bulk-animation frames were pruned. Four exact aliases were restored; missing
paths are explicit in legacy-animation-availability.json. Original animations,
all quantitative/surface histories, active checkpoint and BP3 data remain.
Read the cleanup report limitation before claiming exact figure reproducibility.
