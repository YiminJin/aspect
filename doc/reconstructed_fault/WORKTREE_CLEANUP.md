# Research worktree cleanup · 21 November 2026

The cleanup returned **68.6 GiB** to the laptop filesystem. The research worktree occupies **9.08 GiB**. The canonical frozen reference occupies a further **49.2 MiB** on the same filesystem. The mature and localized controls remain reproducible; the 5–10 ms midpoint trajectory remains a timestep-sensitive diagnostic. [Scientific closeout](particle_free/MIDPOINT_UPDATE_CLOSEOUT.md) · [Selected evidence](particle_free/README.md)

## Scope and accounting

The root was `/home/ein/repository/aspect-pf-rsf-particle-free`, branch `pf-rsf-particle-free`. HEAD and the local origin-tracking ref both pointed to `df02facb9b64ea0d13e888d92c2d9d62fa53081b`. The local tag `research-pre-worktree-cleanup-20261121` preserves that revision. Both incoming tracked diffs were empty. The sibling reference records those diffs, the six Git worktree registrations and the incoming inventory of 9,028 untracked paths. Every incoming untracked path that disappeared is in the deletion ledger. A host process check found no active ASPECT, MPI, build or indexing job.

| Location | Allocated before → after (GiB) | Apparent before → after (GiB) |
|---|---:|---:|
| Research worktree, including build and benchmarks | 77.773 → 9.076 | 77.167 → 8.943 |
| `build-particle-free/` | 68.577 → 5.928 | 68.051 → 5.873 |
| `benchmarks/reconstructed_fault/` | 8.820 → 2.772 | 8.772 → 2.725 |
| `doc/reconstructed_fault/` | 0.014 → 0.014 | 0.013 → 0.013 |
| Old build archives, included above | 29.710 → 0.049 | |
| Sibling reference, including deletion ledger | 0 → 0.048 | 0 → 0.048 |
| Shared Git object and metadata directory | 0.549 → 0.549 | |

Each `du` measurement counts hardlinked storage once within its tree and leaves symlinks unexpanded. Rows overlap. Filesystem available space rose from **1.18 to 69.82 GiB** (`df -B1`), including the cost of the sibling copy and restored checkpoints. The shared Git directory grew by one 4 KiB block. Tracked retirements remain recoverable from Git.

## Recover and check the reference

The frozen package and exact deletion ledger are stored locally beside the checkout. On another machine, copy the sibling reference directory there before running these commands from the research worktree root. The archive SHA-256 is `a82d362c3504be188e9602cf10c93cf51a26544be5e6d4a7677016cc52981c57`.

```sh
cd ../aspect-pf-rsf-reference-20261121
sha256sum -c reference.tar.gz.sha256
dest=$(mktemp -d)
tar -xzf reference.tar.gz -C "$dest"
cd "$dest/reference"
sha256sum --quiet -c SHA256SUMS
OPENBLAS_NUM_THREADS=1 python3 analysis/midpoint_closeout.py runs > "$dest/diagnosis.json"
cmp expected/diagnosis.json "$dest/diagnosis.json"
```

The [sibling README](../../../aspect-pf-rsf-reference-20261121/README.md) and unpacked guide give the native replay commands and source reconstruction. The original build location has one compatibility link. The sibling contains one complete compressed package and 128 KiB of extracted guides, expectations and reducers. The package preserves mature base `3928edaf7` and localized base `c407e612b` with their distinct patches, source identities and matched binaries.

Fresh extraction passed all 144 manifest entries. The archive has no absolute paths or symlinks. Both executables and four DSOs resolved their libraries independently of the retired build directories. The mature export checker gave zero maximum difference, and the oblique localized tangent, power, rollback and publication checks passed. A relocated two-rank restart reproduced all 4,968 final material-point records and the two section-history tables exactly. The independent midpoint diagnosis matched the stored JSON byte for byte. These checks establish artifact integrity and reproducibility within the recorded controls.

## Other retained studies

| Location under the research root | Contents and verification |
|---|---|
| `build-particle-free/transported-objective/` (2.28 GiB) | Four complete A/B/D/C runs and selected short spin/basis controls. Restored 60 missing files across all twelve mature rotating generations before retiring the mixed archives. Markers are 2/3/3/2 in A/B/D/C order. The dated November 12 archive notices in those directories describe an earlier state. All 198 preservation entries passed, including seven mature and two historical runtime identities. All 3,104 compressed field samples matched their recorded sizes; first, middle and last samples in each run also matched their plaintext SHA-256. The four-run accuracy limits and plots remain in the [mature closeout](particle_free/MATURE_FAULT_TEMPORARY_CLOSEOUT.md). |
| `build-particle-free/overnight-recovery/` (0.65 GiB), `guard80/` (0.29 GiB), `bulk-tip/` (0.40 GiB), `strip-width/` (49 MiB) | Historical full-section 80/40 μs inputs, accepted parents, outputs, checker and solver (`5e6bdb65646c` / checker `8b582573e7b1`). The final-interval reduction reproduced its saved JSON exactly after cleanup. The distinct width runtime/checker (`6b758a7b857c` / `bffa8d70fdb3`) and coarse/fine strip inputs survive. Both provenance inventories passed (28 and 36 entries). |
| `benchmarks/reconstructed_fault/particle_free/` | Both original BP3 server returns (158 start files and 62 resumed files), figures, reducers and prepared experiments. The separate surface-cohesive `rsf_episode_server/SHA256SUMS` passed, including the packaged restart. The [retained study hashes](../../../aspect-pf-rsf-reference-20261121/retained-study.sha256) cover all 220 BP3 files and 29 selected runtime/source records; verify from the worktree root. |
| Other substantial build survivors | The surface-cohesive episode and its comparison outputs (0.73 GiB), BP3 startup/qualification (0.35 GiB) and first-event package (0.18 GiB) retain paper evidence. Two optimized BP3 package revisions have different hashes; both survive. Seven historical paper-related compressed runtimes remain with their catalogs. The shared-assembler review runtime and its existing review tag survive. |

The native Bernstein final core and checker match their original hashes in `bulk-tip/controls/native-bernstein-final/`. The mutable `bulk-tip/points/bulk_tip_points` differs from the recorded standalone test binary. Its measured results and source survive; exact executable replay of that particular point test is unverified. The original editor compilation database and headers survive (6 MiB). Three of its 504 entries name source files already absent at the incoming revision.

## Retirement and audit trail

The [action manifest](WORKTREE_CLEANUP_ACTIONS.tsv) groups actions by path, measured allocation, reason and retained reference. The [exact gzip ledger](../../../aspect-pf-rsf-reference-20261121/deleted-paths.tsv.gz) contains 170,530 file, link and directory entries in deletion order. Columns are original relative path, kind, inode, apparent bytes, mtime in ns, allocated path-bytes and group. The ledger SHA-256 is `d368f36977ce73cdb6fa1230b6cbc3b6ea201669e83e0dfde598422040ab96ea`. Group totals can count multiple names for one inode.

Retired products include October dense P2–PF6 and WENO archives, old bulk/cohesive trial fields, November checkpoint forests and material-overnight exports, compiler/index products and duplicate midpoint staging. Earlier restore manifests still describe the historical disk layout. Their retired raw payloads can no longer be recovered locally from those manifests. Earlier losses included 39 of 45 legacy native-pilot frames and temporary logs. [Historical loss record](particle_free/CLEANUP_REPORT.md)

The 26 tracked retirements comprise seven superseded documents, two historical mature-driver scripts, fifteen build/run diagnostics and two old P3 archives. Original tracked content can be read with `git show df02facb9:<original-path>`. Source-tree changes are confined to retirement, navigation, recovery instructions and narrowly scoped Python-cache ignores. The cleanup left production code, CMake, parameter inputs and manuscript sources at the entry revision.
