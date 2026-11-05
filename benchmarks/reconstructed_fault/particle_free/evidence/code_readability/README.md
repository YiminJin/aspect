# Fault-code readability evidence

Tested parent: `47c2aae3be15d3ff60aab2a818c2f6e179077702`.
Formatting-only commit: `0eab303e9` (41 files; unchanged non-comment tokens).
The subsequent implementation commit contains the ownership/API changes.

- `final-provenance.json` records the tested source and six working binaries;
  its parent revision and dirty-source designation describe the verification
  state before these commits. All 57 protected artifacts match their entry hashes.
- `verification.json`, `unit-summary.txt`, `equivalence.json` and
  `auxiliary-equivalence.json` summarize the bounded checks. Existing criteria
  were preserved. The incorrect first restart window remains recorded.
- `runs/` retains small inputs, command/identity records and material summaries.
  Inputs are preserved byte-for-byte, including the stationary input’s original
  whitespace after empty parameter values. Full field exports, checkpoints,
  binaries and build directories remain local.
- `build-log-index.json` identifies local build logs by path and hash;
  `independent-compile-commands.json` retains standalone compiler commands.
- `built-sources.json` and `postbuild-comments.json` identify the final
  comment/whitespace-only edits separately from the core build.

The local `mechanical-format.patch` and `mechanical-readability.patch` snapshots
are deliberately not duplicated in Git. Use `git show 0eab303e9` for the complete
formatting-only diff. Other unrelated pre-existing worktree files remain untracked.

The maintained report is
[CODE_READABILITY_REPORT.md](../../../../../doc/reconstructed_fault/particle_free/CODE_READABILITY_REPORT.md).
