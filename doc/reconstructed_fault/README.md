# Reconstructed-fault documentation

## Start here

- [Current design](current_design.md) and [specification](specification.tex):
  authoritative scientific, architectural and lifecycle contracts.
- [Refactoring guidelines](refactoring.md): code quality and responsibility boundaries.
- Current small stress-history experiment:
  [completed moment-consistency qualification](../../benchmarks/reconstructed_fault/bp5/moment-qualified-final/report.md).
  A/B/C pass four real steps and C passes the two-rank replay after the pressure
  compatibility correction. This qualifies the narrow horizontal fixture, not
  a general production transfer correction.
- [Artifact cleanup and recovery](../../benchmarks/reconstructed_fault/CLEANUP.md).

## Historical evidence

- `benchmarking/`: Stage-K plans, qualification reports and bounded-study limitations.
- `bp3/`: BP3 source references, research decisions and long-run instructions.
- `review_and_cleanup/`: implementation-stage reviews and earlier cleanup decisions.
- Root-level redesign/stage plans, `pf_rsf.tex` and old surface-design notes:
  historical derivations and planning, not substitutes for the authoritative pair.

These small documents remain at their original paths to preserve source/report
links and user instructions. Their numerical conclusions have not been rewritten
or upgraded during cleanup. Large raw benchmark artifacts may require restoration
before an old analysis or restart; a surviving log or checkpoint metadata file
does not establish that its full raw payload is currently extracted.
