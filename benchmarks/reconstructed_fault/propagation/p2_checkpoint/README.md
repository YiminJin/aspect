# P2 reproducibility checkpoint — 2026-10-13

Start with [P2_CLOSEOUT.md](../../../../doc/reconstructed_fault/propagation/P2_CLOSEOUT.md).
This checkpoint preserves bounded verification, not a completed native
0–0.625-s trajectory. Reference: consistent-endpoint **reduced Newton with
FGMRES**, immutable accepted FE/surface histories and previous physical phase
obstacle, original energy/KKT acceptance. Direct mechanics is experimental,
serial and default off; quasi-Newton and unsuccessful coupling alternatives
remain inactive.

## Identities and retained inputs

- Incoming branch `fault-propagation`, HEAD
  `33c0adba59132890b386f3e984b3e1650feffe44`; existing worktree
  `/home/ein/repository/aspect-fault-propagation`; no staged changes on entry.
- [checkpoint.json](checkpoint.json) records exact source, selected artifact,
  local binary and preservation hashes. [selected-benchmarks.txt](selected-benchmarks.txt)
  is the explicit curated benchmark inventory.
- [reference-options.prm](reference-options.prm) is a **parameter fragment**,
  not a standalone case or an authorization to run. It makes reduced Newton
  and FGMRES explicit when appended to an existing qualified endpoint fixture.
- Fresh accepted onset: [economical-onset-coarse.prm](../native_solver_evidence/economical-onset-coarse.prm),
  shared [phase](../native_endpoint_evidence/shared-initial-phase.txt) and
  [particle H](../native_endpoint_evidence/shared-initial-H.txt).
- Retained growth/saddle: [MPI core fixture](../native_mpi_evidence/core.prm),
  [saddle fixture](../native_mpi_evidence/saddle.prm), and their immutable input
  states. The fixtures import histories only for explicitly named entry tests.
- Numerical results: [MPI assessment](../native_mpi_evidence/assessment.json),
  [direct batch](../direct_mechanical_evidence/batch-assessment.json),
  [growth](../direct_mechanical_evidence/growth-assessment.json),
  [saddle](../direct_mechanical_evidence/saddle-assessment.json),
  [single pass](../single_pass_evidence/assessment.json), and
  [cheaper coupling](../cheaper_coupling_evidence/assessment.json).

Run records retain original command, absolute paths, inputs, executable/plugin
hashes, rank/thread settings and timing. Installed paths must be checked before
reuse on another machine. Current local Release build uses GCC 12.4.0 and
deal.II under `/opt/dealii/9.6-local`; the local CMake cache identity is recorded.
The MPI suite and current direct batch use different recorded executable hashes;
complete-entry direct comparisons use retained instrumented FGMRES timings.
Pure factorization-only and standalone direct-entry memory data are unavailable.

## Curation and preservation

All incoming work and raw evidence remain in place. Before closeout edits,
`incoming-status.txt`, `incoming.patch`, `incoming-manifest.json` and
`incoming-source-docs.tar.gz` were written here. They preserve the initial dirty
state, tracked patch, 14,126 file identities and modified/untracked source and
documents. These local preservation files and saved executables are **not Git
artifacts**; their hashes are in checkpoint.json. No reset, cleanup, history
conversion, source-behavior change or numerical rerun was performed for closeout.

The committed selection contains reusable harnesses, original inputs, small
immutable entry states, assessments, final logs/profiling records, failure records,
compact plots and diagnostic tables. `results/<tag>/` contains byte-for-byte
copies of small tables from the original native output path in that tag's run
record. Large field/time-history arrays, expanded parameter dumps, build trees
and executables remain local; the preservation manifest records their identities
where they were repository files, and run records/build hashes identify the
original native outputs. Assessment scripts that read full fields still require
those original outputs; the committed summaries are not substitutes for them.
No accepted field, obstacle or centroid was translated during curation.

## Source checkpoint

The earlier source checkpoint is `b6885c616`; current additions since the
incoming documentation checkpoint are preserved without further source edits.
Detailed file/function purposes are in the [MPI](../../../../doc/reconstructed_fault/propagation/P2_native_MPI_report.md),
[cheaper-search](../../../../doc/reconstructed_fault/propagation/P2_cheaper_coupling_report.md)
and [direct](../../../../doc/reconstructed_fault/propagation/P2_direct_mechanical_report.md)
reports. They separate ownership/numerical fixes from optional experiments:

| Area | Current checkpoint purpose |
|---|---|
| endpoint_mpi.h; FE strain / history publication | Unique owners, ghosts, once-only assembly, extended global energy reduction and collective prepublication validation |
| endpoint phase solver | Distributed search/active sets, coordinated restoration, default-off experimental quasi-Newton directions; reduced Newton retained |
| native Stokes / condensed system / surface Schur | Interchangeable default FGMRES and serial experimental direct inverse; actual independent coupling blocks, pressure checks and operator-valid factor reuse |
| phase parameters / simulator headers | Explicit default-false settings and reusable existing solver scales/types |
| test access / observer / unit tests | Native operators, histories, MPI agreement, direct residual/action/cost and pressure quotient checks |
| harnesses / reports | Reproducible retained controls and unsuccessful alternatives, separately labelled from native trajectories |

There are no additional CMake modifications at this checkpoint; existing
CONFIGURE_DEPENDS source/header/unit-test discovery includes the new files.
The last retained compatibility run passed 21,112 assertions in 55 cases.
Closeout checks source preservation, Python syntax, JSON/links and Git whitespace;
it does not rerun numerical tests. Full native evolution, temporal/spatial
qualification, coupled restart, transport, scalability and propagation remain
deferred to a subsequent selection.
