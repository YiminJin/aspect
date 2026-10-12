# Versioned P1/P2 checkpoint

This checkpoint preserves the optional production implementation, reproducible
verification tools and selected evidence from the work following baseline
`de7beaa65846c7372711f7922abf1c5d9a745ee6`. It does not change the scientific
qualification of any retained result. Start with
[STATUS](../../../doc/reconstructed_fault/propagation/STATUS.md) and the
[latest native solver report](../../../doc/reconstructed_fault/propagation/P2_native_solver_report.md).

The commits separate implementation and governing documentation, verification
tools and fixtures, and reports with compact evidence. The versioned benchmark
content includes:

- Model runners, operator checks, assessment and plotting scripts, parameter
  fixtures, plugin sources and their CMake projects.
- The shared continuum initialization reference, native initial phase/H files,
  and immutable unloaded/growing/asymmetric operator, core-growth and saddle
  entries for repeatable fixed-state tests.
- Selected comparison, qualification, cost and provenance summaries; historical
  failed-result reports and equation diffs; and the figures cited by reports.
- Five compact ASPECT visualization snapshots from the earlier output-only
  rendering control. They are not native coupled-evolution evidence.

Full trajectories (`fields.npz`, large history and stability arrays), repeated
particle/stress CSV exports, checkpoints, executables, plugin binaries, build
logs, temporary source copies and most intermediate solver experiments remain
local artifacts. They are intentionally absent from these commits. Existing
reports and provenance manifests can refer to such local files or historical
paths; those references are not promises that every artifact is versioned.
No excluded artifact was deleted by this checkpoint.

For a new checkout, follow the build/run commands in the corresponding stage
report. Generate the required prototype trajectories before invoking an
assessment script that reads their full fields. The native endpoint tools use
`native_endpoint.py`; the economical schedule uses
`native_economical_schedule.py`. The shared initialization is defined by
`matched_initialization.py`. Output directories are protected against accidental
overwriting; use a fresh destination where the runner supports one, or retain
existing evidence separately before deliberately regenerating a campaign.

The prepared full native `.625`-s input is a costed plan, not a completed run.
The experimental options remain default off. Full native continuation,
tangential stability, coupled restart, postpublication rollback, transport/birth
histories and P3 are not qualified by this checkpoint.
