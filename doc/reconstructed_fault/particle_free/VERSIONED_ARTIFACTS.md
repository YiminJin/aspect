# Versioned implementation and research evidence

This snapshot includes the accumulated objective-stress and transported-state
source, headers, native checker CMake targets, unit tests, benchmark runners,
BP3 plotting/parser tools, design/specification updates and research reports.
Compact Stage A/B/C, isolation and bounded-profile measurements, provenance,
figures and failed-check diagnoses accompany the implementation.

Raw server output, local simulation fields/checkpoints, executable/DSO copies,
build trees, source snapshot archives/patches, temporary logs and plot caches
remain local. They are not removed or converted by this commit. Reports and
provenance may link to those local artifacts; a fresh clone does not contain
all inputs needed to replay reductions of historical server or native runs.
The benchmark drivers and recorded commands provide the regeneration path
when the corresponding environment and input data are available.

Recorded revisions and hashes identify the actual pre-commit worktree and
binaries used for each experiment. They are historical evidence, not pointers
to rewrite to the new commit. In particular, committing source does not rebuild
retained executables or convert checkpoints. Compare source and binary hashes
before reuse. The copied final unit log under bounded-profile evidence retains
the original bytes and supplements the original local log path.

The inherited cleanup record remains authoritative, including the 39/45 lost
animation frames and earlier temporary-log losses. Experimental maturity and
remaining physical-energy/boundary/cohesive limitations are in STATUS.md and
BOUNDED_PROFILE_TRANSPORT_REPORT.md; inclusion in Git is not qualification for
production runs. No server job, production restart or push accompanies this
snapshot.
