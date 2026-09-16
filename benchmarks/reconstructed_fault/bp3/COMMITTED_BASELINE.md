# Recorded K5 development baseline

This commit records the accumulated BP3 implementation dependencies as well as
the bounded within-step state-feedback and two/four-substep verification. The
BP3 plugin and these prerequisite K5 source changes were not previously tracked;
recording only the latest scripts would not preserve a usable source baseline.

Start with [README.md](README.md) and the latest
[coupled-substep report](../../../doc/reconstructed_fault/bp3/stage_K5_coupled_substeps_report.md).
This is a development/evidence snapshot, not approval of every retained opt-in
experiment. The default production state update remains split; the candidate
state is a separately selected bounded benchmark. Airy initialization remains
historical/diagnostic, and the unsuccessful interface-preconditioner prototype
is not enabled by the current fixtures. No long-event or new convergence run
is authorized by this commit.

Recorded: source and focused tests; maintained plugin/diagnostic/analysis tools;
fixture include chains, immutable background/completion/fault/target-mesh inputs;
design and reports; selected compact logs, hashes, accepted profiles, aging/
Jacobian checks and raw last-two-element samples needed for the latest offline
comparison. Discarded audit attempts retain their failure metadata and samples.

Not recorded: build products/shared libraries, checkpoint binaries, full bulk
VTU and particle outputs, most historical raw data and local cleanup archives.
Nothing is deleted. The source step-9 checkpoint remains a **local input**, at
`work-replay-50-local4/restart/01`; its hashes are recorded in provenance and
retiming records. To rerun rather than inspect the bounded comparison, obtain
that exact checkpoint (or explicitly reproduce its baseline), rebuild ASPECT
and the plugin, and select fresh output directories. Do not use a later or
failed in-memory state in its place. See CLEANUP.md for older archived evidence.

Validation is reused from the completed task: Release build with `-j4`, surface
direct tests (545 assertions), and six accepted four-rank coupled substeps with
42 passing fresh-linear checks, single-publication aging audits and independent
final-time comparison. No simulation was rerun solely to create the commit.
