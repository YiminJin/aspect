# Committed evidence summaries

The source commit retains the PF5 `assessment.json` and `verification.json`,
plus the small stage-specific README files describing the original evidence
packages. The full measured results and limitations are also in
[the phase reports](../../../../doc/reconstructed_fault/particle_free/STATUS.md).

Detailed run/source manifests, archived fields, binaries, checkpoints, plots,
logs and earlier development evidence remain local. They are intentionally
excluded from the source commit. In the stage README files, archive-replay
instructions and links to excluded data apply when that original local package
is available; they are not a claim that a fresh checkout contains those data.
Historical phase reports retain their original local-evidence references.

The two committed PF5 JSON files record the measured reference errors, second
order convergence, rejection/publication gates, restart equality and retained
checks at the reported pre-commit working tree. Their artifact hashes, absolute
paths and old revision identifiers describe that historical experiment. They
are not automatically refreshed by a later commit or a claim about untested
changes. Related JSON filenames and raw replay paths in `verification.json`
refer to the original local package.

For fresh verification, build the observer sources and use the bounded runner
and independent assessments in the [benchmark README](../README.md). No build,
transport campaign, threshold tuning or coupled experiment runs automatically.
