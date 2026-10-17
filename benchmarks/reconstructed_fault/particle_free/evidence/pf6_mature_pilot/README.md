# Bounded mature DG pilot evidence

Base `a8707307008401ac9cff702a369e2e2a611fa0f7` plus the preserved incoming
worktree. See [measured report](../../../../../doc/reconstructed_fault/particle_free/PF6_MATURE_PILOT_REPORT.md)
and [scope](../../../../../doc/reconstructed_fault/particle_free/PF6_MATURE_PILOT_FORMULATION.md).
Unlimited domain-free DGQ2 is the candidate. Particle controls are comparison
evidence, not a reference solution. Strict reference matching remains **false**.

`measured-summary.json` records fixed-FF refinement, dt controls at matching
times, signed contrast sensitivity, all 32 native attempt costs including
failures, 13 lifecycle controls, point audits and intended surface initialization.
`source-verification.json` records unchanged defaults, mandatory publication
allowance, limiter/domain sources and original evidence hashes. Native unit
checks pass 124 cases / 118034 assertions; the standalone reference suite passes
55/2625. No solver/publication tolerance was relaxed.

`reference.tar.xz`, `archive-manifest.json` and `provenance.json` retain selected
raw native CSVs, failure inputs/logs, actual restart checkpoints, pinned
content-addressed binaries/libraries, exact source snapshot, preparation tools
and the parser-validated server package. Omitted duplicate/time-intermediate
fields remain in the original local study; no raw output was deleted.
Every archive member has an independent byte count and SHA256. Source build
logs, compiler/runtime records and final fresh/resume parser checks are included.
The supplementary `build-provenance.json` records the actual generated compiler
flags/macros for the core, local-control plugins and ordinary BP3 plugins.

Replay uses Python and NumPy; it never runs ASPECT, MPI or evolution. Use a fresh
destination with space for the uncompressed packet:

```sh
python3 benchmarks/reconstructed_fault/particle_free/replay_pf6_mature_pilot.py benchmarks/reconstructed_fault/particle_free/evidence/pf6_mature_pilot /tmp/pf6-mature-pilot-replay
```

It verifies archive/member hashes, restores artifact hardlinks, reproduces the
aggregate summary, recomputes nine field/history/weak-load comparisons, the
complete Q1 phase-overlay check and the independent full-geometry RSF initializer.
Comparison differences use the full embedded tensor metric; surface loads
subtract each native mass's own background. Neither archive replay nor the
initialization-only control establishes continuum or full-geometry accuracy.

The server pilot is **prepared only**. Regenerate it at its server location,
use a compatible build/runtime, and explicitly launch within the retained
8-step / 400000 s / 7500 s supervisor caps. See
[preparation and limits](../../bp3_dg_pilot/README.md). Keep the recorded
endpoint temporal failure, spatial sensitivity and state-interpolation defect.

Delivered replay passed: **5246 verified members**, exact summary, nine exact
independent comparisons, complete Q1 overlay and both surface-initialization
controls. See `portable-replay.json`/`.log` and `final-checks.json`. Archive SHA256:
`1dd3910246fe529884a9c6efcf45fb9353d9b84c4c9d3fc98f17bd7c8f667f29`.
Compressed/uncompressed packet sizes are 1126579888 / 3504801672 bytes.
The endpoint 1% temporal screen and continuum spatial accuracy remain unpassed.
