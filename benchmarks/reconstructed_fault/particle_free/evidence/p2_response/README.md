# Bounded P2 physical-response evidence

Serial, fixed 2×512-cell bulk mesh and eight-segment through-going cohesive
surface. Domain-free homogeneous DGQ2 independent history, ordinary Q1 phase,
native consistent endpoint/reduced Newton, Gauss12 physical / Gauss3 source,
unlimited DG, fixed diagnostic particles. Production source/defaults and all
BP3 artifacts are preserved. No P3 or general propagation is run.

The report is [P2_PHYSICAL_RESPONSE_REPORT.md](../../../../../doc/reconstructed_fault/particle_free/P2_PHYSICAL_RESPONSE_REPORT.md).
`strict_reference_match_passed=false` and the preceding failed evidence remain.

The selection and practical screens were frozen before trajectories. The
physical initialization is the retained continuum seeded-H BVP interpreted as
pre-existing damage and sampled at native nodes, with unloaded T/Z/slip. The
physical seed is the obstacle; H does not drive the endpoint phase equation.
No independent per-mesh phase relaxation or history reset is used.

Inputs apply the exact interval mean of the retained velocity ramp through
.5625 s, then zero boundary velocity through .59375 s. All histories continue
advancing during the hold. The three native controls are unloaded (two steps),
primary (76), and one timestep halving (152). The recorded accepted sequences
are checked explicitly; lifecycle evidence comes from the preceding native
rejection/retry/publication/restart qualification.

## Contents and replay

- `selection-before-runs.md/json`: immutable interval, meshes, finite budgets
  and practical criteria selected before the trajectories.
- `verification.json`, `comparison.json`: native acceptance/once-only checks,
  unaligned common physical fields, native/reference and targeted spatial
  comparisons, hold response and separate physical-work remainder terms.
- `provenance.json`, `protected-bp3.json`: source/build/input/executable
  identities, native commands, bounded execution records and BP3 preservation.
- `reference.tar.xz`, `archive-manifest.json`: raw native fields and histories,
  selected dense stress/weak-load exports, logs/records/inputs, all inexpensive
  reference trajectories, compact full-DG assessments and entry-source evidence.
  Executables, DSOs and compiler products are omitted. The archive is evidence,
  not a replacement ASPECT or BP3 server package.
- `verify.py`: read-only raw CSV identity, physical admissibility, native
  gate/publication, common-field and energy-summary replay. It launches no job.

Replay in this worktree:

```sh
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/evidence/p2_response/verify.py --protected benchmarks/reconstructed_fault/particle_free/evidence/p2_response/protected-bp3.json --output /tmp/p2-response-replay.json
```

For isolated review, extract the archive into a fresh directory and use the
current worktree's mathematical reference modules:

```sh
mkdir /tmp/p2-response-review
tar -xJf benchmarks/reconstructed_fault/particle_free/evidence/p2_response/reference.tar.xz -C /tmp/p2-response-review
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/evidence/p2_response/verify.py --root /tmp/p2-response-review --output /tmp/p2-response-review/replay.json
```

## Reproduction commands

`run_p2_response.py prepare LABEL --case unloaded|primary|half` writes explicit
fixtures and `command.json`, copying the isolated native executable and matching
PF6a observer. Launch that command through the retained bounded `run_logged.py`,
then use `run_p2_response.py finalize LABEL`. The actual commands and budgets are
recorded in provenance. The native integrator owns mechanics, phase, acceptance
and publication. Only the PF6a DSO is loaded; the transfer observer shares its
registered class name and must not be loaded alongside it.

Analyze each loaded run without writing into native outputs:

```sh
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/assess_p2_response.py build-particle-free/runs/p2-response-primary build-particle-free/p2-response/native-primary --full-work-checks
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/assess_p2_response.py snapshots build-particle-free/runs/p2-response-primary build-particle-free/p2-response/native-primary-snapshots.json
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/assess_p2_response.py compare build-particle-free/p2-response /tmp/p2-response-comparison.json
```

The extractor requires a fresh assessment directory. Repeat the first two
commands with `half` for the timestep-halved control. Every native tensor retains
all nine DGQ2 coefficients and all three independent components; zz=−(xx+yy)
and both xy contributions enter errors. Physical phase gradients, normalization
and loads use the same geometry and measures. Branches are never registered.
External energy remainders and individual mechanical/contact/phase work terms
are reported separately; algebraic ledger closure is not the readiness screen.
