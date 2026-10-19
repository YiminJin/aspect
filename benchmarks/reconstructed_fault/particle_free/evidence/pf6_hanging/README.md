# Serial fixed-mesh constrained cohesive phase evidence

Scope: 67 fixed Cartesian cells, 90 physical Q1 phase nodes / 86 masters,
homogeneous domain-free accepted DGQ2 history, fixed eight-segment surface,
144 stationary diagnostic particles, Gauss12 physical / source3, unlimited DG.
No physical remeshing, MPI local-phase solve, transport, moving geometry or
propagation is qualified. Production defaults and all BP3 artifacts are retained.
See [the report](../../../../../doc/reconstructed_fault/particle_free/PF6_HANGING_REPORT.md).
`strict_reference_match_passed=false` is deliberately retained.

- `verification.json`: independent constrained derivatives/actions, all primary
  and lifecycle controls, full free-master eigenvalues, physical quadrature,
  exact comparisons, interval work ledger and the 34-file BP3 identity check.
- `verify.py`: read-only replay; uses the existing reference modules alongside
  `assess_pf6_hanging.py`. It launches no simulator or MPI job.
- `uniform-baseline.json`: SHA256 identities of all 162 non-timing CSVs from
  the qualified incoming `evolving-consolidated-20261019` control.
- `provenance.json`: source/build/input/binary identities, command records and
  failed diagnostic development attempts. Native raw data are immutable.
- `protected-bp3.json`: original protected-file identities; use only where those
  local BP3 paths are present. No copy/rebuild of those artifacts is required.
- `reference.tar.xz`, `archive-manifest.json`: compact raw inputs/exports,
  accepted checkpoints, logs, records and the revised source/observer snapshots;
  omit executables/DSOs and compiler products. Includes the two failed diagnostic
  attempts and the entry patch preserving incoming work. The archive is evidence,
  not a replacement ASPECT distribution or BP3 server package.

Replay in the existing worktree:

```sh
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/evidence/pf6_hanging/verify.py --protected benchmarks/reconstructed_fault/particle_free/evidence/pf6_hanging/protected-bp3.json --output /tmp/pf6-hanging-replay.json
```

For an isolated raw-data replay, extract into a fresh directory, retaining the
current worktree's reference modules. This avoids replacing local outputs:

```sh
mkdir /tmp/pf6-hanging-review
sha256sum benchmarks/reconstructed_fault/particle_free/evidence/pf6_hanging/reference.tar.xz
tar -xJf benchmarks/reconstructed_fault/particle_free/evidence/pf6_hanging/reference.tar.xz -C /tmp/pf6-hanging-review
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/evidence/pf6_hanging/verify.py --runs /tmp/pf6-hanging-review/build-particle-free/runs --output /tmp/pf6-hanging-review/replay.json
```

To rebuild/rerun, use the worktree's isolated
`build-particle-free/evolving-phase/{native,plugin}` CMake configurations and
`run_pf6_hanging.py prepare/finalize`, then launch `command.json` via the existing
bounded `run_logged.py`, as shown in the report. Final build commands are in
provenance; final native suite uses `aspect-release --test`. The optional observer
parameters default off; load only the transfer DSO for these fixtures. The
shared interval-mean ramp is the existing PF6a formula, not a new controller.
Final guards restrict the new hanging-phase admission to serial, nonperiodic,
2D homogeneous domain-free cohesive DG. All scalar and native criteria are fixed.

The archive includes different diagnostic build identities for earlier passed
controls. Final changes are limited to an explicit periodic exclusion, explicit
header includes, corrected J/m KKT units and a separate contact-KKT export. Final
primary/uniform/restart and native-unit repeats qualify that final source.
Earlier raw metric label `KKT_Pa_m` is preserved; replay relabels that quantity
`KKT_J_per_m` without changing values. A sticking negative raw fault residual is
retained and checked through the native lower-bound contact convention.
