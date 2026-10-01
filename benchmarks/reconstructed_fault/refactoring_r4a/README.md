# R4a: move the dedicated reconstructed-fault Stokes driver

Reference: branch `pf-rsf-refactor`, HEAD `d7b88b25e6e157206b6bdcaf714dac434701f97c`.
The intervening commit after Maxwell cleanup only removed the user's temporary
R2b review. All 5,907 entries of the accepted cleanup source manifest match.
Reference executable: `build-refactor-r3b/aspect-maxwell-qualified`, SHA256
`1cea4bfcc506594374e9b4556d7000fd059360d532070cbd1f190de32cdcc7b5`.
The reference executable retains an older embedded `bef79b31a` version banner;
its qualified source/artifact manifests, not that banner, establish the accepted
post-R3/Maxwell content. The original direct frozen-Maxwell calculation and
restart correction are retained.
Candidate: this uncommitted R4a diff, separate `build-refactor-r4a` directory.

## Definition inventory and scope

| Definition | Consumers | R4a placement |
|---|---|---|
| `Simulator::solve_reconstructed_fault_stokes` and all its local lambdas/types | Existing reconstructed-fault dispatch in `solver_schemes.cc` | Complete unchanged definition in `solver/reconstructed_fault_stokes.cc`; same private member declaration |
| `FaultVector`, `current_slip_rate` | Fault driver only | Unchanged anonymous-namespace helper beside the driver |
| `internal::StokesBlock` | Ordinary Stokes solve; fault affine audit | Class declaration in source-private `solver/stokes_operators.h`; five non-inline definitions remain unchanged in `solver.cc` |
| `SchurComplementOperator`, `WeightedBFBT`, `InverseWeightedMassMatrix` | Ordinary and condensed Stokes solves | Unchanged class/template definitions in that private header; no duplicate implementations |
| `InverseVelocityBlock`, `BlockSchurPreconditioner` | Both solves | Existing `include/aspect/simulator/solver/block_stokes_preconditioner.h`, unchanged |
| Fault pressure/condensation/nonlinear helpers, interface preconditioner and residual audit | Existing fault assembly/solver/test consumers | Existing files unchanged; relocated driver includes them |

No interface, owner, mathematical expression, diagnostic, MPI ordering,
publication or rollback behavior changes. General solver loses the complete
fault algorithm; existing shared operator implementations remain single.
No preconditioner-setup consolidation, step helper or context redesign occurs.

CMake keeps the new file outside unity builds and precompiled headers, preserving
existing unity groups. The first independent driver compilation diagnosed the
previously indirect `NewtonHandler` definition; adding `aspect/newton.h` resolves
it. General solver and the new internal header also compile independently with
no PCH. This include repair is the only dependency addition.

## Qualified result

All 19 recorded build/runtime invocations pass (five build/compile/configure and
14 runtime invocations). One/two-rank unit runs pass 20,149 assertions in 16 cases
per rank. Both versions pass accepted-Newton rollback on one/two ranks. All 16
focused comparisons pass; ordinary AMG statistics and bulk/particle VTU payloads
match exactly after excluding output paths and the VTU generation timestamp.
The four legacy/automatic BP3 trajectories match in all **372 field/history groups**
(maximum absolute difference zero), **24 cache/work checks** and **four detailed
solver-decision comparisons**. No numerical differences or new scientific defects
were identified in the checked paths. `evidence/qualification.json` records these
results and the 2D/3D symbol checks; `driver-symbols.txt` and
`ordinary-symbols.txt` retain the actual definitions.

Qualified candidate: `build-refactor-r4a/aspect-r4a-qualified`, SHA256
`ed827270970996854ae2425e519255422f80a25b0602d1cbf863c88a8d4910be`.
`candidate-source-hashes.json`, `executed-artifacts.json` and
`candidate-artifacts.json` preserve source and executable/plugin/PRM provenance.
Effective candidate parameters confirm every loaded test plugin comes from the
candidate plugin build. The source checks confirm all reference/user artifacts
remain unchanged. The user accepted R4a and requested its local commit; no R4b/R4c changes are included.

## Bounded verification and reproduction

`evidence/reference.json`, `entry-source-hashes.json` and `protected-hashes.json`
record the entry revision, accepted source and 891 reference/local artifacts.
`verify_source.py` checks exact moved and retained blocks and protected hashes.
Build commands, stack, environments and individual outcomes are recorded in
`evidence/*.json`, with paired logs. Do not overwrite an existing evidence label.
The initial `build` log records the missing include; `build-includes` records the
completed build. Initial `sandbox-reference-*` logs record OpenMPI socket denial,
not a solver failure; authorized `reference-*` runs are the executable checks.

- Fresh candidate Release build with GCC 12.4/OpenMPI 5.0.6/deal.II 9.6-local,
  Voro++, `-fno-finite-math-only -ffp-contract=off`, two build jobs.
- `compile_independent.py`: general solver and private header without unity/PCH.
  New driver independently compiles in the normal build; symbol checks cover
  the member's 2D/3D instantiations. No 3D simulation is required or run.
- `plugin/CMakeLists.txt`: build candidate rollback, cache observer and maintained
  BP3 plugins against the candidate build. Candidate PRMs load these libraries.
- `prepare_inputs.py`: reuse existing BP3 and rollback PRMs, changing only paths;
  ordinary `tests/convection_box_particles.prm` selects block AMG explicitly to
  exercise the shared assembled Stokes operators. Physics/tolerances unchanged.
- `run_checks.sh reference|candidate`: existing condensation/Stage-I unit cases
  and accepted-Newton rollback on one/two ranks, ordinary AMG smoke on one rank.
  Candidate additionally runs legacy/automatic BP3 steps 0–6 on one/two ranks.
  Reference BP3 comparisons use preserved qualified Maxwell outputs.
- `compare_bp3.py`: existing exact field/history/solver and cache/work comparison,
  bounded to those four trajectories. `compare_checks.py`: matched rollback/unit
  decisions plus ordinary statistics and output fields. Exclude time/path metadata.

Use `python3 benchmarks/reconstructed_fault/refactoring_r4a/run_logged.py LABEL
SECONDS COMMAND...` from the repository root for recorded invocations. MPI needs
permission for local sockets in this environment. Runs use one compute thread
per rank and the established explicit B/G environment only for BP3.

Qualified restart/checkpoint evidence from `restart_fix`, `refactoring_r3b` and
`maxwell_cleanup` is reused: initialization and restoration semantics are
unchanged, and the moved restoration body is byte-identical and covered by fresh
rollback tests. No checkpoint is changed. The existing Stage-J pressure failure,
cohesive step-two nonconvergence and cross-rank cohesive observer limitation are
separate historical outcomes, not R4a passes or new numerical fixes.

No full earthquake, long performance, M1/M2 campaign, Debug build or 3D simulation.
R4b/R4c are not implemented. Proposed next selection: private extraction of the
existing condensed linear solve/preconditioner operation, with its dependency,
mutation, collective and lifetime contract documented before editing.
