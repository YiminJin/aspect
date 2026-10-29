# Files revised for the native cohesive–RSF task

Compared with the actual entry snapshots, not a clean HEAD checkout. Pre-existing
manual scientific work is preserved and is included where required by the source
snapshot; it is not claimed as a new change from this task.

## Original files changed (12)

- `source/simulator/core.cc`
- `source/simulator/checkpoint_restart.cc`
- `source/simulator/parameters.cc`
- `source/simulator/solver_schemes.cc`
- `include/aspect/simulator.h`
- `include/aspect/parameters.h`
- `include/aspect/reconstructed_fault/cohesive_interface.h`
- `doc/reconstructed_fault/specification.tex`
- `doc/reconstructed_fault/current_design.md`
- `doc/reconstructed_fault/particle_free/COHESIVE_RSF_NATIVE_STAGES.md`
- `doc/reconstructed_fault/particle_free/SOURCE_AUDIT.md`
- `doc/reconstructed_fault/particle_free/STATUS.md`

The first six source/header hooks select the default-off route, invoke its actual
outer lifecycle and conditionally checkpoint its state. `cohesive_interface.h`
adds serialization of all six physical point histories; its point-return `.cc`
and the accepted law are unchanged. Design/specification and status/audit files
record the selected option and measured scope.

## New native implementation files

- `include/aspect/reconstructed_fault/cohesive_bulk_history.h`
- `include/aspect/reconstructed_fault/cohesive_evolution.h`
- `include/aspect/reconstructed_fault/cohesive_geometry.h`
- `include/aspect/reconstructed_fault/cohesive_system.h`
- `include/aspect/reconstructed_fault/cohesive_system_internal.h`
- `source/reconstructed_fault/cohesive_advance.cc`
- `source/reconstructed_fault/cohesive_bulk_history.cc`
- `source/reconstructed_fault/cohesive_continuation.cc`
- `source/reconstructed_fault/cohesive_geometry.cc`
- `source/reconstructed_fault/cohesive_initialization.cc`
- `source/reconstructed_fault/cohesive_mechanics.cc`
- `source/reconstructed_fault/cohesive_output.cc`
- `source/reconstructed_fault/cohesive_state.cc`
- `source/reconstructed_fault/cohesive_system.cc`
- `source/simulator/solver/cohesive_rsf.cc`

## Benchmark, reproducibility and evidence

- `benchmarks/reconstructed_fault/particle_free/cohesive_rsf_native/`: CMake,
  `checks.cc`, `pilot.cc`, nine parameter files, README, bounded `qualify.py`,
  accepted-output `analyze.py`, independent saved-state `verify.py`, `package.py`.
- `benchmarks/reconstructed_fault/particle_free/cohesive_interface_evolution.h`:
  retained include now delegates to the one native signed solver. Its entry
  version is preserved in `build-particle-free/cohesive-rsf-native/retained-solver-source.zip`.
- `doc/reconstructed_fault/particle_free/COHESIVE_RSF_NATIVE_{FORMULATION,STAGES,REPORT}.md`.
- `benchmarks/reconstructed_fault/particle_free/evidence/cohesive_rsf_native/`:
  compact measured/indexed data, figures, source/environment manifests and native
  restart anchor; adjacent `cohesive_rsf_native_evidence.zip` is the final bundle.

No BP3 file changed, no entry-audited file is missing, and no existing failed
scientific evidence was removed. Exact file hashes and entry-to-current core/docs
patch are delivered separately. Build outputs and checkpoints remain outside
source control. No commit was requested for this task.
