# R3a move-only qualification

Reference: committed R2b proposal 3 (`06b70740f`), preserved executable
`build-refactor-r2b-cache/aspect-cache-qualified`. Candidate:
`build-refactor-r3a/aspect-release`. See the pre-edit ownership/lifecycle table
and final disposition in [the rolling review](../../../doc/reconstructed_fault/refactor_review.md).
All reference source/artifact snapshots, commands, exit codes and full logs are
retained under ignored `evidence/`; generated inputs/outputs and plugin builds
also remain local. The separate scientific worktree is not changed.

Configure with the same GCC 12.4.0 / OpenMPI 5.0.6 / deal.II 9.6.2 / Voro++
Release settings as R2b, including `-fno-finite-math-only -ffp-contract=off`.
The exact configure command is in `evidence/configure.json`. CMake compiles the
two new translation units without unity/PCH, preserving all 55 previous unity
groups. The entry file was also compiled separately without unity/PCH. The
38 explicit moved-member definitions for dimensions 2 and 3 are checked in
independent object symbol tables. These facts do not qualify a 3D simulation.

`move_methods.py` is the one-time relocation against the captured entry source.
It refuses an already-edited source. `verify_move.py` checks all 19 complete
method definitions and six helpers byte-for-byte, retained entry code, every
header, both R2 files and all other entry source/test files. No method declaration,
expression, MPI operation, call order, publication or rollback owner changed.

The following commands assume the captured baseline/snapshots exist. Runners
refuse to overwrite evidence logs and input staging refuses an existing inputs
folder. Use a new qualification directory for a repeat run.

```sh
python3 benchmarks/reconstructed_fault/refactoring_r3a/verify_move.py
cmake --build build-refactor-r3a -j2
cmake -S benchmarks/reconstructed_fault/refactoring_r3a/plugin \
  -B benchmarks/reconstructed_fault/refactoring_r3a/plugin-build \
  -DAspect_DIR="$PWD/build-refactor-r3a"
cmake --build benchmarks/reconstructed_fault/refactoring_r3a/plugin-build -j1
python3 benchmarks/reconstructed_fault/refactoring_r3a/stage_inputs.py
bash benchmarks/reconstructed_fault/refactoring_r3a/run_checks.sh reference
bash benchmarks/reconstructed_fault/refactoring_r3a/run_checks.sh candidate
bash benchmarks/reconstructed_fault/refactoring_r3a/run_stage_j_open_top.sh reference
bash benchmarks/reconstructed_fault/refactoring_r3a/run_stage_j_open_top.sh candidate
python3 benchmarks/reconstructed_fault/refactoring_r3a/run_cohesive_restart.py reference
python3 benchmarks/reconstructed_fault/refactoring_r3a/run_cohesive_restart.py candidate
python3 benchmarks/reconstructed_fault/refactoring_r3a/compare_bp3.py
python3 benchmarks/reconstructed_fault/refactoring_r3a/compare_lifecycle.py
python3 benchmarks/reconstructed_fault/refactoring_r3a/compare_cohesive_checkpoint.py
```

The plugin project compiles unchanged existing test sources. Reference and
candidate load the same plugins. Unit, surface-temperature, frozen-stress and
rollback cases run on one/two ranks. The original Stage-J trajectory is retained
with its pressure-compatibility failure. A separately labelled supplemental
Stage-J case uses the existing traction-free-top boundary pattern and explicit
B/G environment, retaining all original numerical tolerances and assertions.
It verifies initial history and one physical Theta/H update, then encounters
the reference's second-step convergence failure. Its accepted checkpoint mesh/
particle/FE data and serialized existing history fingerprint are compared exactly.

The cohesive restart uses that identical reference checkpoint and the shared
restart observer in the create plugin. The resume wrapper's hard-coded copy
cannot be used with these output paths; its failed first invocation is retained.
The reference restored-state checks pass before a segmentation fault in manager
trial-value setup. This is separate, incomplete cohesive-restart qualification,
not a fix or a successful resumed step. Matched failure checks do not turn these
cases into passing scientific tests.

The four short mature/frozen BP3 legacy/automatic one/two-rank trajectories and
one automatic cross-rank checkpoint continuation reuse R2b's qualified inputs,
plugin libraries and exact comparison rules. Only output directories change.
The automatic restart uses the same accepted step-4 reference checkpoint as R2b.
All owned bulk values, particle histories, fault fields, accepted solver decisions
and cache/work diagnostics are compared; elapsed time is excluded.

Inspect individual run JSONs: a suite's shell exit status is not its verdict.
No tolerances, expected numerical results or original test assertions are relaxed.
