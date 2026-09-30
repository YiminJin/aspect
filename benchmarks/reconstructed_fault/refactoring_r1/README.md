# R1 local reference harness

This directory contains only qualification tools. Production code and existing
tests are unchanged. Results and decisions belong in
[the rolling review](../../../doc/reconstructed_fault/refactor_review.md).
Ignored build, input, evidence, output and checkpoint directories are deliberately
retained locally as reference artifacts; do not reuse their names for candidates.

Run from the refactoring worktree root. The source baseline is
`3af7c2aa2590aa8e39eaf2a1e46df0bc6c9d9700`. Record/check source hashes before any
candidate comparison. `stage_inputs.py` recreates the recorded PRMs and copies
only four immutable convex fixture files from the sibling scientific worktree.
It refuses to replace existing inputs. The templates retain the fixture's
physical settings and full bottom constraint. Library paths and output paths
are rebound, full-state observation is enabled, and the old heap/map-size
observation plugin is omitted. No saved scientific run is executed or modified.

Configure/build (fresh directories only):

```sh
export PATH=/opt/gcc/12.4.0/bin:/opt/openmpi/5.0.6/bin:$PATH
cmake -S . -B build-refactor-baseline \
  -DCMAKE_C_COMPILER=/opt/openmpi/5.0.6/bin/mpicc \
  -DCMAKE_CXX_COMPILER=/opt/openmpi/5.0.6/bin/mpic++ \
  -DCMAKE_Fortran_COMPILER=/opt/openmpi/5.0.6/bin/mpifort \
  -DDEAL_II_DIR=/opt/dealii/9.6-local -DASPECT_WITH_VORO=ON \
  -DVORO_DIR=/home/ein/local/voro++/0.4.6 -DASPECT_WITH_NETCDF=OFF \
  '-DASPECT_ADDITIONAL_CXX_FLAGS=-fno-finite-math-only -ffp-contract=off' \
  -DCMAKE_BUILD_TYPE=Release -DASPECT_RUN_ALL_TESTS=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build-refactor-baseline -j2
cmake -S benchmarks/reconstructed_fault/bp3/plugin \
  -B benchmarks/reconstructed_fault/refactoring_r1/plugin-build \
  -DAspect_DIR="$PWD/build-refactor-baseline" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/opt/openmpi/5.0.6/bin/mpic++
cmake --build benchmarks/reconstructed_fault/refactoring_r1/plugin-build -j2
cmake -S benchmarks/reconstructed_fault/refactoring_r1 \
  -B benchmarks/reconstructed_fault/refactoring_r1/probe-build \
  -DAspect_DIR="$PWD/build-refactor-baseline" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/opt/openmpi/5.0.6/bin/mpic++
cmake --build benchmarks/reconstructed_fault/refactoring_r1/probe-build -j1
python3 benchmarks/reconstructed_fault/refactoring_r1/stage_inputs.py
bash benchmarks/reconstructed_fault/refactoring_r1/run_focused.sh
bash benchmarks/reconstructed_fault/refactoring_r1/run_trajectory.sh
python3 benchmarks/reconstructed_fault/refactoring_r1/compare_states.py
```

MPI requires local sockets; these runs needed sandbox escalation. Each bounded
invocation records its command, selected environment, elapsed time and exit
code beside its log. Shell suites continue after known baseline failures;
their final exit status is **not** an aggregate qualification verdict.
The limiter probe catches individual parser outcomes and restores its temporary
Theta change. Its successful process exit does not imply the parser contract
passed. Forced rollback runs intentionally fail the nonlinear solve: require
both an accepted Newton update and the history-restoration marker.

The `ih-converged*`, `ih-no-composition-converged`, and `ih-cell*` wrappers
increase only the phase initialization iteration budget to 50. They keep the
original tolerance and assertion code. The separate `rollback-open-top*`
fixtures reuse the original rollback plugin and its two-iteration forced
failure, with the traction-free-top boundary pattern already used by the
state-limiter fixture. They are supplemental: inspection of the original raw
logs subsequently confirmed that the closed-box fixtures also reached and
passed rollback. Their shell filters fail on the newly printed `alpha` suffix.
No original test or expectation was edited.

The trajectory script runs six physical steps on one and two ranks, reads the
actual retained checkpoint metadata to select accepted step four, branches to
a new directory with the existing BP3 tool, and resumes on one rank. Logs,
full-state CSVs, expanded parameters, native fault output and checkpoints remain
available for later candidate comparisons. Same-rank restart and cross-rank
comparisons must be reported separately; tiny rates must never be scaled by one.
