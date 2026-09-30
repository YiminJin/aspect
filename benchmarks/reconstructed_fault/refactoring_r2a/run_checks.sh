#!/usr/bin/env bash
# Run from the worktree root after building the candidate and matching plugins.
# Each result is recorded separately; final shell status is not a suite verdict.
set -u
export PATH=/opt/gcc/12.4.0/bin:/opt/openmpi/5.0.6/bin:$PATH
export CMAKE_BUILD_PARALLEL_LEVEL=2
root=benchmarks/reconstructed_fault/refactoring_r2a
runner="$root/run_logged.py"
binary=build-refactor-r2a/aspect-release
# Smoke the actual separately linked members before the normal candidate suite.
python3 "$runner" separate-normalization 600 mpirun -np 1 build-refactor-r2a/aspect-separate --test '[phase_field_fault_ih_accuracy],[phase_field_fault_ih_cache]'
for ranks in 1 2; do
  python3 "$runner" "ih-accuracy-$ranks" 600 mpirun -np "$ranks" "$binary" --test '[phase_field_fault_ih_accuracy]'
  python3 "$runner" "ih-cache-$ranks" 300 mpirun -np "$ranks" "$binary" --test '[phase_field_fault_ih_cache]'
  python3 "$runner" "rollback-original-$ranks" 300 mpirun -np "$ranks" "$binary" "$root/inputs/rollback-original-$ranks.prm"
done
python3 "$runner" ordinary-particles 300 ctest --test-dir build-refactor-r2a --output-on-failure -R '^convection_box_particles$'

# Match R1's environment for the supplemental lifecycle and BP3 cases.
source benchmarks/reconstructed_fault/bp3/environment.sh
for name in ih-converged ih-no-composition-converged ih-cell rollback-open-top bp3-one; do
  python3 "$runner" "$name" 600 mpirun -np 1 "$binary" "$root/inputs/$name.prm"
done
for name in ih-converged-two ih-cell-two rollback-open-top-two bp3-two; do
  python3 "$runner" "$name" 600 mpirun -np 2 "$binary" "$root/inputs/$name.prm"
done
# This is the preserved R1-created checkpoint, NOT a new candidate checkpoint.
reference=benchmarks/reconstructed_fault/refactoring_r1/output-bp3-one
python3 - "$reference" <<'PY' || exit 1
import pathlib, sys
assert (pathlib.Path(sys.argv[1]) / 'restart/02/bp3_accepted_state.txt').read_text().split() == ['4', '400']
PY
python3 "$runner" baseline-branch 60 bash benchmarks/reconstructed_fault/bp3/branch_output.sh \
  "$reference" 02 "$root/output-bp3-split" || exit 1
python3 "$runner" baseline-restart 600 mpirun -np 1 "$binary" "$root/inputs/bp3-split.prm"
