#!/usr/bin/env bash
# Preserve individual failures; the suite exit status is not its verdict.
set -u
export PATH=/opt/gcc/12.4.0/bin:/opt/openmpi/5.0.6/bin:$PATH
root=benchmarks/reconstructed_fault/refactoring_r3a
variant=$1
binary=build-refactor-r3a/aspect-release
if [ "$variant" = reference ]; then binary=build-refactor-r2b-cache/aspect-cache-qualified; fi
for ranks in 1 2; do
 python3 "$root/run_logged.py" "$variant-unit-$ranks" 180 mpirun -np "$ranks" "$binary" --test '[phase_field_fault_cohesive],MaxwellStress*,Phase-field physical*,I_h distinguishes*,[fault_state_limiter],Stage-I*'
done
# Original fixtures run under their original numerical environment.
for name in stage-j temperature frozen-stress rollback-original; do
 for ranks in 1 2; do
  python3 "$root/run_logged.py" "$variant-$name-$ranks" 300 mpirun -np "$ranks" "$binary" "$root/inputs/$variant-$name-$ranks.prm"
 done
done
source benchmarks/reconstructed_fault/bp3/environment.sh
for ranks in 1 2; do
 python3 "$root/run_logged.py" "$variant-rollback-open-top-$ranks" 300 mpirun -np "$ranks" "$binary" "$root/inputs/$variant-rollback-open-top-$ranks.prm"
done
if [ "$variant" = candidate ]; then
 for name in legacy-one legacy-two automatic-one automatic-two; do
  ranks=1
  case "$name" in *-two) ranks=2;; esac
  python3 "$root/run_logged.py" "candidate-bp3-$name" 600 mpirun -np "$ranks" "$binary" "$root/inputs/candidate-bp3-$name.prm"
 done
 python3 "$root/run_logged.py" branch-reference-checkpoint 60 bash benchmarks/reconstructed_fault/bp3/branch_output.sh benchmarks/reconstructed_fault/refactoring_r2b_cache/output-reference-bp3-automatic-one 2 "$root/output-candidate-bp3-automatic-split"
 python3 "$root/run_logged.py" candidate-bp3-automatic-split 600 mpirun -np 2 "$binary" "$root/inputs/candidate-bp3-automatic-split.prm"
fi
