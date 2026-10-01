#!/usr/bin/env bash
# Preserve individual failures; the suite exit code alone is not the verdict.
set -u
export PATH=/opt/gcc/12.4.0/bin:/opt/openmpi/5.0.6/bin:$PATH
root=benchmarks/reconstructed_fault/refactoring_r3b
variant=$1
binary=build-refactor-r3b/aspect-release
if [ "$variant" = reference ]; then binary=build-restart-fix/aspect-r3a-corrected-qualified; fi
for ranks in 1 2; do
 python3 "$root/run_logged.py" "$variant-unit-$ranks" 180 mpirun -np "$ranks" "$binary" --test '[phase_field_fault_cohesive],MaxwellStress*,Phase-field physical*,I_h distinguishes*,[fault_state_limiter],Stage-I*,[fault_slip_restart],[fault_prescribed_v],ReconstructedFaultManager checkpoint*'
done
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
 for mode in create resume; do
  python3 "$root/run_logged.py" "cohesive-$mode-one" 180 mpirun -np 1 "$binary" "$root/inputs/cohesive-$mode-one.prm"
 done
 for mode in legacy-one legacy-two automatic-one automatic-two; do
  ranks=1
  case "$mode" in *-two) ranks=2;; esac
  python3 "$root/run_logged.py" "candidate-bp3-$mode" 600 mpirun -np "$ranks" "$binary" "$root/inputs/candidate-bp3-$mode.prm"
 done
 python3 "$root/run_logged.py" branch-reference-checkpoint 60 bash benchmarks/reconstructed_fault/bp3/branch_output.sh benchmarks/reconstructed_fault/refactoring_r2b_cache/output-reference-bp3-automatic-one 2 "$root/output-candidate-bp3-automatic-split"
 python3 "$root/run_logged.py" candidate-bp3-automatic-split 600 mpirun -np 2 "$binary" "$root/inputs/candidate-bp3-automatic-split.prm"
fi
