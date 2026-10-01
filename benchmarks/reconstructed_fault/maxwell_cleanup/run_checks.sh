#!/usr/bin/env bash
set -u
export PATH=/opt/gcc/12.4.0/bin:/opt/openmpi/5.0.6/bin:$PATH
root=benchmarks/reconstructed_fault/maxwell_cleanup
binary=build-refactor-r3b/aspect-release
for ranks in 1 2; do
 python3 "$root/run_logged.py" "unit-$ranks" 180 mpirun -np "$ranks" "$binary" --test '[phase_field_fault_cohesive],MaxwellStress*,[fault_state_limiter],Stage-I*,[fault_slip_restart],[fault_prescribed_v],ReconstructedFaultManager checkpoint*'
 for name in frozen-stress temperature; do
  python3 "$root/run_logged.py" "candidate-$name-$ranks" 180 mpirun -np "$ranks" "$binary" "$root/inputs/candidate-$name-$ranks.prm"
 done
done
source benchmarks/reconstructed_fault/bp3/environment.sh
for mode in legacy-one legacy-two automatic-one automatic-two; do
 ranks=1
 case "$mode" in *-two) ranks=2;; esac
 python3 "$root/run_logged.py" "candidate-bp3-$mode" 300 mpirun -np "$ranks" "$binary" "$root/inputs/candidate-bp3-$mode.prm"
done
python3 "$root/run_logged.py" branch-reference-checkpoint 60 bash benchmarks/reconstructed_fault/bp3/branch_output.sh benchmarks/reconstructed_fault/refactoring_r2b_cache/output-reference-bp3-automatic-one 2 "$root/output-candidate-bp3-automatic-split"
python3 "$root/run_logged.py" candidate-bp3-automatic-split 300 mpirun -np 2 "$binary" "$root/inputs/candidate-bp3-automatic-split.prm"
