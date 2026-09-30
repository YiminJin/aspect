#!/usr/bin/env bash
# Individual recorded exit codes, including expected failures, are the verdict.
set -u
export PATH=/opt/gcc/12.4.0/bin:/opt/openmpi/5.0.6/bin:$PATH
root=benchmarks/reconstructed_fault/restart_fix
binary=build-restart-fix/aspect-fix
python3 "$root/run_logged.py" regression-before-fix 90 mpirun -np 1 build-restart-fix/aspect-before-fix --test '[fault_slip_restart]'
for ranks in 1 2; do
 python3 "$root/run_logged.py" "unit-$ranks" 180 mpirun -np "$ranks" "$binary" --test '[fault_slip_restart],[fault_prescribed_v],ReconstructedFaultManager checkpoint*,Stage-I*'
done
source benchmarks/reconstructed_fault/bp3/environment.sh
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
