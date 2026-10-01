#!/usr/bin/env bash
set -u
export PATH=/opt/gcc/12.4.0/bin:/opt/openmpi/5.0.6/bin:$PATH
root=benchmarks/reconstructed_fault/refactoring_r4b_residual
variant=candidate
binary=build-refactor-r4b-residual/aspect-release
# The extended checks use the original fixture environment, not BP3 switches.
for name in residual exhaustion pressure; do
 for ranks in 1 2; do
  python3 "$root/run_logged.py" "$variant-$name-$ranks" 180 mpirun -np "$ranks" "$binary" "$root/inputs/$variant-$name-$ranks.prm"
 done
done
python3 "$root/run_logged.py" "$variant-gmg-1" 180 mpirun -np 1 "$binary" "$root/inputs/$variant-gmg-1.prm"
if [ "$variant" = candidate ]; then
 for ranks in 1 2; do
  python3 "$root/run_logged.py" "candidate-unit-$ranks" 180 mpirun -np "$ranks" "$binary" --test '[reconstructed_fault_condensation],Stage-I*'
  python3 "$root/run_logged.py" "candidate-rollback-original-$ranks" 180 mpirun -np "$ranks" "$binary" "$root/inputs/candidate-rollback-original-$ranks.prm"
 done
 source benchmarks/reconstructed_fault/bp3/environment.sh
 for mode in legacy-one legacy-two automatic-one automatic-two; do
  ranks=1
  case "$mode" in *-two) ranks=2;; esac
  python3 "$root/run_logged.py" "candidate-bp3-$mode" 300 mpirun -np "$ranks" "$binary" "$root/inputs/candidate-bp3-$mode.prm"
 done
fi
