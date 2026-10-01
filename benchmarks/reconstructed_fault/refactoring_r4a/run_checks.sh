#!/usr/bin/env bash
# Individual runner records are the verdict; continue to record all selected cases.
set -u
export PATH=/opt/gcc/12.4.0/bin:/opt/openmpi/5.0.6/bin:$PATH
root=benchmarks/reconstructed_fault/refactoring_r4a
variant=$1
binary=build-refactor-r4a/aspect-release
if [ "$variant" = reference ]; then binary=build-refactor-r3b/aspect-maxwell-qualified; fi
for ranks in 1 2; do
 python3 "$root/run_logged.py" "$variant-unit-$ranks" 180 mpirun -np "$ranks" "$binary" --test '[reconstructed_fault_condensation],Stage-I*'
 python3 "$root/run_logged.py" "$variant-rollback-original-$ranks" 180 mpirun -np "$ranks" "$binary" "$root/inputs/$variant-rollback-original-$ranks.prm"
done
python3 "$root/run_logged.py" "$variant-ordinary-amg" 180 mpirun -np 1 "$binary" "$root/inputs/$variant-ordinary-amg.prm"
if [ "$variant" = candidate ]; then
 source benchmarks/reconstructed_fault/bp3/environment.sh
 for mode in legacy-one legacy-two automatic-one automatic-two; do
  ranks=1
  case "$mode" in *-two) ranks=2;; esac
  python3 "$root/run_logged.py" "candidate-bp3-$mode" 300 mpirun -np "$ranks" "$binary" "$root/inputs/candidate-bp3-$mode.prm"
 done
fi
