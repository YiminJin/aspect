#!/usr/bin/env bash
set -eu
export PATH=/opt/gcc/12.4.0/bin:/opt/openmpi/5.0.6/bin:$PATH
source benchmarks/reconstructed_fault/bp3/environment.sh
root=benchmarks/reconstructed_fault/refactoring_r2b_cache
for variant in reference candidate; do
 binary=build-refactor-r2b-cache/aspect-release
 if [ "$variant" = reference ]; then binary=build-refactor-boundary/aspect-boundary-qualified; fi
 bash benchmarks/reconstructed_fault/bp3/branch_output.sh "$root/output-reference-bp3-automatic-one" 2 "$root/output-$variant-bp3-automatic-split"
 python3 "$root/run_logged.py" "$variant-bp3-automatic-split" 600 mpirun -np 2 "$binary" "$root/inputs/$variant-bp3-automatic-split.prm"
done
