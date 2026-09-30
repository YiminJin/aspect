#!/usr/bin/env bash
set -eu
export PATH=/opt/gcc/12.4.0/bin:/opt/openmpi/5.0.6/bin:$PATH
source benchmarks/reconstructed_fault/bp3/environment.sh
root=benchmarks/reconstructed_fault/refactoring_r3a
variant=$1
binary=build-refactor-r3a/aspect-release
if [ "$variant" = reference ]; then binary=build-refactor-r2b-cache/aspect-cache-qualified; fi
python3 "$root/run_logged.py" "$variant-stage-j-open-top-2" 300 mpirun -np 2 "$binary" "$root/inputs/$variant-stage-j-open-top-2.prm"
