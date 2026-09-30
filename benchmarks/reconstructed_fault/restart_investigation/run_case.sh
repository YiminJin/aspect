#!/usr/bin/env bash
set -eu
export PATH=/opt/gcc/12.4.0/bin:/opt/openmpi/5.0.6/bin:$PATH
source benchmarks/reconstructed_fault/bp3/environment.sh
root=benchmarks/reconstructed_fault/restart_investigation
case_name=$1
binary=build-refactor-r3a/aspect-r3a-qualified
if [[ "$case_name" == debug-* ]]; then binary=build-restart-investigation/aspect-symbols; fi
python3 "$root/run_logged.py" "$case_name" 180 mpirun -np 1 "$binary" "$root/inputs/$case_name.prm"
