#!/usr/bin/env bash
set -eu
export PATH=/opt/gcc/12.4.0/bin:/opt/openmpi/5.0.6/bin:$PATH
source benchmarks/reconstructed_fault/bp3/environment.sh
root=benchmarks/reconstructed_fault/restart_investigation
name=$1
python3 "$root/run_logged.py" "$name-gdb" 180 mpirun -np 1 gdb --batch -x "$root/inspect_manager.gdb" --args build-restart-investigation/aspect-symbols "$root/inputs/$name.prm"
