#!/usr/bin/env bash
set -eu
source benchmarks/reconstructed_fault/bp3/environment.sh
root=benchmarks/reconstructed_fault/refactoring_boundary
python3 "$root/run_logged.py" qualified-legacy-one 600 /opt/openmpi/5.0.6/bin/mpirun -np 1 build-refactor-boundary/aspect-release "$root/inputs/qualified-legacy-one.prm"
