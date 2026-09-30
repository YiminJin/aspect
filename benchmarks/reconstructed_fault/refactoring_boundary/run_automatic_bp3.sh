#!/usr/bin/env bash
set -eu
source benchmarks/reconstructed_fault/bp3/environment.sh
root=benchmarks/reconstructed_fault/refactoring_boundary
for ranks in 1 2; do
  name=one
  if [ "$ranks" = 2 ]; then name=two; fi
  python3 "$root/run_logged.py" "qualified-bp3-$name" 600 /opt/openmpi/5.0.6/bin/mpirun -np "$ranks" build-refactor-boundary/aspect-release "$root/inputs/final-bp3-$name.prm"
done
bash benchmarks/reconstructed_fault/bp3/branch_output.sh "$root/output-qualified-bp3-one" 2 "$root/output-qualified-bp3-split"
python3 "$root/run_logged.py" qualified-bp3-split 600 /opt/openmpi/5.0.6/bin/mpirun -np 2 build-refactor-boundary/aspect-release "$root/inputs/final-bp3-split.prm"
python3 "$root/compare_automatic.py"
