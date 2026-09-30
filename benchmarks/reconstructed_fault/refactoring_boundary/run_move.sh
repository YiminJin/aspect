#!/usr/bin/env bash
set -eu
export PATH=/opt/gcc/12.4.0/bin:/opt/openmpi/5.0.6/bin:$PATH
source benchmarks/reconstructed_fault/bp3/environment.sh
root=benchmarks/reconstructed_fault/refactoring_boundary
for ranks in 1 2; do
  name=bp3-one
  if [ "$ranks" = 2 ]; then name=bp3-two; fi
  python3 "$root/run_logged.py" "move-$name" 600 mpirun -np "$ranks" \
    build-refactor-boundary/aspect-completion-move "$root/inputs/$name.prm"
done
python3 "$root/compare_move.py"
