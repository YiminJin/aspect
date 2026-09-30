#!/usr/bin/env bash
# Bounded matched-rank checks; stop on the first failed invocation.
set -eu
export PATH=/opt/gcc/12.4.0/bin:/opt/openmpi/5.0.6/bin:$PATH
root=benchmarks/reconstructed_fault/refactoring_r2b
runner="$root/run_logged.py"
binary=build-refactor-r2b/aspect-release
for ranks in 1 2; do
  python3 "$runner" "ih-accuracy-$ranks" 600 mpirun -np "$ranks" "$binary" --test '[phase_field_fault_ih_accuracy]'
  python3 "$runner" "ih-cache-$ranks" 300 mpirun -np "$ranks" "$binary" --test '[phase_field_fault_ih_cache]'
done
source benchmarks/reconstructed_fault/bp3/environment.sh
for name in ih-converged ih-no-composition-converged ih-cell bp3-one; do
  python3 "$runner" "$name" 600 mpirun -np 1 "$binary" "$root/inputs/$name.prm"
done
for name in ih-converged-two ih-cell-two bp3-two; do
  python3 "$runner" "$name" 600 mpirun -np 2 "$binary" "$root/inputs/$name.prm"
done
for ranks in 1 2; do
  name=ih-cell
  if [ "$ranks" = 2 ]; then name=ih-cell-two; fi
  for variant in reference candidate; do
    executable="$binary"
    if [ "$variant" = reference ]; then executable=build-refactor-r2a/aspect-release; fi
    python3 "$runner" "$variant-$name-warm" 600 env ASPECT_DISABLE_IH_VALUE_CACHE=1 \
      mpirun -np "$ranks" "$executable" "$root/inputs/$variant-$name-warm.prm"
  done
done
