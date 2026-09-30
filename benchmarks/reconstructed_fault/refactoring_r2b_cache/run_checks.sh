#!/usr/bin/env bash
set -eu
export PATH=/opt/gcc/12.4.0/bin:/opt/openmpi/5.0.6/bin:$PATH
source benchmarks/reconstructed_fault/bp3/environment.sh
root=benchmarks/reconstructed_fault/refactoring_r2b_cache
variant=$1
binary=build-refactor-r2b-cache/aspect-release
if [ "$variant" = reference ]; then binary=build-refactor-boundary/aspect-boundary-qualified; fi
for ranks in 1 2; do
  python3 "$root/run_logged.py" "$variant-cache-unit-$ranks" 180 mpirun -np "$ranks" "$binary" --test '[phase_field_fault_ih_cache]'
done
for name in cache-remote-one cache-remote-two cache-cell-one cache-cell-two cache-independent cache-no-composition warm-cell-one warm-cell-two bp3-legacy-one bp3-legacy-two bp3-automatic-one bp3-automatic-two; do
  ranks=1
  case "$name" in *-two) ranks=2;; esac
  unset ASPECT_DISABLE_IH_VALUE_CACHE
  case "$name" in warm-*) export ASPECT_DISABLE_IH_VALUE_CACHE=1;; esac
  python3 "$root/run_logged.py" "$variant-$name" 600 mpirun -np "$ranks" "$binary" "$root/inputs/$variant-$name.prm"
done
