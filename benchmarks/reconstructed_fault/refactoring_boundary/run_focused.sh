#!/usr/bin/env bash
# Run after staging inputs and building the focused plugin. Logs are never overwritten.
set -eu
root=benchmarks/reconstructed_fault/refactoring_boundary
for name in interior interior_touch perpendicular oblique reversed left_top multiple curved oblique-two multiple-two; do
  ranks=1
  case "$name" in *-two) ranks=2;; esac
  python3 "$root/run_logged.py" "qualified-$name" 180 /opt/openmpi/5.0.6/bin/mpirun -np "$ranks" build-refactor-boundary/aspect-release "$root/inputs/$name.prm"
done
for ranks in 1 2; do
  python3 "$root/run_logged.py" "qualified-contact-unit-$ranks" 120 /opt/openmpi/5.0.6/bin/mpirun -np "$ranks" build-refactor-boundary/aspect-release --test '[fault_boundary_contact]'
done
for name in corner tangential crossing h-driven material; do
  unset ASPECT_TEST_BOUNDARY_H_DRIVEN
  case "$name" in
    material) expected="Automatic exterior material data are unqualified";;
    corner) expected=corner;;
    tangential) expected=tangential;;
    crossing) expected=topology;;
    h-driven) export ASPECT_TEST_BOUNDARY_H_DRIVEN=1; expected='verified compatible fully';;
  esac
  if python3 "$root/run_logged.py" "qualified-rejection-$name" 120 /opt/openmpi/5.0.6/bin/mpirun -np 2 build-refactor-boundary/aspect-release "$root/inputs/$name.prm"; then
    echo "Expected rejection for $name"; exit 1
  fi
  rg -q "$expected" "$root/evidence/qualified-rejection-$name.log"
done
