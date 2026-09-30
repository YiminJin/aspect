#!/usr/bin/env bash
# Run from the refactoring worktree root. Continue after baseline failures.
set -u
export PATH=/opt/gcc/12.4.0/bin:/opt/openmpi/5.0.6/bin:$PATH
export CMAKE_BUILD_PARALLEL_LEVEL=2
runner=benchmarks/reconstructed_fault/refactoring_r1/run_logged.py
binary=build-refactor-baseline/aspect-release
for ranks in 1 2; do
  python3 "$runner" "ih-accuracy-$ranks" 600 mpirun -np "$ranks" "$binary" --test '[phase_field_fault_ih_accuracy]'
  python3 "$runner" "ih-cache-$ranks" 300 mpirun -np "$ranks" "$binary" --test '[phase_field_fault_ih_cache]'
done
python3 "$runner" unit-mechanics 300 mpirun -np 1 "$binary" --test '[phase_field_fault_cohesive],[reconstructed_fault_condensation],Stage-I*,MaxwellStress*,Phase-field physical*,I_h distinguishes*,[fault_surface_direct]'
python3 "$runner" unit-limiter 120 mpirun -np 1 "$binary" --test '[fault_state_limiter]'
for test in phase_field_fault_ih phase_field_fault_ih_mpi phase_field_fault_ih_no_composition phase_field_fault_frozen_stress phase_field_fault_frozen_stress_mpi phase_field_fault_stage_i_rollback phase_field_fault_stage_i_rollback_mpi convection_box_particles checkpoint_03_particles; do
  python3 "$runner" "ctest-$test" 900 ctest --test-dir build-refactor-baseline --output-on-failure -R "^$test$"
done
