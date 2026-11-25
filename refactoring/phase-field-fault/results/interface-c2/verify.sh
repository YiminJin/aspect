#!/usr/bin/env bash
# From the worktree root: verify.sh GROUP BUILD PREFIX PLUGINS FRESH_OUTPUT MPIEXEC
set -eu
group=$1
B=$(realpath "$2")
I=$(realpath "$3")
P=$(realpath "$4")
D=$(realpath -m "$5")
MPI=$6
A=$I/bin/aspect-release
G=$I/benchmarks/growing_fault
R=$I/benchmarks/fixed_fault/phase_ridge
L=$I/benchmarks/layered_moving_fault
growth=$P/growing_fault/libgrowing_fault_checks.release.so
ridge=$P/fixed_fault/phase_ridge/libphase_ridge.release.so
layered=$P/layered_moving_fault/liblayered_moving_fault.release.so
schedule=$P/growing_fault/timestep/libmidpoint_fields.release.so
export DEAL_II_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p "$D"
case "$group" in
public)
  python3 "$G/check_interface.py" "$A" "$growth" "$D/public"
  # Compile/load only the law. No material or postprocessor registration is
  # linked in this library. Compare both routes to the qualified dashpot runs.
  for c in persistent dashpot-strip; do
    mkdir "$D/solo-$c"
    cp "$D/public/$c/"*.prm "$D/solo-$c/"
    echo "set Additional shared libraries = $P/tests/libdashpot.release.so" >> "$D/solo-$c/input.prm"
    (cd "$D/solo-$c" && "$A" input.prm > log)
  done
  cmp "$D/solo-persistent/output/fault-labels-00002.txt" "$D/public/persistent/output/fault-labels-00002.txt"
  cmp "$D/solo-dashpot-strip/output/sections-0001.csv" "$D/public/dashpot-strip/output/sections-0001.csv"
  ;;
serial)
  (cd "$B/unit_tests" && "$A" --test)
  ctest --test-dir "$B" -R postprocess/phase_field_surface_energy_at2 --output-on-failure --no-tests=error
  python3 "$G/check_adiabatic_pressure.py" "$A" "$growth" "$D/constant" constant --stage-b build-interface-b/final/compression --pressure-plugin "$P/tests/libpressure_constraint.release.so"
  python3 "$G/check_adiabatic_pressure.py" "$A" "$growth" "$D/history" lifecycle
  python3 "$R/check_pressure.py" "$A" "$ridge" "$D/trace-history" --lifecycle
  python3 "$L/check_restart_width.py" "$A" "$layered" "$D/width" --seeds build-interface-a/old-width
  python3 "$G/check_repairs.py" "$A" "$growth" "$D/archives" lifecycle --legacy-snapshot build-5a-repair/before-checkpoint/restart/03 --repaired-checkpoints build-5a-repair/qualified-lifecycle/clean/output/restart
  python3 "$G/check_repairs.py" "$A" "$growth" "$D/cold" friction
  python3 "$R/check_normalization.py" "$A" "$ridge" "$D/normalization"
  python3 "$R/run.py" "$A" "$ridge" "$D/ridge/free-32" free 32
  (cd "$D" && PYTHONPATH="$R" python3 -c 'from pathlib import Path; from check import check_case; check_case(Path("ridge"),"free",32)')
  python3 "$I/benchmarks/fixed_fault/check_stress_output.py" "$A" "$D/stress"
  ;;
mpi)
  python3 "$L/check_vertex_file.py" "$A" "$layered" "$D/vertices" --mpiexec "$MPI"
  python3 "$R/check_pressure.py" "$A" "$ridge" "$D/trace1"
  python3 "$R/check_pressure.py" "$A" "$ridge" "$D/trace2" --mpiexec "$MPI" --ranks 2 --serial "$D/trace1"
  python3 "$I/benchmarks/explicit_dg_transport/check_3d.py" "$A" "$P/explicit_dg_transport" "$D/slabs" --slabs-only --mpiexec "$MPI"
  python3 "$G/check_state_free.py" "$A" "$growth" "$D/law1"
  python3 "$G/check_state_free.py" "$A" "$growth" "$D/law2" --ranks 2 --mpiexec "$MPI"
  ;;
stress)
  W=$PWD
  for spec in dg_stress_transport dg_stress_spin_order; do
    dir=$D/$spec
    mkdir "$dir"
    cp "tests/$spec.prm" "$dir/input.prm"
    echo "set Additional shared libraries = $P/tests/lib$spec.release.so" >> "$dir/input.prm"
    echo "set Output directory = output-$spec" >> "$dir/input.prm"
    (cd "$dir" && "$A" input.prm > log)
  done
  (cd "$D/dg_stress_transport" && bash "$W/tests/dg_stress_transport.sh" </dev/null)
  diff -u tests/dg_stress_spin_order/spin-check "$D/dg_stress_spin_order/output-dg_stress_spin_order/spin-check"
  ;;
moving)
  python3 "$L/run.py" "$A" "$layered" "$P/tests/libfixed_fault.release.so" "$D/layered" --mpiexec "$MPI"
  python3 -O "$L/check.py" "$D/layered" build-4c/donor/output "$D/layered.png" --zero build-4c/donor-zero/output --reverse build-4c/donor-reverse/output
  ;;
profile)
  python3 "$G/check_adiabatic_pressure.py" "$A" "$growth" "$D/profile1" profile
  # Propagate the strict scientific result to the caller.
  python3 "$G/check_adiabatic_pressure.py" "$A" "$growth" "$D/profile2" profile --ranks 2 --mpiexec "$MPI" --serial "$D/profile1"
  ;;
frozen)
  python3 "$G/run.py" "$A" "$growth" "$D/nine" --principal-only --mpiexec "$MPI"
  python3 "$G/compare.py" build-5a-repair/donor-two/output "$D/nine/principal/output"
  python3 "$G/timestep/run.py" parent "$D/strip18/parent" "$schedule" --aspect "$A" --plugin "$growth" --launcher "$MPI"
  python3 "$G/timestep/run.py" coarse "$D/strip18/coarse" "$schedule" --aspect "$A" --plugin "$growth" --launcher "$MPI" --parent "$D/strip18/parent/output"
  python3 "$G/timestep/run.py" half "$D/strip18/half" "$schedule" --aspect "$A" --plugin "$growth" --launcher "$MPI"
  for c in parent coarse half; do ln -s "$PWD/build-5b/donor-$c" "$D/strip18/donor-$c"; done
  # Source-side diagnosis needs the documented independent donor archives.
  python3 benchmarks/growing_fault/timestep/audit.py "$D/strip18" build-5a-repair/reference "$D/strict" --returned-rows build-5b/rows
  ;;
*) exit 2 ;;
esac
