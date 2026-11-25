#!/usr/bin/env bash
# Run the distributed examples from an installed Release prefix in a fresh directory.
# Usage: bash PREFIX/benchmarks/verify_phase_field_fault.sh DEST [MPIEXEC]
set -eu
examples=$(cd "$(dirname "$0")" && pwd -P)
export ASPECT_SOURCE_DIR=$(dirname "$examples")
aspect="$ASPECT_SOURCE_DIR/bin/aspect-release"
mpi=${2:-mpirun}
out=$(realpath -m "${1:?supply a new output directory}")
mkdir "$out"
cd "$out"
export DEAL_II_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
energy="$examples/phase_field_surface_energy"
ridge="$examples/fixed_fault/phase_ridge"
moving="$examples/layered_moving_fault"
growth="$examples/growing_fault"
python3 "$energy/check.py" "$aspect" "$out/energy"
python3 "$energy/check.py" "$aspect" "$out/energy-mpi" --mpi "$mpi"
python3 "$ridge/run.py" "$aspect" "$ridge/libphase_ridge.release.so" "$out/ridge/free-32" free 32
PYTHONPATH="$ridge" python3 -c 'from pathlib import Path; from check import check_case; check_case(Path("ridge"), "free", 32)'
python3 "$moving/check_review.py" "$aspect" "$moving/liblayered_moving_fault.release.so" "$out/moving"
python3 "$growth/run.py" "$aspect" "$growth/libgrowing_fault_checks.release.so" "$out/growing" --mpiexec "$mpi"
python3 "$growth/check_history.py" "$out/growing/principal/output"
scalar="$examples/explicit_dg_transport"
python3 "$scalar/check_3d.py" "$aspect" "$scalar" "$out/scalar" --slabs-only --mpiexec "$mpi"

# The coarse continuation reads the checkpoint saved after the 5 ms solve.
# The fine schedule starts afresh and must publish the same parent.
schedule="$growth/timestep"
mkdir strip18
for case in parent coarse half; do
  extra=()
  if [[ "$case" == coarse ]]; then extra=(--parent "$out/strip18/parent/output"); fi
  python3 "$schedule/run.py" "$case" "$out/strip18/$case" "$schedule/libmidpoint_fields.release.so" \
    --aspect "$aspect" --plugin "$growth/libgrowing_fault_checks.release.so" --launcher "$mpi" "${extra[@]}"
done
for step in 0000 0001; do
  for table in strip18/parent/output/*-"$step"*.csv; do
    cmp "$table" "strip18/half/output/$(basename "$table")"
  done
done
echo '18-strip schedules completed with identical seed and 5 ms parent tables.'
MPLBACKEND=Agg MPLCONFIGDIR="$out/mpl" python3 "$schedule/plot_phase.py" strip18 strip18/figures
