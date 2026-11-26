#!/bin/sh
# Run the same 5B schedules with the ordinary executable and its returned-row
# recorder. The unchanged strict audit returns 1 for the four strip-15 angles.
set -eu
if [ "$#" -ne 7 ]; then
  echo "Usage: sh $0 INSTALL RECORDER GROWTH_PLUGIN MESH_PLUGIN DONOR_RUNS REFERENCE FRESH_RUNS" >&2
  exit 2
fi
install=$(realpath "$1")
recorder=$(realpath "$2")
growth_plugin=$(realpath "$3")
mesh_plugin=$(realpath "$4")
donor_runs=$(realpath "$5")
reference=$(realpath "$6")
runs=$(realpath -m "$7")
test ! -e "$runs"
mkdir -p "$runs"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DEAL_II_NUM_THREADS=1 UCX_LOG_LEVEL=error
bench=$install/benchmarks/growing_fault/timestep
for kind in ordinary rows; do
  case "$kind" in
    ordinary) exe=$install/bin/aspect-release ;;
    rows) exe=$recorder ;;
  esac
  for schedule in parent coarse half; do
    python3 -B "$bench/run.py" "$schedule" "$runs/$kind/$schedule" "$mesh_plugin" \
      --aspect "$exe" --plugin "$growth_plugin" --parent "$runs/$kind/parent/output"
  done
done
for schedule in parent coarse half; do
  ln -s "$donor_runs/donor-$schedule" "$runs/ordinary/donor-$schedule"
done
python3 -B "$bench/audit.py" "$runs/ordinary" "$reference" "$runs/strict" \
  --returned-rows "$runs/rows"
