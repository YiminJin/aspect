#!/usr/bin/env bash
# Run from the worktree root. Each case writes to a fresh output directory.
set -eu
W=$PWD
D=$(realpath "$2")
I=$(realpath "$3")
A=$I/bin/aspect-release
R=$I/benchmarks/fixed_fault/phase_ridge
F=$W/benchmarks/fixed_fault
L=$I/benchmarks/layered_moving_fault
P=$D/control-plugins
M=${MPIEXEC:-mpirun}
export PATH="$(dirname "$(command -v "$M")"):$PATH"
export ASPECT_SOURCE_DIR=$I OMP_NUM_THREADS=1 DEAL_II_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
RP=$P/phase_ridge/libphase_ridge.release.so
LP=$P/layered/liblayered_moving_fault.release.so
TP=$P/tests/libfixed_fault.release.so
case "$1" in
strict)
  # The historical 38-label table retains its original values and tolerances.
  python3 -uBO "$R/check.py" "$D/ridge" --retry
  ;;
independent)
  # Complete the other checks even when the strict historical comparison fails.
  PYTHONPATH="$R" python3 -uBO - "$D/ridge" "$4" "$5" <<'PYTHON'
import sys
from pathlib import Path
from check import check_case, check_retry, donor_check, rank_check, require
from moments import check_moments

root, donor, replay = map(Path, sys.argv[1:])
rank_check(root)
check_retry(root)
check_case(root, "free", 64)
for step in (1, 2):
    errors = [check_moments(root/f"free-{grid}", grid, step, .0125) for grid in (32, 64)]
    require(errors[1] < errors[0] or max(errors) <= .01, "Free cut moments failed refinement")
donor_check(root, donor, replay)
PYTHON
  ;;
plugins)
  mkdir -p "$P/test-src"
  cat > "$P/test-src/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.13.4)
find_package(Aspect 3.1 REQUIRED)
DEAL_II_INITIALIZE_CACHED_VARIABLES()
project(fault_observer)
add_library(fixed_fault SHARED "${TEST_SOURCE}/fixed_fault.cc")
ASPECT_SETUP_PLUGIN(fixed_fault)
CMAKE
  cmake -S "$R" -B "$P/phase_ridge" -DAspect_DIR="$I/lib/cmake/Aspect"
  cmake --build "$P/phase_ridge" -j1
  cmake -S "$P/test-src" -B "$P/tests" -DTEST_SOURCE="$W/tests" -DAspect_DIR="$I/lib/cmake/Aspect"
  cmake --build "$P/tests" -j1
  cmake -S "$L" -B "$P/layered" -DAspect_DIR="$I/lib/cmake/Aspect"
  cmake --build "$P/layered" -j1
  ;;
serial)
  for case in fixed_fault stationary_fault; do
    python3 -BO "$F/check_lifecycle.py" "$A" "$TP" "$D/lifecycle-$case" --case "$case"
  done
  python3 -B "$F/compare_snapshots.py" "$F/stationary-reference" "$D/lifecycle-stationary_fault/clean/output"
  python3 -B "$I/benchmarks/fixed_fault/check_stress_output.py" "$A" "$D/stress"
  ;;
ridge)
  for grid in 32 64; do
    for case in free inclined-fit inclined vertical narrow translated; do
      python3 -B "$R/run.py" "$A" "$RP" "$D/ridge/$case-$grid" "$case" "$grid"
    done
  done
  python3 -B "$R/run.py" "$A" "$RP" "$D/ridge/inclined-q1-64" inclined-q1 64
  ;;
retry)
  python3 -B "$R/run.py" "$A" "$RP" "$D/ridge/free-32-retry" free 32 --retry-plugin "$TP"
  ;;
mpi)
  python3 -B "$R/run.py" "$A" "$RP" "$D/ridge/free-32-mpi2" free 32 --ranks 2
  ;;
layered)
  python3 -B "$L/run.py" "$A" "$LP" "$TP" "$D/layered" --mpiexec "$M"
  python3 -BO "$L/check.py" "$D/layered" "$W/build-4c/donor/output" "$D/layered.png" --zero "$W/build-4c/donor-zero/output" --reverse "$W/build-4c/donor-reverse/output"
  ;;
*) echo "usage: $0 {plugins|serial|ridge|retry|mpi|layered|strict|independent} RUN_DIRECTORY INSTALLED_PREFIX [ARCHIVED_DONOR INTERVAL_REPLAY]" >&2; exit 1 ;;
esac
