#!/bin/sh
# ASPECT runs this hook in the test build directory after a successful solve.
set -eu
cat
benchmark="$(dirname "$0")/../benchmarks/fixed_fault"
python3 "$benchmark/compare_snapshots.py" "$benchmark/stationary-reference" output-stationary_fault
