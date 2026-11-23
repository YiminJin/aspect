#!/bin/sh
# Check the same node and signed face oracles for the serial CTest.
set -eu
cat
PYTHONPATH="$(dirname "$0")/../benchmarks/fixed_fault" python3 -S - <<'PY'
from pathlib import Path
from check_transport import accepted, check_corrupt_jump, read_spatial, spatial_errors
output = Path("output-dg_stress_transport")
spatial_errors(read_spatial(output, 1))
accepted(output, 1)
check_corrupt_jump(output, 1)
PY
