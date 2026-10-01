#!/usr/bin/env python3
"""Prepare matched inputs without modifying either accepted executable or archive."""
from pathlib import Path
import json
root = Path(__file__).resolve().parent
repo = root.parents[2]
relative = root.relative_to(repo)
(root/'inputs').mkdir(exist_ok=True)
(root/'evidence').mkdir(exist_ok=True)
for version in ('reference', 'candidate'):
    (root/f'inputs/{version}.prm').write_text(
        f'include $ASPECT_SOURCE_DIR/{relative}/frozen.prm\n'
        f'set Additional shared libraries = $ASPECT_SOURCE_DIR/{relative}/{version}-plugin-build/libbp3_frozen_reference.release.so, '
        f'$ASPECT_SOURCE_DIR/{relative}/{version}-plugin-build/libreconstructed_fault_frozen_gmg.release.so\n'
        f'set Output directory = $ASPECT_SOURCE_DIR/{relative}/output-{version}\n')
env = dict(ASPECT_SOURCE_DIR=str(repo), ASPECT_FAULT_EXPLICIT_B='1',
           ASPECT_FAULT_EXPLICIT_G='1', ASPECT_FAULT_SURFACE_SOLVER='tridiagonal',
           ASPECT_FAULT_STRESS_SAMPLE_DIAGNOSTIC='1', ASPECT_FAULT_SOURCE_HISTORY_DIAGNOSTIC='1',
           ASPECT_BP3_EXACT_TARGET='1',
           ASPECT_BP3_TARGET_MESH=str(repo/'benchmarks/reconstructed_fault/bp3/fixtures/modified_bp3_wide/target_cells.txt'),
           ASPECT_BP3_EXPECTED_FAULT=str(repo/'benchmarks/reconstructed_fault/bp3/fixtures/modified_bp3/fault.txt'),
           ASPECT_BP3_TIMESTEP_SEQUENCE=str(root/'clock.csv'),
           ASPECT_FAULT_LINEAR_PERFORMANCE='1', ASPECT_FROZEN_GMG_STEP='2', ASPECT_FROZEN_GMG_NEWTON='4')
(root/'evidence/environment.json').write_text(json.dumps(env, indent=2)+'\n')
