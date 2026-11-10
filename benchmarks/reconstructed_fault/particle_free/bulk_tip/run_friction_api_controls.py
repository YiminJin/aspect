#!/usr/bin/env python3
"""Short API-migration controls using retained histories; no growth campaign."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'benchmarks/reconstructed_fault/particle_free/evidence/friction_api'
BUILD = ROOT / 'build-particle-free/friction-api'
CORE = ROOT / 'build-particle-free/material-overnight/native/aspect-release'
CHECKER = ROOT / 'build-particle-free/material-overnight/checks/libbulk_tip_checks.release.so'
RUNNER = ROOT / 'benchmarks/reconstructed_fault/particle_free/bulk_tip/run.py'
MPI = '/opt/openmpi/5.0.6/bin/mpirun'
env = os.environ.copy()
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'DEAL_II_NUM_THREADS'):
    env[key] = '1'
env['ASPECT_SOURCE_DIR'] = str(ROOT)
results = []


def execute(label, command, seconds):
    started = time.monotonic()
    with (OUT / (label + '.log')).open('x') as stream:
        try:
            code = subprocess.run([str(v) for v in command], cwd=ROOT, env=env,
                                  stdout=stream, stderr=subprocess.STDOUT,
                                  timeout=seconds).returncode
        except subprocess.TimeoutExpired:
            code = 124
    results.append(dict(label=label, command=[str(v) for v in command],
                        exit_code=code, seconds=time.monotonic()-started))
    (OUT / 'controls.json').write_text(json.dumps(results, indent=2) + '\n')
    print(label, code, flush=True)
    if code:
        raise SystemExit(code)


execute('units', [CORE, '--test'], 180)
common = ['python3', RUNNER]
options = ['--executable', CORE, '--checker', CHECKER]
execute('rheology', common + ['friction-api-rheology01'] + options + [
    '--fixture', ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/material_overnight/rheology-mixture.prm',
    '--ranks', '2', '--end', '0', '--rheology-audit', '--seconds', '120'], 150)
execute('be-recovery', common + ['friction-api-be01'] + options + [
    '--fixture', ROOT/'benchmarks/reconstructed_fault/particle_free/bulk_tip/synthetic_propagation.prm',
    '--dt', '.0025', '--end', '.06', '--ranks', '2', '--phase-balance',
    '--resume-from', ROOT/'build-particle-free/bulk-tip/runs/synthetic-growth-be-dt0025-local/output',
    '--snapshot-id', '3', '--rejection', '--rejection-step', '24', '--seconds', '300'], 330)
execute('midpoint', common + ['friction-api-midpoint01'] + options + [
    '--fixture', ROOT/'benchmarks/reconstructed_fault/particle_free/bulk_tip/localized.prm',
    '--time-update', 'midpoint', '--dt', '.005', '--end', '.09', '--ranks', '1',
    '--tangent', '--tangent-step', '8', '--seconds', '180'], 210)
execute('midpoint-recovery', common + ['friction-api-midpoint-recovery01'] + options + [
    '--fixture', ROOT/'benchmarks/reconstructed_fault/particle_free/bulk_tip/localized.prm',
    '--time-update', 'midpoint', '--dt', '.005', '--end', '.08', '--ranks', '2',
    '--resume-from', ROOT/'build-particle-free/bulk-tip/runs/tip-time-local01/output',
    '--snapshot-id', '1', '--rejection', '--rejection-step', '16', '--seconds', '180'], 210)
run = BUILD / 'mature'
run.mkdir(exist_ok=False)
text = (ROOT/'build-particle-free/material-overnight/mature-control/input.prm').read_text()
text += '\nset Output directory = ' + str(run/'output') + '\n'
(run/'input.prm').write_text(text)
execute('mature', [MPI, '-np', '1', CORE, run/'input.prm'], 180)
# The historical particle/Q1 friction fixtures are compile-only consumers.
# Reuse the existing limiter oracle with the supported mature configuration.
run = BUILD / 'state-limiter-native'
run.mkdir(exist_ok=False)
text = (BUILD/'mature/input.prm').read_text()
import re
libraries = re.findall(r'set Additional shared libraries = (.*)', text)[-1]
text += f'\nset Additional shared libraries = {libraries}, {BUILD}/consumers-build/libstate_limiter.release.so\n'
text += f'set Output directory = {run}/output\nset End time = 0\nset Maximum time step = 100000\n'
text += 'subsection Postprocess\n set List of postprocessors = layered background, verify fault state limiter\nend\n'
(run/'input.prm').write_text(text)
execute('state-limiter', [MPI, '-np', '2', CORE, run/'input.prm'], 180)
