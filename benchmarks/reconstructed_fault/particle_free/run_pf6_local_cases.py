#!/usr/bin/env python3
"""Sequential capped cases; use through the existing native run logger."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('phase', choices=['quadrature', 'local', 'support'])
p.add_argument('--prefix', default='pf6-local')
a = p.parse_args()
root = Path(__file__).resolve().parents[3]
runner = Path(__file__).with_name('run_pf6_local.py')
logger = root/'benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py'
diag = root/'build-particle-free/quad-local'
runs = root/'build-particle-free/runs'


def run(label, options):
    subprocess.run([sys.executable, str(runner), 'prepare', label]+options, check=True, stdout=subprocess.DEVNULL)
    out = runs/label
    command = json.loads((out/'command.json').read_text())
    completed = subprocess.run([sys.executable, str(logger), label, '190', 'timeout', '--signal=TERM', '--kill-after=5s', '180s']+command, cwd=root)
    subprocess.run([sys.executable, str(runner), 'finalize', label], check=True)
    completed.check_returncode()
    return out


if a.phase == 'quadrature':
    old = runs/'pf6-transfer-checked-r1'
    frozen = run(a.prefix+'-frozen-q', ['--restart', str(old/'run/restart'), '--snapshot', '--quadrature', '6'])
    subprocess.run([sys.executable, str(Path(__file__).with_name('assess_pf6_local.py')), str(frozen), '--frozen', '--output', str(diag/'frozen-quadrature.json')], check=True)
    fresh = run(a.prefix+'-fresh-q12', [])
    subprocess.run([sys.executable, str(Path(__file__).with_name('assess_pf6_local.py')), str(fresh), '--compare', str(old), '--output', str(diag/'fresh-quadrature.json')], check=True)
elif a.phase == 'support':
    label = a.prefix+'-deep-support'
    try:
        run(label, ['--deep-uniform'])
        raise AssertionError('Unsupported deep support unexpectedly passed')
    except subprocess.CalledProcessError:
        record = json.loads((runs/label/'record.json').read_text())
        log = (runs/label/'run.log').read_text()
        assert record['exit_code'] == 1 and 'neighborhood box' in log and 'crosses the boundary' in log
        (diag/'deep-support.json').write_text(json.dumps(dict(native_exit_code=1,
            original_coverage_guard_retained=True, supported=False,
            note='Fresh one-layer particle domains cannot recover every fixed-particle Voronoi cell on this deep grid',
            record=record), indent=2)+'\n')
        print('Expected deep particle-support rejection captured.', flush=True)
else:
    for name in ['frozen-quadrature.json', 'fresh-quadrature.json']:
        assert json.loads((diag/name).read_text())['G12_gate_passed']
    assessor = Path(__file__).with_name('assess_pf6_local.py')
    frozen = run(a.prefix+'-final-frozen-q', ['--restart', str(runs/'pf6-transfer-checked-r1/run/restart'), '--snapshot', '--quadrature', '6'])
    subprocess.run([sys.executable, str(assessor), str(frozen), '--frozen', '--output', str(diag/'final-frozen-quadrature.json')], check=True)
    # A fresh uniform G12 control on the final local-transfer core/observer.
    uniform = run(a.prefix+'-final-q12', [])
    subprocess.run([sys.executable, str(assessor), str(uniform), '--output', str(diag/'final-quadrature.json')], check=True)
    for q in [24, 32]:
        reference = run(a.prefix+f'-trajectory-q{q}', ['--quadrature', str(q)])
        subprocess.run([sys.executable, str(assessor), str(uniform), '--compare', str(reference), '--output', str(diag/f'trajectory-q12-q{q}.json')], check=True)
    half = run(a.prefix+'-uniform-half', ['--end', '.125'])
    continued = run(a.prefix+'-uniform-continued', ['--restart', str(half/'run/restart')])
    subprocess.run([sys.executable, str(Path(__file__).with_name('assess_pf6_transfer.py')), str(uniform), '--compare', str(continued), '--output', str(diag/'uniform-continued.json')], check=True)
    local = {}
    for ranks in [1, 2, 4]:
        local[ranks] = run(a.prefix+f'-r{ranks}', ['--local', '--ranks', str(ranks)])
        subprocess.run([sys.executable, str(assessor), str(local[ranks]), '--local', '--output', str(diag/f'local-r{ranks}.json')], check=True)
        if ranks > 1:
            subprocess.run([sys.executable, str(assessor), str(local[1]), '--compare', str(local[ranks]), '--local', '--output', str(diag/f'local-rank-1-{ranks}.json')], check=True)
    for ranks in [1, 2, 4]:
        restored = run(a.prefix+f'-restart-r{ranks}', ['--local', '--ranks', str(ranks), '--restart', str(local[ranks]/'run/restart')])
        subprocess.run([sys.executable, str(assessor), str(local[ranks]), '--compare', str(restored), '--local', '--restart-comparison', '--output', str(diag/f'local-restart-r{ranks}.json')], check=True)
    cross = run(a.prefix+'-cross-1-to-4', ['--local', '--ranks', '4', '--restart', str(local[1]/'run/restart')])
    subprocess.run([sys.executable, str(assessor), str(local[1]), '--compare', str(cross), '--local', '--restart-comparison', '--output', str(diag/'local-cross-1-to-4.json')], check=True)
