#!/usr/bin/env python3
"""Sequential bounded native cases; execute through the established run logger."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('phase', choices=['serial', 'mpi', 'negative'])
p.add_argument('--prefix', default='pf6-domain-free')
a = p.parse_args()
root = Path(__file__).resolve().parents[3]
runs = root/'build-particle-free/runs'
diag = root/'build-particle-free/domain-free'
if a.prefix != 'pf6-domain-free':
    diag = diag/a.prefix
diag.mkdir(parents=True, exist_ok=True)
runner = Path(__file__).with_name('run_pf6_domain_free.py')
assessor = Path(__file__).with_name('assess_pf6_domain_free.py')
logger = root/'benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py'


def run(suffix, options=(), failure=None):
    label = a.prefix+'-'+suffix
    subprocess.run([sys.executable, str(runner), 'prepare', label]+list(options), check=True, stdout=subprocess.DEVNULL)
    out = runs/label
    command = json.loads((out/'command.json').read_text())
    completed = subprocess.run([sys.executable, str(logger), label, '190', 'timeout', '--signal=TERM', '--kill-after=5s', '180s']+command, cwd=root)
    subprocess.run([sys.executable, str(runner), 'finalize', label], check=True)
    if failure:
        record = json.loads((out/'record.json').read_text())
        assert record['exit_code'] == 1 and failure in (out/'run.log').read_text(), label
        (diag/(suffix+'.json')).write_text(json.dumps(dict(expected_failure=True, retained=True, record=record), indent=2)+'\n')
    else:
        completed.check_returncode()
    return out


def assess(suffix, case, options=()):
    subprocess.run([sys.executable, str(assessor), str(case)]+list(options)+['--output', str(diag/(suffix+'.json'))], check=True)


if a.phase == 'serial':
    legacy = run('legacy-control', ['--legacy', '--no-deep'])
    serial = run('r1')
    assess('r1', serial)
    assess('legacy-comparison', serial, ['--compare', str(legacy), '--legacy-comparison'])
    half = run('half', ['--end', '.125'])
    continued = run('continued', ['--restart', str(half/'run/restart')])
    assess('continued', serial, ['--compare', str(continued)])
    restored = run('restart-r1', ['--restart', str(serial/'run/restart')])
    from assess_pf6_local import compare_exports
    from assess_pf6_domain_free import domain_absence
    c = compare_exports(serial, restored, 'mesh-checkpoint', 'mesh-restored')
    c['domain_absence'] = domain_absence(restored)
    (diag/'restart-r1.json').write_text(json.dumps(c, indent=2)+'\n')
    run('legacy-deep-failure', ['--legacy'], 'crosses the boundary')
elif a.phase == 'mpi':
    from assess_pf6_local import compare_exports
    from assess_pf6_domain_free import domain_absence
    serial = runs/(a.prefix+'-r1')
    assert json.loads((diag/'r1.json').read_text())['deep_uniform_remap']['passed']
    for ranks in [2, 4]:
        parallel = run(f'r{ranks}', ['--ranks', str(ranks)])
        assess(f'r{ranks}', parallel)
        assess(f'rank-1-{ranks}', serial, ['--compare', str(parallel)])
        restored = run(f'restart-r{ranks}', ['--ranks', str(ranks), '--restart', str(parallel/'run/restart')])
        c = compare_exports(parallel, restored, 'mesh-checkpoint', 'mesh-restored')
        c['domain_absence'] = domain_absence(restored)
        (diag/f'restart-r{ranks}.json').write_text(json.dumps(c, indent=2)+'\n')
    cross = run('cross-1-to-4', ['--ranks', '4', '--restart', str(serial/'run/restart')])
    c = compare_exports(serial, cross, 'mesh-checkpoint', 'mesh-restored')
    c['domain_absence'] = domain_absence(cross)
    (diag/'cross-1-to-4.json').write_text(json.dumps(c, indent=2)+'\n')
else:
    markers = dict(domains='requires particle domains and CPDI data disabled',
                   heterogeneous='requires homogeneous G',
                   **{'missing-seed': 'requires accepted DG endpoint history and a prescribed physical phase seed',
                      'particle-fields': 'requires static bulk fields',
                      'fit-geometry': 'requires fixed prescribed fault geometry'})
    for name, marker in markers.items():
        run('negative-'+name, ['--negative', name], marker)
    old = runs/(a.prefix+'-legacy-control/run/restart')
    run('negative-mode-restart', ['--restart', str(old)], 'Cannot change homogeneous mesh data configuration on restart')
