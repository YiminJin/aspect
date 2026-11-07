#!/usr/bin/env python3
"""Bounded shared-assembly operator comparisons; never overwrite old evidence."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('output', type=Path)
parser.add_argument('--candidate', type=Path, required=True)
parser.add_argument('--baseline', type=Path, required=True)
a = parser.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
env = os.environ.copy()
env['PATH'] = '/opt/gcc/12.4.0/bin:/opt/openmpi/5.0.6/bin:' + env['PATH']
for key in ('OMP_NUM_THREADS', 'DEAL_II_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    env[key] = '1'
records = []
def run(label, binary, ranks, args):
    command = ['/opt/openmpi/5.0.6/bin/mpirun', '-np', str(ranks), str(binary), *map(str, args)]
    start = time.monotonic()
    with (a.output / (label + '.log')).open('w') as stream:
        result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=120)
    record = dict(label=label, command=command, exit_code=result.returncode, seconds=time.monotonic()-start)
    records.append(record)
    (a.output / 'runs.json').write_text(json.dumps(records, indent=2)+'\n')
    if result.returncode:
        raise RuntimeError(f'{label} failed; inspect its preserved log')
    lines = (a.output / (label + '.log')).read_text().splitlines()
    value = next(json.loads(line) for line in reversed(lines) if line.startswith('{'))
    record['metrics'] = value
    return value

for geometry in ('cartesian', 'affine', 'bilinear'):
    for ranks, refined in ((1, 0), (2, 1), (4, 1)):
        run(f'weak-{geometry}-{ranks}', a.candidate, ranks, [2, 'weak-forms', 0, .005, .005, refined, geometry])

comparisons = []
for label, ranks, args in (
    ('constant', 2, [2, 'constant', 1, .025, .1, 1, 'bilinear']),
    ('divergence', 2, [3, 'divergence', 1, .025, .1, 1, 'bilinear']),
    ('translate', 2, [3, 'translate', 1, .025, .1, 1, 'bilinear']),
    ('reverse', 4, [3, 'reverse', 1, .025, .1, 1, 'affine']),
    ('smooth', 1, [3, 'translate', 0, .025, .1, 0, 'bilinear']),
    ('elliptic', 2, [2, 'elliptic', 1, .025, .1, 1, 'bilinear']),
    ('elliptic-fine', 4, [3, 'elliptic', 1, .025, .1, 1, 'bilinear']),
    ('obstacle', 2, [2, 'obstacle', 1, .025, .1, 1, 'affine']),
    ('moving-obstacle', 4, [2, 'moving-obstacle', 1, .025, .1, 1, 'bilinear']),
):
    old = run('old-'+label, a.baseline, ranks, args)
    new = run('new-'+label, a.candidate, ranks, args)
    differences = {k: new[k]-old[k] for k in new.keys() & old.keys()
                   if isinstance(new[k], (float, int)) and isinstance(old[k], (float, int))}
    # Limiter activation at alpha=1 is sensitive to roundoff; physical values,
    # limiter strength and subcycle/solver decisions are checked separately.
    observational = {'seconds', 'operator_bytes_rank0', 'limited_cells'}
    for key, difference in differences.items():
        if key not in observational:
            assert abs(difference) < 2e-12 * max(1., abs(old[key])), (label, key, difference)
    comparisons.append(dict(case=label, differences=differences))
(a.output/'runs.json').write_text(json.dumps(records, indent=2)+'\n')
(a.output/'comparisons.json').write_text(json.dumps(comparisons, indent=2)+'\n')
print(json.dumps(dict(attempts=len(records), seconds=sum(r['seconds'] for r in records), output=str(a.output))))
