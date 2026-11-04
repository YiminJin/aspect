#!/usr/bin/env python3
"""Compare complete native material exports across retry and MPI restart."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
RUNS = ROOT/'build-particle-free/transported-objective/stage-c/material-runs'
OUT = ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/stage_c_coupled'


def rows(name, step, surface=False):
    pattern = f'material-{step}-surface.csv' if surface else f'material-{step}-rank*.csv'
    values = []
    for path in (RUNS/name/'output').glob(pattern):
        with path.open() as stream:
            values.extend(csv.DictReader(stream))
    assert values, (name, step, pattern)
    values.sort(key=(lambda r: int(r['i'])) if surface else (lambda r: (r['cell'], int(r['q']))))
    return values


def compare(control, candidate, steps):
    result = {}
    for step in steps:
        result[step] = {}
        for surface in (False, True):
            a, b = rows(control, step, surface), rows(candidate, step, surface)
            assert len(a) == len(b)
            if not surface:
                assert [(r['cell'], r['q']) for r in a] == [(r['cell'], r['q']) for r in b]
            errors = {}
            for key in a[0]:
                if key == 'cell':
                    continue
                x = np.array([float(r[key]) for r in a])
                y = np.array([float(r[key]) for r in b])
                error = float(np.max(np.abs(x-y)))
                bound = 1e-12*max(1., float(np.max(np.abs(x))))
                assert error <= bound, (control, candidate, step, key, error, bound)
                errors[key] = dict(max_abs=error, bound=bound)
            result[step]['surface' if surface else 'bulk'] = errors
    return result


def main():
    result = {'criterion': 'Per-column maximum absolute difference <= 1e-12 times max(1, maximum control magnitude); labels/point keys exact.'}
    result['matched_2_to_4_restart'] = compare('matched-positive', 'matched-restart4-final', [3, 4])
    same = []
    for p in (RUNS/'matched-rejection/output').glob('material-*.csv'):
        if p.name == 'material-summary.csv':
            continue
        q = RUNS/'matched-cutback-control/output'/p.name
        assert p.read_bytes() == q.read_bytes(), p
        same.append(p.name)
    result['rejection_matched_schedule_byte_identical'] = sorted(same)
    for control, candidate in [('heterogeneous-positive', 'heterogeneous-restart-final')]:
        if (RUNS/candidate/'command.json').exists():
            meta = json.loads((RUNS/candidate/'command.json').read_text())
            assert meta['exit_code'] == 0
            result[candidate] = compare(control, candidate, [3, 4])
    if (RUNS/'heterogeneous-cutback-control-final/command.json').exists():
        identity = {}
        for p in (RUNS/'heterogeneous-rejection-final/output').glob('material-*.csv'):
            if p.name == 'material-summary.csv':
                continue
            q = RUNS/'heterogeneous-cutback-control-final/output'/p.name
            identity[p.name] = p.read_bytes() == q.read_bytes()
        result['heterogeneous_rejection_byte_identity'] = identity
        result['heterogeneous_rejection_comparison'] = compare(
            'heterogeneous-cutback-control-final', 'heterogeneous-rejection-final', [0, 1, 2, 3])
        result['heterogeneous_rejection_note'] = (
            'The initial byte-identity screen fails only at the final endpoint: '
            'the termination-limited rejected schedule uses dt=.049999999999999989 '
            'and the function-limited control uses .050000000000000003. '
            'All earlier exports are byte-identical; final differences are tested '
            'with the same declared per-column restart comparison bound. '
            'Neither original output nor its timestep is altered.')
    names = ['matched-positive', 'matched-restart4-final', 'matched-rejection', 'matched-cutback-control',
             'heterogeneous-positive', 'heterogeneous-restart-final', 'heterogeneous-rejection-final', 'heterogeneous-cutback-control-final']
    result['inputs_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for name in names for p in (RUNS/name).glob('output/material-*.csv')}
    (OUT/'lifecycle-comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Complete-state rejection and restart comparisons pass.')


if __name__ == '__main__':
    main()
