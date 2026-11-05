#!/usr/bin/env python3
"""Compare retained auxiliary/lifecycle controls without changing their criteria."""
import csv
import json
import math
import re
from pathlib import Path
from check_aspect_dispatch import compare_csv
from check_stress_history_consolidation import match

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/code_readability'
RUNS = ROOT/'build-particle-free/transported-objective/stage-c/runs'


def main():
    target = OUT/'auxiliary-equivalence.json'
    assert not target.exists(), 'Preserve completed evidence.'
    before = RUNS/'architecture-after-stationary/output'
    after = RUNS/'readability-stationary/output'
    files = ['accepted_steps.csv', 'objective-dg-summary.csv', 'profiles/fault_0.csv',
             'profiles/fault_1.csv', 'objective-dg-final-rank0.csv', 'objective-dg-final-rank1.csv']
    stationary = [compare_csv(before/f, after/f) for f in files]
    pattern = r'Fault linear solve: iterations=(\d+)'
    assert re.findall(pattern, (before.parent/'run.log').read_text()) == re.findall(pattern, (after.parent/'run.log').read_text())
    old = ROOT/'build-particle-free/aspect-architecture'
    new = ROOT/'build-particle-free/code-readability'
    assert (old/'nonfault-after/output/statistics').read_bytes() == (new/'nonfault-after/output/statistics').read_bytes()
    serial = {}
    for name in ('limiter', 'stabilization'):
        rows = []
        for candidate in sorted((new/name/'output').glob('*.csv')):
            reference = old/name/'output'/candidate.name
            a, b = [list(csv.reader(p.open())) for p in (reference, candidate)]
            assert len(a) == len(b) and a[0] == b[0]
            skip = {i for i, h in enumerate(a[0]) if h.endswith('_seconds')}
            maximum = 0.
            for x, y in zip(a[1:], b[1:]):
                assert len(x) == len(y)
                for i, (u, v) in enumerate(zip(x, y)):
                    if i in skip:
                        continue
                    try:
                        f, g = float(u), float(v)
                    except ValueError:
                        assert u == v
                        continue
                    assert math.isfinite(f) and math.isfinite(g)
                    maximum = max(maximum, abs(f-g)/max(1., abs(f), abs(g)))
            assert maximum == 0., (name, candidate, maximum)
            rows.append(dict(file=candidate.name, max_scaled_difference=maximum,
                             excluded_columns=[a[0][i] for i in sorted(skip)]))
        assert rows
        serial[name] = rows
    lifecycle = [match('readability-dl', 'readability-restart-new'),
                 match('true-cutback-control', 'readability-reject'),
                 match('readability-dl', 'readability-midpoint')]
    assert len(lifecycle[0]['steps']) == 2
    for result in lifecycle:
        for step in result['steps']:
            assert step['tensor_L2'] < 5e-12 and step['work_traction_Linf'] < 5e-12
    data = dict(stationary=stationary, nonfault_statistics_byte_identical=True,
                serial=serial, lifecycle=lifecycle)
    target.write_text(json.dumps(data, indent=2)+'\n')
    print(json.dumps(dict(stationary_max=max(r['max_absolute'] for r in stationary),
                          serial_numerics_identical=True, nonfault_statistics_identical=True,
                          restart_max_bulk=max(s['bulk_Linf'] for s in lifecycle[0]['steps']),
                          restart_max_surface=max(s['surface_Linf'] for s in lifecycle[0]['steps']))))


if __name__ == '__main__':
    main()
