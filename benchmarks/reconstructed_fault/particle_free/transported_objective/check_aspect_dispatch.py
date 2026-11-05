#!/usr/bin/env python3
"""Bounded stationary-fault and ordinary solver dispatch equivalence."""
import csv
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/aspect_architecture'
RUNS = ROOT/'build-particle-free/transported-objective/stage-c/runs'


def compare_csv(before, after):
    a, b = [list(csv.reader(p.open())) for p in (before, after)]
    assert len(a) == len(b) and a[0] == b[0], before
    relative = absolute = 0.
    for x, y in zip(a[1:],b[1:]):
        assert len(x) == len(y)
        for u, v in zip(x,y):
            try:
                f, g = float(u), float(v)
            except ValueError:
                assert u == v, (before,u,v)
                continue
            assert math.isfinite(f) and math.isfinite(g), (before,u,v)
            error = abs(f-g)
            # New dispatch screen, fixed before running the changed source.
            # The fixture also retains its original independent native assertions.
            assert error <= 1e-10+1e-12*max(abs(f),abs(g)), (before,u,v)
            absolute = max(absolute,error)
            relative = max(relative,error/max(1.,abs(f),abs(g)))
    return dict(file=str(before.name),max_absolute=absolute,max_scaled=relative,
                byte_identical=before.read_bytes()==after.read_bytes())


def main():
    result = OUT/'dispatch-equivalence.json'
    assert not result.exists(), 'Preserve completed evidence.'
    before = RUNS/'architecture-before-stationary-retry/output'
    after = RUNS/'architecture-after-stationary/output'
    files = ['accepted_steps.csv','objective-dg-summary.csv','profiles/fault_0.csv',
             'profiles/fault_1.csv','objective-dg-final-rank0.csv','objective-dg-final-rank1.csv']
    rows = [compare_csv(before/f,after/f) for f in files]
    for pattern in [r'Fault linear solve: iterations=(\d+)',r'Material projected reaction: outer=(\d+)']:
        assert re.findall(pattern,(before.parent/'run.log').read_text()) == re.findall(pattern,(after.parent/'run.log').read_text())
    nonfault = ROOT/'build-particle-free/aspect-architecture'
    stats = [(nonfault/('nonfault-'+name)/'output/statistics').read_bytes() for name in ('before','after')]
    assert stats[0] == stats[1], 'Ordinary non-fault statistics changed.'
    data = dict(stationary=rows,nonfault_statistics_byte_identical=True,
                stationary_scale_tolerance=1e-12,stationary_absolute_tolerance=1e-10)
    result.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data))


if __name__ == '__main__':
    main()
