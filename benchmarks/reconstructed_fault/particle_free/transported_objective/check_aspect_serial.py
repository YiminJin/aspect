#!/usr/bin/env python3
"""Compare retained serial support outputs; exclude timing columns only."""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/aspect_architecture/serial-comparison.json'
assert not OUT.exists(), 'Preserve completed evidence.'
result = {}
for name in ('limiter','stabilization'):
    before = ROOT/'build-particle-free/stress-history-consolidation'/name/'output'
    after = ROOT/'build-particle-free/aspect-architecture'/name/'output'
    rows = []
    for candidate in sorted(after.glob('*.csv')):
        reference = before/candidate.name
        a,b = [list(csv.reader(p.open())) for p in (reference,candidate)]
        assert len(a)==len(b) and a[0]==b[0], candidate
        skip = {i for i,h in enumerate(a[0]) if h.endswith('_seconds')}
        maximum = 0.
        for x,y in zip(a[1:],b[1:]):
            assert len(x)==len(y)
            for i,(u,v) in enumerate(zip(x,y)):
                if i in skip:
                    continue
                try:
                    f,g = float(u),float(v)
                except ValueError:
                    assert u==v,(candidate,u,v)
                    continue
                assert math.isfinite(f) and math.isfinite(g)
                maximum=max(maximum,abs(f-g)/max(1.,abs(f),abs(g)))
        rows.append(dict(file=candidate.name,max_scaled_difference=maximum,
                         excluded_columns=[a[0][i] for i in sorted(skip)],
                         byte_identical=reference.read_bytes()==candidate.read_bytes()))
    result[name]=dict(files=rows,maximum_scaled_difference=max(r['max_scaled_difference'] for r in rows))
OUT.write_text(json.dumps(result,indent=2)+'\n')
print({name:r['maximum_scaled_difference'] for name,r in result.items()})
