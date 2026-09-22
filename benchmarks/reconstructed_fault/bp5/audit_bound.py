"""Offline arithmetic reproducer, NOT a replay of the unavailable failed trial.

Use saved production coordinates with admissible, equal endpoint rates. Test
both separate operations and possible fused multiply-add contractions. No
production state, residual, or bound is modified.
"""
import csv
import json
import math
from pathlib import Path

OUT=Path(__file__).resolve().parent/'dc010-ell100'
minimum=1e-20
results=[]
examples=[]
for path in sorted((OUT/'candidate-six').glob('work_qp_1_rank*.csv')):
    counts=dict(separate=0,fma_left=0,fma_right=0)
    samples=0
    captured=0
    with path.open() as stream:
        for row in csv.DictReader(stream):
            if row['source_active']!='1': continue
            xi=float(row['xi'])
            if not 0<xi<1: continue
            samples+=1
            a,b=1-xi,xi
            values=dict(separate=a*minimum+b*minimum,
                        fma_left=math.fma(a,minimum,b*minimum),
                        fma_right=math.fma(b,minimum,a*minimum))
            for mode,value in values.items(): counts[mode]+=value<minimum
            representative=path.stem.endswith('rank3') and 13000<float(row['xd'])<18000
            if any(value<minimum for value in values.values()) and captured<2 and (representative or not path.stem.endswith('rank3')):
                captured+=1
                examples.append(dict(rank=path.stem.split('rank')[-1],cell=row['cell'],qp=int(row['qp']),
                    x=float(row['x']),y=float(row['y']),xd=float(row['xd']),xi=xi,
                    left_V=minimum,right_V=minimum,**values,
                    absolute_deficit=minimum-min(values.values()),
                    deficit_ulps=(minimum-min(values.values()))/math.ulp(minimum)))
    results.append(dict(file=path.name,samples=samples,below_minimum=counts))
assert examples
result=dict(status='INTERPOLATION COUNTEREXAMPLE; exact failed trial was not exported',
            Vmin=minimum,ulp=math.ulp(minimum),ranks=results,examples=examples)
(OUT/'bound_arithmetic.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
