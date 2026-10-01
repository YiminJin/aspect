#!/usr/bin/env python3
"""Compare the existing frozen-stress and temperature fixture outcomes to R3b."""
from pathlib import Path
import json
root=Path(__file__).resolve().parent
ref=root.with_name('refactoring_r3b')
checks={}
for name in ('frozen-stress','temperature'):
    for rank in (1,2):
        case=f'candidate-{name}-{rank}'
        records=[json.loads((p/'evidence'/f'{case}.json').read_text()) for p in (ref,root)]
        checks[case+'/pass']=all(r['exit_code']==0 for r in records)
        checks[case+'/statistics']=(ref/f'output-{case}/statistics').read_bytes()==(root/f'output-{case}/statistics').read_bytes()
        markers=('Frozen Maxwell','Stage-J controlled transverse-temperature evaluation:','Fault linear solve:', 'Relative nonlinear residuals','line search accepted after')
        logs=[(p/'evidence'/f'{case}.log').read_text() for p in (ref,root)]
        lines=[[line.strip() for line in text.splitlines() if any(m in line for m in markers)] for text in logs]
        checks[case+'/decisions-and-assertions']=lines[0]==lines[1] and bool(lines[0])
(root/'evidence/frozen-comparison.json').write_text(json.dumps(checks,indent=2)+'\n')
print(len(checks),'frozen/temperature checks; failures:',[k for k,v in checks.items() if not v])
assert all(checks.values())
