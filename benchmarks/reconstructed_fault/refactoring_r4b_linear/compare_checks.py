#!/usr/bin/env python3
"""Compare existing linear-solve fixtures at matching rank counts, without retuning."""
from pathlib import Path
import json
root=Path(__file__).resolve().parent;e=root/'evidence';previous=root.with_name('refactoring_r4a');checks={}
markers=('line search accepted after','Stage-I rollback after an accepted Newton update: verified',
 'Relative nonlinear residuals (bulk, fault)', 'Fault linear solve:', 'Timestep ', 'All tests passed',
 'Affine audit alpha=', 'Affine nonzero-V probe:', 'Affine audit merit:', 'Affine audit active=',
 'Coupled pressure gauge:', 'Reconstructed-fault Stage-I solve:', 'Changed loading:')
names=[f'{name}-{rank}' for name in ('residual','exhaustion','pressure','unit','rollback-original') for rank in (1,2)]+['gmg-1']
for name in names:
    reused=name.startswith(('unit-','rollback-original-'))
    dirs=[previous if reused else root,root];variants=['candidate' if reused else 'reference','candidate']
    records=[json.loads((d/f'evidence/{v}-{name}.json').read_text()) for d,v in zip(dirs,variants)]
    # Exhaustion is an intended failure. Other baseline failures require an
    # explicit disposition; equal failure alone is never declared a pass.
    expected=1 if name.startswith('exhaustion') else 0
    checks[name+'/expected-outcome']=all(r['exit_code']==expected for r in records)
    logs=[(d/f'evidence/{v}-{name}.log').read_text(errors='replace') for d,v in zip(dirs,variants)]
    summaries=[sorted(line.strip() for line in log.splitlines() if any(m in line for m in markers)) for log in logs]
    checks[name+'/exact-decisions']=bool(summaries[0]) and summaries[0]==summaries[1]
    if name.startswith('exhaustion'):
        checks[name+'/budget-and-restoration']=all('Fault linear solve: iterations=1,' in s and 'Reconstructed-fault line search accepted' not in s and 'Stage-I rollback after an accepted Newton update: verified' in s for s in logs)
    if name.startswith('rollback'):
        checks[name+'/accepted-update-and-restoration']=all('line search accepted after' in s and 'Stage-I rollback after an accepted Newton update: verified' in s for s in logs)
    if name.startswith('residual'):
        checks[name+'/affine-and-nonzero-V']=all('Affine audit alpha=' in s and 'Affine nonzero-V probe: block=0' in s and 'Affine nonzero-V probe: block=1' in s and 'Reconstructed-fault Stage-I solve:' in s for s in logs)
    if name.startswith('pressure'):
        states=[{p.name:p.read_bytes() for p in (d/f'output-{v}-{name}').glob('gauge-state-*.txt')} for d,v in zip(dirs,variants)]
        checks[name+'/pressure-and-history']=bool(states[0]) and states[0]==states[1] and all('Coupled pressure gauge:' in s for s in logs)
    if name=='gmg-1':
        checks[name+'/gmg-lifecycle']=all('Reconstructed-fault Stage-I solve:' in s for s in logs)
(e/'focused-comparison.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2));assert all(checks.values())
