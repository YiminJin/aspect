"""Read-only independent checks on the phase precision evidence."""
import json
from pathlib import Path
import re
import sys

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from reference import read_parameters

here=Path(__file__).resolve().parent
audit=json.loads((here/'analysis.json').read_text())
plateau=next(r for r in audit if r['step']==2 and r['iteration']==9 and r['alpha']==1)
assert plateau['J_action_vs_extended'] < 1e-12*plateau['Jrepresented']
assert plateau['extended_unrounded_trial'] < .3*plateau['extended_trial']
assert plateau['extended_affine_error'] < .01*plateau['affine_error']
amplified=next(r for r in audit if r['step']==2 and r['iteration']==9 and r['alpha']==1e6)
assert amplified['extended_affine_error'] < 1e-8*amplified['Jrepresented']
checks=[]
for case,ny in (('one',16),('two',16),('refined',32)):
    log=(here/f'{case}.log').read_text()
    parameters=read_parameters(here/case/'parameters.prm')
    E=float(parameters['Material model/Phase field fault/Critical energy release rates'])/(
        (8/3)*float(parameters['Phase field model/Length scale']))
    # At phi=0,H=Hc: S=2 E times the exact lumped Q1 mass; periodic x,
    # natural y endpoints have half the interior nodal measure.
    exact=8*np.finfo(float).eps*2*E*.25*np.sqrt((ny-.5)/4)/ny
    samples=[dict(zip(('value','initial','allowance','final','target'),map(float,m))) for m in re.findall(
        r'PHASE_PRECISION value=(\S+) initial=(\S+) allowance=(\S+) final=(\S+) target=(\S+)',log)]
    assert len(samples)>=3 and all(s['final']<=s['target'] for s in samples)
    assert abs(samples[0]['allowance']/exact-1)<1e-12
    assert all(s['initial']>1000*s['allowance'] for s in samples if s['value']>=1e-13)
    checks.append(dict(case=case,exact_zero_allowance=exact,samples=samples,
        resources=json.loads((here/f'{case}.resources.json').read_text())))
assert abs(checks[0]['samples'][0]['allowance']/checks[1]['samples'][0]['allowance']-1)<1e-14
report=dict(passed=True,checks=checks,plateau=plateau,
    criterion='max(relative * initial residual, 8 eps ||positive phase term scale||)',
    scale_is_estimate_not_rigorous_bound=True)
(here/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(passed=True,serial_mpi_scale_agreement=True,analytic_mesh_scale_agreement=True)))
