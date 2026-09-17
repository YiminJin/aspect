"""Compare the GMG short trajectory to the saved wide AMG reference.

Reuse the existing 1e-8 per-field maximum-norm comparison coefficient. Raw
pressure/deviatoric stress checks are not scaled by the 50-MPa background.
"""
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-gmg-check-mpl')
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
BP3=REPO/'benchmarks/reconstructed_fault/bp3'
sys.path.insert(0,str(BP3))
import numpy as np
from check_first_cycle_restart import convergence, load_parts
from analyze_fully_frictional import samples
from analyze_uniform_sliding import read

old=BP3/'wide-seven-local4'
new=BP3/'wide-gmg-seven-local4'
result=dict(convergence=convergence(new/'run.log'),steps={},comparison_coefficient=1e-8,
            failures=[],exact_failures=[])
def difference(a,b):
    # Record every field even when one exceeds the existing comparison
    # allowance. Never turn a solver pass into a silently loosened equivalence.
    assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    absolute=float(np.max(np.abs(a-b),initial=0.))
    scale=float(np.max(np.abs(a),initial=0.))
    return dict(absolute=absolute,scale=scale,relative=absolute/scale if scale else 0.,
                passed=absolute<=1e-8*scale)
def exact(a,b,step,field):
    if not np.array_equal(a,b):
        value=difference(a,b)
        value['numerical_allowance_passed']=value.pop('passed')
        result['exact_failures'].append(dict(step=step,field=field,bitwise_equal=False,**value))
a,b=[read(p/'accepted_steps.csv') for p in (old,new)]
for field in ('step','time','dt','free','lower_active'):
    np.testing.assert_array_equal(a[field],b[field])
for k in range(8):
    a,b=[read(p/f'fault_{k}.csv') for p in (old,new)]
    for field in ('xd','x','y','time','dt','Ih','C','prescribed','tau_bg','sigma_n_bg'):
        exact(a[field],b[field],k,'fault/'+field)
    entry={'fault':{field:difference(a[field],b[field]) for field in ('V','Theta','slip')}}
    a,b=[samples(p,k) for p in (old,new)]
    for field in ('x','y','JxW','source_active','segment','xi','phi','Ih','chi'):
        exact(a[field],b[field],k,'bulk-QP/'+field)
    entry['current_stress']={field:difference(a[field],b[field])
                            for field in ('p','tau_xx','tau_yy','tau_xy','tauN','sigma_n','q')}
    a,b=[load_parts(p,f'mature_history_{k}_rank*.csv') for p in (old,new)]
    np.testing.assert_array_equal(a[:,:2],b[:,:2])
    entry['particle_stress']={str(c):difference(a[:,c],b[:,c]) for c in range(2,5)}
    for family,fields in entry.items():
        for field,value in fields.items():
            if not value['passed']: result['failures'].append([k,family,field])
    result['steps'][k]=entry
result['passed']=not result['failures'] and not result['exact_failures']
(new/'gmg_equivalence.json').write_text(json.dumps(result,indent=2)+'\n')
for family in ('fault','current_stress','particle_stress'):
    print(family,max((v['relative'],k,key,v['absolute'])
                    for k,row in result['steps'].items() for key,v in row[family].items()))
print('GMG seven-step equivalence:', 'PASSED' if result['passed'] else 'NOT PASSED',result['failures'])
print('Non-bitwise fields:',result['exact_failures'])
sys.exit(0 if result['passed'] else 1)
