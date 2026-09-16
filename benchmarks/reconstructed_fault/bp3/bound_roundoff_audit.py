"""Offline replay of the failed represented bound trial, not a solver modification."""
from decimal import Decimal, localcontext
import json
from pathlib import Path
import numpy as np

directory=Path(__file__).resolve().parent/'cached_smoke_low_floor'
table=np.genfromtxt(directory/'nonlinear_bounds_2.csv',delimiter=',',names=True)
minimum=1e-20
failures=[]
for row in table:
    v,d,alpha=map(float,(row['V'],row['dV'],row['alpha_max']))
    trial=v+alpha*d
    if trial>=minimum: continue
    with localcontext() as context:
        context.prec=65
        V,D,A,M=map(Decimal.from_float,(v,d,alpha,minimum))
        exact=V+A*D
        failures.append(dict(fault=int(row['fault']),vertex=int(row['vertex']),
            V=v,dV=d,alpha=alpha,minimum=minimum,represented_trial=trial,
            deficit=minimum-trial,epsilon_times_base=np.finfo(float).eps*v,
            current_snap_tolerance=100*np.finfo(float).eps*max(minimum,abs(trial)),
            exact_affine_trial_of_represented_inputs=str(exact),
            exact_contact_alpha=str((V-M)/(-D)),
            represented_alpha=str(A),
            subtract_then_add_exact_minimum=v+(minimum-v)))
assert len(failures)==1 and failures[0]['vertex']==1154
assert failures[0]['deficit']<failures[0]['epsilon_times_base']
assert failures[0]['deficit']>failures[0]['current_snap_tolerance']
assert failures[0]['subtract_then_add_exact_minimum']<minimum
report=dict(failures=failures,
    interpretation='Bound-contact arithmetic loses digits when V_min is eleven orders below V; '
      'the existing snap scale uses the small result instead of the cancelled operands. '
      'Reconstructing a snapped trial from its increment repeats the loss.',
    production_modified=False)
(directory/'bound_roundoff_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
