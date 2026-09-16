"""Reconstruct represented trials from the saved audit, without a mechanics solve."""
import json
from pathlib import Path
import numpy as np

directory=Path(__file__).resolve().parent/'bound_corrected_smoke'
table=np.genfromtxt(directory/'nonlinear_bounds_2.csv',names=True,delimiter=',')
rows=table[table['iteration']==2]
minimum=1e-20
base,direction=rows['V'],rows['dV']
alpha=rows['alpha_max'][0]
contact=np.full(len(base),np.inf)
np.divide(base-minimum,-direction,out=contact,where=direction<0)
trial=base+alpha*direction
trial[contact==alpha]=minimum
old_endpoint_interpolation=trial[:-1]+(trial[1:]-trial[:-1])
failures=np.flatnonzero(old_endpoint_interpolation<minimum)
report=dict(iteration=2,alpha=float(alpha),minimum=float(np.min(trial)),
            infeasible_nodes=np.flatnonzero(trial<minimum).tolist(),
            endpoint_interpolation_failures=[dict(segment=int(i),left=float(trial[i]),
                right=float(trial[i+1]),xi=1.,interpolated=float(old_endpoint_interpolation[i]))
                for i in failures],
            interpretation='Nodal contact is exact. The surface affine interpolation can lose it at xi=1; no solver retuning was applied.')
assert not report['infeasible_nodes']
assert failures.tolist()==[1154]
(directory/'bound_contact_followup.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
