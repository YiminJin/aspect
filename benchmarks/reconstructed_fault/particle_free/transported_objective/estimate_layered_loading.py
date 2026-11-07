import sys,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'benchmarks/reconstructed_fault/particle_free/transported_objective'))
from layered_fields import State,localization
import numpy as np
p=Path('build-particle-free/transported-objective/stage-c/material-runs/layered-L0/output')
s=State(p,0,.125);x,w=s.quadrature(8);v=s.at(x);owner,q,chi,_,_=localization(s,x,v)
def rate(tau,theta):
 lo,hi=0.,10.
 for _ in range(100):
  V=(lo+hi)/2
  f=.025*np.arcsinh(V/.002*np.exp((.6+.015*np.log(theta*.001/.008))/.025))*5+V-tau
  if f>0:hi=V
  else:lo=V
 return (hi+lo)/2
out=dict(assumptions='Initial scalar RSF estimate at compression 5, before coupled elastic backstress; not an imposed rate',bias_Pa=3.7,theta_s=[8,10,12],V=[rate(3.7,t) for t in [8,10,12]],chi_sampled_max=float(chi.max()),estimated_envelope_radians=float(.5*chi.max()*rate(3.7,10)*.8),T=.8,U=2,halfwidth=6,dt=.0125,refined_dt=.00625)
print(json.dumps(out,indent=2))
Path('benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/layered_bulk_campaign/loading-estimate.json').write_text(json.dumps(out,indent=2)+'\n')
