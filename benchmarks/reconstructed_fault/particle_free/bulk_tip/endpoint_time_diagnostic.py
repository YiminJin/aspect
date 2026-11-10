#!/usr/bin/env python3
"""Independent, conditional endpoint stationarity diagnostic; never an acceptance gate.
The last stage V is held fixed when evaluating R(V,theta_1). This does not
reconstruct an unknown instantaneous endpoint velocity. Parameters below belong
to the two bounded homogeneous bulk-tip fixtures, not a general material API.
"""
import argparse,json
from pathlib import Path
import numpy as np
from analyze_time_update import read,frame,balance,RUNS,OUT
p=argparse.ArgumentParser(description=__doc__);p.add_argument('labels',nargs='+');a=p.parse_args()
result={}
for label in a.labels:
 d=RUNS/label/'output';rows=read(d/'bulk-accepted.csv');values=[]
 for row in rows:
  step=int(row.step);f=frame(d,step);b=balance(d,f'tip-balance-{step:04d}-r*.csv');sec=read(d/f'tip-sections-{step:04d}.csv')
  # Signed regularized RSF evaluated directly at accepted age, no age advancement.
  force=1000*.015*np.arcsinh(sec.V/(2*.001)*np.exp((.6+.02*np.log(sec.age*.001/.01))/.015))+1000*sec.V
  ff=force[f.section.astype(int)];c=80*f.phi*(1+f.phi);dd=(1-f.phi)**2;positive=f.phi>0
  Q=np.zeros(len(f));Q[positive]=2e6*f.Axy[positive]*dd[positive]/c[positive]
  Y=80*(f.tauxy-ff)**2/(2e6) # conditional intact analytic reaction
  Y[positive]=2e6*f.Axy[positive]**2*80*(1+3*f.phi[positive])*(1-f.phi[positive])/c[positive]**2
  stationarity=f.tauxy-Q-ff
  stage_load=np.zeros(len(b));end_load=np.zeros(len(b));poly=0.
  for cell in np.unique(f.cell):
   mask=f.cell==cell;idx=np.flatnonzero(b.cell==cell);samples=f[mask];local=b[idx]
   h=np.ptp(samples.x)/(np.polynomial.legendre.leggauss(6)[0][-1]);xx=(samples.x-local.x[0])/h+.5;yy=(samples.y-local.y[0])/h+.5
   bx=np.array([(1-xx)**2,2*xx*(1-xx),xx**2]);by=np.array([(1-yy)**2,2*yy*(1-yy),yy**2]);basis=np.array([bx[int(i)%3]*by[int(i)//3] for i in local.i])
   poly=max(poly,float(np.max(abs(local.B@basis-samples.phi))))
   stage_load[idx]=basis@(samples.weight*samples.Y);end_load[idx]=basis@(samples.weight*Y[mask])
  stage_error=float(max(abs(stage_load-b.driving_load)))
  assert poly<2e-13 and stage_error<2e-10,(poly,stage_error)
  res=(b.fracture_gradient-end_load)/40
  kkt=b.B-np.clip(b.B-res/b.mass,b.lower,1.)
  values.append(dict(step=step,time=float(row.time),conditional_endpoint_phase_KKT=float(max(abs(kkt))),
   conditional_force_max=float(max(abs(stationarity[positive]))) if np.any(positive) else 0.,conditional_force_L2=float(np.sqrt(sum(f.weight[positive]*stationarity[positive]**2))),
   independent_stage_load_error=stage_error,polynomial_error=poly))
 result[label]=values
(OUT/'endpoint-stationarity.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v[-1] for k,v in result.items()},indent=2))
