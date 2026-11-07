#!/usr/bin/env python3
"""Variable-step histories against time-interpolated resolved uniform label histories."""
import argparse,json
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import numpy as np
from assess_layered_campaign import OUT,compare_final
from assess_sheared_moving_fault import masks,metric
from layered_fields import dot3

def main():
 p=argparse.ArgumentParser();p.add_argument('variable');p.add_argument('uniform');a=p.parse_args()
 A=json.loads((OUT/(a.variable+'.json')).read_text());B=json.loads((OUT/(a.uniform+'.json')).read_text())
 initial={k:float(np.max(abs(np.array(v)-np.array(B['records'][0]['profile'][k]))))
          for k,v in A['records'][0]['profile'].items()}
 assert max(initial.values())<1e-11,initial
 times=np.array([r['time'] for r in B['records']]);records=[]
 for r in A['records']:
  t=r['time'];j=max(0,min(len(times)-2,int(np.searchsorted(times,t)-1)));q=(t-times[j])/(times[j+1]-times[j])
  assert -1e-9<=q<=1+1e-9
  ref={k:(1-q)*np.array(B['records'][j]['profile'][k])+q*np.array(B['records'][j+1]['profile'][k]) for k in r['profile']}
  regions={}
  for region,mask in masks(np.array(A['labels'])).items():
   vals={};w=ref['jacobian'][mask]/len(mask)
   for k in ref:
    if k=='jacobian':continue
    v=np.array(r['profile'][k])[mask];b=ref[k][mask];e=v-b
    if k=='stress':num,den=dot3(e,e),dot3(b,b)
    elif v.ndim==2:num,den=np.sum(e*e,axis=1),np.sum(b*b,axis=1)
    else:num,den=e*e,b*b
    vals[k]=metric(num,den,w)
   regions[region]=vals
  records.append(dict(time=t,regions=regions,reference_interval=[float(times[j]),float(times[j+1])]))
 intervals=np.diff([r['time'] for r in records]);history_norms={}
 for region in records[0]['regions']:
  history_norms[region]={}
  for field in records[0]['regions'][region]:
   values=[r['regions'][region][field] for r in records]
   def integral(key):
    square=np.array([v[key]**2 for v in values])
    return float(np.sum(intervals*.5*(square[1:]+square[:-1])))
   error,reference=integral('absolute_L2'),integral('reference_L2')
   history_norms[region][field]=dict(absolute_L2_space_time=error**.5,
       reference_L2_space_time=reference**.5,
       relative_L2_space_time=(error/reference)**.5 if reference else None)
 result=dict(variable=a.variable,uniform=a.uniform,initialization_Linf=initial,
     interpretation='Linear time interpolation of uniform accepted material-label histories; no dense-output accuracy claim. Space-time norms use trapezoidal integration of squared spatial norms at adaptive accepted times.',
     max_reference_interval=float(np.max(np.diff(times))),history=records,history_norms=history_norms,
     final=compare_final(A,B),cost={a.variable:A['cost']|dict(wall_s=A['wall_s']),a.uniform:B['cost']|dict(wall_s=B['wall_s'])})
 (OUT/('schedule-'+a.variable+'.json')).write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(dict(variable=a.variable,steps=len(records)-1,final=result['final']['common_domain'])),flush=True)
if __name__=='__main__':main()
