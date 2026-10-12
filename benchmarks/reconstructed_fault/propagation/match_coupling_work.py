#!/usr/bin/env python3
"""Frozen-record replay at the actual assembly surface rule (Q3).
The original observer used independent Q8 surface integration; retain it as an
accuracy comparator, never confuse it with the matched discrete balance.
"""
from pathlib import Path
import csv,json,math
import numpy as np
from numpy.polynomial.legendre import leggauss
root=Path(__file__).resolve().parents[3];b=root/'build-propagation-integration';e=Path(__file__).parent/'coupling_evidence'
def rows(p):return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(p.open())]
summary={}
for kind in ['work','evolving']:
 for tag in ['125','0625','03125']:
  label=f'coupling_{kind}_dt{tag}';p=b/('output-'+label);work=rows(p/'coupling-work.csv');feedback=rows(p/'feedback.csv');records=[]
  for r in work[1:]:
   k=int(r['step']);dt=feedback[k]['dt'];beta=math.exp(-dt/100);kappa=-1e8*math.expm1(-dt/100);fault=rows(p/f'feedback-fault-{k}.csv');vals=[]
   for order in [3,8]:
    z,w=leggauss(order);power=friction=integrated=jump=0.
    for s in range(10):
     for zz,ww in zip(z,w):
      x=(zz+1)/2;weight=ww/20;interpolate=lambda key:(1-x)*fault[s][key]+x*fault[s+1][key]
      I=interpolate('Ih_current');oldI=interpolate('Ih_incoming');V=interpolate('V')
      Z=(1-x)*fault[s]['Ih_incoming']*fault[s]['q_incoming']+x*fault[s+1]['Ih_incoming']*fault[s+1]['q_incoming']
      q=(beta*Z+kappa*V)/I;steady=1e8*V/I;transient=Z/I-steady
      power+=weight*q*V;friction+=weight*1000*V
      integrated+=weight*V*(steady*dt+transient*(-math.expm1(-dt/100))*100)
      jump+=weight*Z*Z/(2e6)*(1/I-1/oldI)
    vals.append((power,friction,integrated,jump))
   q3,q8=vals
   # Raw records contain Q8. Recover their bulk endpoint power from that record;
   # no mechanical or physical history is recomputed or advanced.
   assert abs(q8[2]-r['cohesive_work'])<1e-9*max(1,abs(q8[2]))
   endpoint=r['endpoint_exchange_defect']+q8[0]+q8[1]-q3[0]-q3[1]
   integrated=r['bulk_crack_work']-r.get('profile_work',0)-q3[2]-q3[1]*dt
   records.append(dict(step=k,time=r['time'],native_Q3_endpoint_exchange_defect=endpoint,native_Q3_integrated_exchange_defect=integrated,Q3_cohesive_work=q3[2],Q3_friction_work=q3[1]*dt,Q3_fault_phase_jump=q3[3],Q8_minus_Q3_endpoint_power=q8[0]-q3[0]))
  out=e/(label+'-matched-work.csv')
  with out.open('w') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
  summary[label]=dict(max_native_endpoint_defect=max(abs(r['native_Q3_endpoint_exchange_defect']) for r in records),integrated_defect=sum(r['native_Q3_integrated_exchange_defect'] for r in records),max_Q8_Q3_power_difference=max(abs(r['Q8_minus_Q3_endpoint_power']) for r in records))
  assert summary[label]['max_native_endpoint_defect']<1e-9
(e/'matched-work.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
