#!/usr/bin/env python3
"""Native-weight local sign/cancellation audit, with no phase or rate cutoff."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,csv,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'first_free_turn'))
from analyze import RUNS,OUT,gather,read
p=argparse.ArgumentParser(description=__doc__);p.add_argument('label');p.add_argument('--step',type=int,default=17);a=p.parse_args()
path=RUNS/a.label/'output';out=OUT.parent/'recover_free_turn_step16'
q=gather(path,f'tip-points-{a.step:04d}-r',['cell','q'])
s=read(path/f'tip-sections-{a.step:04d}.csv');planes=read(path/f'tip-planes-{a.step:04d}.csv')
q=q[np.lexsort((q.y,q.section))]
offsets=np.searchsorted(q.section,np.arange(len(s)+1))
records=[]
for k in range(len(s)):
 v=q[offsets[k]:offsets[k+1]]
 pos=v.w>0;neg=v.w<0
 def components(mask):return int(sum(mask & ~np.r_[False,mask[:-1]]))
 positive=float(sum(v.weight[pos]*v.w[pos]));negative=float(sum(v.weight[neg]*v.w[neg]))
 signed=positive+negative;absolute=positive-negative
 assert abs(signed-s.ds[k]*s.V[k])<2e-14
 power=v.weight*v.w*s.force[k]
 records.append(dict(section=k,x=float(planes.x[k]),formed=int(planes.formed[k]),angle=float(planes.angle[k]),
   phase_max=float(max(v.phi)),positive_integral=positive,negative_integral=negative,
   V=float(s.V[k]),force=float(s.force[k]),section_power=float(s.ds[k]*s.V[k]*s.force[k]),
   positive_local_power=float(sum(power[power>0])),negative_local_power=float(sum(power[power<0])),
   cancellation=1-abs(signed)/absolute if absolute else 0.,
   sampled_positive_components=components(pos),sampled_negative_components=components(neg),
   negative_y_min=float(min(v.y[neg])) if neg.any() else None,negative_y_max=float(max(v.y[neg])) if neg.any() else None))
with (out/f'{a.label}-signed-sections-{a.step:04d}.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
mixed=[r for r in records if r['positive_integral']>0 and r['negative_integral']<0]
result=dict(label=a.label,step=a.step,sign_rule='Exact native quadrature signs; no phase/rate cutoff. Components are sampled sign runs, not continuous-band reconstruction.',
  mixed_sections=len(mixed),maximum_cancellation=max(r['cancellation'] for r in records),
  maximum_positive_components=max(r['sampled_positive_components'] for r in records),
  maximum_negative_components=max(r['sampled_negative_components'] for r in records),
  minimum_section_power=min(r['section_power'] for r in records),
  signed_section_power=sum(r['section_power'] for r in records),
  negative_local_power=sum(r['negative_local_power'] for r in records),
  positive_local_power=sum(r['positive_local_power'] for r in records),
  largest_negative_local_power_sections=sorted(records,key=lambda r:r['negative_local_power'])[:6],
  largest_cancellation_sections=sorted(records,key=lambda r:r['cancellation'],reverse=True)[:6])
(out/f'{a.label}-signed-sections-{a.step:04d}.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if not isinstance(v,list)},indent=2))
