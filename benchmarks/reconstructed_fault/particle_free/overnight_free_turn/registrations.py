#!/usr/bin/env python3
"""Track first plane registration and its eventual slip without any cutoff."""
from observations import *
import argparse
p=argparse.ArgumentParser(description=__doc__);p.add_argument('labels',nargs='+');p.add_argument('--hy',type=float,required=True);p.add_argument('--name',required=True);p.add_argument('--end-time',type=float,default=.08);a=p.parse_args()
registrations={};final=None
z,_=np.polynomial.legendre.leggauss(6);y=((np.arange(round(4/a.hy))[:,None]+(z+1)/2)*a.hy).ravel()
for label in a.labels:
 path=RUNS/label/'output'
 if not (path/'bulk-accepted.csv').exists():continue
 for r in read(path/'bulk-accepted.csv'):
  if r.time>a.end_time+1e-12:continue
  step=int(r.step);plane=read(path/f'tip-planes-{step:04d}.csv');section=read(path/f'tip-sections-{step:04d}.csv');poly=None
  for k in np.flatnonzero((plane.formed>0)&~np.isin(plane.section,list(registrations))):
   if poly is None:poly=field(label,step,a.hy)
   registrations[int(k)]=dict(section=int(k),x=float(plane.x[k]),time=0. if step==1 else float(r.time),first_export_time=float(r.time),angle=float(plane.angle[k]),
       maximum_phase_on_normal_Gauss6_at_first_export=float(max(poly.evaluate(plane.x[k],y))),initial_seed=bool(step==1))
  final=(section,plane)
if final:
 section,plane=final
 for k,v in registrations.items():v.update(final_slip=float(section.slip[k]),final_age=float(section.age[k]),final_ds=float(section.ds[k]))
(OUT/(a.name+'-registrations.json')).write_text(json.dumps(list(registrations.values()),indent=2)+'\n')
print('Registered',len(registrations),'sections; no phase/rate thresholds used.')
