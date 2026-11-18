#!/usr/bin/env python3
"""Compare closed fixed-plane or normal-refined histories at common times."""
from observations import *
import argparse,copy
p=argparse.ArgumentParser(description=__doc__);p.add_argument('case',choices=['b','c']);a=p.parse_args()
labels=['overnight-free-turn-'+a.case+'-'+part for part in (['prefix','half'] if a.case=='b' else ['pilot','prefix','half'])]
labels=[l for l in labels if (RUNS/l/'output/bulk-accepted.csv').exists()]
candidate_by_time={round(float(r.time),12):(label,int(r.step)) for label in labels for r in read(RUNS/label/'output/bulk-accepted.csv')}
hy=.015625 if a.case=='c' else .03125
seedlabel=labels[0];first=int(read(RUNS/seedlabel/'output/bulk-accepted.csv')[0].step)
initial={'label':seedlabel,'field':seed(seedlabel,first,hy)}
coarseinitial={'label':'first-turn-positive-prefix','field':seed('first-turn-positive-prefix',1)}
coarselabels=['first-turn-positive-prefix','first-turn-positive-end','first-turn-positive-fourcore-end','recover-free-turn-positive-half']
base={round(float(r.time),12):(label,int(r.step)) for label in coarselabels for r in read(RUNS/label/'output/bulk-accepted.csv')}
reference_work=work(coarselabels)['steps']
def cumulative(rows,time):
 errors=[r['error'] for r in rows if r['time']<=time+1e-12]
 return dict(signed=float(sum(errors)),absolute=float(sum(abs(v) for v in errors)))
result=dict(case=a.case,seed_error=l2(coarseinitial['field'],initial['field']),seed_width=pulse(initial['field']),comparisons=[],work=work(labels))
result['initial_fracture']=float(read(RUNS/seedlabel/'output/tip-initial.csv')[0].fracture)
result['reference_initial_fracture']=float(read(RUNS/coarseinitial['label']/'output/tip-initial.csv')[0].fracture)
last=max(float(r.time) for label in labels for r in read(RUNS/label/'output/bulk-accepted.csv'))
for label in labels:
 for r in read(RUNS/label/'output/bulk-accepted.csv'):
  time=round(float(r.time),12)
  if time not in base or (time not in [.04,.06,.065,.07,.075,.0775,.08] and abs(time-last)>1e-12):continue
  ref,step=base[time];x,fx=endpoint(label,int(r.step),hy,initial);y,fy=endpoint(ref,step,.03125,coarseinitial)
  gx=copy.copy(fx);gx.coefficients=fx.coefficients-initial['field'].coefficients
  gy=copy.copy(fy);gy.coefficients=fy.coefficients-coarseinitial['field'].coefficients
  comparison=dict(time=time,candidate=x,coarse_free=y,phase_error=l2(fy,fx),increment_error=l2(gy,gx),common_centerlines=[])
  comparison['candidate_work_to_time']=cumulative(result['work']['steps'],time)
  comparison['coarse_work_to_time']=cumulative(reference_work,time)
  if time>.075+1e-12 and .075 in candidate_by_time:
   parent_label,parent_step=candidate_by_time[.075]
   px=field(parent_label,parent_step,hy);ref_parent,ref_step=base[.075]
   py=field(ref_parent,ref_step,.03125)
   comparison['late_endpoint_error']=l2(fy,fx,parent=py)
   dx=copy.copy(fx);dx.coefficients=fx.coefficients-px.coefficients
   dy=copy.copy(fy);dy.coefficients=fy.coefficients-py.coefficients
   comparison['late_increment_error']=l2(dy,dx)
   comparison['late_error_note']='Window starts at.075. Endpoint error includes inherited spatial/closure differences; increment error subtracts each path own.075 field.'
  sx=read(RUNS/label/'output'/f'tip-sections-{int(r.step):04d}.csv');sy=read(RUNS/ref/'output'/f'tip-sections-{step:04d}.csv')
  assert max(abs(sx.s-sy.s))<1e-12
  comparison['section_errors']={n:dict(maximum=float(max(abs(sx[n]-sy[n]))),weighted_L2=float(np.sqrt(sum(sy.ds*(sx[n]-sy[n])**2)))) for n in ['age','slip']}
  if abs(x['stage_time']-y['stage_time'])<1e-13:
   comparison['section_errors']['same_stage_V']=dict(time=x['stage_time'],maximum=float(max(abs(sx.V-sy.V))),weighted_L2=float(np.sqrt(sum(sy.ds*(sx.V-sy.V)**2))))
  else:comparison['section_errors']['stage_rate_note']='Different stage times; no terminal-stage rate comparison.'
  for one,two in zip(x['centerlines'],y['centerlines']):
   if 'center' in one and 'center' in two:comparison['common_centerlines'].append(dict(x=one['x'],candidate_center=one['center'],coarse_center=two['center'],candidate_width=one['width'],coarse_width=two['width']))
  result['comparisons'].append(comparison)
(OUT/(a.case.upper()+'-comparison.json')).write_text(json.dumps(result,indent=2)+'\n')
for c in result['comparisons']:print(c['time'],c['candidate']['front'],c['increment_error'])
