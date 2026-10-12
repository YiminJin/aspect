#!/usr/bin/env python3
"""Reduce the bounded material study without advancing a model."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss
here=Path(__file__).resolve().parent; result=here/'results';result.mkdir(exist_ok=True)
def read(path):
 return np.atleast_1d(np.genfromtxt(path,delimiter=',',names=True))
def dump(path,rows):
 if not rows:return
 with (result/path).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
records=[];phases=[];local=[];norms=[];driving=[];summaries=[];cases={}
for evidence in sorted((here/'evidence').glob('*.json')):
 r=json.loads(evidence.read_text());label=r['label'];out=here/('output-'+label)
 if not (out/'feedback.csv').exists():continue
 data=read(out/'feedback.csv');params=read(out/'study-runtime.csv');m=float(params['m_native'][0]);energy=float(params['energy'][0]);states=[]
 for row in data:
  step=int(row['step']);time=float(row['time'])
  state=read(out/f'study-state-accepted-{step}.csv')[0]
  combined=dict(case=label,Gc=r['gc'],a=r['amplitude'],**{k:float(row[k]) for k in data.dtype.names},phi_max_x=float(state['x_max']),phi_max_y=float(state['y_max']))
  records.append(combined);states.append(combined)
  fault=read(out/f'study-fault-{step}.csv')
  for z in fault:
   local.append(dict(case=label,step=step,time=time,imposed_displacement=.5*r['amplitude']*time*time,**{k:float(z[k]) for k in fault.dtype.names}))
  grid=read(out/f'study-phase-accepted-{step}.csv')
  for z in grid:
   if z['y']==.5 or z['x'] in (0,.5,1):
    phases.append(dict(case=label,step=step,time=time,**{k:float(z[k]) for k in grid.dtype.names}))
  # Independent analytic transverse integral of the captured Q1 field.
  arr=np.zeros((33,33))
  for z in grid:arr[round(z['x']*32),round(z['y']*32)]=z['phi']
  native_g_error=float(max(abs(grid['g']-(1-grid['phi'])**2/((1-grid['phi'])**2+m*grid['phi']*(1+grid['phi'])))))
  assert native_g_error<5e-16
  xs=np.unique(np.r_[np.linspace(0,1,3201),np.geomspace(1e-8,.03125,100),1-np.geomspace(1e-8,.03125,100)])
  ix=np.minimum((xs*32).astype(int),31);t=(xs*32-ix)[:,None]
  col=((1-t)*arr[ix]+t*arr[ix+1]).astype(np.longdouble);u=1-col;u0=u[:,:-1];u1=u[:,1:];z=(u1-u0)/u0
  log=np.divide(np.log1p(z),z,out=np.ones_like(z),where=z!=0)/u0
  direct=np.sum(m/32*(2/(u0*u1)-3*log+1),axis=1).astype(float)
  nf=read(out/f'study-normalization-accepted-{step}.csv');native=np.interp(xs,nf['x'],nf['I_native'])
  # Independent Q64 quadrature at exact endpoints and center.
  gq,w=leggauss(64);theta=(gq+1)/2
  qerr=[]
  for x in (0.,.5,1.):
   j=min(int(x*32),31);p=x*32-j;c=(1-p)*arr[j]+p*arr[j+1]
   sample=c[:-1,None]+(c[1:]-c[:-1])[:,None]*theta
   numerical=np.sum(w/64*m*sample*(1+sample)/(1-sample)**2)
   qerr.append(abs(numerical/np.interp(x,xs,direct)-1))
  norms.append(dict(case=label,step=step,time=time,I_min=float(min(native)),I_max=float(max(native)),relative_L1=float(np.trapezoid(abs(native-direct),xs)/np.trapezoid(direct,xs)),max_source_error=float(max(abs(direct/native-1))),left_source_integral=float(direct[0]/native[0]),right_source_integral=float(direct[-1]/native[-1]),Q64_analytic_relative=max(qerr),native_g_error=native_g_error))
  particles=read(out/f'study-particles-{step}.csv')
  for region,mask in [('left',particles['x']<1/32),('center',abs(particles['x']-.5)<1/64),('right',particles['x']>31/32)]:
   # Separate all-normal-depth mean from near-core samples; H absolute is initialization-relative only.
   for band,bm in [('full',np.ones(len(particles),dtype=bool)),('core',abs(particles['y']-.5)<1/32)]:
    z=particles[mask&bm];v=z['volume'];v=v/v.sum()
    driving.append(dict(case=label,step=step,time=time,region=region,band=band,delta_H_mean=float(v@(z['H_committed']-z['H_initial'])),delta_H_max=float(max(z['H_committed']-z['H_initial'])),incoming_H_increment_mean=float(v@(z['H_committed']-z['H_incoming'])),driving_initial_mean=float(v@z['driving_initial']),driving_incoming_mean=float(v@z['driving_incoming']),driving_committed_mean=float(v@z['driving_committed']),driving_incoming_max=float(max(z['driving_incoming'])),driving_committed_max=float(max(z['driving_committed'])),cpdi_point_max=float(max(abs(z['phi_cpdi']-z['phi_point'])))))
 active=[z for z in states[1:] if z['phase_change']>1e-6]
 norm=[z for z in norms if z['case']==label]
 failed=[]
 accepted={int(z['step']) for z in states}
 for p in sorted(out.glob('phase-entry-*.csv')):
  step=int(p.stem.split('-')[-1])
  if step not in accepted:failed.append(dict(step=step,**{k:float(v) for k,v in zip(read(p).dtype.names,read(p)[0])}))
 s=dict(case=label,Gc=r['gc'],a=r['amplitude'],dt=r['dt'],exit_code=r['exit_code'],seconds=r['seconds'],accepted_physical_steps=len(states)-1,last_time=states[-1]['time'],phi0=states[0]['phi_max'],phi_final=states[-1]['phi_max'],delta_phi_max=states[-1]['phi_max']-states[0]['phi_max'],max_increment=max(z['phase_change'] for z in states[1:]) if len(states)>1 else 0,evolving_cycles=len(active),first_evolving_time=active[0]['time'] if active else None,maximum_healing=max(z['phase_healing'] for z in states),maximum_KKT=max(z['phase_kkt'] for z in states),maximum_mechanical_residual=max(z['mechanical_residual'] for z in states),normalization_max_source_error=max(z['max_source_error'] for z in norm),normalization_final_relative_L1=norm[-1]['relative_L1'],normalization_final_endpoint_ratio=norm[-1]['left_source_integral'],failed_entries=failed)
 summaries.append(s);cases[label]=dict(meta=r,states=states,faults=[z for z in local if z['case']==label])
dump('accepted.csv',records);dump('local.csv',local);dump('phase_profiles.csv',phases);dump('normalization.csv',norms);dump('driving.csv',driving)
(result/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
# Compact figures contain every completed material/calibration case, labeled honestly.
plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.2})
fig,axes=plt.subplots(1,3,figsize=(12,3.5))
for label,c in cases.items():
 s=c['states'];title=f"{label}: Gc={c['meta']['gc']:.3g}, a={c['meta']['amplitude']:g}, dt={c['meta']['dt']:g}"
 axes[0].plot([x['time'] for x in s],[x['phi_max'] for x in s],'.-',label=title)
 for ax,xlocation in zip(axes[1:],(0,.5)):
  z=[x for x in c['faults'] if abs(x['x']-xlocation)<1e-10]
  ax.plot([x['slip'] for x in z],[x['phi'] for x in z],'.-',label=title)
for ax,title,xlabel in zip(axes,['Maximum phase','Left endpoint phase','Center phase'],['Time (s)','Accumulated local slip (m)','Accumulated local slip (m)']):
 ax.set(title=title,xlabel=xlabel,ylabel='Phase')
axes[0].legend(fontsize=6);fig.tight_layout();fig.savefig(result/'phase_time_slip.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(12,3.5))
for label,c in cases.items():
 for ax,xlocation in zip(axes,(0,.1,.5)):
  z=[x for x in c['faults'] if abs(x['x']-xlocation)<1e-10]
  ax.plot([x['slip'] for x in z],[x['q'] for x in z],'.-',label=label)
for ax,x in zip(axes,(0,.1,.5)):ax.set(title=f'x={x:g} m',xlabel='Accumulated local slip (m)',ylabel='Cohesive traction q (Pa)')
axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(result/'traction_slip.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(12,3.5))
for label,c in cases.items():
 steps=sorted({0,int(c['states'][-1]['step'])})
 for step in steps:
  z=[x for x in phases if x['case']==label and x['step']==step];style='--' if step==0 else '-'
  for ax,coord,key,value in [(axes[0],'x','y',.5),(axes[1],'y','x',0),(axes[2],'y','x',.5)]:
   p=sorted([x for x in z if x[key]==value],key=lambda x:x[coord]);ax.plot([x[coord] for x in p],[x['phi'] for x in p],style,label=f'{label}, step {step}')
for ax,title,xlabel in zip(axes,['Along fault','Across left endpoint','Across center'],['x (m)','y (m)','y (m)']):ax.set(title=title,xlabel=xlabel,ylabel='Phase')
axes[0].legend(fontsize=6);fig.tight_layout();fig.savefig(result/'spatial_phase.png',dpi=180);plt.close(fig)
print(json.dumps(summaries,indent=2))
