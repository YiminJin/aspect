#!/usr/bin/env python3
"""Frozen-entry comparison, including pointwise source, without advancing histories."""
from pathlib import Path
import csv,json,os
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-irrev-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[3];b=root/'build-propagation-integration';e=Path(__file__).parent/'irreversibility_evidence'
def rows(p):return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(p.open())]
def field(rs,key):
 v=np.zeros((33,33))
 for r in rs:v[round(32*r['x']),round(32*r['y'])]=r[key]
 return v
result={};fig,axes=plt.subplots(2,3,figsize=(12,6.5),constrained_layout=True)
for col,(tag,dt,step) in enumerate([('125',.125,8),('0625',.0625,14),('03125',.03125,28)]):
 label='irreversibility_replay_guard_dt'+tag;p=b/('output-'+label);run=json.loads((e/(label+'.json')).read_text())
 assert run['code']==1 and 'frozen-H irreversibility replay completed' in (e/(label+'.log')).read_text()
 r=rows(p/'irreversibility-replay.csv');prior=rows(b/('output-coupling_evolving_dt'+tag)/f'material-phase-entry-{step}.csv')
 box=field(r,'box');lower=field(r,'lower');new=field(r,'irreversible');oldbox=field(prior,'phi');oldlower=field(prior,'old')
 assert np.array_equal(box,oldbox) and np.array_equal(lower,oldlower), 'Incoming trajectory not identical'
 assert np.all(new>=lower) and np.max(new)<1
 rr=field(r,'residual');active=new==lower;free=(new>lower)&(new<1)
 assert np.max(rr[active])<=1e-11 and np.max(np.abs(rr[free]))<=1e-11
 delta=new-box;active_xy=[[r['x'],r['y']] for r in r if r['lower_active'] and r['lower']>0]
 metrics=rows(p/'irreversibility-replay-metrics.csv')[0];norms={k:rows(p/f'replay-normalization-{k}.csv') for k in ['box','irreversible']}
 n0,n1=norms.values();I0=np.array([r['projected'] for r in n0]);I1=np.array([r['projected'] for r in n1]);
 # The full source difference uses fixed bulk Q1 phase and retained surface I.
 x=np.linspace(0,1,201);y=np.arange(33)/32;i=np.minimum((32*x).astype(int),31);s=32*x-i
 phase=lambda v:(1-s[:,None])*v[i]+s[:,None]*v[i+1]
 h=lambda v:128.00000000016*phase(v)*(1+phase(v))/(1-phase(v))**2
 source0=h(box)/I0[:,None];source1=h(new)/I1[:,None]
 result[tag]=dict(dt=dt,identical_box_and_lower=True,metrics=metrics,full_difference_inf=float(abs(delta).max()),full_difference_rms=float(np.sqrt(np.mean(delta**2))),core_difference_inf=float(abs(delta[:,16]).max()),active_positive_locations=active_xy,active_sign_max=float(rr[active].max()),free_residual_max=float(abs(rr[free]).max()),I_relative_change=float(abs(I1/I0-1).max()),I_min_new=float(I1.min()),source_absolute_change=float(abs(source1-source0).max()),source_relative_to_max=float(abs(source1-source0).max()/source0.max()),normalized_integral_error={k:max(abs(r['normalized_source_integral']-1) for r in n) for k,n in norms.items()},fracture_difference=metrics['fracture_irreversible']-metrics['fracture_box'])
 im=axes[0,col].imshow(delta.T,origin='lower',extent=[0,1,0,1]);fig.colorbar(im,ax=axes[0,col],label='Irreversible minus box phase');axes[0,col].set(title=f'Frozen H, dt={dt}',xlabel='x (m)',ylabel='y (m)')
 im=axes[1,col].imshow((active&(lower>0)).T,origin='lower',extent=[0,1,0,1],vmin=0,vmax=1);axes[1,col].set(title='Positive lower-bound contacts',xlabel='x (m)',ylabel='y (m)')
(e/'replays.json').write_text(json.dumps(result,indent=2)+'\n');fig.savefig(e/'replays.png',dpi=170);plt.close(fig)
print(json.dumps({t:{k:v for k,v in z.items() if k not in ['active_positive_locations']} for t,z in result.items()},indent=2))
