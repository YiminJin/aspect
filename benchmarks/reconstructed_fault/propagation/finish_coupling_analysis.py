#!/usr/bin/env python3
"""Failed entries and common-history-time comparisons; no physical updates."""
from pathlib import Path
import csv,json,os
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-coupling-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[3];b=root/'build-propagation-integration';e=Path(__file__).parent/'coupling_evidence'
def rows(p):return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(p.open())]
def field(p,key='phi'):
 v=np.zeros((33,33))
 for r in rows(p):v[round(32*r['x']),round(32*r['y'])]=r[key]
 return v
failed={};data={}
for tag,dt in [('125',.125),('0625',.0625),('03125',.03125)]:
 p=b/('output-coupling_evolving_dt'+tag);d=b/('output-coupling_diagnosis_dt'+tag);f=rows(p/'feedback.csv');k=int(f[-1]['step'])+1
 m=rows(p/f'material-phase-metrics-entry-{k}.csv')[0];phase=rows(p/f'material-phase-entry-{k}.csv');worst=max(phase,key=lambda x:x['old']-x['phi']);diag=rows(d/'healing-diagnosis.csv')[0]
 assert m['healing']>4.922e-12 and diag['correction_inf']<1e-12 and diag['repeat1']==diag['repeat2']==0
 assert diag['repeated_healing']==m['healing']
 files=list(p.glob('*.csv'));assert all((d/x.name).read_bytes()==x.read_bytes() for x in files)
 failed[tag]=dict(dt=dt,last_accepted_time=f[-1]['time'],entry=m,worst_node=worst,diagnosis=diag,identical_replay_csvs=len(files))
 data[tag]=(p,f,rows(p/'material-local.csv'),rows(p/'recoverable-energy.csv'),rows(p/'coupling-work.csv'),k)
lag=[]
for t in [.625,.75]:
 z={}
 for tag,dt in [('125',.125),('0625',.0625),('03125',.03125)]:
  f=data[tag][1];r=next((r for r in f if abs(r['time']-t-dt)<1e-12),None)
  if r:z[tag]=dict(phase_time=r['time'],consumed_H_through=t,phi_max=r['phi_max'])
 lag.append(dict(history_time=t,cases=z))
(e/'failed-entries.json').write_text(json.dumps(dict(failed=failed,lag_comparison=lag),indent=2)+'\n')
fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
for col,(tag,d) in enumerate(data.items()):
 p,f,loc,en,work,k=d;dt=failed[tag]['dt'];entry=field(p/f'material-phase-entry-{k}.csv');old=field(p/f'material-phase-entry-{k}.csv','old');time=failed[tag]['entry']['time']
 # Same physical time in the top row, separate failed fields in the bottom.
 common=field(p/f'material-phase-accepted-{round(.75/dt)}.csv')-field(p/'material-phase-accepted-0.csv')
 im=axes[0,col].imshow(common.T,origin='lower',extent=[0,1,0,1],vmin=0,vmax=.025);axes[0,col].set(title=f'Accepted t=0.75 s, dt={dt}',xlabel='x (m)',ylabel='y (m)');fig.colorbar(im,ax=axes[0,col],label='Phase gain')
 change=entry-old;scale=max(1e-12,-change.min());im=axes[1,col].imshow(np.minimum(change,0).T,origin='lower',extent=[0,1,0,1],vmin=-scale,vmax=0,cmap='RdBu');axes[1,col].set(title=f'Failed entry t={time} s',xlabel='x (m)',ylabel='y (m)');fig.colorbar(im,ax=axes[1,col],label='Phase decrease only')
fig.savefig(e/'coupling-spatial-and-failed.png',dpi=170);fig.savefig(e/'coupling-spatial-and-failed.pdf');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(12,3.6),constrained_layout=True)
for tag,(p,f,loc,en,work,k) in data.items():
 label=f"dt={failed[tag]['dt']} s"
 for ax,keys in [(axes[0],['bulk_stored','fault_stored']),(axes[1],['bulk_viscous','fault_viscous']),(axes[2],['fracture_change'])]:
  for key in keys:ax.plot([r['time'] for r in en],[r[key] for r in en],label=label+', '+key)
for ax,title in zip(axes,['Stored energy (parent bulk / surface fault)','Viscous dissipation (separate fields)','Fracture expenditure']):ax.set(title=title,xlabel='Time (s)',ylabel='J/m out of plane');ax.legend(fontsize=6);ax.grid(alpha=.25)
fig.savefig(e/'coupling-energy-components.png',dpi=170);plt.close(fig)
print(json.dumps(failed,indent=2))
