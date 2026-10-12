#!/usr/bin/env python3
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
r=Path(__file__).resolve().parent;e=r/'native_tangent_evidence';p=e/'economical-1';ref=r/'native_endpoint_evidence/dt-1-2048'
f,ff=[np.load(q/'fields.npz') for q in [p,ref]]
h,hf=[json.loads((q/'history.json').read_text()) for q in [p,ref]]
fig,ax=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
for q,fields,hist,label in [(p,f,h,'economical: 154 steps'),(ref,ff,hf,'uniform: 1280 steps')]:
 t=np.r_[0,[x['time'] for x in hist]];slip=np.r_[0,[x['slip'] for x in hist]];traction=np.r_[0,[x['q'] for x in hist]]
 ax[0,0].plot(t,fields['phase'][:,256],label=label)
 ax[0,1].plot(slip,traction,label=label)
 ax[0,2].plot(fields['y'],fields['phase'][-1],label=label)
 ledger=json.loads((q/'energy.json').read_text())
 for j,k in enumerate(['mechanical_power_absolute','friction_residual_power_absolute','phase_power_absolute']):
  ax[1,j].plot(t[1:],np.cumsum([x[k] for x in ledger]),label=label)
for a,title,x,y in zip(ax.ravel(),['Core phase','Traction–slip','Final physical profile','Mechanical residual work','Friction residual work','Phase residual work'],['t (s)','slip (m)','y (m)','t (s)','t (s)','t (s)'],['phase','q (Pa)','phase','absolute work (J/m)','absolute work (J/m)','absolute work (J/m)']):
 a.set(title=title,xlabel=x,ylabel=y);a.legend(fontsize=7);a.grid(alpha=.2)
fig.savefig(e/'economical-schedule.png',dpi=160);fig.savefig(e/'economical-schedule.pdf')
