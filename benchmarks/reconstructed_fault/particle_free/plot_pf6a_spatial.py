#!/usr/bin/env python3
"""Standalone cancellation, frozen spatial error, and common-entry diagnostics."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from assess_pf6a_spatial import fields
from assess_pf6a import csv

p=argparse.ArgumentParser(description=__doc__);p.add_argument('assessment',type=Path);p.add_argument('runs',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
d=json.loads(a.assessment.read_text());fig,ax=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
selected=[r for r in d['runs'] if r['case']['backend']=='DG' and r['case']['quadrature']==12]
for r in selected:
 n=r['resolved']['cells'];f=fields(a.runs/r['label']/'run/resolved');f=f[f['x']==min(f['x'])];f=np.sort(f,order='y')
 if n==32:
  for k,label in [('Txy',r'$T_{xy}$'),('Zchi',r'$Z\chi$')]:ax[0,0].plot(f['y'],f[k]/1000,label=label)
 ax[0,1].plot(f['y'],f['xy']/1000,label=str(n),lw=1)
 ax[1,0].loglog(n,r['resolved']['whole_full_tensor_L2_Pa'],'o',color='C0')
 ax[1,0].loglog(n,r['resolved']['localized_full_tensor_RMS_Pa'],'s',color='C1')
 event=a.runs/f'pf6a-spatial-event-DG{n}'/'run/event-4/gradient.csv'
 if event.exists():
  g=csv(event);g=g[g['x']==.5];g=np.sort(g,order='y');ax[1,1].plot(g['y'],g['phi'],'.-',label=str(n),markersize=2)
ax[0,0].set(xlim=(.2,.8),xlabel='normal coordinate y (m)',ylabel='kPa',title='Large cancelling terms, 32-cell solve');ax[0,0].legend()
ax[0,1].axhline(d['continuum']['mean_tau']/1000,color='k',ls='--',label='continuum');ax[0,1].set(xlim=(.2,.8),xlabel='y (m)',ylabel=r'$\tau_{xy}$ (kPa)',title='Frozen physical phase and incoming histories');ax[0,1].legend(title='normal cells')
for k,mark,label,col in [('whole_full_tensor_L2_Pa','o','whole-domain L2','C0'),('localized_full_tensor_RMS_Pa','s','localization-weighted RMS','C1')]:
 ax[1,0].plot([r['resolved']['cells'] for r in selected],[r['resolved'][k] for r in selected],'-'+mark,label=label,color=col)
ax[1,0].set(xlabel='normal cells',ylabel='full-tensor error (Pa)',title='Mechanical projection remainder');ax[1,0].legend()
ax[1,1].set(xlim=(.2,.8),xlabel='y (m)',ylabel='physical phase',title='Short common-entry event at t=0.5625 s');ax[1,1].legend(title='normal cells')
for x in ax.flat:x.grid(alpha=.2)
a.output.parent.mkdir(parents=True,exist_ok=True)
for suffix in ['png','pdf']:fig.savefig(a.output.with_suffix('.'+suffix),dpi=180)
