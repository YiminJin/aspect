#!/usr/bin/env python3
"""Plot the physical work balance, keeping the BE ledger distinct (SI)."""
import argparse,csv,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/cohesive-rsf-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(__doc__);p.add_argument('review',type=Path);args=p.parse_args()
rows=list(csv.DictReader((args.review/'indexed-histories.csv').open()))
cases=list(dict.fromkeys(r['case'] for r in rows));fig,ax=plt.subplots(2,2,figsize=(11.5,7))
for label in cases:
 h=[r for r in rows if r['case']==label]
 v=lambda key:np.array([float(r[key]) for r in h])
 t=v('time_s');weakening=label.startswith('State weakening');half=label.startswith('Half')
 if weakening or label.startswith('Matched'):
  color='tab:blue' if weakening else 'gray'
  ax[0,0].plot(t,np.cumsum(v('trapezoid_work_J_m')),color=color,label=label+' work')
  ax[0,0].plot(t,np.cumsum(v('bulk_increment_J_m')),'--',color=color,label=label+' Δbulk')
 if not (weakening or half):continue
 mask=t>=374.56704168613635;ids=np.flatnonzero(mask);time=t[mask]-t[ids[0]]
 def cumulative(key):
  z=v(key)[mask].copy();z[0]=0.;return np.cumsum(z)
 tag='Primary' if weakening else 'Half limits';style='-' if weakening else '--'
 for key,name,color in [('bulk_increment_J_m','Δbulk','tab:blue'),('bond_increment_J_m','Δcohesive','tab:orange'),('friction_J_m','friction','tab:green'),('radiation_J_m','radiation','tab:red')]:
  ax[0,1].plot(time,cumulative(key),color=color,ls=style,label=tag+' '+name)
 rem=v('physical_remainder_J_m')[mask].copy();rem[0]=0
 absolute=np.cumsum(abs(rem));sliding=cumulative('friction_J_m')+cumulative('radiation_J_m');fracture=cumulative('fracture_increment_J_m')
 color='tab:blue' if weakening else 'tab:orange'
 ax[1,0].plot(time,np.cumsum(rem),color=color,ls='-',label=tag+' signed physical remainder')
 ax[1,0].plot(time,cumulative('BE_bulk_J_m')+cumulative('surface_remainder_J_m'),color=color,ls=':',label=tag+' BE + surface remainder')
 for data,name,color in [(absolute,'|physical remainder| sum','tab:red'),(.01*sliding,'1% sliding dissipation','gray'),(.01*fracture,'1% new fracture','tab:green')]:
  ax[1,1].semilogy(time[1:],np.maximum(data[1:],1e-20),color=color,ls=style,label=tag+' '+name)
for aa,title,xx in zip(ax.flat,['Slow loading stores energy','Rapid replay window releases bulk storage','Physical remainder versus numerical ledger terms','Separate physical-energy screens'],['Time (s)']+['Time − 374.567041686 s']*3):
 aa.set(title=title,xlabel=xx,ylabel='Energy (J/m)');aa.grid(alpha=.2);aa.legend(fontsize=6.5)
fig.suptitle('Physical work uses trapezoidal boundary reaction work; exact BE closure is a separate check')
fig.tight_layout();fig.savefig(args.review/'energy.png',dpi=160);plt.close(fig)
