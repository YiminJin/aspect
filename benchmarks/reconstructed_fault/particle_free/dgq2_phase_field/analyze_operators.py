#!/usr/bin/env python3
"""Compact tables and a standalone PNG from native operator evidence."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-dgphase-mpl')
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[4]
out=root/'benchmarks/reconstructed_fault/particle_free/evidence/dgq2_phase_field'
rows=[]
for p in sorted((out/'transport02').glob('*.log')):
 for line in p.read_text().splitlines():
  if line.startswith('{'):
   row=json.loads(line);row['log']=str(p.relative_to(root));rows.append(row)
with (out/'transport-summary.csv').open('w') as stream:
 writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
q1=json.loads((out/'q1-controls.json').read_text())
selected=lambda mode,limited=1:sorted([r for r in rows if r['mode']==mode and r['limited']==limited and r['interval']==.05],key=lambda r:r['level'])
plus=selected('translate');unlimited=selected('translate',0)
q=[r for r in q1 if r['speed']==1 and r['dt']==.05]
fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
ax=axes[0,0]
for data,key,label in [(q,'final_l2','Q1 MacCormack replay'),(unlimited,'final_l2','DGQ2 unlimited'),(plus,'final_l2','DGQ2 + bounds'),(plus,'initial_l2','DGQ2 initialization')]:
 ax.loglog([4/2**r['level'] for r in data],[r[key] for r in data],'-o',label=label)
ax.set(xlabel='Cell width',ylabel='L2 error',title='Compact translation, travel 0.75');ax.legend(fontsize=8);ax.grid(alpha=.2)
ax=axes[0,1]
for data,label in [(q,'Q1 MacCormack replay'),(plus,'DGQ2 + bounds')]:
 ax.semilogy([4/2**r['level'] for r in data],[abs(r['final_integral']/r['initial_integral']-1) for r in data],'-o',label=label)
ax.semilogy([4/2**r['level'] for r in plus],[r['balance']/r['initial_integral'] for r in plus],'--o',label='DG accumulated balance defect')
ax.set(xlabel='Cell width',ylabel='Relative integral change / defect',title='Compact profile; numerical outflow retained');ax.legend(fontsize=8);ax.grid(alpha=.2)
ax=axes[1,0]
for mode in ('translate','reverse','rotation','divergence','inflow'):
 data=selected(mode);ax.loglog([4/2**r['level'] for r in data],[r['final_l2'] for r in data],'-o',label=mode)
ax.set(xlabel='Cell width',ylabel='L2 error',title='Limited native material transport');ax.legend(fontsize=8);ax.grid(alpha=.2)
ax=axes[1,1]
for mode,limiting,label in [('translate',0,'Translation, unlimited'),('translate',1,'Translation, bounds'),('rotation',0,'Rotation, unlimited'),('rotation',1,'Rotation, bounds')]:
 data=selected(mode,limiting);ax.loglog([r['dofs'] for r in data],[r['seconds'] for r in data],'-o',label=label)
ax.set_xticks([2304,9216,36864],['2.3k','9.2k','36.9k']);ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set(xlabel='Phase DoFs',ylabel='Transport wall seconds',title='Native C++ costs (one rank)');ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.savefig(out/'operator-comparison.png',dpi=180);plt.close(fig)
summary=dict(cases=len(rows),maximum_accumulated_absolute_balance=max(r['balance'] for r in rows),maximum_step_balance=max(r['maximum_step_balance'] for r in rows),
             translation_l2=[r['final_l2'] for r in plus],translation_orders=[float(np.log2(plus[i]['final_l2']/plus[i+1]['final_l2'])) for i in range(2)],
             smooth_limiter_activations=sum(r['limited_cells'] for r in rows if r['mode'] in ('inflow','quadratic-shear')),
             temporal_control=[r for r in rows if r['mode']=='inflow' and r['level']==4 and r['limited'] and r['interval']!=.05])
(out/'operator-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='temporal_control'}))
