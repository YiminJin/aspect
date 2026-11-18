#!/usr/bin/env python3
"""Plot closed diagnostic evidence; no trial is presented as accepted history."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-recover-free-turn-matplotlib')
import json,csv,re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[4]
out=root/'benchmarks/reconstructed_fault/particle_free/evidence/recover_free_turn_step16'
label='recover-free-turn-trace-complete'
rows=list(csv.DictReader((out/(label+'-row-summary.csv')).open()))
fd=json.loads((out/(label+'-one-sided.json')).read_text())
fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
for ax,iteration in zip(axes[0],(30,39)):
 r=sorted([a for a in rows if int(a['iteration'])==iteration and float(a['shift'])==0],key=lambda a:float(a['alpha']))
 x=[float(a['alpha']) for a in r]
 ax.loglog(x,[abs(float(a['current_trial_residual'])) for a in r],'o-',label='Current worst row, actual trial')
 ax.loglog(x,[float(a['trial_max']) for a in r],'s-',label='Worst actual trial row')
 ax.loglog(x,[float(a['max_linear_predicted']) for a in r],'--',label='Largest sampled linear prediction')
 ax.axhline(float(r[0]['current_max']),color='.5',ls=':',label='Current maximum')
 ax.axhline(1e-7,color='black',lw=.7,ls='--',label='Acceptance gate')
 ax.set(title=f'Unpublished Newton iteration {iteration}',xlabel='Trial fraction',ylabel='Absolute projected phase residual')
 ax.grid(alpha=.2)
axes[0,0].legend(fontsize=7)
ax=axes[1,0]
for row,style in [(120060,'o-'),(27395,'s--')]:
 r=sorted([a for a in fd if a['row']==row],key=lambda a:a['alpha'])
 ax.loglog([a['alpha'] for a in r],[a['relative_error'] for a in r],style,label=f'Near-intact row {row}')
ax.set(title='One-sided tangent check at iteration36',xlabel='Feasible perturbation fraction',ylabel='Relative directional discrepancy')
ax.legend(fontsize=8);ax.grid(alpha=.2)
ax.text(.03,.03,'Very small probes are limited by assembly arithmetic.',transform=ax.transAxes,fontsize=8)
ax=axes[1,1]
runroot=root/'build-particle-free/bulk-tip/runs'
for name,display in [('first-turn-positive-fourcore-end','Original .005 failure'),(label,'Diagnostic replay'),('recover-free-turn-positive-half','Half-step attempt')]:
 path=runroot/name/'output/log.txt'
 if not path.exists():continue
 step=None;data={}
 for line in path.read_text().splitlines():
  m=re.search(r'\*\*\* Timestep (\d+)',line)
  if m:step=int(m[1])
  m=re.search(r'Bulk tip coupled Newton (\d+): mechanics=([^,]+), phase KKT=([^,]+)',line)
  if m and step and step>=16:data.setdefault(step,[]).append((int(m[1]),float(m[3])))
 for step,points in data.items():
  points=np.array(points);ax.semilogy(points[:,0],points[:,1],label=f'{display}, step{step}')
ax.axhline(1e-7,color='black',lw=.7,ls='--')
ax.set(title='Nonlinear trials, not accepted phase evolution',xlabel='Newton iteration',ylabel='Maximum phase KKT')
ax.legend(fontsize=7);ax.grid(alpha=.2)
for ext in ('png','pdf'):fig.savefig(out/f'recovery-diagnosis.{ext}',dpi=180)
plt.close(fig)
