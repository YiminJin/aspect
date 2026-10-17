#!/usr/bin/env python3
"""Standalone common-time spatial errors and onset/growth field snapshots."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from assess_pf6a_reference import load,sample

p=argparse.ArgumentParser(description=__doc__);p.add_argument('assessment',type=Path);p.add_argument('cases',type=Path);p.add_argument('output',type=Path);a=p.parse_args();d=json.loads(a.assessment.read_text());a.output.mkdir(exist_ok=True)
fig,axes=plt.subplots(3,3,figsize=(12,10),constrained_layout=True)
labels=[('whole_stress','Whole tensor stress error'),('localized_stress','Localized tensor stress error'),('phase_L2','Phase shape L2 error'),('q','Cohesive traction error'),('slip','Cumulative slip error'),('energy','Total stored energy error'),('driving','Common phase-driving error')]
for n in [64,128,256,512]:
 key=f'mechanics_{n}';rows=d['comparisons'][key]['by_time'];t=[r['time']for r in rows]
 for ax,(name,title)in zip(axes.flat,labels):ax.semilogy(t,np.maximum(1e-13,[r[name]for r in rows]),label=str(n));ax.set(title=title,ylabel='fraction of fixed peak scale')
 c=load(a.cases/f'p{n}-m{n}-d128-q12');axes[2,1].plot(t,np.max(c['fields']['phase'],axis=1),label=str(n));ph=c['fields']['phase'];y=c['fields']['y'];mass=np.full(n+1,1/n);mass[[0,-1]]*=.5;centroid=(ph@(mass*y))/(ph@mass);axes[2,2].plot(t,centroid-.5,label=str(n))
for i,(ax,(name,title))in enumerate(zip(axes.flat,labels)):ax.axhline(d['targets'][name],color='k',ls='--',lw=.8)
axes[2,1].set(title='Core growth from physical initialization',ylabel='phase maximum');axes[2,2].set(title='Unaligned physical phase position',ylabel='phase centroid minus .5 m')
for ax in axes.flat:ax.set(xlim=(.5,.5625),xlabel='time (s)');ax.grid(alpha=.2)
axes[0,0].legend(title='mechanical/phase cells',fontsize=8)
fig.suptitle('Projected Q2 versus continuum mechanics at matching phase resolution\nNative 512 confirmation failed publication before onset; curves are independent 1D results',fontsize=11)
for suffix in ['png','pdf']:fig.savefig(a.output/('history.'+suffix),dpi=170)
plt.close(fig)
fig,axes=plt.subplots(2,4,figsize=(13,6),constrained_layout=True);y=np.linspace(.2,.8,4097)
for n,nm,label in [(256,256,'Q2/256'),(512,512,'Q2/512'),(1024,None,'continuum/1024')]:
 c=load(a.cases/f'p{n}-m{nm or "exact"}-d128-q12');phase,chi,stress,_=sample(c,y)
 for j,t in enumerate([0.,.53125,.546875,.5625]):
  k=round(t*128);axes[0,j].plot(y,phase[k],label=label);axes[1,j].plot(y,stress[k]/1000,label=label,lw=.8);axes[0,j].set_title(f't = {t:g} s')
for ax in axes.flat:ax.set(xlabel='normal y (m)');ax.grid(alpha=.2)
for ax in axes[0]:ax.set_ylabel('physical phase')
for ax in axes[1]:ax.set_ylabel('shear stress (kPa)')
axes[0,0].legend(fontsize=8)
for suffix in ['png','pdf']:fig.savefig(a.output/('snapshots.'+suffix),dpi=170)

fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for n,nm,label in [(512,512,'Q2/512'),(512,None,'continuum/512'),(1024,None,'continuum/1024')]:
 c=load(a.cases/f'p{n}-m{nm or "exact"}-d128-q12');f=c['fields'];growth=np.max(f['phase']-f['phase'][0],axis=1)
 axes[0].semilogy(f['time'],np.maximum(growth,1e-16),label=label)
 axes[1].plot(f['y'],f['phase'][1]-f['phase'][0],label=label)
axes[0].axhline(1e-5,color='k',ls='--',label='declared onset threshold');axes[0].set(xlabel='time (s)',ylabel='max growth from prescribed seed',ylim=(1e-8,.1));axes[0].legend(fontsize=8)
axes[1].set(xlim=(0.,1.),xlabel='normal y (m)',ylabel='first-step phase increment')
for ax in axes:ax.grid(alpha=.2)
fig.suptitle('Same continuum seed prescription' if 'physical_initialization' in d else 'Literal table-interpolant refinement control')
for suffix in ['png','pdf']:fig.savefig(a.output/('onset.'+suffix),dpi=170)
