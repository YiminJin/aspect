#!/usr/bin/env python3
"""Plot the measured native confirmation and its unregistered 1D comparison."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__);p.add_argument('assessment',type=Path);p.add_argument('run',type=Path);p.add_argument('reference',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
r=json.loads(a.assessment.read_text());rows=r['by_time'];t=[v['time']for v in rows]
m=np.genfromtxt(a.run/'run/metrics.csv',delimiter=',',names=True);ref=json.loads((a.reference/'history.json').read_text());f=np.load(a.reference/'fields.npz')
fig,axes=plt.subplots(2,3,figsize=(12,6.3),layout='constrained')
axes[0,0].plot(m['time'],m['phase_max'],'o-',ms=2,label='native DGQ2');axes[0,0].plot(f['time'],f['phase'].max(axis=1),'--',label='selected 1D Q2');axes[0,0].set_ylabel('maximum phase')
surface=np.array([np.genfromtxt(a.run/'run'/f"state-{v['step']}"/'accepted-surface.csv',delimiter=',',names=True)for v in rows])
axes[0,1].plot(t,(surface['Z']/surface['I']).mean(axis=1),'o-',ms=2);axes[0,1].plot(f['time'],np.r_[0,[v['q']for v in ref]],'--');axes[0,1].set_ylabel('cohesive traction (Pa)')
axes[0,2].plot(t,surface['slip'].mean(axis=1),'o-',ms=2);axes[0,2].plot(f['time'],f['slip'],'--');axes[0,2].set_ylabel('cumulative slip (m)')
for key,label in [('whole_stress','whole stress'),('localized_stress','localized stress'),('phase_L2','phase L2')]:axes[1,0].semilogy(t,np.maximum([v[key]for v in rows],1e-16),label=label)
axes[1,0].axhline(1e-6,c='gray',ls=':',label='native/1D target');axes[1,0].set_ylabel('scaled differences')
axes[1,1].plot(m['time'],m['bulk']+m['cohesive']+m['fracture'],'o-',ms=2);axes[1,1].plot(f['time'][1:],[v['bulk']+v['cohesive']+v['fracture']for v in ref],'--');axes[1,1].set_ylabel('recoverable energy (J/m)')
for key,label in [('q','traction'),('slip','slip'),('energy','energy'),('driving','phase driving')]:axes[1,2].semilogy(t,np.maximum([v[key]for v in rows],1e-16),label=label)
axes[1,2].set_ylabel('scaled differences')
for ax in axes.ravel():ax.set_xlabel('time (s)');ax.grid(alpha=.2);ax.axvline(.53125,c='gray',lw=.6,ls=':')
for ax in [axes[0,0],axes[1,0],axes[1,2]]:ax.legend(fontsize=8)
fig.suptitle('Fixed 2×512 PF6a: native publication repair and selected 1D reference')
for ext in ['png','pdf']:fig.savefig(a.output.with_suffix('.'+ext),dpi=170)
