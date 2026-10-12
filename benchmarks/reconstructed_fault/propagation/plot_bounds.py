#!/usr/bin/env python3
"""Reference and constrained-equilibrium comparisons from saved P1 evidence."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from check_bound_evidence import rows

folder=Path(__file__).resolve().parent/'bounds_evidence'
fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
for case,ax in [('reference32',axes[0,1]),('reference64',axes[1,0])]:
    data=rows(folder/(case+'.reference-cuts.csv'),None)
    tail=[r for r in data if .16<=r['y']<=.22]
    for field,label,style in [('cpdi','Production CPDI (verified 1D reduction)','-'),
                              ('coarse_sampled_gauss8','Same mesh, sampled H, Gauss 8','--'),
                              ('fine_sampled_8192','Fine reference, sampled H','-'),
                              ('fine_native_8192','Fine reference, native H','-')]:
        ax.plot([r['y'] for r in tail],[r[field] for r in tail],style,label=label)
    ax.axhline(0,color='gray',lw=.6)
    ax.set(xlabel='y',ylabel='Phase field phi',ylim=(-.00025,.001),title=case+' tail')
    ax.legend(fontsize=7)
    if case=='reference32':
        for field,label in [('H_native','Native pointwise H'),('H_sampled','Particle-domain H')]:
            axes[0,0].plot([r['y'] for r in tail],[r[field] for r in tail],label=label)
        axes[0,0].axvline(.188664692008851,color='gray',ls=':',label='Native activation edge')
        axes[0,0].set(xlabel='y',ylabel='H',yscale='log',title='Actual support-edge transition',xlim=(.175,.205))
        axes[0,0].legend(fontsize=8)
data=rows(folder/'bounds_padded_final.bound-nodes.csv',None)
cut=sorted((r for r in data if r['y']==.5),key=lambda r:r['x'])
for field,label,style in [('rounded','Rounded starting guess','k--'),('relaxed','Committed box equilibrium','-'),
                         ('equality_final','Equality, +10% H','-'),('bounded_final','Box constrained, +10% H','-')]:
    axes[1,1].plot([r['x'] for r in cut],[r[field] for r in cut],style,label=label)
axes[1,1].set(xlabel='x at y=0.5',ylabel='Phase field phi',xlim=(.75,1.14),title='Padded cap: no seed constraint')
axes[1,1].axhline(0,color='gray',lw=.6)
axes[1,1].legend(fontsize=7)
for ax in axes.flat:ax.grid(alpha=.2)
fig.savefig(folder/'reference-and-bounds.svg',metadata={'Date':None})
