#!/usr/bin/env python3
"""Compact charts of scoped MPI entries and cost; no trajectory qualification."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

E = Path(__file__).resolve().parent/'native_mpi_evidence'


def plot():
    result=json.loads((E/'assessment.json').read_text())
    plt.rcParams.update({'font.size':9})
    for family in ['growth','saddle']:
        data=result[family]
        fig,axes=plt.subplots(2,2,figsize=(9,6),layout='constrained')
        for run,check in zip(data['runs'],data['results']):
            out=Path(run['output']);name=check['entry']['case'];label=f"{run['ranks']} ranks"
            phase=np.genfromtxt(out/(name+'-phase.csv'),delimiter=',',names=True)
            center=phase[phase['x']==.5];center=np.sort(center,order='y')
            stress=np.genfromtxt(out/(name+'-stress.csv'),delimiter=',',names=True)
            x=np.unique(stress['x']);profile=stress[stress['x']==x[len(x)//2]];profile=np.sort(profile,order='y')
            axes[0,0].plot(center['y'],center['phase'],label=label)
            axes[0,1].plot(center['y'],center['phase']-center['lower'],label=label)
            axes[1,0].plot(profile['y'],profile['chi'],label=label)
            axes[1,1].plot(profile['y'],profile['tau_xy'],label=label)
        axes[0,0].plot(center['y'],center['lower'],'k:',label='incoming obstacle')
        phase_title = 'Accepted phase' if family == 'growth' else 'Converged trial phase (not committed)'
        increment_title = 'Phase increment' if family == 'growth' else 'Trial phase change'
        for ax,title,unit in zip(axes.flat,[phase_title,increment_title,'Localization','Off-quadrature shear stress'],
                                ['φ','Δφ','χ (1/m)','τxy (Pa)']):
            ax.set(title=title,xlabel='normal coordinate y (m)',ylabel=unit);ax.grid(alpha=.25)
        axes[0,0].legend()
        fig.suptitle(f"{family.capitalize()}: retained fixed-entry diagnostic, t={check['entry']['time']:g} s; unaligned fields")
        fig.savefig(E/(family+'-rank-profiles.png'),dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
    for family in ['growth','saddle']:
        runs=result[family]['runs'];ranks=[run['ranks'] for run in runs]
        times=[(run['accepted_entry_seconds'] or run['whole_entry_seconds'])[0] for run in runs]
        actions=[run['accepted_entry_actions']['calls'] for run in runs]
        axes[0].plot(ranks,np.array(times)/60,'o-',label=family)
        axes[1].plot(ranks,actions,'o-',label=family)
    axes[0].set(ylabel='Successful solve (minutes)',xlabel='MPI ranks',title='Whole entry, excluding forced failure')
    axes[1].set(ylabel='Tangent actions',xlabel='MPI ranks',title='Iteration changes reported separately')
    for ax in axes:ax.set_xticks([1,2,4]);ax.grid(alpha=.25);ax.legend()
    fig.savefig(E/'rank-cost.png',dpi=140);plt.close(fig)


if __name__=='__main__':
    plot()
