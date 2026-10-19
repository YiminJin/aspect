#!/usr/bin/env python3
"""Plot measured, unregistered P2 physical response and work uncertainty."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    cases={name:json.loads((a.root/name/'assessment.json').read_text()) for name in ['native-primary','native-half']}
    fig,axes=plt.subplots(2,3,figsize=(13,7.2),layout='constrained')
    for name,style,label in [('native-primary','-','dt=1/128 s'),('native-half','--','dt=1/256 s')]:
        rows=cases[name]['metrics'];t=np.array([r['time'] for r in rows])
        axes[0,0].plot(t,[r['phase_max'] for r in rows],style,label=label)
        axes[0,2].plot(t,[r['q_mean'] for r in rows],style,label=label)
        axes[1,0].plot(t,[r['slip_mean'] for r in rows],style,label=label)
        w=cases[name]['work'];axes[1,2].plot([r['time'] for r in w],np.cumsum([abs(r['external_remainder']) for r in w]),style,label=label)
    native=np.load(a.root/'native-primary/fields.npz');y=np.linspace(0,1,native['node_grid'].shape[0])
    for t in [0.,.53125,.5625]:
        k=np.flatnonzero(native['time']==t)[0]
        axes[0,1].plot(y,native['phase'][k,native['node_grid']].mean(axis=1),label=f'{t:g} s')
    ref=np.load(a.root/'ref1024-continuum-d256/fields.npz')
    k=np.flatnonzero(ref['time']==.5625)[0]
    axes[0,1].plot(ref['y'],ref['phase'][k],':',color='black',label='1024 / continuum, .5625 s')
    axes[0,1].set_xlim(.25,.75)
    rows=cases['native-primary']['metrics'];t=[r['time'] for r in rows]
    for key,label in [('bulk','bulk'),('cohesive','cohesive'),('fracture','fracture')]:
        axes[1,1].plot(t,[r[key]-rows[0][key] for r in rows],label=label)
    for ax in [axes[0,0],axes[0,2],*axes[1]]:
        ax.axvspan(.5625,.59375,color='.9',label=None);ax.set_xlabel('time (s)')
    for ax,title,ylabel in [(axes[0,0],'Accepted phase peak','phase'),(axes[0,1],'Physical phase profiles (unaligned)','phase'),
        (axes[0,2],'Cohesive traction Z/I','Pa'),(axes[1,0],'Accumulated physical slip','m'),
        (axes[1,1],'Energy changes from initial state','J/m'),(axes[1,2],'Cumulative absolute external remainder','J/m')]:
        ax.set_title(title);ax.set_ylabel(ylabel);ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0,1].set_xlabel('normal coordinate y (m)')
    fig.suptitle('Fixed cohesive fault: resolved seed, native loading, then motion hold (grey)')
    fig.savefig(a.output.with_suffix('.png'),dpi=170)
    fig.savefig(a.output.with_suffix('.pdf'))
    plt.close(fig)


if __name__=='__main__':main()
