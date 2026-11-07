#!/usr/bin/env python3
"""Plot saved inflow/model, mechanical-corner and energy comparisons."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MPLCONFIGDIR']='/tmp/corner-isolation-matplotlib'
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/corner_isolation'


def main():
    p=json.loads((OUT/'prescribed.json').read_text());c=json.loads((OUT/'coupled.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    ax=axes[0,0]
    for held,color,label in [(False,'#0072b2','Exact exterior'),(True,'#d55e00','Held interior trace')]:
        rows=sorted([r for r in p['cases'] if r['cells']==1024 and r['reaction'] and r['held_trace']==held],key=lambda r:r['dt'])
        ax.loglog([r['dt'] for r in rows],[r['final']['corner_relative'] for r in rows],'o-',color=color,label=label)
    ax.set(title='Prescribed flow: corner error against discrete reference',xlabel='Physical timestep (s)',ylabel='Relative tensor L2 error');ax.legend(fontsize=8)
    ax=axes[0,1]
    for held,color,label in [(False,'#0072b2','Exact exterior'),(True,'#d55e00','Held interior trace')]:
        rows=sorted([r for r in p['cases'] if r['cells']==1024 and r['reaction'] and r['held_trace']==held],key=lambda r:r['dt'])
        ax.loglog([r['dt'] for r in rows],[r['final']['relative_continuous'] for r in rows],'o-',color=color,label=label)
    ax.set(title='Prescribed flow: error against continuous Maxwell law',xlabel='Physical timestep (s)',ylabel='Relative whole-domain tensor L2 error');ax.legend(fontsize=8)
    ax=axes[1,0];x=np.arange(3)
    for key,off,color,label in [('old_time',-.18,'#d55e00','Original zero top traction'),('compatible_time',.18,'#0072b2','Compatible top traction')]:
        r=c['comparisons'][key]
        vals=[r['regional_stress']['whole']['relative_L2'],r['regional_stress']['corner']['relative_L2'],r['final']['stress']['all']['relative_L2']]
        ax.bar(x+off,100*np.array(vals),width=.36,color=color,label=label)
    ax.set(xticks=x,xticklabels=['Whole domain','Upper-left corner','Fault tube'],ylabel='Relative difference (%)',title='Coupled dt=.05 versus .025, same held-trace inflow',yscale='log');ax.legend(fontsize=8)
    ax=axes[1,1]
    names=['shear32-dt050','shear32-dt025','corner-coupled050-r1','corner-coupled025'];x=np.arange(4)
    ax.bar(x-.17,[100*c['cases'][n]['endpoint_relative'] for n in names],width=.34,color='#0072b2',label='Endpoint')
    ax.bar(x+.17,[100*c['cases'][n]['trapezoid_relative'] for n in names],width=.34,color='#e69f00',label='Trapezoidal diagnostic')
    ax.set(xticks=x,xticklabels=['Original\n.05','Original\n.025','Compatible\n.05','Compatible\n.025'],ylabel='Accumulated |remainder| / active work (%)',title='Coupled energy accuracy');ax.legend(fontsize=8)
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.suptitle('Corner isolation — unchanged native DGQ2 / SSPRK3, four-rank bounded controls')
    for ext in ('png','pdf'):fig.savefig(OUT/f'corner-isolation.{ext}',dpi=170)
    plt.close(fig)


if __name__=='__main__':main()
