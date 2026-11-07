#!/usr/bin/env python3
"""Surface/time sensitivity, signed energy and cost for the selected two runs."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='1'
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/positive_refinement'
os.environ['MPLCONFIGDIR']=str(OUT/'matplotlib-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    result=json.loads((OUT/'comparison.json').read_text());rows=result['cases']
    names=['.10 / .025','.05 / .025','.05 / .0125'];colors=['#666666','#0072b2','#d55e00']
    fig,ax=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    x=np.arange(3)
    ax[0,0].semilogy(x,[100*r['endpoint_relative'] for r in rows],'o-',label='Endpoint')
    ax[0,0].semilogy(x,[100*r['trapezoid_relative'] for r in rows],'s--',label='Trapezoidal diagnostic')
    ax[0,0].set(xticks=x,xticklabels=names,xlabel='Label spacing (m) / timestep (s)',ylabel='Accumulated |remainder| / active work (%)',title='Energy accuracy at 64² cells')
    keys=['stress_relative_L2','point_traction_relative_L2','work_traction_relative_L2','slip_rate_relative_L2','cumulative_slip_relative_L2']
    x=np.arange(len(keys))
    for key,offset,color in [('surface',-.18,colors[1]),('time',.18,colors[2])]:
        ax[0,1].bar(x+offset,[100*result['comparisons'][key]['final'][k] for k in keys],width=.36,label=key.capitalize()+' refinement',color=color)
    ax[0,1].set(xticks=x,xticklabels=['Stress','Point traction','Work traction','Slip rate','Slip'],ylabel='Relative pairwise difference (%)',title='Sensitivity of coupled fields')
    ax[0,1].tick_params(axis='x',labelrotation=15)
    for r,name,color in zip(rows,names,colors):
        steps=r['steps'][1:];t=[s['time'] for s in steps];defect=np.array([s['endpoint_remainder'] for s in steps])
        ax[1,0].plot(t,1e9*np.cumsum(defect),color=color,label=name+' signed')
        ax[1,0].plot(t,1e9*np.cumsum(abs(defect)),color=color,ls='--',label=name+' absolute')
    ax[1,0].set(xlabel='Physical time (s)',ylabel='Cumulative endpoint remainder (nJ/m)',title='Signed and accumulated absolute errors')
    ax[1,1].bar(np.arange(3),[r['wall_s'] for r in rows],color=colors)
    for i,r in enumerate(rows):ax[1,1].text(i,r['wall_s']+2,f"{r['linear_solves']} solves",ha='center',fontsize=9)
    ax[1,1].set(xticks=np.arange(3),xticklabels=names,xlabel='Label spacing (m) / timestep (s)',ylabel='Wall time (s), four MPI ranks',title='Cost including initialization and output')
    ax[1,1].set_ylim(top=max(r['wall_s'] for r in rows)*1.15)
    for a in ax.flat:a.grid(alpha=.2)
    for a in [ax[0,0],ax[0,1],ax[1,0]]:a.legend(fontsize=8)
    fig.savefig(OUT/'positive-refinement.png',dpi=180);plt.close(fig)

if __name__=='__main__':main()
