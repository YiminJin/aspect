#!/usr/bin/env python3
"""Profile balance, localization and actual state resolution of completed cases."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-layered-mpl')
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from assess_layered_campaign import OUT


def main():
    parser=argparse.ArgumentParser();parser.add_argument('cases',nargs='+')
    args=parser.parse_args()
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    costs=[]
    for i,name in enumerate(args.cases):
        case=json.loads((OUT/(name+'.json')).read_text())
        rows=case['records'];time=np.array([r['time'] for r in rows])
        energy=[r['energy'] for r in rows];color=f'C{i%10}'
        label=name.removeprefix('layered-bulk-')
        mass=np.array([e['profile_mass'] for e in energy])
        axes[0,0].plot(time,100*(mass/mass[0]-1),color=color,label=label)
        axes[0,1].plot(time,[e['peak_phase'] for e in energy],color=color)
        axes[0,2].plot(time,[100*e['normalization_relative'] for e in energy],color=color)
        theta=np.array([r['age_vertices'] for r in rows])
        actual=np.max(abs(np.log(theta[1:]/theta[:-1])),axis=1)
        axes[1,0].plot(time[1:],actual,color=color)
        defect=np.array([e.get('endpoint_remainder',0) for e in energy])
        axes[1,1].plot(time,np.cumsum(abs(defect)),color=color)
        axes[1,1].plot(time,np.cumsum(defect),'--',color=color)
        costs.append((label,case['wall_s']/60,color))
    for axis,title in zip(axes.flat[:5],[
        'Profile mass change (%)','Sampled Q1 phase peak',
        'Localization normalization\n(measure / curve length - 1, %)',
        'Maximum actual log-age change\nper accepted step',
        'Endpoint remainder (J/m)\nsolid: absolute; dashed: signed']):
        axis.set(title=title,xlabel='time (s)');axis.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    axes[1,2].barh([r[0] for r in costs],[r[1] for r in costs],color=[r[2] for r in costs])
    axes[1,2].set(title='Native cost, four MPI ranks',xlabel='wall time (minutes)')
    axes[1,2].grid(axis='x',alpha=.2)
    fig.savefig(OUT/'layered-accuracy-cost.png',dpi=160)
    fig.savefig(OUT/'layered-accuracy-cost.pdf')


if __name__=='__main__':
    main()
