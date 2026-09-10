#!/usr/bin/env python3
"""Summarize endpoint moment identities without changing a trajectory."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    base=Path(__file__).resolve().parent
    out=base/'endpoint'
    report=dict(volume_wall_model=[],adjacent_projection_split=[])
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for nx,directory in ((32,base/'refinement/space32'),(64,base/'output'),(128,base/'refinement/space128')):
        def get(k):
            a=np.loadtxt(directory/f'particles_{k}.csv',delimiter=',',skiprows=1,usecols=(0,1,2,3))
            return a[np.argsort(a[:,0])]
        initial,first,second=[get(k) for k in (0,1,2)]
        spacing=.25/(3*nx)
        for k,old,new in ((1,initial,first),(2,first,second)):
            if not np.array_equal(old[:,0],new[:,0]):
                raise ValueError('Particle IDs changed')
            for side,sign in (('left',1),('right',-1)):
                mask=(initial[:,1]<spacing*.51) if side=='left' else (initial[:,1]>.25-spacing*.51)
                mask &= abs(initial[:,2])<.3088215939070757
                actual=new[mask,3]-old[mask,3]
                # First-column Voronoi areas are bounded by the fixed box wall;
                # a shared row translation moves their inner face by delta_x.
                predicted=sign*spacing*(new[mask,1]-old[mask,1])
                report['volume_wall_model'].append(dict(nx=nx,time_s=.5*k,side=side,
                    count=int(sum(mask)),max_relative_volume_change=float(max(abs(actual/old[mask,3]))),
                    correlation=float(np.corrcoef(actual,predicted)[0,1]),
                    relative_rms_model_error=float(np.linalg.norm(actual-predicted)/np.linalg.norm(actual)),
                    max_model_error_m2=float(max(abs(actual-predicted)))))
    for k in (1,2):
        read=lambda name:np.genfromtxt(out/name/f'weak_terms_{k}.csv',delimiter=',',names=True)
        a,b=read('replay64-analysis'),read('saved128')
        x=b['s'];g,w=np.polynomial.legendre.leggauss(3)
        gx=((x[:-1,None]+x[1:,None])/2+np.diff(x)[:,None]*g/2).ravel()
        gw=(np.diff(x)[:,None]*w/2).ravel()
        terms={name:np.interp(x,a['s'],a[name])-b[name] for name in
               ('q','geometry_only_current_q_change','geometry_only_old_history_change','published_transfer_error')}
        for name in terms:
            terms[name]-=np.trapezoid(terms[name],x)/.25
        terms['previous_geometry_remainder']=terms['q']-terms['geometry_only_current_q_change']
        record=dict(time_s=.5*k,terms={})
        for name,values in terms.items():
            record['terms'][name]=dict(left_Pa=float(values[0]),right_Pa=float(values[-1]),
                rms_Pa=float(np.sqrt(np.average(np.interp(gx,x,values)**2,weights=gw))))
        report['adjacent_projection_split'].append(record)
        if k==2:
            for name,label in (('q','actual 64 - 128'),('geometry_only_current_q_change','current - previous measure'),
                               ('previous_geometry_remainder','algebraic remainder')):
                axes[0].plot(x,terms[name],'.-',label=label)
            for n,d in ((64,a),(128,b)):
                axes[1].plot(d['s'],d['published_transfer_error'],'.-',label=str(n))
            np.savetxt(out/'adjacent_split_at_1s.csv',np.column_stack((x,*terms.values())),delimiter=',',
                       header='s,'+','.join(terms),comments='')
    axes[0].set(title='Traction-anomaly difference at 1 s',xlabel='fault coordinate (m)',ylabel='Pa')
    axes[1].set(title='Direct published FE-minus-particle history load',xlabel='fault coordinate (m)',ylabel='projected beta delta tau (Pa)')
    for ax in axes:
        ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/'endpoint_diagnosis.png',dpi=160)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
