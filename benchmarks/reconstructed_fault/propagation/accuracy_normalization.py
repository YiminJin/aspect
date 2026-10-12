#!/usr/bin/env python3
"""Frozen Q1 bulk fields: consistent surface representation only, no histories."""
import csv,json,os
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize_scalar
os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-accuracy-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
BUILD=HERE.parents[2]/'build-propagation-integration'
OUT=HERE/'accuracy_energy_evidence';OUT.mkdir(exist_ok=True)
# Existing independently checked Q1 transverse antiderivative, with no phase cap.
helpers=(HERE/'analyze_irreversibility.py').read_text().split('import argparse')[0]
ns={'__file__':str(HERE/'analyze_irreversibility.py')};exec(helpers,ns)
field,integral,at,rows=(ns[x] for x in ['field','integral','at','rows'])
MATERIAL_M=128.00000000016

def project(v,ds,order=3,split_bulk=False):
    n=round(1/ds);nodes=np.linspace(0,1,n+1)
    edges=np.unique(np.r_[nodes,np.arange(33)/32]) if split_bulk else nodes
    z,w=leggauss(order);x=(edges[:-1,None]+(edges[1:]-edges[:-1])[:,None]*(z+1)/2).ravel()
    W=((edges[1:]-edges[:-1])[:,None]*w/2).ravel()
    j=np.minimum((x/ds).astype(int),n-1);t=(x-nodes[j])/ds
    N=np.zeros((len(x),n+1));N[np.arange(len(x)),j]=1-t;N[np.arange(len(x)),j+1]=t
    mass=N.T@(W[:,None]*N);rhs=N.T@(W*integral(v,x,MATERIAL_M))
    values=np.linalg.solve(mass,rhs)
    constant_error=float(np.max(abs(np.linalg.solve(mass,N.T@W)-1)))
    assert constant_error<2e-14 and np.min(N)>=0 and np.min(W)>0
    conservation=abs(np.sum(mass@values)-np.sum(rhs))/np.sum(rhs)
    assert conservation<2e-15
    return nodes,values,conservation,constant_error

def assess(v,nodes,I):
    error=lambda x: integral(v,np.asarray(x),MATERIAL_M)/np.interp(x,nodes,I)-1
    # Include all bulk/surface breakpoints and bracket every smooth subinterval.
    edges=np.unique(np.r_[np.arange(33)/32,nodes])
    candidates=list(edges)
    for a,b in zip(edges[:-1],edges[1:]):
        for sign in [-1,1]:
            opt=minimize_scalar(lambda x:sign*error(x),bounds=(a,b),method='bounded',options={'xatol':1e-14})
            candidates.append(opt.x)
    errors=error(np.array(candidates));idx=np.argmax(abs(errors));interior=[k for k,x in enumerate(candidates) if 0<x<1]
    interior_idx=max(interior,key=lambda k:abs(errors[k]))
    return dict(min_I=float(min(I)),max_source_error=float(abs(errors[idx])),worst_x=float(candidates[idx]),signed_worst=float(errors[idx]),interior_worst_x=float(candidates[interior_idx]),endpoint_errors=[float(error(0)),float(error(1))],constant_error_bound=2e-14)

summary=[];fig,axes=plt.subplots(2,3,figsize=(12,6),constrained_layout=True)
for col,label in enumerate(['irreversibility_dt125','irreversibility_dt0625','irreversibility_dt03125']):
    folder=BUILD/('output-'+label);feedback=rows(folder/'feedback.csv')
    onset=next(int(r['step']) for r in feedback if r['phase_change']>4.922e-12)
    for row,step in enumerate([onset,int(feedback[-1]['step'])]):
        v=field(folder/f'material-phase-accepted-{step}.csv');assert np.max(v)<1
        for ds in [.1,.05,.025]:
            nodes,I,cons,const=project(v,ds)
            _,high,_,_=project(v,ds,32,True);_,higher,_,_=project(v,ds,48,True)
            native=assess(v,nodes,I);accurate=assess(v,nodes,high)
            native.update(label=label,step=step,time=feedback[step]['time'],state='onset' if row==0 else 'strong',spacing=ds,phi_max=float(v.max()),conservation_error=cons,constant_error=const,reference_projection=accurate,projection_quadrature_uncertainty=float(np.max(abs(high/higher-1))),native_Q3_vs_split_Q32=float(np.max(abs(I/high-1))))
            assert native['projection_quadrature_uncertainty']<2e-12 and native['min_I']>0 and accurate['min_I']>0
            if ds==.1:
                retained=rows(folder/f'material-normalization-accepted-{step}.csv')
                err=max(abs(np.interp(r['x'],nodes,I)/r['projected']-1) for r in retained)
                assert err<1e-8, f'Native reproduction exceeds configured integration tolerance {err}'
                native['retained_native_error']=err
            summary.append(native)
            x=np.linspace(0,1,4001);direct=integral(v,x,MATERIAL_M);projected=np.interp(x,nodes,I)
            axes[row,col].plot(x,100*(direct/np.interp(x,nodes,high)-1),label=f'{ds:g} m')
            with (OUT/f'normalization-{label}-{step}-{ds:g}.csv').open('w') as f:
                writer=csv.writer(f);writer.writerow(['x','direct','consistent_Q3','consistent_split_Q32','source_relative_error'])
                writer.writerows(zip(x,direct,projected,np.interp(x,nodes,high),direct/projected-1))
        axes[row,col].set(title=f"dt={feedback[1]['dt']:g}, t={feedback[step]['time']:g}",xlabel='x (m)',ylabel='Source error (%)');axes[row,col].axhline(.5,color='k',ls=':');axes[row,col].axhline(-.5,color='k',ls=':');axes[row,col].legend(fontsize=8)
fig.suptitle('Fixed integration accuracy: split Q32 surface RHS, analytic transverse integrals')
(OUT/'normalization.json').write_text(json.dumps(summary,indent=2)+'\n')
fig.savefig(OUT/'normalization.png',dpi=160);fig.savefig(OUT/'normalization.pdf')
for r in summary:print(r['label'],r['state'],r['spacing'],f"{100*r['max_source_error']:.6f}%",f"x={r['worst_x']:.8f}",f"space-only={100*r['reference_projection']['max_source_error']:.6f}%",f"Q3={100*r['native_Q3_vs_split_Q32']:.6f}%")
