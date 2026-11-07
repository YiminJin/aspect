#!/usr/bin/env python3
"""Selected 64² positive/constant surface and timestep comparison; read-only data."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','DEAL_II_NUM_THREADS']:os.environ[key]='1'
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from compare_bounded_profile import ROOT,RUNS,load,reduce,tensor,quadrature,dot3,work_traction

HERE=Path(__file__).resolve().parent
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/positive_refinement'
CASES=['bpf-c-p-m-ref','bpr-c-p-m-surface64','bpr-c-p-m-halfdt64']

def comparison(a,b):
    """Physical bulk L2; surface diagnostics on the finer material-label grid."""
    x,w=quadrature(max(a.n,b.n),4);va,vb=a.at(x),b.at(x);ta,tb=tensor(a,x),tensor(b,x)
    result={}
    for key,err,base in [('profile',(va[:,0]-vb[:,0])**2,vb[:,0]**2),
                         ('stress',dot3(ta-tb,ta-tb),dot3(tb,tb)),
                         ('velocity',np.sum((va[:,1:3]-vb[:,1:3])**2,axis=1),np.sum(vb[:,1:3]**2,axis=1))]:
        numerator=float(w@err);denominator=float(w@base)
        result[key+'_absolute_L2']=float(np.sqrt(numerator))
        result[key+'_relative_L2']=float(np.sqrt(numerator/denominator)) if denominator>0 else None
    la=np.linspace(0,1,len(a.surface));lb=np.linspace(0,1,len(b.surface))
    for col,key in [(3,'age'),(4,'memory'),(5,'normalization'),(6,'slip_rate')]:
        aa=np.interp(lb,la,a.surface[:,col]);bb=b.surface[:,col];den=np.linalg.norm(bb)
        result[key+'_relative_L2']=float(np.linalg.norm(aa-bb)/den) if den>0 else None
        result[key+'_Linf']=float(np.max(abs(aa-bb)))
    xa=np.stack([np.interp(lb,la,a.surface[:,c]) for c in [1,2]],axis=1)
    result['geometry_Linf']=float(np.linalg.norm(xa-b.surface[:,1:3],axis=1).max())
    wa,_=work_traction(a);wb,M=work_traction(b);delta=np.interp(lb,la,wa)-wb;den=float(wb@M@wb)
    result['work_traction_relative_L2']=float(np.sqrt((delta@M@delta)/den)) if den>0 else None
    g,gw=np.polynomial.legendre.leggauss(3);q=.5*(g+1)
    labels=(lb[:-1,None]+q[None,:]*np.diff(lb)[:,None]).ravel()
    def traction(state):
        l=np.linspace(0,1,len(state.surface));segment=np.minimum(np.searchsorted(l,labels,side='right')-1,len(l)-2)
        u=(labels-l[segment])/(l[segment+1]-l[segment]);X=state.surface[:,1:3]
        d=X[segment+1]-X[segment];t=d/np.linalg.norm(d,axis=1)[:,None];n=np.stack([-t[:,1],t[:,0]],axis=1)
        tau=tensor(state,X[segment]+u[:,None]*d)
        return t[:,0]*(tau[:,0]*n[:,0]+tau[:,2]*n[:,1])+t[:,1]*(tau[:,2]*n[:,0]+tau[:,1]*n[:,1])
    weights=(.5*np.linalg.norm(np.diff(b.surface[:,1:3],axis=0),axis=1)[:,None]*gw).ravel()
    aa,bb=traction(a),traction(b);den=float(weights@(bb*bb))
    result['point_traction_relative_L2']=float(np.sqrt((weights@((aa-bb)**2))/den)) if den>0 else None
    return result

def main():
    lock=(RUNS.parent/'simulation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    rows=[];states={};initial={};counters={}
    for name in CASES:
        result,final=reduce(name);rows.append(result);states[name]=final
        command,summary,meta,start=load(name,final_only=False);initial[name]=start[0]
        for row in summary:
            assert int(row['history_publications'])==int(row['step']),row
            assert int(row['geometry_publications'])==int(row['step']),row
            assert int(row['retries'])==0,row
        expected=round(.2/command['arguments']['dt']);assert len(summary)-1==expected,(name,len(summary))
        assert abs(float(summary[-1]['time'])-.2)<1e-14
        counters[name]=summary
        print(name, 'physical_steps',expected,'endpoint',result['endpoint_relative'],'trapezoid',result['trapezoid_relative'],flush=True)
        del start
    by={r['case']:r for r in rows};pairs={}
    for key,a,b in [('surface',CASES[0],CASES[1]),('time',CASES[1],CASES[2])]:
        final=comparison(states[a],states[b]);first=comparison(initial[a],initial[b])
        la=np.linspace(0,1,len(states[a].surface));lb=np.linspace(0,1,len(states[b].surface))
        aa=np.interp(lb,la,by[a]['endpoint_cumulative_slip']);bb=np.array(by[b]['endpoint_cumulative_slip'])
        final['cumulative_slip_relative_L2']=float(np.linalg.norm(aa-bb)/np.linalg.norm(bb))
        final['storage_relative']=abs(by[a]['steps'][-1]['storage']-by[b]['steps'][-1]['storage'])/by[b]['steps'][-1]['storage']
        final['net_flux_difference']=by[a]['trapezoid_integrals']['flux']-by[b]['trapezoid_integrals']['flux']
        final['flux_difference_over_active_scale']=abs(final['net_flux_difference'])/by[b]['power_scale']
        pairs[key]=dict(cases=[a,b],initial=first,final=final)
    result=dict(cases=rows,comparisons=pairs,publication_counters=counters,
                norms='Full 3D plane-strain bulk L2 at common Gauss4 quadrature; scalar nodal l2 on reference material labels; segment Gauss3 point traction; common-volume Gauss8 work-traction mass norm.',
                interpretation='Pairwise differences, not exact-continuum errors. Surface comparison includes initialization-field discretization. Time comparison holds initialization interval and label grid fixed.')
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    subprocess.run(['python3',str(HERE/'diagnose_bounded_energy.py'),*CASES,'--output',str(OUT/'coupled-work.json')],cwd=ROOT,check=True)
    print('Surface/time comparisons and pressure-aware work reduction complete.',flush=True)

if __name__=='__main__':main()
