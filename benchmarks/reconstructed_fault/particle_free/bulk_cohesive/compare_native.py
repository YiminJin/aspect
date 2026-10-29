#!/usr/bin/env python3
"""Common physical times/fields; spatial histories use actual weighted moments."""
import argparse,json
from pathlib import Path
import numpy as np
from assess_native import csv,norm3,connected_front


def times(path,field=False):
    f=csv(path/'fixture.csv');result={}
    for row in f:
        step=int(row['step'])
        if not field or (path/f'bulk-{step:04d}.csv').exists():result[round(float(row['time']),10)]=step
    return result


def grid(path,step):
    f=csv(path/f'phase-{step:04d}.csv');f.sort(order=['y','x'])
    x,y=np.unique(f['x']),np.unique(f['y'])
    return x,y,f['phi'].reshape(len(y),len(x))


def interpolate(g,x,y):
    px,py,p=g;i=np.clip(np.searchsorted(px,x,side='right')-1,0,len(px)-2);j=np.clip(np.searchsorted(py,y,side='right')-1,0,len(py)-2)
    u=(x-px[i])/(px[i+1]-px[i]);v=(y-py[j])/(py[j+1]-py[j])
    return (1-u)*(1-v)*p[j,i]+u*(1-v)*p[j,i+1]+(1-u)*v*p[j+1,i]+u*v*p[j+1,i+1]


def area(g):
    x,y,p=g
    return float(np.sum(np.diff(y)[:,None]*np.diff(x)[None,:]*(p[:-1,:-1]+p[1:,:-1]+p[:-1,1:]+p[1:,1:])/4))


def moments(f,nx,ny):
    cell=np.minimum((f['x']*nx).astype(int),nx-1)*ny+np.minimum((f['y']*ny).astype(int),ny-1)
    volume=np.bincount(cell,weights=f['JxW'],minlength=nx*ny)
    tensor=np.array([np.bincount(cell,weights=f['JxW']*f[k],minlength=nx*ny)/volume for k in ['tauxx','tauyy','tauxy']]).T
    return volume,tensor


def compare(first,reference,kind):
    a,b=times(first),times(reference);common=sorted(set(a)&set(b))
    initialA=grid(first,0);initialB=grid(reference,0)
    xx,yy=np.meshgrid(initialB[0],initialB[1])
    initial_difference=interpolate(initialA,xx,yy)-initialB[2]
    phase=growth=0.;phase_time=0.;stress=0.;stress_time=0.;fields=[]
    for t in common:
        ga,gb=grid(first,a[t]),grid(reference,b[t]);difference=interpolate(ga,xx,yy)-gb[2]
        error=float(np.max(abs(difference)))
        if error>phase:phase=error;phase_time=t
        growth=max(growth,float(np.max(abs(difference-initial_difference))))
    ca,cb=times(first,True),times(reference,True)
    for t in sorted(set(ca)&set(cb)):
        f=csv(first/f'bulk-{ca[t]:04d}.csv');r=csv(reference/f'bulk-{cb[t]:04d}.csv')
        if kind=='temporal':
            f.sort(order=['cell','x','y']);r.sort(order=['cell','x','y']);assert np.array_equal(f[['cell','x','y']],r[['cell','x','y']])
            A=np.stack([f[k] for k in ['tauxx','tauyy','tauxy']],axis=-1)
            B=np.stack([r[k] for k in ['tauxx','tauyy','tauxy']],axis=-1);weight=r['JxW']
        else:
            nx,ny=len(initialA[0])-1,len(initialA[1])-1
            weight,A=moments(f,nx,ny);_,B=moments(r,nx,ny)
        error=float(np.sqrt(np.sum(weight*norm3(A-B))/max(1e6*np.sum(weight),np.sum(weight*norm3(B)))))
        if error>stress:stress=error;stress_time=t
        fields.append(t)
    final=common[-1];gA=grid(first,a[final]);gB=grid(reference,b[final])
    areaA=area(gA)-area(initialA);areaB=area(gB)-area(initialB)
    area_error=abs(areaA-areaB)/abs(areaB) if areaB else float(areaA!=0.)
    result=dict(kind=kind,common_nodal_times=len(common),common_field_times=fields,
                initial_phase_Linf_difference=float(np.max(abs(initial_difference))),
                phase_Linf_difference=phase,phase_maximum_time=phase_time,
                evolution_phase_Linf_difference=growth,
                full_tensor_error=stress,stress_maximum_time=stress_time,
                tensor_normalization='max(reference tensor L2 norm, 1000 Pa times sqrt(volume)); full 3D plane strain',
                stress_measure='matched actual quadrature-point tensor L2' if kind=='temporal' else 'coarse-cell weighted full-tensor moment L2',
                first_growth_area=areaA,reference_growth_area=areaB,growth_area_relative_difference=area_error)
    if kind=='temporal':result['phase_screen_passed']=phase<=.01
    if kind=='spatial':
        result['stress_screen_passed']=stress<=.1;result['growth_area_screen_passed']=area_error<=.1
        startA=csv(first/'phase-0000.csv');startB=csv(reference/'phase-0000.csv')
        endA=csv(first/f'phase-{a[final]:04d}.csv');endB=csv(reference/f'phase-{b[final]:04d}.csv')
        result['initial_connected_contour_differences']={str(level):connected_front(startA,level)-connected_front(startB,level) for level in [.001,.01,.1,.3,.5]}
        result['final_connected_contour_differences']={str(level):connected_front(endA,level)-connected_front(endB,level) for level in [.001,.01,.1,.3,.5]}
        result['front_screen_passed']=bool(max(abs(v) for v in result['final_connected_contour_differences'].values())<=np.max(np.diff(initialA[0])))
    if kind=='quadrature':
        result['phase_screen_passed']=phase<=.001
        ledgerA=csv(first/'bulk-accepted.csv');ledgerB=csv(reference/'bulk-accepted.csv')
        error=0.
        for k in ['bulk','cohesive','fracture']:
            error=max(error,float(abs(ledgerA[k][-1]-ledgerB[k][-1])/max(1.,abs(ledgerB[k][-1]))))
        error=max(error,abs(sum(ledgerA['Wext'])-sum(ledgerB['Wext']))/max(1.,abs(sum(ledgerB['Wext']))))
        result['storage_work_error']=error;result['storage_work_screen_passed']=error<=.001
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('first',type=Path);p.add_argument('reference',type=Path)
    p.add_argument('--kind',choices=['temporal','spatial','quadrature'],required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=compare(a.first,a.reference,a.kind);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
