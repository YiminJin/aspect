#!/usr/bin/env python3
"""Common-coordinate K2 differences: total fields and along-fault anomalies.

The reference is another full coupled solve with the same physical support.
Bulk fields are evaluated from their native cell polynomials, not by smoothing
raw stresses. Different initialized profiles/histories are reported separately.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analyze import h,read


def interpolate_q2(bulk, values, points):
    """Exact cellwise Q2 polynomial from the native 3x3 Gauss values."""
    xs,ys=np.unique(bulk['x']),np.unique(bulk['y'])
    nx,ny=len(xs)//3,len(ys)//3
    grid=np.empty((len(xs),len(ys)))
    grid[np.searchsorted(xs,bulk['x']),np.searchsorted(ys,bulk['y'])]=values
    scaled=np.column_stack((points[:,0]*nx/.25,(points[:,1]+.5)*ny))
    cell=np.floor(scaled).astype(int)
    cell[:,0]=np.clip(cell[:,0],0,nx-1); cell[:,1]=np.clip(cell[:,1],0,ny-1)
    unit=scaled-cell
    gauss=(np.polynomial.legendre.leggauss(3)[0]+1)/2
    basis=np.ones((len(points),2,3))
    for i in range(3):
        for j in range(3):
            if i!=j:
                basis[:,:,i]*=(unit-gauss[j])/(gauss[i]-gauss[j])
    result=np.zeros(len(points))
    for i in range(3):
        for j in range(3):
            result+=basis[:,0,i]*basis[:,1,j]*grid[3*cell[:,0]+i,3*cell[:,1]+j]
    return result


def bulk_at(bulk,surface,width,dt,points):
    result={name:interpolate_q2(bulk,bulk[name],points) for name in
            ('ux','uy','p','phi','ux_y','uy_x','old_tau_xy','history')}
    v=np.interp(points[:,0],surface['x'],surface['V'])
    ih=np.interp(points[:,0],surface['x'],surface['Ih'])
    chi=np.where(abs(points[:,1])<=width,h(result['phi'])/ih,0)
    result['q']=-1e8*math.expm1(-dt/100)*(result['ux_y']+result['uy_x']-chi*v-result['history'])
    result['q']+=math.exp(-dt/100)*result['old_tau_xy']
    return result


def norms(error,weights):
    return dict(L2=float(np.sqrt(np.average(error**2,weights=weights))),
                sample_max=float(max(abs(error))),mean=float(np.average(error,weights=weights)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('case',type=Path); p.add_argument('reference',type=Path)
    p.add_argument('case_measurements',type=Path); p.add_argument('reference_measurements',type=Path)
    args=p.parse_args()
    metadata=lambda directory: {float(read(directory,'time',int(f.stem.split('_')[-1]),())[0]['time']):int(f.stem.split('_')[-1])
                                 for f in directory.glob('time_*.csv')}
    ct,rt=metadata(args.case),metadata(args.reference)
    common=sorted(set(ct)&set(rt))
    report=dict(case=str(args.case),reference=str(args.reference),
                same_support_reference_only=True,steps=[])
    cwidth=read(args.case,'segments',0,())['half_width_plus'][0]
    rwidth=read(args.reference,'segments',0,())['half_width_plus'][0]
    if abs(cwidth-rwidth)>1e-12:
        raise ValueError('Support width changed')
    for time in common:
        ci,ri=ct[time],rt[time]
        cb,rb=read(args.case,'bulk',ci,()),read(args.reference,'bulk',ri,())
        cs,rs=read(args.case,'surface',ci,()),read(args.reference,'surface',ri,())
        cp=np.genfromtxt(args.case_measurements/f'surface_balance_{ci}.csv',delimiter=',',names=True)
        rp=np.genfromtxt(args.reference_measurements/f'surface_balance_{ri}.csv',delimiter=',',names=True)
        cd=float(read(args.case,'time',ci,())[0]['dt']); rd=float(read(args.reference,'time',ri,())[0]['dt'])
        # Exact squared-Q1 differences on the union of surface breakpoints.
        breaks=np.unique(np.r_[cs['x'],rs['x']]); gauss,gweights=np.polynomial.legendre.leggauss(3)
        sx=((breaks[:-1,None]+breaks[1:,None])/2+np.diff(breaks)[:,None]*gauss/2).ravel()
        sw=(np.diff(breaks)[:,None]*gweights/2).ravel()
        row=dict(time_s=time,surface={},bulk={})
        for name in ('V','Theta','C_retained','slip','particle_q_Q1'):
            c=np.interp(sx,cp['s'],cp[name]); r=np.interp(sx,rp['s'],rp[name])
            error=c-r; mean=np.average(error,weights=sw)
            row['surface'][name]=dict(total=norms(error,sw),anomaly=norms(error-mean,sw))
            vertex_error=np.interp(breaks,cp['s'],cp[name])-np.interp(breaks,rp['s'],rp[name])
            row['surface'][name]['total']['Linf']=float(max(abs(vertex_error)))
            row['surface'][name]['anomaly']['Linf']=float(max(abs(vertex_error-mean)))
        points=np.column_stack((rb['x'],rb['y'])); bw=rb['weight']
        c=bulk_at(cb,cs,cwidth,cd,points); r=bulk_at(rb,rs,rwidth,rd,points)
        by,index=np.unique(rb['y'],return_inverse=True)
        sums=np.bincount(index,weights=bw)
        for name in ('ux','uy','p','q','phi'):
            error=c[name]-r[name]
            mean_x=np.bincount(index,weights=bw*error)/sums
            row['bulk'][name]=dict(total=norms(error,bw),anomaly=norms(error-mean_x[index],bw))
        report['steps'].append(row)
    report['initialization']=report['steps'][0]
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
