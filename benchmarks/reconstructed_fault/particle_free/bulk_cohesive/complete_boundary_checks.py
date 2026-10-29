#!/usr/bin/env python3
"""Additional physical-region/hold checks, using saved states without evolution."""
from pathlib import Path
import json,numpy as np
from assess_spatial import q1
from assess_native import norm3,csv
OUT=Path('benchmarks/reconstructed_fault/particle_free/evidence/bulk_parallel')
OLD=Path('build-particle-free/bulk-continued/runs/continued')
NEW=Path('build-particle-free/bulk-parallel/runs/mixed')

def main():
    d=dict(np.load(OUT/'movie-data.npz'));g=dict(np.load(OUT/'remote-fields.npz'))
    points=g['points'];weights=g['weights'];radius=.30882159
    base={k:d[k] for k in ['x','y']};base.update(dx=base['x'][1]-base['x'][0],dy=base['y'][1]-base['y'][0])
    masks=dict(original_cap=(points[...,0]>.5)&(np.linalg.norm(points-.5,axis=-1)<radius),
               cap_outer_ring=(points[...,0]>.5)&(np.linalg.norm(points-.5,axis=-1)>=radius)&(np.linalg.norm(points-.5,axis=-1)<.35),
               remote=points[...,0]>.9,whole=np.ones(weights.shape,bool))
    result={}
    for name,grid,phases in [('affine',d['old_grid_ids'],d['old_phase']),('released',d['grid_ids'],d['phase'])]:
        L={**base,'grid_ids':grid};dp=q1(L,phases[-1]-phases[0],points)[0]
        X,Y=np.meshgrid(d['x'],d['y']);initial=phases[0][grid];change=(phases[-1]-phases[0])[grid]
        node_original=(X>.5)&(np.hypot(X-.5,Y-.5)<radius)
        result[name]=dict(final_integrals={k:float(np.sum(weights[m]*dp[m])) for k,m in masks.items()},original_cap_peak=float(change[node_original].max()),hold_phase_change=float(abs(phases[140:]-phases[140]).max()))
    delta=np.diff(d['phase'],axis=0);peak=np.max(delta,axis=1);step=int(peak.argmax()+1)
    result['largest_step']=dict(step=step,time=float(d['time'][step]),peak=float(peak[step-1]),interval=[float(d['time'][step-1]),float(d['time'][step])])
    result['flagged_steps_at_001']=list(map(int,np.flatnonzero(peak>=.001)+1))
    ledger=csv(NEW/'bulk-accepted.csv');beta=float(np.prod(1/(1+ledger['dt'][140:]*.01)))
    col=[3,8,9,10,11,12,13,14] # JxW,tauxx,tauyy,tauxy,a,q,k,L
    a=np.loadtxt(NEW/'bulk-0140.csv',delimiter=',',skiprows=1,usecols=col)
    b=np.loadtxt(NEW/'bulk-0148.csv',delimiter=',',skiprows=1,usecols=col)
    diff=b[:,1:4]-beta*a[:,1:4]
    result['hold']=dict(beta=beta,stress_relative_L2=float(np.sqrt(np.sum(a[:,0]*norm3(diff))/np.sum(a[:,0]*norm3(beta*a[:,1:4])))),stress_max_absolute_Pa=float(abs(diff).max()),signed_shear_change=float(abs(a[:,6]-b[:,6]).max()),absolute_shear_change=float(abs(a[:,7]-b[:,7]).max()))
    assert result['released']['hold_phase_change']==0 and result['hold']['signed_shear_change']==0 and result['hold']['absolute_shear_change']==0
    # Independently reconstruct the local rate from consecutive accepted a
    # histories, then compare the original-QP mean/max and accumulated records.
    previous=np.loadtxt(NEW/'bulk-0120.csv',delimiter=',',skiprows=1,usecols=(3,11,13,14))
    current=np.loadtxt(NEW/'bulk-0121.csv',delimiter=',',skiprows=1,usecols=(3,11,13,14))
    dt=.00125;rate=((1+dt*.01)*current[:,1]-previous[:,1])/dt
    native=np.loadtxt(NEW/'means-0121.csv',delimiter=',',skiprows=1,usecols=(6,7))
    mass=previous[:,0].reshape(-1,64)
    average=np.sum(abs(rate.reshape(-1,64))*mass,axis=1)/mass.sum(axis=1)
    maximum=abs(rate.reshape(-1,64)).max(axis=1)
    result['rate_history']=dict(weighted_mean=float(max(abs(average-native[:,0]))),maximum_point=float(max(abs(maximum-native[:,1]))),signed_increment=float(max(abs(current[:,2]-previous[:,2]-dt*rate))),absolute_increment=float(max(abs(current[:,3]-previous[:,3]-dt*abs(rate)))))
    assert max(result['rate_history'].values())<1e-12
    np.savez_compressed(OUT/'region-checks.npz',points=points,weights=weights)
    (OUT/'boundary-extra.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
