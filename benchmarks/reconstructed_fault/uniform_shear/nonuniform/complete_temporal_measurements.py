#!/usr/bin/env python3
"""Measure completed temporal runs, reusing verified fixed-profile integrals."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import time
import numpy as np
from measure_case import summarize
from compare_cases import norms

def read(path, fields=None):
    with path.open() as stream:
        names = stream.readline().strip().split(',')
        if fields is None: fields = names
        data = np.atleast_2d(np.loadtxt(stream,delimiter=',',usecols=[names.index(k) for k in fields]))
    assert np.isfinite(data).all(), path
    return np.rec.fromarrays(data.T,names=fields)

def digest(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('case',choices=('time025','time0125'))
    args=p.parse_args()
    start=time.monotonic()
    base=Path(__file__).resolve().parent
    directory=base/'domain-convergence-completion'/args.case
    previous=base/'domain-convergence'/args.case
    destination=directory.with_name(directory.name+'-measurements')
    resources=json.loads(directory.with_suffix('.resources.json').read_text())
    assert resources['exit_status']==0
    logged=summarize(directory.with_suffix('.log'))
    assert logged[-1]['time']==2.0
    for step in logged:
        last=step['nonlinear'][-1]
        assert last['bulk']<last['bulk target'] and last['surface']<1e-8*last['surface scale']
        assert all(r['fresh']<=r['target'] for r in step['linear'])
        assert all(r['bulk scale']==last['bulk scale'] and r['surface scale']==last['surface scale'] for r in step['nonlinear'])
    # Prefix verification establishes identical initialized histories and FE
    # profile; do not rerun the already verified independent profile integral.
    prefixes=json.loads((directory.parent/'prefix-checks.json').read_text())[args.case]
    last_saved=7 if args.case=='time025' else 8
    assert set(prefixes)==set(map(str,range(last_saved+1)))
    old_report=json.loads((base/'domain-convergence'/f'{args.case}-early-report.json').read_text())
    reference_profile=digest(previous/'phase_0.csv')
    reference_geometry=digest(previous/'segments_0.csv')
    ih=old_report['independent_Ih_m']
    report={k:v for k,v in old_report.items() if k not in ('steps','completed_to_2_seconds','allowances_pass')}
    report.update(profile_reference='verified identical saved FE profile; prior independent integration reused',steps=[])
    old_normalization={r['time_s']:r for r in json.loads((base/'domain-convergence/temporal-saved-bulk.json').read_text())['normalization'] if r['case']==args.case}
    destination.mkdir(exist_ok=True)
    initial=read(directory/'surface_0.csv')
    old_theta=initial['Theta'].copy()
    slip=np.zeros(len(initial))
    for k,iteration in enumerate(logged):
        assert digest(directory/f'phase_{k}.csv')==reference_profile
        assert digest(directory/f'segments_{k}.csv')==reference_geometry
        surface=read(directory/f'surface_{k}.csv')
        metadata=read(directory/f'time_{k}.csv')[0]
        assert metadata['time']==iteration['time']
        dt=float(metadata['dt']); t=float(metadata['time'])
        s=surface['x']
        sw=np.r_[np.diff(s)/2,0]+np.r_[0,np.diff(s)/2]
        weak=read(directory/f'surface_weak_{k}.csv')
        mass=np.diag(weak['Mdiag'])+np.diag(weak['Moff'][:-1],1)+np.diag(weak['Moff'][:-1],-1)
        rhs=np.column_stack([weak[name] for name in ('q','C','friction','damping','F')])
        assert max(abs(rhs[:,0]-rhs[:,1:4].sum(axis=1)-rhs[:,4]))<1e-10
        projected=np.linalg.solve(mass,rhs)
        strong=float(np.sqrt(max(0,rhs[:,-1]@projected[:,-1])/mass.sum()))
        assert abs(strong-iteration['nonlinear'][-1]['surface'])<1e-9
        if k:
            slip+=dt*surface['V']
            expected_theta=old_theta*np.exp(-surface['V']*dt/.001)+.001/surface['V']*(-np.expm1(-surface['V']*dt/.001))
        else: expected_theta=old_theta
        theta_error=float(max(abs(surface['Theta']-expected_theta)))
        assert theta_error<1e-10
        data=read(directory/f'bulk_{k}.csv',('x','y','weight','ux','uy','ux_y','uy_x','chi','V','history','old_tau_xy'))
        data.sort(order=['x','y'])
        xs,index=np.unique(data['x'],return_inverse=True)
        w=data['weight']
        if k<=1:
            norm_error=old_report['steps'][k]['normalization_max_error']
            global_ratio=old_report['steps'][k]['global_normalization']
        elif t in old_normalization:
            norm_error=old_normalization[t]['maximum_error']; global_ratio=old_normalization[t]['global_ratio']
        else:
            crack=data['chi']*data['V']+data['history']
            normal=np.bincount(index,weights=w*crack)/np.bincount(index,weights=w)
            norm_error=float(max(abs(normal/np.interp(xs,s,surface['V'])-1)))
            global_ratio=float(np.dot(w,crack)/np.dot(sw,surface['V']))
        assert norm_error<=1e-4 and abs(global_ratio-1)<=1e-4
        # Exact native Q2 boundary traces, vectorized over the fixed box columns.
        ny=len(data)//len(xs)
        ys=data['y'][:ny]
        ux=data['ux'].reshape(len(xs),ny); uy=data['uy'].reshape(len(xs),ny)
        boundary_error=0.
        for indices,boundary in ((np.arange(3),-.5),(np.arange(ny-3,ny),.5)):
            nodes=ys[indices]
            basis=np.array([np.prod([(boundary-nodes[j])/(nodes[i]-nodes[j]) for j in range(3) if j!=i]) for i in range(3)])
            boundary_error=max(boundary_error,float(max(abs(ux[:,indices]@basis-boundary*metadata['U']))),float(max(abs(uy[:,indices]@basis))))
        stress=-1e8*math.expm1(-dt/100)*(data['ux_y']+data['uy_x']-data['chi']*data['V']-data['history'])+math.exp(-dt/100)*data['old_tau_xy']
        _,yi=np.unique(data['y'],return_inverse=True)
        mean_x=np.bincount(yi,weights=w*stress)/np.bincount(yi,weights=w)
        row=dict(time_s=t,dt_s=dt,surface_rule='domain integrated',normalization_max_error=norm_error,
                 measured_normalization_locations=len(xs),
                 global_normalization=global_ratio,Ih_relative_error=float(max(abs(surface['Ih']/ih-1))),
                 theta_update_error_s=theta_error,boundary_error_m_s=boundary_error,strong_F_rms_Pa=strong,
                 production_F_rms_Pa=iteration['nonlinear'][-1]['surface'],
                 raw_bulk_stress=norms(stress,w),raw_bulk_stress_anomaly=norms(stress-mean_x[yi],w))
        report['steps'].append(row)
        np.savetxt(destination/f'surface_balance_{k}.csv',np.column_stack((s,projected,surface['V'],surface['Theta'],surface['C'],slip)),delimiter=',',
                   header='s,particle_q_Q1,C_evaluated_Q1,friction_Q1,radiation_Q1,F_Q1,V,Theta,C_retained,slip',comments='')
        old_theta=surface['Theta'].copy()
    report.update(completed_to_2_seconds=True,allowances_pass=old_report['omitted_fraction']<=1e-4,
                  analysis_wall_seconds=time.monotonic()-start,analysis_peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    assert report['allowances_pass']
    (destination/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
