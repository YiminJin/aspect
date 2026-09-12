"""K4.2 independent-continuum comparison; retain transient and raw errors."""
import argparse
import json
import math
from pathlib import Path
import re

import numpy as np

from production import HERE, read
from reference_check import Y, SCALES, dump


def q2_profile(yq, values, samples):
    """Evaluate the represented cellwise Q2 profile, not a smoothed fit."""
    yy, vv = yq.reshape(-1,3), values.reshape(-1,3)
    n = len(yy)
    cell = np.clip(((samples+.5)*n).astype(int),0,n-1)
    y, v = yy[cell], vv[cell]
    result = np.zeros(len(samples))
    for i in range(3):
        basis = np.ones(len(samples))
        for j in range(3):
            if j!=i: basis *= (samples-y[:,j])/(y[:,i]-y[:,j])
        result += v[:,i]*basis
    return result


def residual_checks(log):
    fresh=[]; final=[]; last=None
    for line in log.read_text().splitlines():
        if 'Fault linear solve:' in line:
            data = dict((k,float(v)) for k,v in re.findall(r'([a-z ]+)=([+\-\deE.]+)',line))
            data = {k.strip():v for k,v in data.items()}
            fresh.append(data['fresh']/data['target'])
        if 'Fault nonlinear residual:' in line:
            data = dict((k.strip(),float(v)) for k,v in re.findall(r'([a-z ]+)=([+\-\deE.]+)',line))
            last=dict(bulk=data['bulk'],bulk_target=data['bulk target'],
                      surface=data['surface'],surface_scale=data['surface scale'])
        if 'K4 accepted-state guard:' in line:
            assert last is not None
            final.append(last)
    return dict(linear_checks=len(fresh),max_fresh_over_requested=max(fresh,default=0),
                final=final,passed=bool(fresh and final and max(fresh)<=1 and all(
                    r['bulk']<=r['bulk_target'] and r['surface']/r['surface_scale']<1e-8 for r in final)))


def analyze(case):
    path=HERE/case
    name='ell0' if case=='k42_A' else 'half'
    reference=HERE/'k41'/f'{name}-4096'
    r=json.loads((reference/'dt0.125.json').read_text())
    ru=np.load(reference/'dt0.125.npz')['velocity']
    finer=json.loads((reference/'dt0.0625.json').read_text())
    finer_u=np.load(reference/'dt0.0625.npz')['velocity']
    initial=json.loads((reference/'initialization.json').read_text())
    refphi=np.load(reference/'profile.npz')['phi']
    s0=read(path,'surface',0)
    lengths=np.hypot(np.diff(s0['x']),np.diff(s0['y']))
    weights=.5*(np.r_[lengths,0]+np.r_[0,lengths])
    length=sum(lengths)
    rows=[]; profiles=[]; slip=0.
    for step in range(49):
        gate=path/f'k4_guard_{step}.json'
        if not gate.exists(): break
        guard=json.loads(gate.read_text())
        bulk=read(path,'bulk',step)
        history=read(path,'history_transfer',step)
        s=read(path,'surface',step)
        mean=lambda k:float(weights@s[k]/length)
        time=read(path,'time',step)[0]
        assert abs(time['time']-r[step]['time_s'])<1e-12
        if step: slip+=time['dt']*mean('V')
        # Use the constrained FE history actually consumed by mechanics, not
        # the distinct visualization/publication history or new particle stress.
        q=bulk['kappa']*(bulk['ux_y']+bulk['uy_x']-bulk['chi']*bulk['V']-bulk['history'])
        q+=math.exp(-time['dt']/100)*history['assembly_xy']
        rms=lambda v:float(np.sqrt(np.average(v*v,weights=bulk['weight'])))
        y,index=np.unique(bulk['y'],return_inverse=True)
        wy=np.bincount(index,weights=bulk['weight'])
        mean_u=np.bincount(index,weights=bulk['weight']*bulk['ux'])/wy
        u=q2_profile(y,mean_u,Y)
        reference_u=np.interp(bulk['y'],Y,ru[step])
        ue=bulk['ux']-reference_u
        profiles.append(u)
        observed=dict(V=mean('V'),q=guard['actual_weak_q_mean'],C=mean('C'),Theta=mean('Theta'),slip=slip,
                      Ih=mean('Ih'),supported_integral=float(np.dot(bulk['weight'],bulk['chi']*bulk['V']+bulk['history'])/length))
        errors={k:observed[k]-r[step][k] for k in ('V','q','C','Theta','slip','Ih')}
        # At zero C is retained while the reference row contains evaluated C.
        # Preserve this distinction, including in the recorded initial error.
        if step==0: errors['C']=observed['C']-initial['retained_initial']['C']
        field_errors={k:float(max(abs(s[k]-r[step][k]))) for k in ('V','C','Theta')}
        checks={k:abs(errors[k])<=.002*abs(r[step][k])+1e-5*SCALES[k]
                for k in ('V','q','C','Theta','slip')}
        for k in ('V','C','Theta'):
            checks[k]=field_errors[k]<=.002*abs(r[step][k])+1e-5*SCALES[k]
        checks['raw_stress']=max(abs(q-r[step]['q']))<=.002*abs(r[step]['q'])+.015
        checks['velocity_max']=bool(np.all(abs(ue)<=.002*abs(reference_u)+1e-9))
        checks['velocity_rms']=rms(ue)<=.002*rms(reference_u)+1e-9
        rf=finer[2*step]
        uf=np.interp(bulk['y'],Y,finer_u[2*step])
        fine_checks={k:abs(observed[k]-rf[k])<=.002*abs(rf[k])+1e-5*SCALES[k]
                     for k in ('V','q','C','Theta','slip')}
        for k in ('V','C','Theta'):
            fine_checks[k]=max(abs(s[k]-rf[k]))<=.002*abs(rf[k])+1e-5*SCALES[k]
        fine_checks['raw_stress']=max(abs(q-rf['q']))<=.002*abs(rf['q'])+.015
        fine_checks['velocity_max']=np.all(abs(bulk['ux']-uf)<=.002*abs(uf)+1e-9)
        fine_checks['velocity_rms']=rms(bulk['ux']-uf)<=.002*rms(uf)+1e-9
        rows.append(dict(step=step,time_s=float(time['time']),observed=observed,errors=errors,
                         max_surface_errors=field_errors,
                         raw_stress_error_max=float(max(abs(q-r[step]['q']))),
                         raw_stress_error_rms=rms(q-r[step]['q']),
                         raw_stress_range=[float(min(q)),float(max(q))],
                         velocity_error_max=float(max(abs(ue))),velocity_error_rms=rms(ue),
                         velocity_profile_error_max=float(max(abs(u-ru[step]))),
                         checks={k:bool(v) for k,v in checks.items()},
                         fine_reference_checks={k:bool(v) for k,v in fine_checks.items()},guards=guard))
    phase=read(path,'phase',0)
    particles0=read(path,'particles',0)
    phi_error=phase['phi']-np.interp(phase['y'],Y,refphi)
    ys=np.unique(phase['y'])
    center=phase[phase['x']==np.unique(phase['x'])[len(np.unique(phase['x']))//2]]
    py,pi=np.unique(center['y'],return_index=True)
    pv=center['phi'][pi]
    gauss,gw=np.polynomial.legendre.leggauss(8)
    qy=py[:-1,None]+np.diff(py)[:,None]*(gauss+1)/2
    pphi=np.maximum(np.interp(qy,py,pv),0.)
    hh=initial['m']*pphi*(1+pphi)/(1-pphi)**2
    mw=np.diff(py)[:,None]*gw/2*hh
    moment=float(np.sum(mw*qy*qy)/np.sum(mw))
    result=dict(case=case,reference=str(reference),reference_histories_reset=False,
                completed=len(rows)==49 and all(a['guards']['passed'] for a in rows),
                original_K41_all_time_pass=False,accuracy_interval=[4,6],
                initial=dict(phi_error_max=float(max(abs(phi_error))),
                             H0_range=[float(min(particles0['H'])),float(max(particles0['H']))],
                             reference_H0_max=initial['H0_max'],
                             phi_range=rows[0]['guards']['phi_range'],
                             normal_cells=len(ys)-1,center_phi=float(np.interp(0,py,pv)),
                             localization_second_moment_m2=moment,
                             reference_second_moment_m2=initial['localization_second_moment_m2'],
                             observed=rows[0]['observed'],errors=rows[0]['errors'],
                             independent=initial['retained_initial']),
                residuals=residual_checks(path.with_suffix('.log')),rows=rows,
                post_transient_checks={k:len(rows)==49 and all(row['checks'][k] for row in rows if row['time_s']>=4)
                                       for k in rows[0]['checks']},
                post_transient_fine_reference_checks={k:len(rows)==49 and all(row['fine_reference_checks'][k] for row in rows if row['time_s']>=4)
                                                      for k in rows[0]['checks']})
    dump(HERE/f'{case}-analysis.json',result)
    np.savez_compressed(HERE/f'{case}-profiles.npz',velocity=profiles,y=Y,phase_y=py,phase=pv)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',choices=['k42_A','k42_B','k42_C'])
    args=parser.parse_args()
    analyze(args.case)
