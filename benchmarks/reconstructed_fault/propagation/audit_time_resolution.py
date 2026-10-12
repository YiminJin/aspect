#!/usr/bin/env python3
"""Frozen diagnostics and a separately labeled exact-bulk spatial comparator."""
import argparse,json,time
import numpy as np
from time_resolution import E,directory,load,Reference,G,ETA,h,A
from assess_time_resolution import trajectory,difference

def exact():
    out=E/'exact-bulk-dt-1-1024';out.mkdir(exist_ok=False);start=time.perf_counter()
    r=Reference(256,1.,6,True);reference=np.load(E/'baseline-diagnostics/initial.npz')
    assert np.array_equal(r.initial,reference['phi']) and np.array_equal(r.initial_particle_H,reference['H'])
    hist,fields,stationary,terms,summary=r.run(1/1024,.625,out)
    np.savez_compressed(out/'fields.npz',y=r.y,phase=fields,initial=r.initial,final_Rq=r.rq,final_Z=r.Z)
    (out/'history.json').write_text(json.dumps(hist,indent=2)+'\n')
    summary.update(runtime_seconds=time.perf_counter()-start,purpose='Separate spatial diagnostic; never the temporal reference for the fixed FE model')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

def audit():
    r=Reference(256,1.,6);high=Reference(256,1.,24);higher=Reference(256,1.,48);report={}
    for d in [64,128,256,512,1024]:
        a=trajectory(d);dt=1/d;beta=np.exp(-dt/100);kappa=-ETA*np.expm1(-dt/100)
        # Integral of (beta(t)-beta(dt))/(1-beta(dt)); no small-dt assumption.
        work_factor=100-beta*dt/(-np.expm1(-dt/100))
        predictions=[];old_force=0.;incoming_force=0.;recurrence=0.;replay_error=0.;replay_Z=0.;normalization=0.;checks=[]
        for j in range(1,a['steps']+1):
            old=a['phi'][j-1];ph=a['phi'][j];R=a['r'][j-1];Z=a['Z'][j-1]
            I=r.direct(ph)[0];Io=r.direct(old)[0];oldtau=G*r.D@R-Z*h(r.B@old)/Io;jump=G*r.D@R-Z*h(r.B@ph)/I
            force=r.D.T@(r.w*jump);old_force=max(old_force,float(max(abs(r.D[:,1:-1].T@(r.w*oldtau)))))
            incoming_force=max(incoming_force,float(max(abs(force[1:-1]))))
            u=(a['r'][j]-beta*R)*G/kappa;lift=.256*((j-.5)/d)*(np.linspace(0,1,513)-.5)
            prediction=work_factor*np.dot(u-lift,force);predictions.append(float(prediction))
            independent=high.w@h(high.B@ph);normalization=max(normalization,float(abs(independent/I-1)))
            if j%(d//64)==0:
                # Prototype Maxwell function replay in independent scratch
                # state, using retained inputs; no phase/history is advanced.
                r.r=R.copy();r.Z=Z;r.slip=a['slip'][j-1];r.phi=ph.copy();v=r.mechanical(j/d,dt)
                expected=beta*R+kappa/G*r.velocity
                recurrence=max(recurrence,float(max(abs(expected-r.r))))
                replay_error=max(replay_error,float(max(abs(r.r-a['r'][j]))));replay_Z=max(replay_Z,float(abs(r.Z-a['Z'][j])))
            if j in [d//2,round(.546875*d),round(.625*d)]:
                for obj in [r,high,higher]:obj.r=R.copy();obj.Z=Z
                val,g,t=r.objective(ph);vv,gg,tt=high.objective(ph);hhh=higher.objective(ph)[2]
                H=r.phase_hessian(ph);HH=high.phase_hessian(ph);free=np.flatnonzero((ph-old>4.922e-12)&(ph<1))
                eig=np.linalg.eigvalsh(H[np.ix_(free,free)]) if len(free) else []
                checks.append(dict(time=j/d,incoming_time=(j-1)/d,bulk_energy_6_24=float(abs(t[0]-tt[0])),bulk_energy_24_48=float(abs(tt[0]-hhh[0])),gradient_error_6_24=float(max(abs(g-gg))),Hessian_relative_error_6_24=float(np.max(abs(H-HH))/max(1.,np.max(abs(HH)))),free_nodes=len(free),minimum_free_curvature=float(min(eig)) if len(eig) else None))
        actual=[v['bulk_work']-v['boundary_work'] for v in a['energy']]
        report[str(d)]=dict(work_difference_prediction_signed=sum(predictions),work_difference_measured_signed=sum(actual),maximum_prediction_error=float(max(abs(np.array(predictions)-actual))),old_weak_force_Pa=old_force,phase_redistributed_weak_force_Pa=incoming_force,coefficient_recurrence_error=recurrence,mechanical_replay_R_error=replay_error,mechanical_replay_Z_error=replay_Z,direct_normalization_error=normalization,selected_incoming_checks=checks)
    ef=np.load(E/'exact-bulk-dt-1-1024/fields.npz');eh=load(E/'exact-bulk-dt-1-1024/history.json');a=trajectory(1024)
    spatial=dict(purpose='Matched finest timestep, separate exact-bulk spatial comparator',phi=float(np.max(abs(a['phi'][1:]-ef['phase']))),q_Pa=float(max(abs(a['q'][1:]-[v['q'] for v in eh]))),slip=float(max(abs(a['slip'][1:]-[v['slip'] for v in eh]))),bulk_storage=float(max(abs(a['bulk'][1:]-[v['bulk'] for v in eh]))))
    report['separate_spatial_comparison']=spatial
    (E/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--exact',action='store_true');a=p.parse_args()
    exact() if a.exact else audit()
