#!/usr/bin/env python3
"""Fixed Q2 spatial model, existing endpoint step, and audited work measures."""
import argparse,json,time,resource
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
from localization_1d import Reference
from total_energy_step import G,ETA,FRACTURE_SCALE as A,MU,PRESSURE,h

HERE=Path(__file__).resolve().parent
E=HERE/'time_resolution_evidence'
BASE=HERE/'normal_resolution_evidence/one-d-n256-d2-q6'
FINAL=.625

def load(p):return json.loads(p.read_text())
def directory(d):return BASE if d==64 else E/f'dt-1-{d}'

def energy(r,oldphi,phi,anchor,Z,newr,newZ,dt,V,time):
    """Frozen-state diagnostic; exactly the existing spatial/time work rules.

    General exponential Maxwell integration at 12 time Gauss points; bulk
    and boundary-lifting work are distinct. No history is accumulated here.
    """
    oldI,_=r.direct(oldphi);I,_=r.direct(phi)
    oldtau=G*r.D@anchor-Z*h(r.B@oldphi)/oldI
    tau0=G*r.D@anchor-Z*h(r.B@phi)/I
    beta=np.exp(-G*dt/ETA);kappa=-ETA*np.expm1(-G*dt/ETA)
    du=r.D@((newr-beta*anchor)*G/kappa)
    forcing=du-h(r.B@phi)/I*V
    z,w=leggauss(12);t=(z+1)*dt/2;wt=w*dt/2
    b=np.exp(-G*t/ETA);k=-ETA*np.expm1(-G*t/ETA)
    tau=b[:,None]*tau0+k[:,None]*forcing
    zz=b*Z+k*V
    end=G*r.D@newr-newZ*h(r.B@phi)/I
    oldbulk=r.w@(oldtau*oldtau)/(2*G);jumpbulk=r.w@(tau0*tau0)/(2*G)
    bulk=r.w@(end*end)/(2*G);oldcoh=Z*Z/(2*G*oldI);jumpcoh=Z*Z/(2*G*I);coh=newZ*newZ/(2*G*I)
    fracture0=r.fracture(oldphi);fracture=r.fracture(phi)
    release=oldbulk+oldcoh+fracture0-jumpbulk-jumpcoh-fracture
    Db=wt@((tau*tau)@r.w)/ETA;Dc=wt@(zz*zz)/(ETA*I);Df=dt*MU*PRESSURE*V
    Wbulk=wt@(tau@(r.w*du));Wboundary=.256*(time-dt/2)*(wt@(tau@r.w));Wcoh=wt@(zz*V/I)
    expenditure=bulk+coh+fracture-oldbulk-oldcoh-fracture0+release+Db+Dc+Df
    return {k:float(v) for k,v in dict(bulk=bulk,cohesive=coh,fracture=fracture,bulk_change=bulk-oldbulk,cohesive_change=coh-oldcoh,fracture_change=fracture-fracture0,D_relax=release,bulk_viscous=Db,cohesive_viscous=Dc,friction=Df,bulk_work=Wbulk,boundary_work=Wboundary,cohesive_work=Wcoh,bulk_profile_change=jumpbulk-oldbulk,cohesive_profile_change=jumpcoh-oldcoh,bulk_work_error=Wbulk-expenditure,boundary_work_error=Wboundary-expenditure).items()}

def summarize(rows):
    terms=['bulk_change','cohesive_change','fracture_change','D_relax','bulk_viscous','cohesive_viscous','friction','bulk_work','boundary_work','cohesive_work','bulk_profile_change','cohesive_profile_change','bulk_work_error','boundary_work_error']
    return {k:dict(signed=float(sum(v[k] for v in rows)),absolute=float(sum(abs(v[k]) for v in rows)),maximum=float(max(abs(v[k]) for v in rows))) for k in terms}

def initial_snapshot(r,out):
    np.savez_compressed(out/'initial.npz',phi=r.initial,r=r.r,Z=r.Z,slip=r.slip,particle_y=r.initial_particle_y,H=r.initial_particle_H,y=r.y)

def baseline_energy():
    """Reuse accepted fields; do not replay/advance the baseline history."""
    out=E/'baseline-diagnostics';out.mkdir(exist_ok=False)
    r=Reference(256,1.,6);f=np.load(BASE/'fields.npz');hist=load(BASE/'history.json')
    assert np.array_equal(r.initial,f['initial']) and np.count_nonzero(f['incoming_r'][0])==0 and f['incoming_Z'][0]==0
    initial_snapshot(r,out);rows=[];old=f['initial']
    for j,v in enumerate(hist):
        newr=f['incoming_r'][j+1] if j+1<len(hist) else f['final_r'];newZ=f['incoming_Z'][j+1] if j+1<len(hist) else f['final_Z']
        a=energy(r,old,f['phase'][j],f['incoming_r'][j],f['incoming_Z'][j],newr,newZ,1/64,v['V'],v['time'])
        a.update(time=v['time'],increment=v['increment']);rows.append(a);old=f['phase'][j]
        assert abs(a['bulk']-v['bulk'])<1e-10 and abs(a['cohesive']-v['cohesive'])<1e-10
        assert abs(a['bulk_work_error']-v['mechanical_error'])<1e-10
    (out/'energy.json').write_text(json.dumps(rows,indent=2)+'\n')
    (out/'totals.json').write_text(json.dumps(summarize(rows),indent=2)+'\n')

def case(denominator,frozen=False):
    out=E/(f'frozen-dt-1-{denominator}' if frozen else f'dt-1-{denominator}');out.mkdir(exist_ok=False)
    start=time.perf_counter();cpu=time.process_time();dt=1/denominator;steps=round(FINAL/dt)
    r=Reference(256,1.,6);initial_snapshot(r,out);ref=np.load(E/'baseline-diagnostics/initial.npz')
    assert all(np.array_equal(a,ref[k]) for k,a in [('phi',r.initial),('r',r.r),('Z',r.Z),('slip',r.slip),('particle_y',r.initial_particle_y),('H',r.initial_particle_H)])
    fields=[];anchors=[];zs=[];rows=[];history=[];release=0.;cumulative_displacement=0.;onsets={}
    try:
        for step in range(1,steps+1):
            t=step*dt;old=r.phi.copy();old.flags.writeable=False;anchor=r.r.copy();Z=r.Z
            if frozen:kkt=0.
            else:r.phi,kkt=r.phase(old)
            assert np.array_equal(r.r,anchor) and r.Z==Z,'Phase trial advanced accepted mechanical history'
            assert min(r.phi-old)>=-4.922e-12 and min(r.phi)>=0 and max(r.phi)<1
            data=r.mechanical(t,dt);a=energy(r,old,r.phi,anchor,Z,r.r,r.Z,dt,data['V'],t)
            assert abs(a['bulk_work_error']-data['mechanical_error'])<1e-10
            assert abs(data['loss']-a['bulk_viscous']-a['cohesive_viscous']-a['friction'])<1e-10
            assert data['force']<1e-7 and abs(data['stick_slip'])<1e-7
            release+=a['D_relax'];increment=float(max(r.phi-old));a.update(time=t,increment=increment);rows.append(a)
            history.append(dict(time=t,phi_max=float(max(r.phi)),increment=increment,healing=float(max(old-r.phi)),KKT=float(kkt),I=float(r.direct(r.phi)[0]),fracture=a['fracture'],release=release,**{k:float(v) for k,v in data.items()}))
            fields.append(r.phi.copy());anchors.append(anchor);zs.append(Z)
            cumulative_displacement+=dt*.256*(t-dt/2)
            for key,value in [('full_field',max(r.phi-r.initial)),('core',r.phi[128]-r.initial[128])]:
                if key not in onsets and value>1e-5:onsets[key]=[t-dt,t]
    except Exception as exc:
        np.savez_compressed(out/'failed.npz',phi=r.phi,lower=old,r=anchor,Z=Z)
        (out/'failure.json').write_text(json.dumps(dict(step=step,time=t,error=repr(exc)),indent=2)+'\n');raise
    assert abs(cumulative_displacement-.05)<1e-14
    summary=dict(dt=dt,steps=steps,final=FINAL,cells=256,degree=2,normal_points=6,initialization_identical=True,loading_displacement=cumulative_displacement,onset_threshold=1e-5,onset_brackets=onsets,maximum_phase_increment=max(v['increment'] for v in history),runtime_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,peak_memory_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,frozen_phase=frozen,energy_totals=summarize(rows))
    np.savez_compressed(out/'fields.npz',y=r.y,phase=np.array(fields),initial=r.initial,incoming_r=np.array(anchors),incoming_Z=np.array(zs),final_r=r.r,final_Z=r.Z)
    for name,data in [('history',history),('energy',rows),('summary',summary)]: (out/(name+'.json')).write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--denominator',type=int,choices=[64,128,256,512,1024]);p.add_argument('--baseline',action='store_true');p.add_argument('--frozen',action='store_true');a=p.parse_args()
    if a.baseline:baseline_energy()
    else:
        assert a.denominator is not None and (a.denominator!=64 or a.frozen);case(a.denominator,a.frozen)
