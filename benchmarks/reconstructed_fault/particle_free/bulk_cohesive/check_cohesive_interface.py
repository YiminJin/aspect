#!/usr/bin/env python3
"""Intact onset, mature identity, pressure tangent and history/energy checks."""
import json,copy,time
import numpy as np
from scipy.optimize import brentq
from cohesive_interface import *
from structural_rsf import solve_speed,tensor,inner,plane
from structural_sliding import SlidingBand


def march(dt,mature=False):
    model=UniformBlock();old=initial(damage=1. if mature else 0.);rows=[];cum={k:0. for k in ['endpoint_work','trapezoid_work','fracture_increment','Maxwell','friction','radiation','bulk_BE','surface_remainder']}
    old_energy=0.;absolute=0.;onset=None;maxforce=0.;maxledger=0.;ref_s=0.;ref_theta=1.;ref_j=0.;identity=0.
    end=6. if mature else 1.1
    for step in range(round(end/dt)):
        t=(step+1)*dt;mid=(step+.5)*dt
        speed=(.001 if mid<2 else .002 if mid<4 else 0. if mid<4.5 else -.001) if mature else .001
        du=np.array([speed*dt]);trial=solve(model,old,dt,du);energy=account(model,old,trial,dt,du)
        if not mature and onset is None and trial['damage'][0]>0:onset=(step*dt,t)
        for key in cum:cum[key]+=energy[key]
        physical=energy['Maxwell']+energy['friction']+energy['radiation']
        change=energy['bulk']+energy['cohesive']-old_energy+energy['fracture_increment']
        absolute+=abs(energy['trapezoid_work']-change-physical);old_energy=energy['bulk']+energy['cohesive']
        maxforce=max(maxforce,trial['force_error_Pa']);maxledger=max(maxledger,abs(energy['ledger_defect']))
        if mature:
            beta=1/(1+RELAX*dt);f=beta*(ref_s+model.K[0,0]*du[0])
            V,mu,sl,age,err=solve_speed(abs(f),ETA_RAD+beta*dt*model.K[0,0],P0,ref_theta,dt)
            v=np.sign(f)*float(V);ref_s=beta*(ref_s+model.K[0,0]*(du[0]-dt*v));ref_j+=dt*v;ref_theta=float(age)
            identity=max(identity,abs(ref_s-trial['stress'][0]))
            assert abs(ref_j-trial['jump'][0])<1e-10
        old=commit(old,trial)
        rows.append(dict(mode='mature' if mature else 'formation',dt_s=dt,time_s=t,damage=old['damage'][0],jump_m=old['jump'][0],
            shear_Pa=old['stress'][0],pn_Pa=old['pn'][0],Theta=old['Theta'][0],V_m_s=old['V'][0],bulk_J_m2=energy['bulk'],cohesive_J_m2=energy['cohesive'],
            **{k+'_J_m2':v for k,v in cum.items()},absolute_work_remainder_J_m2=absolute))
    D=cum['friction']+cum['radiation'];balance=cum['trapezoid_work']-old_energy-cum['fracture_increment']-cum['Maxwell']-D
    return rows,dict(mode='mature' if mature else 'formation',dt_s=dt,onset=onset,final_damage=float(old['damage'][0]),max_force_Pa=maxforce,
        max_ledger_J_m2=maxledger,reference_stress_error_Pa=identity,absolute_slip_Dc=float(old['absolute_slip'][0]/DC),
        physical_balance_J_m2=balance,absolute_remainders_J_m2=absolute,absolute_remainders_over_scale=absolute/(D if mature else GC),
        final_fracture_J_m2=cum['fracture_increment'],final_cohesive_J_m2=energy['cohesive']),old


def serialize(state):
    return json.dumps({k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in state.items()},sort_keys=True)
def restore(s):return {k:np.array(v) if isinstance(v,list) else v for k,v in json.loads(s).items()}


def mapping(peak):
    b=SlidingBand(128,peak);S=plane([0,1,0])[2];e=.0002;delta=.0012
    A=b.chi[:,None,None]*(e*S+tensor(xx=2e-5,mean=1e-5));K=b.chi[:,None,None]*(delta*S+tensor(xx=3e-5,mean=-1e-5))
    physical=tensor(xx=200,xy=1200,mean=100);T=np.tile(physical,(len(b.y),1,1))+2*G*A
    q=b.coherent_C*e;a=b.I;c=C0*a/(1-a);enew=c*q
    perpendicular=A-b.chi[:,None,None]*e*S
    residual_energy=G*np.einsum('n,nij,nij->',b.mass*b.ih,perpendicular,perpendicular)
    old_energy=G*np.einsum('n,nij,nij->',b.mass*b.ih,A,A)
    new_energy=.5*q*enew+residual_energy
    mapped_tau=T-2*G*A # Full tensor primary stress, no projection.
    return dict(peak_phi=peak,damage=a,old_cohesive_J_m2=old_energy,new_cohesive_and_residual_J_m2=new_energy,
        switch_work_J_m2=new_energy-old_energy,stress_error_Pa=float(np.max(abs(mapped_tau-physical))),reference_offset_m=delta-enew,
        signed_jump_m=delta,q_Pa=q,residual_tensor_energy_J_m2=residual_energy,
        original_T=T.tolist(),original_A=A.tolist(),original_K=K.tolist(),retained_residual_A=perpendicular.tolist(),retained_residual_K=(K-b.chi[:,None,None]*delta*S).tolist(),new_physical_stress=mapped_tau.tolist(),old_frictional_age=.4,new_frictional_age=.4)


def main():
    start=time.perf_counter();OUT.mkdir(exist_ok=True,parents=True);rows=[];summaries=[]
    for mature in [False,True]:
        for dt in ([.002,.001,.0005] if not mature else [.002,.001]):
            r,m,state=march(dt,mature);stride=round(.01/dt);rows+=r[stride-1::stride];summaries.append(m);print(json.dumps(m),flush=True)
    exact=-np.log(1-RELAX*TAUP/(G/.125*.001))/RELAX
    for m in summaries:
        if m['mode']=='formation':
            # BE onset is first order: exact threshold lies within one step of its bracket.
            assert abs(np.mean(m['onset'])-exact)<=2*m['dt_s']
            assert m['final_damage']==1 and abs(m['final_fracture_J_m2']-GC)<1e-12
        assert m['absolute_remainders_over_scale']<=.01
    # Local force and constitutive derivatives are checked on a current-pressure layer system.
    model=Layers(16,length=1.,top=.075,bottom=.05);old=initial(16,damage=.6)
    old['jump']=.0001*np.cos(2*np.pi*model.x);old['reference']=old['jump'].copy()
    old['memory']=.0001+.00002*np.sin(2*np.pi*model.x)
    trial=solve(model,old,.001,np.full(16,1e-6));saved=serialize(old)
    again=solve(model,old,.001,np.full(16,1e-6));assert serialize(again)==serialize(trial) and serialize(old)==saved
    accepted=commit(old,trial)
    try:commit(accepted,trial)
    except ValueError:pass
    else:raise AssertionError('duplicate commit')
    (OUT/'reference_checkpoint.json').write_text(serialize(accepted)+'\n')
    resumed=restore((OUT/'reference_checkpoint.json').read_text())
    after=solve(model,accepted,.001,np.full(16,1e-6));restart=solve(model,resumed,.001,np.full(16,1e-6))
    assert serialize(after)==serialize(restart)
    # Derivative against an independently reevaluated complete mixed residual.
    j=trial['jump'];dt=.001;beta=1/(1+RELAX*dt)
    def residual(x):
        mem=beta*(old['memory']+1e-6-(x-old['jump']));s=model.K@mem;pn=P0-model.N@mem
        v=(x-old['jump'])/dt;mu,_,_=state_friction(abs(v),old['Theta'],dt)
        a,q,*_=bond(x,old['damage'],old['reference']);return s-q-a*np.sign(v)*(pn*mu+ETA_RAD*abs(v))
    fd_errors=[]
    for eps in [1e-11,5e-12,2.5e-12]:
        fd=np.column_stack([(residual(j+np.eye(16)[k]*eps)-residual(j-np.eye(16)[k]*eps))/(2*eps) for k in range(16)])
        fd_errors.append(float(np.linalg.norm(fd-trial['tangent'])/np.linalg.norm(fd)))
    derivative_error=fd_errors[-1];assert derivative_error<1e-6,fd_errors
    # Exact intact one-sided driving is independent of dt, Gamma and a phase floor.
    intact=[dict(trial_shear_Pa=t,Y_J_m2=.5*C0*t*t,Gc_J_m2=GC,admitted=t>TAUP) for t in [2000,2999,3000,3001,4000]]
    peak=brentq(lambda p:1/SlidingBand(128,p).coherent_C-C0*SlidingBand(128,p).I/(1-SlidingBand(128,p).I),.86,.87,xtol=1e-15)
    matched=mapping(peak);general=mapping(.8)
    assert abs(matched['switch_work_J_m2'])<1e-11 and matched['stress_error_Pa']<1e-10
    assert abs(general['switch_work_J_m2'])>1e-6
    (OUT/'history_mapping.json').write_text(json.dumps(dict(compatible=matched,nonzero_switch_work_required=general),indent=2)+'\n')
    # Same physical sample times; independent of output stride or timestep.
    comparisons=[]
    for mode,pairs in [('formation',[(.002,.001),(.001,.0005)]),('mature',[(.002,.001)])]:
        for coarse,fine in pairs:
            left=[r for r in rows if r['mode']==mode and r['dt_s']==coarse]
            right=[r for r in rows if r['mode']==mode and r['dt_s']==fine]
            assert len(left)==len(right)
            assert max(abs(a['time_s']-b['time_s']) for a,b in zip(left,right))<1e-12
            for key in ['damage','jump_m','shear_Pa','Theta']:
                x=np.array([r[key] for r in left]);y=np.array([r[key] for r in right]);err=float(max(abs(x-y)));scale=float(max(abs(y)))
                comparisons.append(dict(mode=mode,coarse_dt_s=coarse,fine_dt_s=fine,quantity=key,max_absolute_error=err,reference_scale=scale,relative_error=err/scale))
    write_csv('point_comparison.csv',comparisons)
    write_csv('point_history.csv',rows);write_csv('intact_admission.csv',intact)
    restored_mapping=json.loads((OUT/'history_mapping.json').read_text())
    for label,mapped in [('compatible',matched),('nonzero_switch_work_required',general)]:
        for key in ['original_T','original_A','original_K','retained_residual_A','retained_residual_K','new_physical_stress']:
            assert np.array_equal(restored_mapping[label][key],mapped[key])
    result=dict(passed=True,seconds=time.perf_counter()-start,analytic_onset_s=exact,runs=summaries,mixed_derivative_error=derivative_error,finite_difference_errors=fd_errors,
        normal_pressure_range_Pa=float(np.ptp(trial['pn'])),mapping_switch_work_J_m2=general['switch_work_J_m2'],mapping_compatible_work_J_m2=matched['switch_work_J_m2'],
        retry_bitwise=True,restart_bitwise=True,duplicate_commit_rejected=True)
    (OUT/'point_summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
