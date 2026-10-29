#!/usr/bin/env python3
"""Independent weak-admission, native KKT and activation/restart checks."""
import json,csv,sys
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
import analyze_cohesive_interface_native as prior
import cohesive_interface_front as reference
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_front'
read=prior.read

def write(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def weak_examples():
    n=4;Q,w,x,P=prior.surface_rule(n);R=Q.T*w[None,:]*n
    cases={'uniform_below':np.full(n,.8),'alternating_below_nodal_peak':.9*(-1.)**np.arange(n),'redistributed_admissible':R@np.repeat([1,1,-1,-1],3),'uniform_above':np.full(n,1.01)}
    rows=[]
    for name,f in cases.items():
        lp=linprog(np.zeros(3*n),A_eq=R,b_eq=f,bounds=(-1,1),method='highs')
        v=(-1.)**np.arange(n)
        rows.append(dict(case=name,feasible=bool(lp.success),max_nodal_load_over_peak=float(max(abs(f))),max_projected_gauss_load_over_peak=float(max(abs(P@f))),alternating_release_over_peak=float(f@v/n-w@abs(Q@v))))
    assert rows[0]['feasible'] and not rows[1]['feasible'] and rows[2]['feasible'] and not rows[3]['feasible']
    write('weak_admission_examples.csv',rows)
    return rows

def verify_native(name,independent=False):
    prior.OUT=OUT;K,N,_=prior.operators(name);n=len(K);Q,w,x,P=prior.surface_rule(n);R=Q.T*w[None,:]*n
    h=read(OUT/name/'history.csv');a=read(OUT/name/'states.csv');nodes=read(OUT/name/'nodal_history.csv');ng=3*n
    state=reference.initialize(n);max_weak=0.;max_tied=0.;max_bound=0.;max_compare=0.;first=None;max_pressure_term=0.
    for step,row in enumerate(h):
        q=a[step*ng:(step+1)*ng];nd=nodes[step*n:(step+1)*n];time=row['time_s'];dt=row['dt_s']
        jump=np.array([v['jump_m'] for v in nd]);z=np.array([v['relaxed_jump_m'] for v in nd]);vel=np.array([v['V_m_s'] for v in q]);damage=np.array([v['damage'] for v in q]);theta=np.array([v['Theta'] for v in q]);pn=np.array([v['pn_Pa'] for v in q]);bond=np.array([v['q_Pa'] for v in q]);tied=damage==0
        mu=prior.ref.state_friction(abs(vel),np.array([v['Theta'] for v in q]),0.)[0]
        resistance=bond+damage*np.sign(vel)*(pn*mu+prior.ref.ETA_RAD*abs(vel));s=prior.ref.G/.125*row['wall_memory_m']-K@z
        err=max(abs(s-R@resistance));max_weak=max(max_weak,float(err));roundoff=64*np.finfo(float).eps*(max(abs(s))+max(abs(resistance)))
        assert err<=1e-8+roundoff,(time,err,roundoff)
        max_tied=max(max_tied,float(max(abs((Q@jump)[tied]),default=0.)));max_bound=max(max_bound,float(max(abs(bond[tied])-prior.ref.TAUP,default=0.)))
        assert max_tied==0. and max_bound<=0.
        assert np.max(abs(Q@jump-np.array([v['jump_m'] for v in q])))<1e-15
        assert np.all(damage>=state['a']-1e-14)
        e=np.array([v['jump_m']-v['reference_m'] for v in q]);expected_a=np.maximum(state['a'],np.minimum(abs(e)/prior.ref.DF,1.))
        assert max(abs(expected_a-damage))<1e-13
        expected_age=(state['age']+prior.ref.V0*dt/prior.ref.DC)/(1+abs(vel)*dt/prior.ref.DC)
        assert max(abs(expected_age-theta))<1e-12
        expected_q=np.zeros(ng);damaged=damage>0;expected_q[damaged]=e[damaged]*(1-damage[damaged])/(prior.ref.C0*damage[damaged])
        assert max(abs(expected_q[damaged]-bond[damaged]),default=0.)<1e-8
        assert abs(w@(.5*expected_q*e)-row['bond_J_m'])<1e-12
        if first is None and np.any((state['a']==0)&(damage>0)):first=[time-dt,time]
        if independent:
            du=.001*(min(time,.84)-min(time-dt,.84));new,diag=reference.solve_front(K,N,state,dt,du)
            max_compare=max(max_compare,float(max(abs(new['a']-damage))),float(max(abs(new['jump']-Q@jump))/prior.ref.DF))
            assert max_compare<1e-8,(time,max_compare)
            state=new
        else:state['a']=damage;state['age']=theta
        max_pressure_term=max(max_pressure_term,float(np.ptp(pn)))
    assert first is not None
    return dict(case=name,onset_bracket_s=first,max_recomputed_weak_Pa=max_weak,max_tied_jump_m=max_tied,max_tied_bound_excess_Pa=max_bound,max_independent_normalized_history_error=max_compare,max_pressure_range_Pa=max_pressure_term,min_curvature_Pa_m=min(r['min_curvature_Pa_m'] for r in h),min_release_curvature_Pa_m=min(r['release_curvature_Pa_m'] for r in h),max_certificate_iterations=max(r['certificate_iterations'] for r in h))

def qualification():
    examples=weak_examples();native=verify_native('qualification',True)
    loaded=read(OUT/'resume/restart_loaded.csv')[0]['accepted_time_s'];restart={}
    for stem in ['history','states','nodal_history']:
        a=read(OUT/f'qualification/{stem}.csv');b=[r for r in read(OUT/f'split/{stem}.csv') if r['time_s']<=loaded]+read(OUT/f'resume/{stem}-resume.csv');assert len(a)==len(b)
        errors={k:max(abs(x[k]-y[k]) for x,y in zip(a,b)) for k in a[0] if k!='step_seconds'}
        assert max(errors.values())==0.,errors
        restart[stem]=errors
    lifecycle=read(OUT/'qualification/lifecycle.csv');assert any(r['activation']==1 and r['rejected_unchanged']==1 and r['retry_bitwise']==1 for r in lifecycle)
    derivatives=read(OUT/'qualification/derivative_checks.csv');assert min(r['relative_error'] for r in derivatives)<1e-6
    prior.OUT=OUT;_,identity=prior.response('uniform-control','formation')
    K,N,_=prior.operators('uniform-control');state=reference.initialize(len(K));state['a'][:]=0.
    scalar=prior.ref.UniformBlock();old=prior.ref.initial();uniform_error=0.
    for h in read(OUT/'uniform-control/history.csv'):
        state,diagnostic=reference.solve_front(K,N,state,h['dt_s'],.001*h['dt_s'])
        trial=prior.ref.solve(scalar,old,h['dt_s'],np.array([.001*h['dt_s']]));old=prior.ref.commit(old,trial)
        uniform_error=max(uniform_error,float(max(abs(diagnostic['traction']-trial['stress'][0]))))
    assert uniform_error<1e-7
    result=dict(passed=True,independent_weak_uniform_error_Pa=uniform_error,weak_examples=examples,native=native,restart_bitwise=True,identity=identity,lifecycle=lifecycle,derivatives=derivatives)
    (OUT/'qualification_summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':qualification()
