#!/usr/bin/env python3
"""Temporal work audit of the unchanged native-kernel admission reference.
No corrected energy metric: diagnostics partition the raw physical remainder.
"""
import argparse, copy, csv, json, time
from pathlib import Path
import numpy as np
import cohesive_interface_admission as a

ROOT=a.ROOT
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_energy'
BASE=a.OUT
DT=.0025/128
TIMES=(.25,.58,.585,.6,.625,.65,.66)

def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def table(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def decomposition(k,s,t,d):
    K=k.K.astype(np.longdouble)
    c=np.array([t['wall_memory']]+t['memory'],dtype=np.longdouble)
    old=np.array([s['wall_memory']]+s['memory'],dtype=np.longdouble)
    dc=c-old;dx=np.array(t['delta'],dtype=np.longdouble)-s['delta']
    rm=-.5*np.longdouble(d['dt_s']*a.G/a.ETA)*(dc@K@c)
    rg=.5*(K@dc)[1:]@dx+np.longdouble(d['surface_remainder_J_m'])
    return dict(maxwell_remainder_J_m=float(rm),surface_time_remainder_J_m=float(rg),decomposition_error_J_m=float(d['physical_remainder_J_m']-rm-rg))
def summarize(rows):
    sums={key:float(sum(r[key] for r in rows)) for key in ['fracture_increment_J_m','sliding_loss_J_m','maxwell_loss_J_m','endpoint_work_J_m','trapezoid_work_J_m','physical_remainder_J_m','maxwell_remainder_J_m','surface_time_remainder_J_m']}
    absolute=sum(abs(r['physical_remainder_J_m']) for r in rows)
    sums.update(absolute_remainder_J_m=absolute,remainder_over_fracture=absolute/sums['fracture_increment_J_m'],remainder_over_sliding=absolute/sums['sliding_loss_J_m'],max_ledger_J_m=max(abs(r['ledger_error_J_m']) for r in rows),max_decomposition_error_J_m=max(abs(r['decomposition_error_J_m']) for r in rows),max_weak_Pa=max(r['weak_error_Pa'] for r in rows),max_FD_relative=max(r['tangent_relative'] for r in rows),minimum_normal_Pa=min(r['minimum_normal_Pa'] for r in rows),minimum_free_tangent_Pa_m=min(r['minimum_symmetric_tangent_Pa_m'] for r in rows if r['free_dimensions']),final=rows[-1])
    return sums

def archived(k):
    for name in ['coarse','fine']:
        states=json.loads((BASE/(name+'-states.json')).read_text())
        rows=list(csv.DictReader((BASE/(name+'-history.csv')).open()))
        s=a.initial(k)
        for t,r in zip(states,rows):
            r.update({key:float(val) for key,val in r.items()})
            r.update(decomposition(k,s,t,r));s=t
        table(name+'-history.csv',rows);save(name+'-summary.json',summarize(rows))

def analytic(k):
    """Exact tied Maxwell ramp/hold, evaluated directly in long double.
    Integrate each entire analytic loading segment, independently of the grid.
    """
    ld=np.longdouble;tau=ld(a.ETA/a.G);K=ld(k.K[0,0]);w0=ld('.0015');rate=ld('.001')
    ramp=ld('.45');hold=ld('.01');A=rate*tau;B=w0-A
    em=-np.expm1(-ramp/tau);em2=-np.expm1(-2*ramp/tau)
    wh=w0+(A-w0)*em;wf=wh*np.exp(-hold/tau)
    ew=K*rate*(A*ramp+B*tau*em)
    el=K/tau*(A*A*ramp+2*A*B*tau*em+B*B*tau*.5*em2)+K*wh*wh*.5*(-np.expm1(-2*hold/tau))
    rows=[]
    for dt0 in [.005,.0025,DT,DT/2]:
        dt=ld(str(dt0));w=w0;work=loss=remabs=ld(0);maxerr=ld(0)
        n=round((a.END-a.START)/dt0)
        for i in range(n):
            elapsed=dt*(i+1);r=rate if i*dt<ramp-ld('1e-12') else ld(0)
            old=w;w=(w+r*dt)/(1+dt/tau)
            exact=w0+(A-w0)*(-np.expm1(-elapsed/tau)) if elapsed<=ramp else wh*np.exp(-(elapsed-ramp)/tau)
            ww=ld('.5')*K*(w+old)*r*dt;ll=K/tau*dt*w*w
            rem=ww-ld('.5')*K*(w-old)*(w+old)-ll
            work+=ww;loss+=ll;remabs+=abs(rem);maxerr=max(maxerr,abs(w-exact))
        row=dict(dt_s=dt,final_memory=w,exact_final_memory=wf,max_memory_error=maxerr,relative_final_memory_error=abs(w-wf)/abs(wf),work_J_m=work,exact_work_J_m=ew,work_error_J_m=work-ew,maxwell_J_m=loss,exact_maxwell_J_m=el,maxwell_error_J_m=loss-el,absolute_remainder_J_m=remabs,exact_balance_J_m=ew-.5*K*(wf*wf-w0*w0)-el)
        rows.append({key:float(value) for key,value in row.items()})
    table('analytic-maxwell.csv',rows)
    assert max(abs(r['exact_balance_J_m']) for r in rows)<1e-12
    return rows

def run(k,dt,name,incoming=None):
    solver=a.Admission(k);s=copy.deepcopy(incoming) if incoming else a.initial(k)
    start=time.perf_counter();rows=[];events=[];snapshots={};bracketed=False;checked=False
    startclock=s['clock_s'];startpub=s['physical_publications']
    grid=[round(startclock+i*dt,12) for i in range(1,round((a.END-startclock)/dt)+1)]
    def advance(target):
        nonlocal s,checked
        old=copy.deepcopy(s)
        try:
            p=solver.solve(s,target,verify=bracketed)
            if max(p['state']['histories']['damage'])>0 and not checked:
                before=a.encode(s);retry=solver.solve(s,target,verify=True)
                assert a.encode(retry)==a.encode(p) and a.encode(s)==before
                bad=copy.deepcopy(p);bad['state']['histories']['damage'][0]=-1
                try:solver.publish(s,bad)
                except AssertionError:pass
                else:raise AssertionError('invalid trial published')
                restart=solver.solve(json.loads(before),target,verify=True)
                assert a.encode(restart)==a.encode(p) and a.encode(s)==before
                checked=True
            s=solver.publish(s,p)
            assert s['physical_publications']==old['physical_publications']+1
            try:solver.publish(s,p)
            except AssertionError:pass
            else:raise AssertionError('duplicate publication')
            r=p['diagnostics'];r.update(decomposition(k,old,s,r));rows.append(r)
        except Exception:
            save(name+'-last-accepted.json',old);table(name+'-partial-history.csv',rows)
            save(name+'-failure.json',dict(target_s=target,seconds=time.perf_counter()-start));raise
    for i,target in enumerate(grid):
        if s['generation']==0 and s['clock_s']>=a.INSERT-1e-12:
            s,birth=solver.insert(s);save(name+'-birth.json',birth)
        if not bracketed and np.all(np.array(s['histories']['damage'])==0) and solver.tied_trial(s,target)[3]>a.PEAK:
            lo,hi=solver.event(s,target);events.append(dict(lower_s=lo,upper_s=hi));bracketed=True
            if lo>s['clock_s']+1e-12:advance(lo)
            save(name+'-preactivation.json',s)
        advance(target)
        if any(abs(target-t)<1e-11 for t in TIMES):
            snapshots[f'{target:.6f}']=copy.deepcopy(s);save(name+'-snapshots.json',snapshots)
        if i%2000==0:print(name,i+1,len(grid),target,round(time.perf_counter()-start,2),flush=True)
    assert checked
    table(name+'-history.csv',rows);save(name+'-final.json',s)
    result=summarize(rows);result.update(dt_s=dt,seconds=time.perf_counter()-start,steps=s['physical_publications']-startpub,events=events,rejection_retry_restart_once=True)
    save(name+'-summary.json',result);print(json.dumps({key:result[key] for key in ['seconds','steps','remainder_over_sliding','remainder_over_fracture']}),flush=True)
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['audit','fine','replay']);args=parser.parse_args()
    OUT.mkdir(exist_ok=True,parents=True);a.OUT=OUT;k=a.Kernel('native-16-axial')
    if args.mode=='audit':archived(k);analytic(k)
    elif args.mode=='fine':run(k,DT,'selected')
    else:
        full=json.loads((OUT/'selected-summary.json').read_text())
        assert full['remainder_over_sliding']<.01 and full['remainder_over_fracture']<.01
        incoming=json.loads((OUT/'selected-snapshots.json').read_text())['0.580000']
        run(k,DT/2,'replay',incoming)
if __name__=='__main__':main()
