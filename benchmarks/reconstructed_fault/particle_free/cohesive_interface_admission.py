#!/usr/bin/env python3
"""Event-resolved signed admission on the retained native fitted mixed kernel."""
import copy,csv,json,time
from pathlib import Path
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import linprog
from check_cohesive_interface_kink import Kernel,point,encode,G,ETA,GC,PEAK,DF,C0,V0,DC,P0
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_admission'
WORK=ROOT/'build-particle-free/cohesive-interface-admission'
RATE=.001;START=.2;INSERT=.25;HOLD=.65;END=.66

def save(name,data): (OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def table(name,rows):
    with (OUT/name).open('w') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)

def initial(k,installed=False):
    n=len(k.nodes);nq=len(k.s)
    h={key:np.zeros(nq).tolist() for key in ['jump','damage','reference','relaxed_jump','absolute_slip']};h['theta']=np.full(nq,1.001).tolist()
    return dict(clock_s=START,gamma=.0015,wall_memory=.0015,delta=np.zeros(n).tolist(),memory=np.zeros(n).tolist(),histories=h,physical_publications=0,generation=int(installed),publications=int(installed),installed=[0,1] if installed else [0],geometry=[[.25,0],[.5,0],[.75,.125]] if installed else [[.25,0],[.5,0]],virgin_theta=1.001)

def bounded_multiplier(k,force,tied,rows=None):
    if rows is None:rows=np.ones(len(k.nodes),bool)
    L=(k.Q[tied].T*k.w[tied])[rows]/k.mass[rows,None]
    target=-force[rows]/k.mass[rows]
    if not np.any(tied):return np.empty(0),float(np.max(abs(target))),0.
    nq=L.shape[1]
    result=linprog(np.r_[np.zeros(nq),1.],A_eq=np.c_[L*PEAK,np.zeros(sum(rows))],b_eq=target,A_ub=np.c_[np.r_[np.eye(nq),-np.eye(nq)],-np.ones(2*nq)],b_ub=np.zeros(2*nq),bounds=[(None,None)]*nq+[(0,None)],method='highs')
    if not result.success:raise RuntimeError('Weak multiplier solve: '+result.message)
    lam=PEAK*result.x[:-1];error=float(max(abs(L@lam-target)))
    return lam,error,float(result.x[-1]*PEAK)

class Admission:
    def __init__(self,k):self.k=k
    def tied_trial(self,s,end):
        dt=end-s['clock_s'];du=RATE*(min(end,HOLD)-min(s['clock_s'],HOLD));beta=1/(1+G*dt/ETA)
        c=np.r_[beta*(s['wall_memory']+du),beta*np.array(s['memory'])]
        force=(self.k.K@c)[1:];lam,err,peak=bounded_multiplier(self.k,force,np.ones(len(self.k.s),bool))
        return c,lam,err,peak
    def event(self,s,end):
        lo=s['clock_s'];hi=end
        assert self.tied_trial(s,hi)[3]>PEAK
        for _ in range(60):
            if hi-lo<=1e-8:break
            mid=.5*(lo+hi)
            if self.tied_trial(s,mid)[3]<=PEAK:lo=mid
            else:hi=mid
        return lo,hi
    def insert(self,s):
        before=encode(s);assert s['installed']==[0]
        c=np.r_[s['wall_memory'],s['memory']];lam,err,peak=bounded_multiplier(self.k,(self.k.K@c)[1:],np.ones(len(self.k.s),bool))
        assert err<=1e-8 and peak<PEAK
        t=copy.deepcopy(s);t.update(installed=[0,1],geometry=[[.25,0],[.5,0],[.75,.125]],generation=1,publications=1)
        assert s['histories']==t['histories'] and encode(s)==before
        return t,dict(time_s=s['clock_s'],required_multiplier_Pa=peak,weak_error_Pa=err,storage_change_J_m=0.,geometry_publications=1)
    def solve(self,s,end,verify=False):
        k=self.k;dt=end-s['clock_s'];assert dt>0
        du=RATE*(min(end,HOLD)-min(s['clock_s'],HOLD));beta=1/(1+G*dt/ETA)
        old={key:np.array(v) for key,v in s['histories'].items()};virgin=old['damage']==0
        wall=beta*(s['wall_memory']+du);mem=np.array(s['memory']);n=len(k.nodes);qv=k.Q[virgin]
        # Signed semismooth primal/dual predictor; lambda is the nonsmooth
        # peak*sign(jump) part only. Smooth friction/softening stays in h.
        def evaluate(inc,tied=None):
            dq=k.Q@inc
            if tied is not None:dq[tied]=0. # Exact Gauss equality, not a damage floor.
            c=np.r_[wall,beta*(mem+inc)];pn=P0+k.N@c
            if min((P0+k.Nm@c).min(),(P0+k.Np@c).min())<=0:raise RuntimeError('lost contact')
            r=point(old,dq,pn,dt)
            smooth=r['resistance'].copy();smooth[virgin]-=PEAK*np.sign(dq[virgin]);smooth[virgin&(dq==0)]=0.
            f=(k.K@c)[1:]+k.Q.T@(k.w*smooth)
            J=beta*k.K[1:,1:]+k.Q.T@(k.w[:,None]*(r['derivative'][:,None]*k.Q+r['pressure_derivative'][:,None]*beta*k.N[:,1:]))
            return f,J,r,c,dq
        inc=np.zeros(n);f,J,r,c,dq=evaluate(inc)
        if np.all(virgin):lam,err,peak=bounded_multiplier(k,f,virgin)
        else:
            L=(k.Q[virgin].T*k.w[virgin])/k.mass[:,None]
            lam=np.linalg.lstsq(L,-f/k.mass,rcond=None)[0];err=float('inf');peak=float('inf')
        all_tied=bool(np.all(virgin) and peak<=PEAK and err<=1e-8)
        predictor_iterations=0
        if not all_tied:
            x=np.zeros(n);y=np.clip(lam/PEAK,-1,1);L=(k.Q[virgin].T*k.w[virgin])/k.mass[:,None]
            def residual(x,y,jac=False):
                f,J,r,c,dq=evaluate(DF*x);z=qv@x;u=z+y;outside=abs(u)>=1.
                comp=z-np.sign(u)*np.maximum(abs(u)-1.,0.)
                R=np.r_[f/(k.mass*PEAK)+L@y,comp]
                if not jac:return R
                C=(~outside)[:,None]*qv;D=-np.diag(outside.astype(float))
                return R,np.block([[J*DF/(k.mass[:,None]*PEAK),L],[C,D]])
            for it in range(100):
                R,A=residual(x,y,True);predictor_iterations=it+1
                if max(abs(R[:n]))*PEAK<=2e-9 and max(abs(R[n:]),default=0)<2e-13:break
                step=np.linalg.lstsq(A,-R,rcond=None)[0];scale=1.;norm=np.linalg.norm(R)
                for _ in range(35):
                    rn=residual(x+scale*step[:n],y+scale*step[n:])
                    if np.linalg.norm(rn)<norm or max(abs(rn))<1e-13:break
                    scale*=.5
                else:raise RuntimeError(('semismooth line search',end,norm))
                x+=scale*step[:n];y+=scale*step[n:]
            else:raise RuntimeError(('semismooth iteration limit',end,max(abs(R))))
            inc=DF*x;dq=k.Q@inc
            # Remove only arithmetic remnants of exact tie constraints. The
            # bound scales with the computed jump, not a fixed phase cutoff.
            roundoff=256*np.finfo(float).eps*np.max(abs(dq))
            tied=virgin&(abs(dq)<=roundoff)
        else:tied=virgin.copy()
        C=null_space(k.Q[tied]) if np.any(tied) else np.eye(n)
        inc=C@(C.T@inc)
        for polish in range(12):
            f,J,r,c,dq=evaluate(inc,tied)
            # r includes the sign part on released points; tied resistance is
            # solved independently as a bounded weak multiplier.
            force=(k.K@c)[1:]+k.Q[~tied].T@(k.w[~tied]*r['resistance'][~tied])
            free=C.T@force
            if np.max(abs(free),initial=0)<1e-12:break
            inc-=C@np.linalg.solve(C.T@J@C,free)
        f,J,r,c,dq=evaluate(inc,tied)
        force=(k.K@c)[1:]+k.Q[~tied].T@(k.w[~tied]*r['resistance'][~tied])
        lam,err,peak=bounded_multiplier(k,force,tied,rows=k.old if s['generation']==0 else None)
        if err>1e-8 or peak>PEAK+1e-8:raise RuntimeError(('full weak certificate',end,err,peak))
        if np.max(abs((k.Q@inc)[tied]),initial=0)>256*np.finfo(float).eps*max(np.max(abs(inc)),1e-300):raise RuntimeError('tie projection arithmetic')
        if C.shape[1]:
            M=C.T@np.diag(k.mass)@C;d,V=np.linalg.eigh(M);inverse=(V/np.sqrt(d))@V.T
            reduced=inverse@(C.T@(.5*(J+J.T))@C)@inverse
            curvature=float(np.linalg.eigvalsh(reduced).min())
            if curvature<=0:
                save('rejected-curvature.json',dict(end_s=end,increment=inc.tolist(),tied=tied.tolist(),curvature=curvature));raise RuntimeError('negative admissible mechanical symmetric tangent')
        else:curvature=0.
        tangent_error=0.
        if verify and C.shape[1]:
            direction=C@np.random.default_rng(7).normal(size=C.shape[1]);direction/=np.linalg.norm(direction)
            scale=max(np.linalg.norm(inc),1e-20)*1e-5
            def physical_force(v):
                f,j,rr,cc,qq=evaluate(v,tied)
                return (k.K@cc)[1:]+k.Q[~tied].T@(k.w[~tied]*rr['resistance'][~tied])
            fd=C.T@(physical_force(inc+scale*direction)-physical_force(inc-scale*direction))/(2*scale)
            exact=C.T@J@direction;tangent_error=float(np.linalg.norm(fd-exact)/max(np.linalg.norm(fd),1.))
            assert tangent_error<1e-6,tangent_error
        t=copy.deepcopy(s);t['clock_s']=end;t['gamma']+=du;t['wall_memory']=wall;t['delta']=(np.array(s['delta'])+inc).tolist();t['memory']=c[1:].tolist();t['virgin_theta']+=V0*dt/DC
        for key in old:
            if key=='jump':val=old[key]+dq
            elif key=='damage':val=r['new_damage'].copy();val[tied]=0.
            elif key=='theta':val=r['new_theta']
            elif key=='reference':val=old[key]
            elif key=='relaxed_jump':val=beta*(old[key]+dq)
            elif key=='absolute_slip':val=old[key]+abs(dq)
            t['histories'][key]=val.tolist()
        assert np.all(np.array(t['histories']['damage'])>=old['damage'])
        # Exact BE ledger and independent physical trapezoidal work.
        oldc=np.r_[s['wall_memory'],s['memory']];dc=c-oldc
        bulk=.5*c@k.K@c;dbulk=bulk-.5*oldc@k.K@oldc
        a=old['damage'];e=old['jump']-old['reference'];oldbond=.5*(1-a)/(C0*np.where(a==0,1,a))*e*e
        storage=r['storage'].copy();storage[tied]=0.;bond=k.w@storage;dbond=k.w@(storage-oldbond)
        fracture=k.w@(GC*(np.array(t['histories']['damage'])-a));sliding=k.w@(r['friction']+r['radiation']);maxwell=2*G/ETA*dt*bulk
        q=(1-np.array(t['histories']['damage']))*(np.array(t['histories']['jump'])-old['reference'])/(C0*np.where(np.array(t['histories']['damage'])==0,1,np.array(t['histories']['damage'])))
        surface_rem=k.w@(q*dq)-dbond-fracture;be=.5*dc@k.K@dc
        reaction=(k.K@c)[0];work=reaction*du;trap=.5*(reaction+(k.K@oldc)[0])*du
        ledger=work-(dbulk+dbond+fracture+sliding+maxwell+be+surface_rem)
        assert abs(ledger)<=1e-10*max(1,abs(work)),(end,ledger)
        data=dict(time_s=end,dt_s=dt,gamma=t['gamma'],bulk_J_m=float(bulk),bond_J_m=float(bond),fracture_increment_J_m=float(fracture),sliding_loss_J_m=float(sliding),maxwell_loss_J_m=float(maxwell),endpoint_work_J_m=float(work),trapezoid_work_J_m=float(trap),BE_bulk_J_m=float(be),surface_remainder_J_m=float(surface_rem),physical_remainder_J_m=float(trap-dbulk-dbond-fracture-sliding-maxwell),ledger_error_J_m=float(ledger),weak_error_Pa=err,peak_tied_multiplier_Pa=peak,minimum_normal_Pa=float(min((P0+k.Nm@c).min(),(P0+k.Np@c).min())),minimum_symmetric_tangent_Pa_m=curvature,tangent_relative=tangent_error,predictor_iterations=predictor_iterations,free_dimensions=C.shape[1],integrated_damage_m=float(k.w@np.array(t['histories']['damage'])),peak_damage=float(max(t['histories']['damage'])),integrated_absolute_jump_m2=float(k.w@abs(np.array(t['histories']['jump']))),peak_absolute_jump_m=float(max(abs(np.array(t['histories']['jump'])))),fully_decohered_length_m=float(k.w@(np.array(t['histories']['damage'])==1)))
        payload=dict(base_publications=s['physical_publications'],base_generation=s['generation'],state=t,tied=tied.tolist(),multipliers=lam.tolist(),diagnostics=data)
        return payload
    def publish(self,s,payload):
        assert payload['base_publications']==s['physical_publications'] and payload['base_generation']==s['generation'],'stale publication'
        t=copy.deepcopy(payload['state']);assert t['clock_s']>s['clock_s']
        assert np.all(np.array(t['histories']['damage'])>=np.array(s['histories']['damage'])) and max(t['histories']['damage'])<=1
        assert payload['diagnostics']['weak_error_Pa']<=1e-8
        t['physical_publications']+=1
        return t

def run(dt,name,preinstalled=False):
    k=Kernel('native-16-axial');solver=Admission(k);s=initial(k,preinstalled);start=time.perf_counter();rows=[];states=[];events=[];birth=None;activation_checked=False
    grid=list(np.round(np.arange(START+dt,END+dt/2,dt),12));bracketed=False
    for target in grid:
        if not preinstalled and s['generation']==0 and s['clock_s']>=INSERT-1e-12:s,birth=solver.insert(s)
        if not bracketed and np.all(np.array(s['histories']['damage'])==0) and solver.tied_trial(s,target)[3]>PEAK:
            lo,hi=solver.event(s,target);events.append(dict(lower_s=lo,upper_s=hi,incoming_time_s=s['clock_s'],target_s=target));bracketed=True
            if lo>s['clock_s']+1e-12:
                p=solver.solve(s,lo);s=solver.publish(s,p);rows.append(p['diagnostics']);states.append(copy.deepcopy(s))
                save(name+'-preactivation.json',s)
        try:p=solver.solve(s,target,verify=bracketed)
        except Exception:
            save(name+'-last-accepted.json',s);save(name+'-stopped.json',dict(target_s=target,rows=rows,events=events));raise
        growing=max(p['state']['histories']['damage'])>0
        if growing and not activation_checked:
            before=encode(s);retry=solver.solve(s,target,verify=True);assert encode(retry)==encode(p) and encode(s)==before
            bad=copy.deepcopy(p);bad['state']['histories']['damage'][0]=-1
            try:solver.publish(s,bad)
            except AssertionError:pass
            else:raise AssertionError('bad trial published')
            assert encode(s)==before
            checkpoint=json.loads(encode(s));restarted=solver.solve(checkpoint,target,verify=True);assert encode(restarted)==encode(p)
            save(name+'-activation-trial.json',p);activation_checked=True
        old=s;s=solver.publish(s,p)
        if growing and s['physical_publications']==old['physical_publications']+1:
            try:solver.publish(s,p)
            except AssertionError:pass
            else:raise AssertionError('duplicate trial published')
        rows.append(p['diagnostics']);states.append(copy.deepcopy(s))
    assert activation_checked and birth is not None or preinstalled
    save(name+'-states.json',states);table(name+'-history.csv',rows)
    result=dict(dt_s=dt,seconds=time.perf_counter()-start,events=events,birth=birth,accepted_steps=s['physical_publications'],final_damage=max(s['histories']['damage']),rejection_retry_restart=activation_checked)
    save(name+'-summary.json',result);print(json.dumps(result),flush=True);return rows,states,result

def main():
    OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
    coarse,states,_=run(.005,'coarse')
    run(.0025,'fine')
    _,control,_=run(.005,'preinstalled',True)
    for a,b in zip(states,control):
        for key in ['histories','gamma','wall_memory','delta','memory','clock_s','physical_publications']:assert a[key]==b[key],key
    save('preinstalled-identity.json',dict(all_physical_states_bitwise=True,compared_states=len(states)))
if __name__=='__main__':main()
