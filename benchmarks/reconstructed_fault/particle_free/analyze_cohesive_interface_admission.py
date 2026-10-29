#!/usr/bin/env python3
"""Common-time response, physical/BE work separation and constraint references."""
import json,numpy as np
from scipy.linalg import null_space
from cohesive_interface_admission import OUT,WORK,Kernel,Admission,initial,save,table,G,ETA,PEAK,DF,P0,HOLD,RATE,START
from check_cohesive_interface_kink import point

def partial_tie_check(k):
    old={key:np.array(v) for key,v in initial(k,True)['histories'].items()}
    tied=k.oq.copy();tied[np.flatnonzero(k.nq)[1]]=True
    C=null_space(k.Q[tied]);rng=np.random.default_rng(924);inc=2e-6*C@rng.normal(size=C.shape[1]);direction=C@rng.normal(size=C.shape[1]);direction/=np.linalg.norm(direction)
    dt=.002;beta=1/(1+G*dt/ETA)
    def evaluate(v):
        dq=k.Q@v;dq[tied]=0.;c=np.r_[.001875,beta*v];r=point(old,dq,P0+k.N@c,dt)
        F=(k.K@c)[1:]+k.Q[~tied].T@(k.w[~tied]*r['resistance'][~tied])
        J=beta*k.K[1:,1:]+k.Q[~tied].T@(k.w[~tied,None]*(r['derivative'][~tied,None]*k.Q[~tied]+r['pressure_derivative'][~tied,None]*beta*k.N[~tied,1:]))
        return C.T@F,C.T@J@C
    eps=1e-11;fd=(evaluate(inc+eps*direction)[0]-evaluate(inc-eps*direction)[0])/(2*eps);exact=evaluate(inc)[1]@(C.T@direction);err=np.linalg.norm(fd-exact)/np.linalg.norm(exact)
    assert err<1e-6 and np.linalg.norm(k.Q[tied]@C)<1e-12 and C.shape[1]==6
    save('partial-tie-check.json',dict(free_dimensions=C.shape[1],constraint_error=float(np.linalg.norm(k.Q[tied]@C)),full_reduced_tangent_relative=float(err),note='independent algebraic partial-Gauss tie fixture; not an additional evolved physical state'))

def bulk_energy_diagnostic(k):
    summary={};rows=[]
    for name in ['coarse','fine']:
        h=np.genfromtxt(OUT/(name+'-history.csv'),delimiter=',',names=True);w=.0015;t=START;error=[]
        for r in h:
            dt=r['dt_s'];du=RATE*(min(r['time_s'],HOLD)-min(t,HOLD));v=(w+du)/(1+G*dt/ETA)
            E=.5*k.K[0,0]*v*v;old=.5*k.K[0,0]*w*w
            e=.5*k.K[0,0]*(w+v)*du-(E-old)-2*G/ETA*dt*E;error.append(e);w=v;t=r['time_s']
            rows.append(dict(case=name,time_s=t,bulk_only_remainder_J_m=e,interface_remainder_J_m=r['physical_remainder_J_m'],difference_J_m=r['physical_remainder_J_m']-e))
        summary[name]=dict(bulk_only_absolute_remainder_J_m=float(sum(abs(np.array(error)))),absolute_difference_J_m=float(sum(abs(h['physical_remainder_J_m']-error))))
    save('bulk_energy_diagnostic.json',dict(results=summary,note='Analytic homogeneous Maxwell block without fracture, same BE time/load schedule. Diagnostic only; no subtraction from acceptance error.'))
    table('bulk_energy_diagnostic.csv',rows)

def main():
    k=Kernel('native-16-axial');partial_tie_check(k);bulk_energy_diagnostic(k);E0=.5*k.K[0,0]*.0015**2;metrics={};histories=[];profiles=[];states={}
    selected=[.25,.58,.585,.6,.625,.65,.66]
    for name in ['coarse','fine']:
        h=np.genfromtxt(OUT/(name+'-history.csv'),delimiter=',',names=True);states[name]=json.loads((OUT/(name+'-states.json')).read_text())
        physical=h['physical_remainder_J_m'];mx_trap=G/ETA*h['dt_s']*(h['bulk_J_m']+np.r_[E0,h['bulk_J_m'][:-1]])
        alternate=physical+h['maxwell_loss_J_m']-mx_trap
        fd=sum(h['fracture_increment_J_m']);sl=sum(h['sliding_loss_J_m']);work=sum(h['trapezoid_work_J_m']);error=sum(abs(physical))
        metrics[name]=dict(final_time_s=float(h[-1]['time_s']),integrated_damage_m=float(h[-1]['integrated_damage_m']),peak_damage=float(h[-1]['peak_damage']),integrated_absolute_jump_m2=float(h[-1]['integrated_absolute_jump_m2']),peak_absolute_jump_m=float(h[-1]['peak_absolute_jump_m']),added_fracture_J_m=float(fd),sliding_loss_J_m=float(sl),maxwell_loss_J_m=float(sum(h['maxwell_loss_J_m'])),trapezoidal_work_J_m=float(work),absolute_physical_remainder_J_m=float(error),signed_physical_remainder_J_m=float(sum(physical)),absolute_remainder_over_fracture=float(error/fd),absolute_remainder_over_sliding=float(error/sl),absolute_remainder_over_work=float(error/abs(work)),trapezoidal_Maxwell_alternative_absolute_remainder_J_m=float(sum(abs(alternate))),max_BE_ledger_error_J_m=float(max(abs(h['ledger_error_J_m']))),max_weak_error_Pa=float(max(h['weak_error_Pa'])),max_coupled_tangent_error=float(max(h['tangent_relative'])),minimum_normal_Pa=float(min(h['minimum_normal_Pa'])),minimum_free_symmetric_tangent_Pa_m=float(min(h['minimum_symmetric_tangent_Pa_m'][h['free_dimensions']>0])),fully_decohered_length_m=float(h[-1]['fully_decohered_length_m']))
        for i,row in enumerate(h):
            histories.append(dict(case=name,**{key:float(row[key]) for key in h.dtype.names},cumulative_fracture_J_m=float(sum(h['fracture_increment_J_m'][:i+1])),cumulative_sliding_J_m=float(sum(h['sliding_loss_J_m'][:i+1])),cumulative_Maxwell_J_m=float(sum(h['maxwell_loss_J_m'][:i+1])),cumulative_trapezoid_work_J_m=float(sum(h['trapezoid_work_J_m'][:i+1])),cumulative_absolute_remainder_J_m=float(sum(abs(physical[:i+1]))),cumulative_signed_remainder_J_m=float(sum(physical[:i+1])),Maxwell_trapezoidal_alternative_J_m=float(mx_trap[i])))
        solver=Admission(k)
        for target in selected:
            i=int(np.argmin([abs(s['clock_s']-target) for s in states[name]]));s=states[name][i];assert abs(s['clock_s']-target)<1e-10
            prev=initial(k) if i==0 else states[name][i-1];dt=s['clock_s']-prev['clock_s'];c=np.r_[s['wall_memory'],s['memory']];pn=P0+k.N@c;physical_plane=np.loadtxt(k.path/'shear.csv',delimiter=',')@c
            old={key:np.array(v) for key,v in prev['histories'].items()};dq=np.array(s['histories']['jump'])-old['jump'];r=point(old,dq,pn,dt)
            replay=solver.solve(prev,s['clock_s']);lam=np.zeros(len(k.s));lam[np.array(replay['tied'])]=replay['multipliers']
            aa=np.array(s['histories']['damage']);ee=np.array(s['histories']['jump'])-old['reference'];bondq=(1-aa)*ee/((2/PEAK**2)*np.where(aa==0,1,aa))
            for q in range(len(k.s)):
                profiles.append(dict(case=name,time_s=s['clock_s'],gamma=s['gamma'],x_m=float(k.s['x_m'][q]),y_m=float(k.s['y_m'][q]),weight_m=float(k.w[q]),branch=int(k.s['branch'][q]),damage=s['histories']['damage'][q],jump_m=s['histories']['jump'][q],velocity_m_s=float(dq[q]/dt),Theta=s['histories']['theta'][q],normal_pressure_Pa=float(pn[q]),physical_plane_shear_Pa=float(physical_plane[q]),cohesive_traction_Pa=float(bondq[q] if aa[q]>0 else lam[q]),total_resistance_Pa=float(r['resistance'][q] if aa[q]>0 else lam[q]),installed=int(s['generation']>0 or k.oq[q]),tied=int(s['histories']['damage'][q]==0),fully_decohered=int(s['histories']['damage'][q]==1)))
    comparisons=[]
    for t in [.585,.6,.625,.65,.66]:
        a=min(states['coarse'],key=lambda s:abs(s['clock_s']-t));b=min(states['fine'],key=lambda s:abs(s['clock_s']-t));da=np.array(a['histories']['damage']);db=np.array(b['histories']['damage']);ja=np.array(a['histories']['jump']);jb=np.array(b['histories']['jump'])
        comparisons.append(dict(time_s=t,damage_L2_relative=float(np.sqrt(k.w@(da-db)**2/(k.w@db**2))),integrated_damage_relative=float(abs(k.w@(da-db))/(k.w@db)),peak_damage_relative=float(abs(max(da)-max(db))/max(db)),integrated_slip_relative=float(abs(k.w@(abs(ja)-abs(jb)))/(k.w@abs(jb))),peak_slip_relative=float(abs(max(abs(ja))-max(abs(jb)))/max(abs(jb)))))
    tau=ETA/G;critical=PEAK/(1.6*G);exact=START-tau*np.log((critical-RATE*tau)/(.0015-RATE*tau))
    events={n:json.loads((OUT/(n+'-summary.json')).read_text())['events'][0] for n in ['coarse','fine']}
    save('verification.json',dict(metrics=metrics,events=events,homogeneous_tied_continuum_event_s=float(exact),event_shift_s=events['coarse']['lower_s']-events['fine']['lower_s'],late_temporal_response_passed=all(max(r['integrated_damage_relative'],r['integrated_slip_relative'])<.02 for r in comparisons),fine_fracture_energy_screen_passed=metrics['fine']['absolute_remainder_over_fracture']<.01,fine_sliding_energy_screen_passed=metrics['fine']['absolute_remainder_over_sliding']<.01,native_owner=json.loads((OUT/'native-owner.json').read_text())))
    table('histories.csv',histories);table('profiles.csv',profiles);table('temporal_comparison.csv',comparisons)
    # Actual native physical stress fields, with no remapping of history.
    raw=np.genfromtxt(k.path/'field-response.csv',delimiter=',',names=True).reshape(-1,len(k.nodes)+1);rows=[]
    for t in [.58,.6,.65,.66]:
        s=min(states['fine'],key=lambda s:abs(s['clock_s']-t));c=np.r_[s['wall_memory'],s['memory']]
        v={key:raw[key]@c for key in ['tau_xx_Pa','tau_yy_Pa','tau_zz_Pa','tau_xy_Pa','pprime_Pa']}
        for i in range(len(raw)):
            rows.append(dict(time_s=t,x_m=float(raw['x_m'][i,0]),y_m=float(raw['y_m'][i,0]),weight_m2=float(raw['weight_m2'][i,0]),**{key:float(a[i]) for key,a in v.items()}))
    table('physical_fields.csv',rows)
    print(json.dumps(dict(metrics=metrics,temporal=comparisons,event_reference_s=exact),indent=2))
if __name__=='__main__':main()
