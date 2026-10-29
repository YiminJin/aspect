#!/usr/bin/env python3
"""Bounded point qualification of explicit material-plane continuation."""
import copy,csv,json,time
import numpy as np
from scipy.optimize import minimize_scalar
from structural_rsf import *


def csvwrite(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def encode(state):
    return json.dumps({k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in state.items()},sort_keys=True)


def decode(data):
    s=json.loads(data)
    return {k:np.array(v) if isinstance(v,list) else v for k,v in s.items()}


def trajectory(phi,n,dt,nonproportional=False):
    s=initial(phi,n);rows=[];work=0.;loss=0.
    for step in range(round(.3/dt)):
        midpoint=(step+.5)*dt;t=(step+1)*dt
        de=deviator(gamma=(.02 if midpoint<.15 else -.02)*dt,
                    xx=(.004*dt if nonproportional and .1<midpoint<.2 else 0.),
                    yy=(-.004*dt if nonproportional and .1<midpoint<.2 else 0.))
        p=P0+500*np.sin(2*np.pi*t/.3)
        trial=point_trial(s,de,dt,p);energy=point_work(s,trial,de,dt)
        s=publish(s,trial);work+=energy['work_J_m3'];loss+=sum(energy[k] for k in ['Maxwell_J_m3','friction_J_m3','radiation_J_m3'])
        competing=directions(s['tau'],s['Q'],p,s['mu'])['maximum']-ETA_RAD*s['V']
        rows.append(dict(phi=phi,preference_deg=np.degrees(np.arctan2(n[1],n[0])),nonproportional=int(nonproportional),dt_s=dt,time_s=t,
            tau_xx_Pa=s['tau'][0,0],tau_yy_Pa=s['tau'][1,1],tau_zz_Pa=s['tau'][2,2],tau_xy_Pa=s['tau'][0,1],
            A_xx=s['A'][0,0],A_xy=s['A'][0,1],K_xx=s['K'][0,0],K_xy=s['K'][0,1],
            slip_m=2*inner(s['K'],plane(n)[2])/s['chi'],signed_V_m_s=s['signed_V'],pn_Pa=s['pn'],Theta=s['Theta'],
            other_plane_excess_Pa=competing,work_J_m3=work,dissipation_J_m3=loss,
            rate_error_Pa=s['rate_error_Pa'],ledger_error_J_m3=energy['ledger_defect_J_m3']))
    return rows,s


def main():
    start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    virgin=initial(.8);virgin['T']=tensor(xy=1100.)
    before=encode(virgin);left,info=form_direction(virgin,P0,[0,1,0]);right,info2=form_direction(virgin,P0,[1,0,0])
    ambiguous,ambinfo=form_direction(virgin,P0)
    assert ambiguous is None and left is not None and right is not None
    assert encode(virgin)==before
    # Independent maximization using actual pn on each oriented physical plane,
    # rather than the inherited orthogonal-pair enumeration.
    def actual_force(angle):
        n=np.array([np.cos(angle),np.sin(angle),0.]);S=plane(n)[2]
        return abs(inner(virgin['T'],S))-info['mu']*(P0-inner(virgin['T'],np.outer(n,n)))
    maxima=[-minimize_scalar(lambda x:-actual_force(x),bounds=(a,b),method='bounded',options={'xatol':1e-14}).fun
            for a,b in zip(np.linspace(0,np.pi,33)[:-1],np.linspace(0,np.pi,33)[1:])]
    formation_error=abs(max(maxima)-ETA_RAD*info['V'])
    assert formation_error<1e-8,formation_error
    for assigned in [left,right]:
        for key in ['T','A','K','Theta','clock','publications','phi']:
            assert np.array_equal(assigned[key],virgin[key])
    # Advance the actual admitted loaded state: no fresh-history substitution.
    formation_rows=[];formation_errors=[]
    for phi in [.2,.8,.99,1.]:
        for branch,preference in enumerate([[0,1,0],[1,0,0]]):
            incoming=initial(phi);incoming['T']=tensor(xy=1100.)
            assigned,admission=form_direction(incoming,P0,preference)
            sequences=[]
            for dt in [.0125,.00625,.003125]:
                state=copy.deepcopy(assigned);seq=[]
                for step in range(round(.025/dt)):
                    de=deviator(gamma=.001*dt)
                    candidate=point_trial(state,de,dt,P0);point_work(state,candidate,de,dt)
                    state=publish(state,candidate);seq.append(state['tau'].copy())
                    formation_rows.append(dict(phi=phi,branch=branch,dt_s=dt,time_s=state['clock'],
                        tau_xx_Pa=state['tau'][0,0],tau_xy_Pa=state['tau'][0,1],A_xy=state['A'][0,1],K_xy=state['K'][0,1],
                        normal_x=state['normal'][0],normal_y=state['normal'][1],Theta=state['Theta']))
                sequences.append(np.array(seq))
            errors=[np.linalg.norm(a-b[1::2])/np.linalg.norm(b[1::2]) for a,b in zip(sequences[:-1],sequences[1:])]
            assert errors[1]<errors[0] and errors[1]<.02,errors
            formation_errors.append(errors[1])
    csvwrite('formation_history.csv',formation_rows)
    allrows=[];comparisons=[];cases={}
    for phi in [.2,.8,.99,1.]:
        for branch,n in enumerate([left['normal'],right['normal']]):
            trajectories=[]
            for dt in [.0125,.00625,.003125]:
                rows,state=trajectory(phi,n,dt);allrows+=rows;trajectories.append(rows)
            errs=[]
            for coarse,fine in zip(trajectories[:-1],trajectories[1:]):
                fields=['tau_xx_Pa','tau_yy_Pa','tau_zz_Pa','tau_xy_Pa','tau_xy_Pa']
                a=np.array([[r[k] for k in fields] for r in coarse]);b=np.array([[r[k] for k in fields] for r in fine[1::2]])
                stress=np.linalg.norm(a-b)/np.linalg.norm(b)
                sa=np.array([r['slip_m'] for r in coarse]);sb=np.array([r['slip_m'] for r in fine[1::2]])
                slip=np.linalg.norm(sa-sb)/np.linalg.norm(sb);errs.append((stress,slip))
                comparisons.append(dict(phi=phi,branch=branch,coarse_dt_s=coarse[0]['dt_s'],stress_relative_L2=stress,slip_relative_L2=slip))
            assert errs[1][0]<errs[0][0] and errs[1][1]<errs[0][1] and errs[1][0]<=.02,errs
            cases[(phi,branch)]=trajectories[-1]
    # Symmetry has two different physical plane records, not a failed root.
    a,b=cases[(.8,0)],cases[(.8,1)]
    symmetry=max(abs(x['tau_xy_Pa']-y['tau_xy_Pa']) for x,y in zip(a,b))/G
    assert symmetry<1e-10
    nonprop=[]
    for dt in [.0125,.00625,.003125]:
        rows,s=trajectory(.8,left['normal'],dt,True);allrows+=rows;nonprop.append(rows)
    errs=[]
    for coarse,fine in zip(nonprop[:-1],nonprop[1:]):
        fields=['tau_xx_Pa','tau_yy_Pa','tau_zz_Pa','tau_xy_Pa','tau_xy_Pa']
        a=np.array([[r[k] for k in fields] for r in coarse]);b=np.array([[r[k] for k in fields] for r in fine[1::2]])
        stress=np.linalg.norm(a-b)/np.linalg.norm(b)
        sa=np.array([r['slip_m'] for r in coarse]);sb=np.array([r['slip_m'] for r in fine[1::2]])
        slip=np.linalg.norm(sa-sb)/np.linalg.norm(sb);errs.append((stress,slip))
        comparisons.append(dict(phi=.8,branch=2,coarse_dt_s=coarse[0]['dt_s'],stress_relative_L2=stress,slip_relative_L2=slip))
    assert errs[1][0]<errs[0][0] and errs[1][1]<errs[0][1] and errs[1][0]<=.02,errs
    # Carry a non-proportional full tensor memory; no orientation reset.
    old=initial(.8,left['normal']);old['A']=tensor(xx=2e-5,xy=5e-5,mean=1e-5);old['K']=tensor(xx=3e-5,xy=8e-5,mean=2e-5)
    old['T']=tensor(xx=180.,xy=1300.,mean=20.);old['Theta']=.9
    de=deviator(xx=1e-5,yy=-2e-5,gamma=2e-4);dt=.00625;p=10200.
    saved=encode(old);new=point_trial(old,de,dt,p,True);energy=point_work(old,new,de,dt)
    S=plane(old['normal'])[2];beta=1/(1+RELAX*dt)
    perpendicular=lambda a:a-2*inner(a,S)*S
    memory_error=np.linalg.norm(perpendicular(new['A'])-beta*perpendicular(old['A']))
    assert memory_error<1e-15
    fd=[];step=1e-8
    for j,E in enumerate([deviator(xx=1),deviator(yy=1),deviator(gamma=1),tensor()]):
        plus=point_trial(old,de+step*E,dt,p+(G*step if j==3 else 0.))
        minus=point_trial(old,de-step*E,dt,p-(G*step if j==3 else 0.))
        fd.append((plus['tau']-minus['tau'])/(2*step*G))
    fd=np.stack(fd,axis=-1);tangent_error=np.linalg.norm(fd-new['tangent'])/np.linalg.norm(fd)
    assert tangent_error<=1e-6,tangent_error
    R=rotation(.371);rot=copy.deepcopy(old)
    for k in ['T','A','K']:rot[k]=R@old[k]@R.T
    rot['normal']=R@old['normal'];rn=point_trial(rot,R@de@R.T,dt,p)
    covariance=max(np.linalg.norm(rn[k]-R@new[k]@R.T)/max(1.,np.linalg.norm(new[k])) for k in ['T','A','K','tau','Q'])
    assert covariance<=1e-10
    try:point_trial(old,de,dt,-20000.)
    except ValueError:pass
    else:raise AssertionError('Opening was silently admitted')
    assert encode(old)==saved
    retry=point_trial(old,de,dt,p,True);assert encode(retry)==encode(new)
    accepted=publish(old,new)
    try:publish(accepted,new)
    except ValueError:pass
    else:raise AssertionError('Duplicate accepted publication')
    (OUT/'point_checkpoint.json').write_text(encode(accepted)+'\n')
    resumed=decode((OUT/'point_checkpoint.json').read_text())
    follow=point_trial(accepted,-de,dt,p);restart=point_trial(resumed,-de,dt,p)
    assert encode(follow)==encode(restart)
    inconsistent=copy.deepcopy(old);inconsistent['normal']=None
    try:form_direction(inconsistent,p,[0,1,0])
    except ValueError:pass
    else:raise AssertionError('Unassigned existing tensor history was reinterpreted')
    csvwrite('point_history.csv',allrows);csvwrite('point_timestep.csv',comparisons)
    result=dict(passed=True,seconds=time.perf_counter()-start,formation=info,other_preference=info2,
                unresolved_symmetry=ambinfo,tangent_relative_error=tangent_error,covariance=covariance,
                loaded_formation_fine_stress_error=max(formation_errors),independent_formation_error_Pa=formation_error,pressure_tangent_norm=float(np.linalg.norm(new['tangent'][...,3])),
                symmetric_shear_error_over_G=symmetry,perpendicular_memory_error=memory_error,
                max_rate_error_Pa=max(r['rate_error_Pa'] for r in allrows),
                max_energy_defect_J_m3=max(abs(r['ledger_error_J_m3']) for r in allrows),
                maximum_other_plane_excess_Pa=max(r['other_plane_excess_Pa'] for r in allrows),
                max_fine_stress_error=max(r['stress_relative_L2'] for r in comparisons if r['coarse_dt_s']==.00625),
                max_fine_slip_error=max(r['slip_relative_L2'] for r in comparisons if r['coarse_dt_s']==.00625),
                retry_bitwise=True,restart_bitwise=True,duplicate_rejected=True,opening_rejected=True,unassigned_memory_rejected=True)
    (OUT/'point_summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
