#!/usr/bin/env python3
"""Experimental material-plane RSF closure; no production or native edits."""
from pathlib import Path
import copy
import numpy as np
from scipy.optimize import brentq
from feng_direction import (G, ETA, P0, MU, tensor, inner, directions, select,
                           tolerance, h_of, rotation, deviator)
from reconcile_feng_rsf import A_RSF, B_RSF, V0, DC, ELL, ETA_RAD, friction, chi_center

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/directional_continuation'
WORK=ROOT/'build-particle-free/directional-continuation'
RELAX=G/ETA


def plane(normal):
    n=np.asarray(normal,dtype=float);n=n/np.linalg.norm(n)
    if abs(n[2])>1e-14:raise ValueError('Only an in-plane structural normal is supported')
    t=np.array([n[1],-n[0],0.])
    return n,t,(np.outer(n,t)+np.outer(t,n))/2


def initial(phi,normal=None):
    return dict(T=tensor(),A=tensor(),K=tensor(),Theta=1.,phi=phi,
                normal=None if normal is None else plane(normal)[0],clock=0.,publications=0)


def form_direction(old,p,preference=None):
    """Conditional first-record admission at a specified formation event.

    Maximizes instantaneous speed on the unchanged incoming state. This is
    not a phase/birth trigger. Existing tensors/clocks must never be reset.
    """
    if old['normal'] is not None:raise ValueError('Existing material plane cannot be replaced by a new maximum')
    if np.any(old['A']!=0) or np.any(old['K']!=0):
        raise ValueError('Unassigned material carries fault history; an explicit structural transfer is required')
    tau=old['T']-2*G*old['A'];Q=tensor() # virgin fault history; no invented intact Q reaction
    if p<=max(np.linalg.eigvalsh(tau[:2,:2])):raise ValueError('New-direction test requires closed contact')
    if np.linalg.norm(tau[:2,:2]-np.eye(2)*np.trace(tau[:2,:2])/2)==0:
        if preference is None:return None,dict(reason='continuum symmetry',candidates=[])
        answer=copy.deepcopy(old);answer['normal']=plane(preference)[0]
        return answer,dict(reason='declared geometry in zero-drive symmetry',V=0.,candidates=[])
    def force(logq):
        q=np.exp(logq);mu=friction(q,old['Theta'])[0]
        return directions(tau,Q,p,mu)['maximum']-ETA_RAD*V0*q
    logq=brentq(force,-100.,30.,xtol=1e-13)
    V=V0*np.exp(logq);mu=friction(np.exp(logq),old['Theta'])[0]
    candidates=directions(tau,Q,p,mu);chosen,reason=select(candidates,preference)
    info=dict(reason=reason,V=V,mu=mu,global_error_Pa=candidates['maximum']-ETA_RAD*V,
              candidates=[c['normal'].tolist() for c in candidates['candidates']])
    if chosen is None:return None,info
    answer=copy.deepcopy(old);answer['normal']=chosen['normal'].copy()
    return answer,info


def state_friction(V,Theta_old,dt):
    V=np.asarray(V);Theta=(Theta_old+V0/DC*dt)/(1+V*dt/DC)
    exponent=np.exp((MU+B_RSF*np.log(Theta))/A_RSF)
    z=V/(2*V0)*exponent;fraction=z/np.hypot(1.,z)
    mu=A_RSF*np.arcsinh(z)
    slope=np.zeros_like(V,dtype=float)
    np.divide(A_RSF*fraction,V,out=slope,where=V>0)
    slope=np.where(V>0,slope,A_RSF*exponent/(2*V0))
    slope-=B_RSF*fraction*(dt/DC)/(1+V*dt/DC)
    return mu,slope,Theta


def solve_speed(drive,stiffness,pn,Theta_old,dt):
    """Vectorized monotone scalar solve with explicit global slope bound.

    stiffness includes radiation plus the appropriate elastic/cohesive term.
    No speed floor or physical force tolerance enlargement.
    """
    drive,stiffness,pn,Theta_old=np.broadcast_arrays(np.asarray(drive,dtype=float),
        np.asarray(stiffness,dtype=float),np.asarray(pn,dtype=float),np.asarray(Theta_old,dtype=float))
    if np.any(pn<=0):raise ValueError('Closed contact lost on the recorded plane')
    if np.any(Theta_old<=0):raise ValueError('Nonpositive accepted frictional age')
    lower_slope=stiffness-pn*B_RSF*dt/DC
    if np.any(lower_slope<=0):raise ValueError('Monotonicity bound fails; possible kinetic branches require analysis, not forced acceptance')
    V=np.zeros_like(drive);positive=drive>0
    if not np.any(positive):
        mu,slope,Theta=state_friction(V,Theta_old,dt)
        return V,mu,stiffness+pn*slope,Theta,0.
    f=drive[positive];c=stiffness[positive];p=pn[positive];age=Theta_old[positive]
    upper=np.log(f/c)
    xforce=f/(p*A_RSF)
    log_sinh=xforce-np.log(2.)+np.log(-np.expm1(-2*xforce))
    estimate=np.log(2*V0)+log_sinh-(MU+B_RSF*np.log(age+V0/DC*dt))/A_RSF
    lo=np.minimum(estimate-10.,upper-100.)
    x=np.minimum(estimate,upper);hi=upper.copy()
    for iteration in range(90):
        velocity=np.exp(x);mu,slope,age_new=state_friction(velocity,age,dt)
        residual=c*velocity+p*mu-f
        if max(abs(residual))<=1e-9:break
        hi=np.where(residual>0,x,hi);lo=np.where(residual<0,x,lo)
        trial=x-residual/((c+p*slope)*velocity)
        # Leave converged entries unchanged; their bracket can have collapsed.
        trial=np.where((trial<=lo)|(trial>=hi)|~np.isfinite(trial),.5*(lo+hi),trial)
        x=np.where(abs(residual)<=1e-10,x,trial)
    else:raise RuntimeError('Bounded scalar rate solve did not converge')
    V[positive]=np.exp(x)
    mu,slope,Theta=state_friction(V,Theta_old,dt)
    error=float(np.max(abs(stiffness*V+pn*mu-drive)))
    if error>1e-8:raise RuntimeError(f'Rate residual {error} Pa exceeds retained allowance')
    return V,mu,stiffness+pn*slope,Theta,error


def point_trial(old,de,dt,p,tangent=False):
    if old['normal'] is None:raise ValueError('A fault record or explicit formation event is required')
    phi=old['phi'];h=h_of(phi);chi=chi_center(phi)
    n,t,S=plane(old['normal']);beta=1/(1+RELAX*dt)
    T=beta*(old['T']+2*G*de);trial=T-2*G*beta*old['A'];qref=2*G*beta*old['A']/h
    drive=inner(trial-qref,S);pn=p-inner(trial,np.outer(n,n))
    kappa=G*beta*dt*chi*(1+1/h)
    V,mu,slope,Theta,error=solve_speed(abs(drive),kappa+ETA_RAD,pn,old['Theta'],dt)
    V,mu,slope,Theta=map(float,[V,mu,slope,Theta]);sgn=np.sign(drive);v=sgn*V
    dK=dt*chi*v*S;A=beta*(old['A']+dK);tau=T-2*G*A;Q=2*G*A/h
    result=dict(T=T,A=A,K=old['K']+dK,Theta=Theta,normal=n.copy(),phi=phi,
                clock=old['clock']+dt,publications=old['publications'],tau=tau,Q=Q,
                signed_V=v,V=V,p=p,pn=pn,chi=chi,mu=mu,rate_error_Pa=error,W=chi*v*S)
    if tangent:
        columns=[]
        for j,E in enumerate([deviator(xx=1),deviator(yy=1),deviator(gamma=1),tensor()]):
            dtrial=2*G*beta*E;dp=G if j==3 else 0.
            dv=(inner(dtrial,S)-sgn*mu*(dp-inner(dtrial,np.outer(n,n))))/slope
            columns.append((dtrial-2*G*beta*dt*chi*S*dv)/G)
        result['tangent']=np.stack(columns,axis=-1)
    return result


def publish(old,trial):
    if trial['publications']!=old['publications'] or trial['clock']<=old['clock']:
        raise ValueError('Stale or duplicate point publication')
    answer=copy.deepcopy(trial);answer['publications']=old['publications']+1
    return answer


def point_work(old,new,de,dt):
    h=h_of(old['phi']);tau0=old['T']-2*G*old['A'];tau=new['tau'];A=new['A']
    energy0=inner(tau0,tau0)/(4*G)+G*inner(old['A'],old['A'])/h
    energy=inner(tau,tau)/(4*G)+G*inner(A,A)/h
    maxwell=dt*(inner(tau,tau)+h*inner(new['Q'],new['Q']))/(2*ETA) if np.isfinite(h) else dt*inner(tau,tau)/(2*ETA)
    friction_loss=dt*new['chi']*new['pn']*new['mu']*new['V']
    radiation=dt*new['chi']*ETA_RAD*new['V']**2
    remainder=inner(tau-tau0,tau-tau0)/(4*G)+G*inner(A-old['A'],A-old['A'])/h
    work=inner(tau,de);defect=work-(energy-energy0+maxwell+friction_loss+radiation+remainder)
    assert abs(defect)<=1e-10*max(1.,abs(work)),defect
    return dict(work_J_m3=work,storage_change_J_m3=energy-energy0,Maxwell_J_m3=maxwell,
                friction_J_m3=friction_loss,radiation_J_m3=radiation,BE_J_m3=remainder,ledger_defect_J_m3=defect)
