#!/usr/bin/env python3
"""Nonlinear periodic oblique laminate with exact mixed equilibrium elimination.

Physical material plane stays horizontal. Perturbation layers are oblique;
pressure varies and is solved jointly, not prescribed or lagged.
"""
import json,time
import numpy as np
from scipy.linalg import expm
from structural_rsf import G,ETA,P0,RELAX,DC,V0,ETA_RAD,plane,inner,state_friction,tensor
from structural_sliding import OUT,write_csv
from structural_mixed import base,reduced


def run(N,dt):
    start=time.perf_counter();angle=.04;Vbase=.0005;chi=16.;length=.25
    k=np.array([np.cos(angle),np.sin(angle),0.]);q=np.array([-k[1],k[0],0.]);E=(np.outer(k,q)+np.outer(q,k))/2
    n,t,S=plane([0,1,0]);P=np.outer(n,n);kk=np.outer(k,k);ES=inner(E,S);projected=S-2*ES*E;R=2*ES*E
    x=np.arange(N)*length/N;age0=V0/Vbase
    mu=state_friction(np.array(Vbase),np.array(age0),dt)[0];tau0=2*(P0*mu+ETA_RAD*Vbase)*S
    old=dict(tau=np.tile(tau0,(N,1,1)),A=np.zeros((N,3,3)),K=np.zeros((N,3,3)),Theta=age0*(1+.01*np.cos(2*np.pi*x/length)))
    old['T']=old['tau'].copy();rows=[];max_res=0.;max_energy=0.;max_eq=0.;profiles=[]
    initial=old['Theta'].copy();beta=1/(1+RELAX*dt);c=2*G*beta*dt*chi
    dlocal=inner(S,projected);dmean=inner(S,R);plocal=-c*inner(kk-P,projected);pmean=c*(inner(kk,projected)+inner(P,R))
    debar=dt*(RELAX*tau0/(2*G)+chi*Vbase*S)
    M,BB,CC=reduced(base(1.,Vbase),angle)
    for step in range(round(1./dt)):
        trial=beta*(old['tau']+2*G*debar);mean=trial.mean(axis=0)
        ptrial=P0+np.einsum('nij,ij->n',trial-mean,kk)-np.einsum('nij,ij->n',trial,P)
        drive=np.einsum('nij,ij->n',trial,S);v=np.full(N,Vbase)
        for iteration in range(25):
            pn=ptrial+plocal*v+pmean*v.mean();assert pn.min()>0
            mu,slope,age=state_friction(abs(v),old['Theta'],dt)
            residual=drive-c*(dlocal*v+dmean*v.mean())-np.sign(v)*pn*mu-ETA_RAD*v
            if np.max(abs(residual))<=1e-8:break
            diagonal=-c*dlocal-np.sign(v)*mu*plocal-pn*slope-ETA_RAD
            rankone=-c*dmean-np.sign(v)*mu*pmean
            rhs=-residual/diagonal;u=rankone/diagonal
            increment=rhs-u*rhs.mean()/(1+u.mean());v+=increment
        else:raise RuntimeError('Nonlinear mixed laminate did not converge')
        tau=trial-c*(v[:,None,None]*projected+v.mean()*R)
        p=P0+np.einsum('nij,ij->n',tau-tau.mean(axis=0),kk)
        dgamma=2*dt*chi*ES*(v-v.mean());de=debar+dgamma[:,None,None]*E
        A=beta*(old['A']+dt*chi*v[:,None,None]*S);K=old['K']+dt*chi*v[:,None,None]*S
        T=beta*(old['T']+2*G*de)
        assert np.max(abs(T-2*G*A-tau))<1e-8
        storage=np.mean(np.sum(tau*tau,axis=(1,2)))/(4*G);storage0=np.mean(np.sum(old['tau']**2,axis=(1,2)))/(4*G)
        maxwell=dt*np.mean(np.sum(tau*tau,axis=(1,2)))/(2*ETA)
        friction=dt*chi*np.mean(pn*mu*abs(v));radiation=dt*chi*ETA_RAD*np.mean(v*v)
        remainder=np.mean(np.sum((tau-old['tau'])**2,axis=(1,2)))/(4*G)
        work=inner(tau.mean(axis=0),debar);defect=work-(storage-storage0+maxwell+friction+radiation+remainder)
        assert abs(defect)<1e-10*max(1.,abs(work)),defect
        normal=np.einsum('nij,ij->n',tau,kk)-p
        shear=np.einsum('nij,ij->n',tau,E)
        eq=max(np.ptp(normal),np.ptp(shear));assert eq<1e-8,eq
        max_res=max(max_res,max(abs(residual)));max_eq=max(max_eq,eq);max_energy=max(max_energy,abs(defect))
        time_now=(step+1)*dt
        if (step+1)%round(.01/dt)==0:
            linear=expm(M*time_now)@np.array([0,0,0,0,0,.01])
            predicted=age0*(1+linear[-1]*np.cos(2*np.pi*x/length))
            rows.append(dict(cells=N,dt_s=dt,time_s=time_now,Theta_cosine=2*np.mean((age/age0-1)*np.cos(2*np.pi*x/length)),linear_Theta_cosine=linear[-1],
                pressure_range_Pa=np.ptp(p),normal_pressure_range_Pa=np.ptp(pn),V_range_m_s=np.ptp(v),energy_defect_J_m3=defect,
                nonlinear_linear_difference=np.linalg.norm(age-predicted)/np.linalg.norm(predicted-age0)))
        old=dict(tau=tau,A=A,K=K,T=T,Theta=age)
    for j in range(N):profiles.append(dict(cells=N,dt_s=dt,coordinate_m=x[j],Theta=age[j],pressure_Pa=p[j],pn_Pa=pn[j],V_m_s=v[j],tau_xx_Pa=tau[j,0,0],tau_xy_Pa=tau[j,0,1],A_xy=A[j,0,1],K_xy=K[j,0,1]))
    return rows,profiles,dict(cells=N,dt_s=dt,seconds=time.perf_counter()-start,max_force_residual_Pa=max_res,max_equilibrium_error_Pa=max_eq,max_ledger_error_J_m3=max_energy,
         final_state_amplitude=rows[-1]['Theta_cosine'],final_linear_amplitude=rows[-1]['linear_Theta_cosine'],pressure_range_Pa=rows[-1]['pressure_range_Pa'],pn_range_Pa=rows[-1]['normal_pressure_range_Pa'])


def main():
    rows=[];profiles=[];results=[]
    for cells,dt in [(32,.002),(64,.002),(128,.002),(128,.001),(128,.0005)]:
        r,p,m=run(cells,dt);rows+=r;profiles+=p;results.append(m);print(json.dumps(m),flush=True)
    write_csv('oblique_history.csv',rows);write_csv('oblique_profiles.csv',profiles)
    (OUT/'oblique_summary.json').write_text(json.dumps(results,indent=2)+'\n')

if __name__=='__main__':main()
