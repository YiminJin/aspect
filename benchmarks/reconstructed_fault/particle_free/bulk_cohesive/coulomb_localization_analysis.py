#!/usr/bin/env python3
"""Actual mixed constitutive symbols and an independent active-plane rate limit."""
from pathlib import Path
import csv,json
import numpy as np
from scipy.optimize import brentq, minimize_scalar
ROOT=Path(__file__).resolve().parents[4]
WORK=ROOT/'build-particle-free/coulomb-localization'
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/coulomb_localization'
def read(path):return list(csv.DictReader(path.open()))
def write(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def save(name,data):(OUT/name).write_text(json.dumps(data,indent=2)+'\n')
def columns(row):return np.array([[float(row[f'd{j}_{key}']) for j in range(5)] for key in ['xx','yy','zz','xy','Y']])
def rate_columns(row):
    phi=float(row['phi']);h=float(row['h']);mu=float(row.get('mu',.1))
    strain=[]
    for x,y,xy in [(1.,0.,0.),(0.,1.,0.),(0.,0.,.5)]:
        E=np.array([[x,xy,0],[xy,y,0],[0,0,0.]])-np.eye(3)*(x+y)/3;strain.append(E)
    active=[i for i in range(2) if int(row['active'])&(1<<i)];M=[];N=[]
    for i in active:
        theta=float(row['theta_plus' if i==0 else 'theta_minus']);k=1 if i==0 else -1
        t=np.array([np.cos(theta),np.sin(theta),0]);n=np.array([-np.sin(theta),np.cos(theta),0])
        m=k*.5*(np.outer(t,n)+np.outer(n,t));M.append(m);N.append(m+mu*np.outer(n,n))
    C=np.zeros((5,5))
    if h==0. or not active:
        for j,E in enumerate(strain):C[:4,j]=[2*E[0,0],2*E[1,1],2*E[2,2],2*E[0,1]]
        return C
    M,N=np.array(M),np.array(N)
    H=np.einsum('aij,bij->ab',N,M)+np.einsum('aij,bij->ab',M,M)/h
    for j,E in enumerate(strain):
        lam=np.linalg.solve(H,np.einsum('aij,ij->a',N,E));d=2*(E-np.einsum('a,aij->ij',lam,M))
        C[:4,j]=[d[0,0],d[1,1],d[2,2],d[0,1]]
    dp=mu*np.einsum('a,aij->ij',np.linalg.solve(H,np.ones(len(active))),M)
    C[:4,3]=[dp[0,0],dp[1,1],dp[2,2],dp[0,1]]
    return C
def symbol(C,theta,kell=None):
    theta=np.asarray(theta);n=np.stack([np.cos(theta),np.sin(theta)],axis=-1);a=np.stack([-np.sin(theta),np.cos(theta)],axis=-1)
    E=np.stack([a[...,0]*n[...,0],a[...,1]*n[...,1],a[...,0]*n[...,1]+a[...,1]*n[...,0]],axis=-1)
    s=E@C[:4,:3].T
    traction=np.stack([s[...,0]*n[...,0]+s[...,3]*n[...,1],s[...,3]*n[...,0]+s[...,1]*n[...,1]],axis=-1)
    b=np.stack([(C[0,3]-1)*n[...,0]+C[3,3]*n[...,1],C[3,3]*n[...,0]+(C[1,3]-1)*n[...,1]],axis=-1)
    if kell is not None:
        c=np.stack([C[0,4]*n[...,0]+C[3,4]*n[...,1],C[3,4]*n[...,0]+C[1,4]*n[...,1]],axis=-1)
        denominator=2*kell*kell-C[4,4]
        traction=traction+c*(E@C[4,:3])[...,None]/denominator
        b=b+c*C[4,3]/denominator
    aa=np.sum(a*traction,axis=-1);na=np.sum(n*traction,axis=-1);ab=np.sum(a*b,axis=-1);nb=np.sum(n*b,axis=-1)
    return aa-ab*na/nb,aa,aa*nb-ab*na,nb
def minimum(C):
    theta=np.linspace(0,np.pi,4097);L,fixed,det,b=symbol(C,theta);index=np.argmin(L)
    result=minimize_scalar(lambda x:float(symbol(C,x)[0]),bounds=(theta[max(0,index-1)],theta[min(len(theta)-1,index+1)]),method='bounded',options={'xatol':1e-14})
    roots=[]
    for i in np.where(det[:-1]*det[1:]<0)[0]:roots.append(brentq(lambda x:float(symbol(C,x)[2]),theta[i],theta[i+1],xtol=1e-14))
    return float(result.fun),float(result.x),float(min(fixed)),roots,float(min(abs(b)))
def points():
    OUT.mkdir(exist_ok=True,parents=True);rows=read(WORK/'point-states.csv');results=[]
    for row in rows:
        actual=columns(row);rate=rate_columns(row);a=minimum(actual);r=minimum(rate)
        results.append({**{k:row[k] for k in ['kind','normal_sign','mu','phi','h','step','dt','active','Y','Wxx','Wxy']},
                        'phase_obstacle_residual_Pa':8-float(row['Y']),
                        'algorithmic_mixed_min_over_G':a[0],'algorithmic_normal_rad':a[1],'algorithmic_fixed_P_min_over_G':a[2],
                        'rate_mixed_min_over_G':r[0],'rate_normal_rad':r[1],'rate_fixed_P_min_over_G':r[2],
                        'rate_localization_normals_rad':';'.join(map(str,r[3])),'min_pressure_normal_action':a[4]})
    write('point-symbols.csv',results)
    mu=.1;root=brentq(lambda t:mu-2*t-3*mu*t*t+2*t**3,0.,mu)
    excess=4*root*(mu-root)/(1+root*root)**2;hc=1/excess
    pc=brentq(lambda p:4*p*(1+p)/(1-p)**2-hc,.1,.99)
    save('analytic-critical.json',{'mu':mu,'critical_h':hc,'critical_phi_m4':pc,'critical_wave_normal_to_active_tangent_rad':float(np.arctan(root)),
         'criterion':'Single active system, continuous rate, full incompressible pressure elimination. Not fixed-pressure stiffness.'})
    worst=min(results,key=lambda a:a['rate_mixed_min_over_G']);save('point-summary.json',{'states':len(results),'worst':worst,
        'negative_rate_states':sum(float(r['rate_mixed_min_over_G'])<-1e-8 for r in results),'negative_algorithmic_states':sum(float(r['algorithmic_mixed_min_over_G'])<-1e-8 for r in results)})
    print(json.dumps({'critical_h':hc,'critical_phi':pc,'worst':worst},indent=2))
if __name__=='__main__':points()
