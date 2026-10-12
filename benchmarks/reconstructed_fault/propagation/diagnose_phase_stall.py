#!/usr/bin/env python3
"""Read-only exact-tangent and independently reordered extended arithmetic audit."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy import sparse
from total_energy_step import G,M,N,ELL,FRACTURE_SCALE,basis,surface_basis
from total_energy_time_comparison import make_energy

def thomas(rhs,ns,dtype):
    rhs=np.asarray(rhs,dtype=dtype).copy();ds=dtype(1)/ns
    diag=np.full(ns+1,2*ds/3,dtype=dtype);diag[[0,-1]]=ds/3
    off=ds/6
    for i in range(1,ns+1):q=off/diag[i-1];diag[i]-=q*off;rhs[i]-=q*rhs[i-1]
    rhs[-1]/=diag[-1]
    for i in range(ns-1,-1,-1):rhs[i]=(rhs[i]-off*rhs[i+1])/diag[i]
    return rhs

def gradient(phase,r,Z,energy,dtype):
    B=energy.B.astype(dtype);T=energy.T.astype(dtype);S=energy.S.astype(dtype)
    p=phase.astype(dtype);m=dtype(M);Gd=dtype(G)
    l=(energy.L@p).astype(dtype);rr=(energy.R@p).astype(dtype)
    u,v=1-l,1-rr;z=(v-u)/u;small=abs(z)<dtype('1e-3')
    ratio=np.empty_like(z);prime=np.empty_like(z);q=z[small]
    ratio[small]=sum(dtype((-1)**k)*q**k/dtype(k+1) for k in range(20))
    prime[small]=sum(dtype((-1)**k)*k*q**(k-1)/dtype(k+1) for k in range(1,20))
    q=z[~small];ratio[~small]=np.log1p(q)/q;prime[~small]=(q/(1+q)-np.log1p(q))/q**2
    vals=m/N*(2/(u*v)-3*ratio/u+1)
    left=m/N*(2/(u*u*v)-3/u**2*(ratio+(1+z)*prime));right=m/N*(2/(u*v*v)+3/u**2*prime)
    J=sparse.csr_matrix((np.ones(len(l),dtype=dtype),(np.repeat(np.arange(len(energy.xs)),N),np.arange(len(l)))),shape=(len(energy.xs),len(l)))@(sparse.diags(left)@energy.L.astype(dtype)+sparse.diags(right)@energy.R.astype(dtype))
    ws=energy.ws.astype(dtype);wb=energy.wb.astype(dtype)
    direct=vals.reshape(-1,N).sum(axis=1,dtype=dtype)
    I=thomas(S.T@(ws*direct),energy.ns,dtype);Ib=T@I;Is=S@I
    ph=B@p;hh=m*ph*(1+ph)/(1-ph)**2;hp=m*(4/(1-ph)**3-3/(1-ph)**2)
    zb=T@Z.astype(dtype);zs=S@Z.astype(dtype)
    tau=Gd*r.astype(dtype)-zb*hh/Ib
    c=-wb*zb*tau/Gd
    local=B.T@(c*hp/Ib)
    forceI=T.T@(-c*hh/Ib**2)+S.T@(-ws*zs*zs/(2*Gd*Is*Is))
    adjoint=J.T@(ws*(S@thomas(forceI,energy.ns,dtype)))
    pc=(np.arange(3*N)+.5)/(3*N);px,py=np.meshgrid(pc,pc,indexing='ij')
    P,Dx,Dy=[v.astype(dtype) for v in basis(px,py)]
    fscale=dtype(FRACTURE_SCALE);factor=2*fscale*dtype(ELL)**2/dtype((3*N)**2)
    fract=P.T@np.full(P.shape[0],fscale/(3*N)**2,dtype=dtype)+factor*(Dx.T@(Dx@p)+Dy.T@(Dy@p))
    return local+adjoint+fract,dict(local=local,adjoint=adjoint,fracture=fract,I=I)

def shear_encoding(file,x,y):
    a=np.genfromtxt(file,delimiter=',',names=True)
    grid=np.zeros((2*N+1,2*N+1,2))
    for v in a:grid[round(v['x']*2*N),round(v['y']*2*N),int(v['component'])]=v['d_R']
    ix=np.minimum((x*N).astype(int),N-1);iy=np.minimum((y*N).astype(int),N-1);s=x*N-ix;t=y*N-iy
    def b(z):return np.array([(1-z)*(1-2*z),4*z*(1-z),z*(2*z-1)])
    def d(z):return N*np.array([-3+4*z,4-8*z,-1+4*z])
    bs,bt,ds,dt=b(s),b(t),d(s),d(t)
    value=np.zeros_like(x)
    for i in range(3):
        for j in range(3):value+=bs[i]*dt[j]*grid[2*ix+i,2*iy+j,0]+ds[i]*bt[j]*grid[2*ix+i,2*iy+j,1]
    return value

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run');ap.add_argument('--step',type=int,default=20);a=ap.parse_args();run=Path(a.run)
    data=np.genfromtxt(run/f'phase-stall-{a.step}.csv',delimiter=',',names=True);H=np.loadtxt(run/f'phase-stall-{a.step}-jacobian.csv',delimiter=',')
    data=data[np.argsort(data['i'].astype(int))]
    x,rhs,lower=data['phase'],data['rhs'],data['lower'];pdas=x+rhs/abs(np.diag(H));free=(pdas>lower)&(pdas<1)
    indices=np.flatnonzero(free);hf=H[np.ix_(free,free)];eigen=np.linalg.eigvalsh(hf)
    correction=np.zeros_like(x);correction[free]=np.linalg.solve(hf,rhs[free]);projected=np.where(x==lower,np.maximum(rhs,0.),rhs)
    order=np.lexsort((data['y'],data['x']));energy=make_energy(True)
    phase=x[order];r=shear_encoding(run/f'fe-encoding-{a.step-1}.csv',energy.x,energy.y)
    surf=np.genfromtxt(run/f'fe-surface-{a.step-1}.csv',delimiter=',',names=True)
    gd,terms=gradient(phase,r,surf['Z'],energy,np.float64);gl,_=gradient(phase,r,surf['Z'],energy,np.longdouble)
    # Reorder to native dofs; compare the independently evaluated total energy.
    inverse=np.argsort(order);gd=gd[inverse];gl=gl[inverse]
    rounding=abs(gd-gl).astype(float);native=abs(rhs+gl).astype(float)
    input_floor=abs(H)@np.maximum(abs(np.spacing(x)),np.finfo(float).tiny)
    perturb_contact=[]
    for direction in [-1,1]:
        xp=np.nextafter(x,np.full_like(x,direction*np.inf));rp=rhs-H@(xp-x)
        perturb_contact.append(int(np.sum(((xp+rp/abs(np.diag(H))>lower)&(xp+rp/abs(np.diag(H))<1))!=free)))
    worst=int(np.argmax(abs(projected)))
    result=dict(kkt=max(abs(projected)),target=float(data['target'][0]),worst=dict(i=worst,x=data['x'][worst],y=data['y'][worst],phase=x[worst]),free=len(indices),min_eigenvalue=eigen[0],max_eigenvalue=eigen[-1],negative_modes=int(sum(eigen<0)),condition=float(max(abs(eigen))/min(abs(eigen))),max_exact_correction=max(abs(correction)),max_search_correction=max(abs(data['search_update'])),max_correction_ulps=max(abs(correction)/np.maximum(abs(np.spacing(x)),np.finfo(float).tiny)),active_changes_one_ulp=perturb_contact,independent_double_longdouble_max=max(rounding),native_longdouble_difference=max(native),free_native_longdouble_difference=max(native[free]),input_one_ulp_floor=max(input_floor),free_input_one_ulp_floor=max(input_floor[free]),independent_KKT=max(abs(np.where(x==lower,np.minimum(gl,0.),gl))),longdouble_mantissa=np.finfo(np.longdouble).nmant)
    # Independently verify feasible negative-curvature energy reductions.
    ev,vectors=np.linalg.eigh(hf);mode=np.zeros_like(x);mode[free]=vectors[:,0]
    def independent_energy(xx):
        p=xx[order];profile=energy.profile(p);zb=energy.T@surf['Z'];zs=energy.S@surf['Z']
        tau=G*r-zb*profile['chi']
        return energy.wb@(tau*tau)/(2*G)+energy.ws@(zs*zs/profile['isurf'])/(2*G)+energy.fracture(p)
    base=independent_energy(x);curvature=[]
    for amplitude in [-1e-4,1e-4]:
        trial=x+amplitude*mode;assert min(trial-lower)>=0 and max(trial)<1
        curvature.append(dict(amplitude=amplitude,energy_change=float(independent_energy(trial)-base),quadratic_prediction=float(.5*ev[0]*amplitude**2)))
    result['feasible_negative_curvature']=curvature
    result['newton_slope']=float(-rhs@correction)
    out=Path(__file__).resolve().parent/'localization_evidence';(out/'stall-assessment.json').write_text(json.dumps(result,indent=2,default=float));print(json.dumps(result,indent=2,default=float))
