#!/usr/bin/env python3
"""Experimental extrinsic recoverable bonds + geometric RSF/contact.

No production dependency or checkpoint edit. Surface damage is not bulk phi.
"""
from pathlib import Path
import copy,json,csv
import numpy as np
from scipy.optimize import brentq
from structural_rsf import G,ETA,P0,ELL,V0,DC,ETA_RAD,RELAX,state_friction
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface'
WORK=ROOT/'build-particle-free/cohesive-interface'
GC=1.;TAUP=3000.;C0=2*GC/TAUP**2;DF=2*GC/TAUP


def write_csv(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def bond(jump,old_damage,reference=0.,sign_hint=1.):
    e=np.asarray(jump)-reference;previous=np.asarray(old_damage)
    damage=np.maximum(previous,np.minimum(abs(e)/DF,1.))
    growing=(abs(e)>DF*previous)&(damage<1)
    mature=damage==1
    stiffness=np.zeros_like(e,dtype=float)
    np.divide(1-damage,C0*damage,out=stiffness,where=(damage>0)&~mature)
    q=stiffness*e;derivative=stiffness.copy();da=np.zeros_like(e,dtype=float)
    derivative=np.where(growing,-TAUP/DF,derivative)
    da=np.where(growing,np.sign(e)/DF,da)
    intact=(damage==0)
    # One-sided graph at first admission, never an artificial compliance.
    q=np.where(intact,TAUP*np.asarray(sign_hint),q)
    derivative=np.where(intact,-TAUP/DF,derivative)
    da=np.where(intact,np.asarray(sign_hint)/DF,da)
    storage=np.where(intact,0.,.5*q*e)
    return damage,q,derivative,da,storage


def initial(n=1,damage=0.):
    return dict(jump=np.zeros(n),damage=np.full(n,damage),Theta=np.ones(n),memory=np.zeros(n),
                reference=np.zeros(n),signed_slip=np.zeros(n),absolute_slip=np.zeros(n),time=0.,publications=0)


def solve(model,old,dt,du):
    """Exact mechanical Schur complement; simultaneous current contact pressure."""
    beta=1/(1+RELAX*dt);N=len(old['jump']);j=np.zeros(N)
    trial_memory=beta*(old['memory']+du)
    trial_stress=model.K@trial_memory
    active=(old['damage']>0)|(abs(trial_stress)>TAUP)
    sign_hint=np.where(trial_stress>=0,1.,-1.)
    def evaluate(increment):
        j=old['jump']+increment
        memory=trial_memory-beta*increment;stress=model.K@memory
        pn=P0-model.N@memory
        if pn.min()<=0:raise ValueError('Closed contact lost')
        v=increment/dt
        mu,muv,theta=state_friction(abs(v),old['Theta'],dt)
        R=np.sign(v)*(pn*mu+ETA_RAD*abs(v))
        a,q,dq,da,storage=bond(j,old['damage'],old['reference'],sign_hint)
        force=stress-q-a*R
        diagonal=dq+da*R+a*(pn*muv+ETA_RAD)/dt
        jacobian=-beta*model.K-np.diag(diagonal)-beta*(a*np.sign(v)*mu)[:,None]*model.N
        return force,jacobian,dict(jump=j.copy(),increment=increment.copy(),damage=a,Theta=theta,memory=memory,reference=old['reference'].copy(),
            signed_slip=old['signed_slip']+increment,absolute_slip=old['absolute_slip']+abs(increment),
            stress=stress,pn=pn,V=v,mu=mu,q=q,storage=storage,time=old['time']+dt,publications=old['publications'])
    iteration=0
    for outer in range(8):
        for inner in range(50):
            iteration+=1;f,J,new=evaluate(j)
            error=max(abs(f[active])) if np.any(active) else 0.
            if error<=1e-8:break
            increment=np.zeros(N);increment[active]=np.linalg.solve(J[np.ix_(active,active)],-f[active])
            scale=1.
            for ls in range(24):
                candidate=j+scale*increment
                # A newly admitted branch must keep its actual sign.
                if np.any((old['jump']+candidate)[(old['damage']==0)&active]*sign_hint[(old['damage']==0)&active]<0):
                    scale*=.5;continue
                try:fc,_,_=evaluate(candidate)
                except ValueError:scale*=.5;continue
                if max(abs(fc[active]))<error or max(abs(fc[active]))<=1e-8:j=candidate;break
                scale*=.5
            else:raise RuntimeError(f'Line search failed, residual={error:g}')
        else:raise RuntimeError(f'Newton limit, residual={error:g}')
        f,J,new=evaluate(j)
        add=(~active)&(abs(new['stress'])>TAUP+1e-8)
        if not np.any(add):break
        active|=add;sign_hint[add]=np.sign(new['stress'][add])
    else:raise RuntimeError('Admission active-set limit')
    assert np.all(new['damage']>=old['damage']) and np.all(new['damage']<=1)
    new['q'][~active]=new['stress'][~active]
    new['active']=active;new['force_error_Pa']=float(max(abs(f[active])) if np.any(active) else 0.)
    new['intact_excess_Pa']=float(max(np.maximum(abs(new['stress'][~active])-TAUP,0)) if np.any(~active) else 0.)
    new['iterations']=iteration;new['tangent']=J
    return new


def account(model,old,new,dt,du):
    w=model.weights;jump=new['increment'];dm=new['memory']-old['memory']
    old_bond=bond(old['jump'],old['damage'],old['reference'])[4]
    bulk=.5*np.dot(w*new['memory'],new['stress']);oldbulk=.5*np.dot(w*old['memory'],model.K@old['memory'])
    cohesive=np.dot(w,new['storage']);oldcohesive=np.dot(w,old_bond)
    fracture=GC*np.dot(w,new['damage']-old['damage'])
    maxwell=RELAX*dt*np.dot(w*new['memory'],new['stress'])
    friction=dt*np.dot(w,new['damage']*new['pn']*new['mu']*abs(new['V']))
    radiation=dt*np.dot(w,new['damage']*ETA_RAD*new['V']**2)
    be_bulk=.5*np.dot(w*dm,model.K@dm)
    # Signed endpoint quadrature error of the recoverable cohesive/damage path.
    surface_remainder=np.dot(w,new['q']*jump)-(cohesive-oldcohesive)-fracture
    endpoint=np.dot(w*new['stress'],du);trapezoid=np.dot(w*.5*(new['stress']+model.K@old['memory']),du)
    defect=endpoint-(bulk-oldbulk+cohesive-oldcohesive+fracture+maxwell+friction+radiation+be_bulk+surface_remainder)
    assert abs(defect)<1e-10*max(1.,abs(endpoint)),defect
    return dict(bulk=bulk,cohesive=cohesive,fracture_increment=fracture,Maxwell=maxwell,friction=friction,radiation=radiation,
                endpoint_work=endpoint,trapezoid_work=trapezoid,bulk_BE=be_bulk,surface_remainder=surface_remainder,ledger_defect=defect)


def commit(old,trial):
    if trial['publications']!=old['publications'] or trial['time']<=old['time']:raise ValueError('Duplicate/stale publication')
    result=copy.deepcopy(trial);result['publications']+=1;return result


class UniformBlock:
    def __init__(self,height=.125):
        self.K=np.array([[G/height]]);self.N=np.zeros((1,1));self.weights=np.ones(1);self.height=height


def layer_DtN(k,height):
    """Exact incompressible plane-strain layer, outer wall clamped.

    Stable exponential basis for (d_y²-k²)² psi=0. All vectors are in global
    x,y coordinates; this is the upper layer's outward interface traction.
    """
    z=k*height;ex=np.exp(-z)
    matrix=np.array([[1,0,ex,z*ex],[-1,1,ex,(z-1)*ex],[ex,z*ex,1,0],[-ex,(1-z)*ex,1,-1]],complex)
    answer=np.empty((2,2),complex)
    for j in range(2):
        rhs=np.array([1j/k if j==1 else 0.,1/k if j==0 else 0.,0,0])
        a,b,c,d=np.linalg.solve(matrix,rhs)
        psi=a+(c+d*z)*ex;pz=-a+b+(c+d*(z-1))*ex
        pzz=a-2*b+(c+d*(z-2))*ex;pzzz=-a+3*b+(c+d*(z-3))*ex
        answer[:,j]=[-G*k*k*(pzz+psi),-1j*G*k*k*(pzzz-3*pz)]
    assert np.linalg.norm(answer-answer.conj().T)<1e-7*max(1.,np.linalg.norm(answer))
    assert np.linalg.eigvalsh(answer).min()>0
    return answer


class Layers:
    def __init__(self,n,length=2.,top=.075,bottom=.05):
        self.length=length;self.n=n;self.x=(np.arange(n)+.5)*length/n-length/2
        self.weights=np.full(n,length/n);self.top=top;self.bottom=bottom
        frequencies=2*np.pi*np.fft.fftfreq(n,d=length/n)
        kt=np.zeros(n,complex);kn=np.zeros(n,complex);kt[0]=G/(top+bottom)
        reflection=np.diag([1.,-1.])
        for j,k in enumerate(frequencies):
            if k==0:continue
            up=layer_DtN(abs(k),top);down=reflection@layer_DtN(abs(k),bottom)@reflection
            full=np.linalg.inv(np.linalg.inv(up)+np.linalg.inv(down))
            kt[j]=full[0,0];kn[j]=full[1,0] if k>0 else full[1,0].conjugate()
        if n%2==0:kn[n//2]=0. # Real Nyquist convention; no representable sine partner.
        self.symbol_t=kt;self.symbol_n=kn
        raw_K=np.fft.ifft(kt[:,None]*np.fft.fft(np.eye(n),axis=0),axis=0).real
        raw_N=np.fft.ifft(kn[:,None]*np.fft.fft(np.eye(n),axis=0),axis=0).real
        # Enforce analytically exact reciprocal/odd circulant structure at
        # construction, rather than let separate column FFT roundoff break it.
        tk=np.fft.ifft(kt).real;nk=np.fft.ifft(kn).real
        distance=(np.arange(n)[:,None]-np.arange(n)[None,:])%n
        absolute=np.minimum(distance,n-distance)
        self.K=tk[absolute]
        self.N=np.where(distance<=n//2,1.,-1.)*nk[absolute]
        self.N[distance==0]=0.
        if n%2==0:self.N[distance==n//2]=0.
        self.reciprocity_roundoff=dict(K_max_Pa_m=float(np.max(abs(raw_K-self.K))),N_max_Pa_m=float(np.max(abs(raw_N-self.N))))
        assert np.linalg.norm(raw_K-self.K)<=32*n*np.finfo(float).eps*np.linalg.norm(raw_K)
        assert np.linalg.norm(raw_N-self.N)<=32*n*np.finfo(float).eps*np.linalg.norm(raw_N)
        assert np.linalg.norm(self.K-self.K.T)<1e-6
        assert np.linalg.eigvalsh(self.K).min()>0
