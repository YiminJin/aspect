#!/usr/bin/env python3
"""Independent 1-D strip: equilibrium envelopes and native-checked BE evolution.

No surface constitutive law is substituted for the bulk law. The homogeneous
traction variable enforces 1-D equilibrium; local sticking/sliding and the Q1
physical obstacle are retained. The actual C++ point law checks each accepted
strip update through a small, separately compiled C interface.
"""
import argparse,ctypes,json
from pathlib import Path
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq
from scipy.linalg import solve

G,ETA,R=1.e6,1.e8,1000.
SETS={'retained':dict(Gc=26.6666666667,ell=.15625,drop=1000.),
      'candidate':dict(Gc=1.,ell=.046875,drop=2000.)}
HISTORY_HEADER='step,time_s,grip_increment_m,traction_Pa,accumulated_slip_m,elastic_slip_m,peak_phi,bulk_J_m2,cohesive_J_m2,fracture_J_m2,endpoint_work_J_m2,trapezoidal_work_J_m2,bulk_viscous_J_m2,cohesive_viscous_J_m2,friction_J_m2,endpoint_remainder_J_m2,trapezoidal_remainder_J_m2,KKT_scaled,stress_loss_J_m2,traction_loss_J_m2,phase_remainder_J_m2,fracture_loss_J_m2,phase_residual_work_J_m2'

def compliance(phi,m):
    return m*phi*(1+phi)/(1-phi)**2,m*(1+3*phi)/(1-phi)**3,6*m*(1+phi)/(1-phi)**4

def seed_table(ell,peak=.6,n=5000):
    theta=np.linspace(0,np.pi/2,n);phi=peak*np.cos(theta)**2
    radius=np.r_[0.,cumulative_trapezoid(2*ell*(1-phi)/np.sqrt((3-peak)/(1+peak)-phi),theta)]
    phi[-1]=0.
    return radius,phi

def envelope(p,order):
    z,w=np.polynomial.legendre.leggauss(order);theta=(z+1)*np.pi/4;w=w*np.pi/4
    # Endpoint refinement also resolves the finite-slip limit analytically.
    peak=np.unique(np.r_[np.linspace(1e-6,.99,1601),1-np.geomspace(.01,1e-6,801)])
    phi=peak[:,None]*np.cos(theta)**2
    v=2*p['ell']*(1-phi)/np.sqrt((3-peak[:,None])/(1+peak[:,None])-phi)
    h,_,_=compliance(phi,p['m'])
    q=p['drop']*(1-peak)/np.sqrt(1+peak)
    I=2*np.sum(w*h*v,axis=1);slip=q*I/G
    f=phi-peak[:,None]*h/(p['m']*peak[:,None]*(1+peak[:,None])/(1-peak[:,None])**2)
    fracture=2*p['Kf']*np.sum(w*(phi+f)*v,axis=1)
    cohesive=q*q*I/(2*G)
    work=np.r_[0.,cumulative_trapezoid(q,slip)]
    return np.stack([peak,R+q,q,slip,I,fracture,cohesive,work],axis=-1)

class Strip:
    def __init__(self,p,library,n=256):
        self.p=p;self.n=n;self.x=np.linspace(0,1,n+1);self.dy=1/n
        z,w=np.polynomial.legendre.leggauss(8);self.N=np.stack([(1-z)/2,(1+z)/2],axis=-1)
        self.weights=np.broadcast_to(w*self.dy/2,(n,8));self.ids=np.stack([np.arange(n),np.arange(1,n+1)],axis=-1)
        self.qx=self.x[:-1,None]+(z+1)*self.dy/2
        r,phi=seed_table(p['ell']);self.phase=np.interp(abs(self.x-.5),r,phi,right=0.)
        self.state=np.zeros((n,8,6));self.state[:,:,4]=self.phase[self.ids]@self.N.T
        self.mass=np.zeros(n+1);self.K=np.zeros((n+1,n+1))
        for i in range(n):
            self.mass[i:i+2]+=self.dy/2
            self.K[i:i+2,i:i+2]+=2*p['Kf']*p['ell']**2/self.dy*np.array([[1,-1],[-1,1]])
        self.fracture0=self.fracture(self.phase);self.tau=0.;self.energy=self.fracture0
        self.lib=ctypes.CDLL(str(library.resolve()));self.fn=self.lib.bulk_strip_points
        array=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        self.fn.argtypes=[ctypes.c_uint,array,array,array,array];self.fn.restype=ctypes.c_int
        self.max_oracle=0.;self.max_ledger=0.;self.initial_kkt=float(np.min((p['Kf']*self.mass+self.K@self.phase)/(p['Kf']*self.mass)))

    def fracture(self,phase):
        return self.p['Kf']*(self.mass@phase+self.p['ell']**2*np.sum(np.diff(phase)**2)/self.dy)

    def evaluate(self,phase,dU,dt,tangent=False):
        p=self.p;beta=1/(1+dt*G/ETA);pq=phase[self.ids]@self.N.T
        h,hp,hpp=compliance(pq,p['m']);old=self.state;abar=beta*old[:,:,1]
        qstick=np.divide(G*abar,h,out=np.zeros_like(h),where=h>0)
        def local(tau):
            q=np.clip(qstick,tau-R,tau+R)
            q=np.where(h>0,q,np.sign(tau)*max(abs(tau)-R,0.))
            a=h*q/G
            return a,q
        def mechanical(tau):
            a,_=local(tau)
            return ((1+dt*G/ETA)*tau-self.tau)/G+np.sum(self.weights*((1+dt*G/ETA)*a-old[:,:,1]))-dU
        tau=brentq(mechanical,-1e6,1e6,xtol=1e-10);a,q=local(tau)
        Y=hp*q*q/(2*G);source=np.einsum('cq,qj->cj',self.weights*Y,self.N)
        g=p['Kf']*self.mass+self.K@phase
        for j in range(2):np.add.at(g,self.ids[:,j],-source[:,j])
        if not tangent:return g,tau,pq,a,q
        sliding=(tau-qstick>R)|(tau-qstick<-R)
        ap=sliding*hp*q/G;at=sliding*h/G
        qp=np.divide(-q*hp,h,out=np.zeros_like(q),where=(h>0)&~sliding)
        Yp=(hpp*q*q+2*hp*q*qp)/(2*G)
        H=self.K.copy();sourceH=np.einsum('cq,qi,qj->cij',self.weights*Yp,self.N,self.N)
        v=np.zeros(len(phase));dv=np.einsum('cq,qj->cj',self.weights*ap,self.N)
        for i in range(self.n):H[i:i+2,i:i+2]-=sourceH[i]
        for j in range(2):np.add.at(v,self.ids[:,j],dv[:,j])
        H+=np.outer(v,v)/(1/G+np.sum(self.weights*at))
        return g,H,tau,pq,a,q

    def step(self,dU,dt):
        lower=self.phase.copy();phase=lower.copy();p=self.p
        def norm(g,x):
            projected=np.where((x<=lower+1e-13)&(g>=0),0,g)/(p['Kf']*self.mass)
            return np.max(abs(projected))
        for iteration in range(80):
            g,H,tau,pq,a,q=self.evaluate(phase,dU,dt,True);error=norm(g,phase)
            self.last_trial=(phase.copy(),g.copy(),H.copy(),tau)
            if error<1e-7:break
            free=np.where(~((phase<=lower+1e-13)&(g>=0)))[0]
            dx=np.zeros_like(phase);dx[free]=solve(H[np.ix_(free,free)],-g[free],assume_a='sym')
            alpha=1.
            if np.any(dx>0):alpha=min(alpha,.99*np.min((1-phase[dx>0])/dx[dx>0]))
            for _ in range(35):
                trial=np.maximum(lower,phase+alpha*dx)
                gt,*_=self.evaluate(trial,dU,dt)
                if norm(gt,trial)<error*(1-1e-4*alpha) or norm(gt,trial)<1e-7:break
                alpha/=2
            else:raise RuntimeError(f'1-D phase line search stopped, KKT={error:.9g}')
            phase=trial
        else:raise RuntimeError('1-D Newton iteration limit')
        old=self.state
        dgamma=((1+dt*G/ETA)*tau-self.tau)/G+(1+dt*G/ETA)*a-old[:,:,1]
        inc=np.stack([dgamma,pq,np.full_like(pq,dt)],axis=-1)
        result=np.zeros((self.n,8,19))
        args=np.array([G,ETA,p['m'],R])
        assert self.fn(self.n*8,args,np.ascontiguousarray(old),np.ascontiguousarray(inc),result)==0
        discrepancy=max(np.max(abs(result[:,:,6]-tau))/max(1000,abs(tau)),np.max(abs(result[:,:,1]-a)),np.max(abs(result[:,:,5]-q))/1000)
        assert discrepancy<1e-12,discrepancy;self.max_oracle=max(self.max_oracle,float(discrepancy))
        self.max_ledger=max(self.max_ledger,float(np.max(abs(result[:,:,14]))))
        weights=self.weights;Ebulk=np.sum(weights*result[:,:,9]);Ecoh=np.sum(weights*result[:,:,10]);Ef=self.fracture(phase)
        Db,Dc,Df=[np.sum(weights*result[:,:,j])*dt for j in [11,12,13]]
        work=tau*dU;trwork=(tau+self.tau)*dU/2
        delta=Ebulk+Ecoh+Ef-self.energy
        slip=np.sum(weights*result[:,:,2]);elas=np.sum(weights*result[:,:,1])
        record=[tau,slip,elas,float(max(phase)),Ebulk,Ecoh,Ef,work,trwork,Db,Dc,Df,work-delta-Db-Dc-Df,trwork-delta-Db-Dc-Df,error]
        hn=compliance(old[:,:,4],p['m'])[0];hp=compliance(pq,p['m'])
        numerical=[(tau-self.tau)**2/(2*G),float(np.sum(weights*hn*(q-old[:,:,5])**2/(2*G))),
            float(np.sum(weights*q*q/(2*G)*(hp[0]-hn-hp[1]*(pq-old[:,:,4])))),
            float(p['Kf']*p['ell']**2*np.sum(np.diff(phase-lower)**2)/self.dy),float(g@(phase-lower))]
        assert abs(record[12]-(sum(numerical[:4])-numerical[4]))<1e-10
        record+=numerical
        self.phase=phase;self.state=result[:,:,:6].copy();self.tau=tau;self.energy=Ebulk+Ecoh+Ef
        return record

def main(out,library,calibration=False):
    out.mkdir(exist_ok=True,parents=True);summary={}
    for name,p0 in SETS.items():
        p=dict(p0);p['Kf']=p['Gc']/((8/3)*p['ell']);p['m']=2*G*p['Kf']/p['drop']**2
        e=envelope(p,512);fine=envelope(p,1024)
        qerror=np.max(abs(e[:,1:7]-fine[:,1:7])/np.maximum(1,abs(fine[:,1:7])))
        np.savetxt(out/f'{name}-strip-envelope.csv',fine,delimiter=',',comments='',header='peak_phi,traction_Pa,excess_traction_Pa,elastic_slip_m,I_m,fracture_J_m2,cohesive_J_m2,excess_work_J_m2')
        r,seed=seed_table(p['ell']);seedrow=np.argmin(abs(fine[:,0]-.6))
        # Evaluate the exact .6 profile without depending on envelope sampling.
        z,w=np.polynomial.legendre.leggauss(512);th=(z+1)*np.pi/4;pp=.6*np.cos(th)**2
        v=2*p['ell']*(1-pp)/np.sqrt(1.5-pp);h,_,_=compliance(pp,p['m']);hh=compliance(np.array(.6),p['m'])[0]
        f=pp-.6*h/hh;Fseed=2*p['Kf']*np.sum(w*np.pi/4*(pp+f)*v)
        strip=Strip(p,library);history=[];failure=None
        for n in range(1,309 if calibration else 149):
            t=n*.00125;stop=.375 if calibration else .175
            dU=.018*.00125 if .05+1e-12<t<=stop+1e-12 else 0.
            try:history.append([n,t,dU,*strip.step(dU,.00125)])
            except RuntimeError as error:failure=dict(step=n,time_s=t,reason=str(error));break
        tag='calibration' if calibration else 'history'
        np.savetxt(out/f'{name}-strip-{tag}.csv',history,delimiter=',',comments='',header=HISTORY_HEADER)
        summary[name]=dict(**p,G_Pa=G,eta_Pa_s=ETA,residual_Pa=R,peak_Pa=R+p['drop'],
            cohesive_length_m=G*p['Gc']/p['drop']**2,characteristic_slip_m=p['Gc']/p['drop'],
            final_elastic_slip_limit_m=3*np.pi/(2*np.sqrt(2))*p['Gc']/p['drop'],
            seed_radius_m=float(r[-1]),seed_full_width_m=float(2*r[-1]),seed_fracture_J_m2=float(Fseed),
            seed_fracture_fraction=float(Fseed/p['Gc']),equilibrium_seed_excess_Pa=p['drop']*.4/np.sqrt(1.6),
            envelope_quadrature_relative_difference=float(qerror),full_path_excess_work_J_m2=float(fine[-1,-1]),
            full_path_fracture_J_m2=float(fine[-1,5]),remaining_cohesive_J_m2=float(fine[-1,6]),
            strip_initial_KKT_min_scaled=strip.initial_kkt,strip_initial_fracture_J_m2=float(strip.fracture0),
            strip_max_native_point_discrepancy=strip.max_oracle,strip_max_native_point_ledger_error=strip.max_ledger,
            strip_steps=len(history),strip_stop=failure)
        print(name,summary[name],flush=True)
    (out/('strip-calibration-summary.json' if calibration else 'strip-summary.json')).write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--library',type=Path,required=True);p.add_argument('--calibration',action='store_true');a=p.parse_args();main(a.output,a.library,a.calibration)
