#!/usr/bin/env python3
"""Rich Q2 ux(y), Q1 phi(y), endpoint Maxwell/slip, exact normal I.

The constant along-fault I is exactly reproduced by the retained consistent
surface projection. No surface discretization or constitutive parameter changes.
Independent 1-D reference, never modifies native histories.
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.linalg import solve
from scipy.integrate import cumulative_trapezoid
from numpy.polynomial.legendre import leggauss
from total_energy_step import G, ETA, GC, ELL, M, FRACTURE_SCALE as A, PRESSURE, MU, h, dh

HERE=Path(__file__).resolve().parent

def native_profile(y,core=.6):
    # Source PhaseFieldProfile's change of variables; high-resolution trapezoid.
    theta=np.linspace(0,np.pi/2,100001)
    phi=core*np.cos(theta)**2; hh=h(core)
    den=hh*phi-core*h(phi)
    integrand=np.zeros_like(theta)
    integrand[1:-1]=2*ELL*core*np.sin(theta[1:-1])*np.cos(theta[1:-1])*np.sqrt(hh/den[1:-1])
    integrand[0]=2*ELL*np.sqrt(core*hh/(core*dh(core)-hh))
    integrand[-1]=2*ELL*np.sqrt(core*hh/(hh-core*dh(0.)))
    distance=cumulative_trapezoid(integrand,theta,initial=0.)
    return np.interp(abs(y-.5),distance,phi,right=0.)

class Reference:
    def __init__(self,n,height,quadrature=6,exact_bulk=False,degree=2):
        self.exact_bulk=exact_bulk
        assert degree in [2,3]
        self.degree=degree
        self.n=n; self.height=height; self.dy=height/n
        self.y=np.linspace(.5-height/2,.5+height/2,n+1)
        z,w=leggauss(quadrature); s=(z+1)/2
        self.w=np.tile(w*self.dy/2,n)
        self.yq=(self.y[:-1,None]+self.dy*s).ravel()
        rows=np.repeat(np.arange(n*len(z)),2)
        cols=(np.repeat(np.arange(n),len(z))[:,None]+np.arange(2)).ravel()
        vals=np.tile(np.c_[1-s,s],(n,1)).ravel()
        self.B=sparse.csr_matrix((vals,(rows,cols)),shape=(n*len(z),n+1))
        if degree==2:
            d=np.c_[-3+4*s,4-8*s,-1+4*s]/self.dy
        else:
            # Native FE_Q uses Gauss-Lobatto support points.
            nodes=np.r_[0.,(1-1/np.sqrt(5))/2,(1+1/np.sqrt(5))/2,1.]
            coefficients=np.linalg.solve(np.vander(nodes,4,increasing=True),np.eye(4))
            d=np.polynomial.polynomial.polyval(s,np.arange(1,4)[:,None]*coefficients[1:]).T/self.dy
        rc=np.repeat(np.arange(n*len(z)),degree+1)
        cc=(degree*np.repeat(np.arange(n),len(z))[:,None]+np.arange(degree+1)).ravel()
        self.D=sparse.csr_matrix((np.tile(d,(n,1)).ravel(),(rc,cc)),shape=(n*len(z),degree*n+1)).toarray()
        self.mass=np.full(n+1,self.dy);self.mass[[0,-1]]*=.5
        self.flinear=A*self.mass
        self.F=sparse.diags([np.full(n,-1.),np.r_[1.,np.full(n-1,2.),1.],np.full(n,-1.)],[-1,0,1]).toarray()*2*A*ELL**2/self.dy
        self.r=np.zeros(degree*n+1);self.rq=np.zeros(len(self.w));self.Z=0.; self.slip=0.;self.phi=self.initialize()
        self.initial=self.phi.copy()
    def anchor_shear(self):return self.rq if self.exact_bulk else self.D@self.r
    def direct(self,phi):
        u,v=1-phi[:-1],1-phi[1:];z=(v-u)/u;small=abs(z)<1e-3
        ratio=np.empty_like(z);prime=np.empty_like(z)
        q=z[small]
        ratio[small]=sum((-1)**k*q**k/(k+1) for k in range(13))
        prime[small]=sum((-1)**k*k*q**(k-1)/(k+1) for k in range(1,13))
        q=z[~small];ratio[~small]=np.log1p(q)/q;prime[~small]=(q/(1+q)-np.log1p(q))/q**2
        I=M*self.dy*np.sum(2/(u*v)-3*ratio/u+1)
        left=M*self.dy*(2/(u*u*v)-3/u**2*(ratio+(1+z)*prime))
        right=M*self.dy*(2/(u*v*v)+3/u**2*prime)
        J=np.r_[left,0.]+np.r_[0.,right]
        return I,J
    def initialize(self):
        # Native prescribed H at 3 regular CPDI points/cell, same strength rule.
        pc=(np.arange(3*self.n)+.5)*self.dy/3+self.y[0]
        guess=native_profile(self.y); profile=native_profile(pc)
        H=np.full(len(pc),.5);active=profile>1e-6
        H[active]=A*.6/h(.6)*(1+h(profile[active]))**2
        # Diagnostic snapshots of the actual native-initialization samples.
        self.initial_particle_y=pc.copy();self.initial_particle_H=H.copy()
        cell=np.minimum(((pc-self.y[0])/self.dy).astype(int),self.n-1);s=(pc-self.y[cell])/self.dy
        P=sparse.csr_matrix((np.c_[1-s,s].ravel(),(np.repeat(np.arange(len(pc)),2),(cell[:,None]+[0,1]).ravel())),shape=(len(pc),self.n+1))
        def obj(x):
            p=P@x;den=(1-p)**2+M*p*(1+p);g=(1-p)**2/den;gp=-M*(1-p)*(1+3*p)/den**2
            return (self.fracture(x)+self.dy/3*np.dot(H,g))/A,(self.flinear+self.F@x+P.T@(self.dy/3*H*gp))/A
        result=minimize(obj,guess,jac=True,method='L-BFGS-B',bounds=[(0.,1.)]*len(guess),options=dict(gtol=1e-12,ftol=1e-16,maxiter=5000,maxls=50))
        x=self.refine(result.x,np.zeros_like(guess),obj)
        # Prescribed band protected by obstacle, total-energy unloaded stability.
        unloaded_gradient=self.flinear+self.F@x
        self.startup_defect=float(max(np.maximum(-unloaded_gradient,0.))/A)
        if self.startup_defect>1e-9: raise RuntimeError(('unloaded seed is not stable',self.startup_defect))
        return x
    def fracture(self,x): return self.flinear@x+.5*x@(self.F@x)
    def objective(self,x):
        if max(x)>=1 or min(x)<0: return np.inf,np.zeros_like(x),None
        I,J=self.direct(x);ph=self.B@x;chi=h(ph)/I
        tau=G*self.anchor_shear()-self.Z*chi
        eb=np.dot(self.w,tau*tau)/(2*G);ec=self.Z**2/(2*G*I);f=self.fracture(x)
        gb=self.B.T@(-self.w*self.Z*tau/G*dh(ph)/I)+J*np.dot(self.w,self.Z*tau/G*h(ph)/I**2)
        gc=-self.Z**2/(2*G*I**2)*J;gf=self.flinear+self.F@x
        return (eb+ec+f)/A,(gb+gc+gf)/A,(eb,ec,f,gb,gc,gf,I,chi,tau)
    @staticmethod
    def defect(g,x,lower):
        return max(abs(np.where(x<=lower+1e-13,np.minimum(g,0.),g)))
    def phase_hessian(self,x):
        I,J=self.direct(x);ph=self.B@x
        hh=h(ph);hp=dh(ph);hpp=M*(12/(1-ph)**4-6/(1-ph)**3)
        tau=G*self.anchor_shear()-self.Z*hh/I
        c=-self.w*self.Z*tau/G;a=self.w*self.Z**2/G
        pp=a*hp*hp/I**2+c*hpp/I
        cross=self.B.T@(-a*hp*hh/I**3-c*hp/I**2)
        ii=np.sum(a*hh*hh/I**4+2*c*hh/I**3)+self.Z**2/(G*I**3)
        force=np.sum(-c*hh/I**2)-self.Z**2/(2*G*I**2)
        # Independently refined integration of the analytic second derivative
        # of I; the objective and residual still use the exact antiderivative.
        z,w=leggauss(24);s=(z+1)/2;p=(1-s)*x[:-1,None]+s*x[1:,None]
        hpp=M*(12/(1-p)**4-6/(1-p)**3)
        HI=np.zeros((len(x),len(x)))
        for i in range(self.n):
            for j in range(2):
                for k in range(2):HI[i+j,i+k]+=self.dy/2*np.dot(w,hpp[i]*(s if j else 1-s)*(s if k else 1-s))
        return (self.F+(self.B.T@sparse.diags(pp)@self.B).toarray()+np.outer(cross,J)+np.outer(J,cross)+ii*np.outer(J,J)+force*HI)/A
    def refine(self,x,lower,objective,hessian=None):
        for _ in range(40):
            value,g,*_=objective(x)
            if self.defect(g,x,lower)<=1e-11: break
            free=np.flatnonzero((x>lower+1e-13)|(g<0.))
            if hessian is not None: H=hessian(x)[np.ix_(free,free)]
            else:
                H=np.empty((len(free),len(free)))
                for j,i in enumerate(free):
                    eps=1e-5;plus=x.copy();minus=x.copy();plus[i]+=eps;minus[i]-=eps
                    if minus[i]<0: H[:,j]=(objective(plus)[1][free]-g[free])/eps
                    else: H[:,j]=(objective(plus)[1][free]-objective(minus)[1][free])/(2*eps)
                H=(H+H.T)/2
            try: step=solve(H,-g[free],assume_a='sym')
            except Exception: break
            for ls in range(40):
                trial=x.copy();trial[free]=np.maximum(lower[free],x[free]+2.**(-ls)*step)
                vt,gt,*_=objective(trial)
                if vt<=value+32*np.finfo(float).eps*max(1.,abs(value)) and self.defect(gt,trial,lower)<self.defect(g,x,lower):
                    x=trial;break
            else: break
        return x
    def phase(self,lower):
        objective=lambda x:self.objective(x)
        answer=minimize(lambda x:objective(x)[:2],lower.copy(),jac=True,method='L-BFGS-B',bounds=list(zip(lower,np.ones_like(lower))),options=dict(gtol=2e-13,ftol=2e-16,maxiter=3000,maxls=60,maxcor=30))
        x=self.refine(answer.x,lower,objective,self.phase_hessian)
        defect=self.defect(objective(x)[1],x,lower)
        if defect>1e-11: raise RuntimeError(('phase KKT',defect))
        return x,defect
    def mechanical(self,time,dt,imposed_rate=None):
        beta=np.exp(-G*dt/ETA);kappa=ETA*(-np.expm1(-G*dt/ETA))
        I,_=self.direct(self.phi);chi=h(self.B@self.phi)/I
        tau0=G*self.anchor_shear()-self.Z*chi
        rate=.256*(time-dt/2) if imposed_rate is None else imposed_rate
        left,right=-self.height/2*rate,self.height/2*rate
        if self.exact_bulk:
            # Rich 1-D continuum bulk balance: shear stress is spatially
            # constant. Retain C=quadrature(chi), rather than assuming C=1.
            C=np.dot(self.w,chi);mean=np.dot(self.w,tau0)/self.height
            V=max(0.,(C*(beta*mean+kappa*(right-left)/self.height)-beta*self.Z/I-MU*PRESSURE)
                      /(kappa*(C*C/self.height+1/I)))
            T=beta*mean+kappa*((right-left)/self.height-C*V/self.height)
            du=(T-beta*tau0)/kappa+chi*V
            u=None
        else:
            fixed=np.array([left,right]);free=np.arange(1,self.degree*self.n)
            D=self.D[:,free];lift=self.D[:,[0,self.degree*self.n]]@fixed
            K=kappa*(D.T@(self.w[:,None]*D));B=-kappa*(D.T@(self.w*chi))
            rhs=-D.T@(self.w*(beta*tau0+kappa*lift))
            rV=np.dot(self.w*chi,beta*tau0+kappa*lift)-beta*self.Z/I-MU*PRESSURE
            ss=kappa*(np.dot(self.w,chi*chi)+1/I)
            u0=solve(K,rhs,assume_a='pos');b=solve(K,B,assume_a='pos')
            V=max(0.,(rV-B@u0)/(ss-B@b))
            u=np.r_[left,u0-b*V,right];du=self.D@u
        rate=du-chi*V
        tau=beta*tau0+kappa*rate;Zn=beta*self.Z+kappa*V
        endpoint_free=np.array([max(abs(tau-T))]) if self.exact_bulk else D.T@(self.w*tau)
        surface=np.dot(self.w*chi,tau)-Zn/I
        # General exponential Maxwell exact work/loss on the same QPs.
        z,w=leggauss(12);tq=(z+1)*dt/2;tw=w*dt/2
        tt=np.exp(-G*tq[:,None]/ETA)*tau0+ETA*(-np.expm1(-G*tq[:,None]/ETA))*rate
        zz=np.exp(-G*tq/ETA)*self.Z+ETA*(-np.expm1(-G*tq/ETA))*V
        loss=np.dot(tw,(tt*tt)@self.w/ETA+zz*zz/(ETA*I)+MU*PRESSURE*V)
        work=np.dot(tw,tt@(self.w*du))
        bulk0=np.dot(self.w,tau0*tau0)/(2*G);coh0=self.Z**2/(2*G*I)
        bulk1=np.dot(self.w,tau*tau)/(2*G);coh1=Zn*Zn/(2*G*I)
        error=work-(bulk1+coh1-bulk0-coh0+loss)
        # Slip equation enforced at endpoint, not throughout the interval.
        if self.exact_bulk:
            self.rq=beta*self.rq+(kappa/G)*du
            # Diagnostic integral only; never fed back to the accepted R_q.
            self.velocity=left+cumulative_trapezoid(du,self.yq,initial=0.)
        else:
            self.r=beta*self.r+(kappa/G)*u;self.velocity=u.copy()
        self.Z=Zn;self.slip+=dt*V;self.tau=tau.copy()
        return dict(q=Zn/I,V=V,slip=self.slip,stress_min=min(tau),stress_max=max(tau),bulk=bulk1,cohesive=coh1,loss=loss,work=work,mechanical_error=error,force=max(abs(endpoint_free)),stick_slip=surface-MU*PRESSURE if V>0 else max(surface-MU*PRESSURE,0.))
    def run(self,dt,final,failure_directory=None):
        history=[];fields=[];release=0.;abs_error=0.;onset=None
        for step in range(1,round(final/dt)+1):
            time=step*dt;old=self.phi.copy();e0=self.objective(old)[0]*A
            try:self.phi,defect=self.phase(old)
            except Exception as error:
                out=Path(failure_directory) if failure_directory is not None else HERE/'localization_evidence'/f'one-d-h{self.height:g}-n{self.n}-failed'
                out.mkdir(exist_ok=True)
                np.savez_compressed(out/'accepted.npz',phi=old,r=self.r,Z=self.Z,slip=self.slip,y=self.y,Rq=self.rq)
                (out/'history.json').write_text(json.dumps(history,indent=2))
                (out/'failure.json').write_text(json.dumps(dict(time=time,error=repr(error),target=1e-11),indent=2))
                raise
            en=self.objective(self.phi)[0]*A
            release+=e0-en;increment=max(self.phi-old)
            if onset is None and max(self.phi-self.initial)>1e-5:onset=time
            data=self.mechanical(time,dt);abs_error+=abs(data['mechanical_error'])
            I,_=self.direct(self.phi);chi=h(self.phi)/I
            chi_width=2*np.sqrt(np.trapezoid((self.y-.5)**2*chi,self.y)/np.trapezoid(chi,self.y)) if hasattr(np,'trapezoid') else 2*np.sqrt(np.trapz((self.y-.5)**2*chi,self.y)/np.trapz(chi,self.y))
            history.append(dict(time=time,phi_max=max(self.phi),phi_boundary=max(self.phi[[0,-1]]),increment=increment,kkt=defect,I=I,chi_width=chi_width,fracture=self.fracture(self.phi),release=release,absolute_mechanical_error=abs_error,**data))
            fields.append(self.phi.copy())
        stationary,skkt=self.phase(self.phi)
        _,_,terms=self.objective(self.phi)
        return history,np.array(fields),stationary,terms,dict(onset=onset,startup_kkt=self.startup_defect,stationary_kkt=skkt,stationary_difference=max(abs(stationary-self.phi)))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--cells',type=int,default=64);ap.add_argument('--height',type=float,default=1.);ap.add_argument('--dt',type=float,default=.015625);ap.add_argument('--final',type=float,default=.5);ap.add_argument('--quadrature',type=int,default=6);ap.add_argument('--exact-bulk',action='store_true');args=ap.parse_args()
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r=Reference(args.cells,args.height,args.quadrature,args.exact_bulk);history,fields,stationary,terms,summary=r.run(args.dt,args.final)
    out=HERE/'localization_evidence'/f'one-d-h{args.height:g}-n{args.cells}-dt{args.dt:g}'
    if args.quadrature!=6:out=out.with_name(out.name+f'-q{args.quadrature}')
    if args.exact_bulk:out=out.with_name(out.name+'-exact-bulk')
    out.mkdir(exist_ok=True)
    np.savez_compressed(out/'fields.npz',y=r.y,phase=fields,initial=r.initial,stationary=stationary,chi=h(r.phi)/terms[6],r=r.r,Rq=r.rq,Z=r.Z,yq=r.yq,stress=r.tau,velocity=r.velocity,chi_q=terms[7],bulk_driving=-terms[3]/r.mass,cohesive_driving=-terms[4]/r.mass,fracture_driving=-terms[5]/r.mass)
    (out/'history.json').write_text(json.dumps(history,indent=2));(out/'summary.json').write_text(json.dumps(summary,indent=2))
    fig,ax=plt.subplots(2,2,figsize=(10,7));ax[0,0].plot(r.y,r.initial,label='initial');ax[0,0].plot(r.y,r.phi,label='transient');ax[0,0].plot(r.y,stationary,'--',label='stationary, final R and Z fixed');ax[0,0].set_ylabel('phase');ax[0,0].legend()
    ax[0,1].plot(r.y,h(r.phi)/terms[6]);ax[0,1].set_ylabel('chi [1/m]')
    for key,name in [(3,'bulk'),(4,'cohesive'),(5,'fracture')]:ax[1,0].plot(r.y,-terms[key]/r.mass,label=name)
    ax[1,0].set_ylabel('weak driving / nodal measure [Pa]');ax[1,0].legend()
    ax[1,1].plot([x['time'] for x in history],[x['q'] for x in history],label='q [Pa]');ax[1,1].set_ylabel('q [Pa]');av=ax[1,1].twinx();av.plot([x['time'] for x in history],[x['V'] for x in history],'r',label='interval V');av.set_ylabel('V [m/s]')
    for a in ax.ravel()[:3]:a.set_xlabel('y [m]')
    ax[1,1].set_xlabel('time [s]');fig.tight_layout();fig.savefig(out/'profiles.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(summary=summary,last=history[-1]),indent=2))
