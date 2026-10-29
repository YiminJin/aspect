#!/usr/bin/env python3
"""Independent endpoint and single-direction kinetic references, not native code."""
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from scipy.optimize import least_squares
from scipy.linalg import eigvals
from feng_direction import (G, ETA, P0, MU, EPS, tensor, inner, v2, J,
                           geometry, directions, tolerance, h_of, rotation)

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/constitutive_reconciliation'
ELL=.046875
A_RSF,B_RSF,V0,DC,RHO=.005,.01,5e-6,.001,2670.
ETA_RAD=G/(2*np.sqrt(G/RHO))
RELAX=G/ETA


def paper_angles(tau_trial,Qref,p,h,mu=MU):
    """Literal closed-contact Feng 78/79a/80/81, with both assumed signs.

    atan(C2/C1), not atan2. The paper tries -1 first, then +1 if outside
    [-pi/4,pi/4]. No global test is silently folded into this formula.
    """
    d=v2(tau_trial);s=v2(tau_trial-Qref)
    principal=.5*np.arctan2(d[1],d[0]);rs=np.linalg.norm(s);rt=np.linalg.norm(d)
    relative=.5*np.arctan2(s[1],s[0])-principal
    average=.5*np.trace(tau_trial[:2,:2])-p
    reduction=1. if np.isinf(h) else h/(1+h)
    rows=[]
    for sign in [-1.,1.]:
        k=sign*reduction*mu
        A=sign*rs*np.cos(2*relative);B=sign*rs*np.sin(2*relative)+rt*mu
        C1=B-k*A;C2=A+k*B
        if C1==0.:
            rows.append(dict(assumed_sign=sign,relative_angle=np.nan,theta=np.nan,
                             in_range=False,C=np.nan,defined=False,C1=C1,C2=C2))
            continue
        C=-np.sign(C1)*k*average*mu/np.hypot(C1,C2)
        if abs(C)>1.:continue
        angle=.5*(np.arcsin(C)-np.arctan(C2/C1))
        rows.append(dict(assumed_sign=sign,relative_angle=angle,
                         theta=(principal+angle)%np.pi,
                         in_range=bool(abs(angle)<=np.pi/4+8*EPS),C=C,
                         defined=True,C1=C1,C2=C2))
    return rows


def complete_stationary_angles(tau_trial,Qref,p,h,mu=MU):
    """All trigonometric branches of Feng E7/E10, not the printed shortcut.

    C1 sin(2Theta)+C2 cos(2Theta)=-k average mu. This equation remains
    defined at C1=0; both sine branches and the assumed shear sign are checked.
    Final GLOBAL admissibility is still a separate, required test.
    """
    d=v2(tau_trial);s=v2(tau_trial-Qref)
    principal=.5*np.arctan2(d[1],d[0]);relative=.5*np.arctan2(s[1],s[0])-principal
    rs=np.linalg.norm(s);rt=np.linalg.norm(d);average=.5*np.trace(tau_trial[:2,:2])-p
    reduction=1. if np.isinf(h) else h/(1+h);answer=[]
    for sign in [-1.,1.]:
        k=sign*reduction*mu;A=sign*rs*np.cos(2*relative);B=sign*rs*np.sin(2*relative)+rt*mu
        C1=B-k*A;C2=A+k*B;radius=np.hypot(C1,C2);rhs=-k*average*mu/radius
        if abs(rhs)>1:continue
        shift=np.arctan2(C2,C1)
        for root in [np.arcsin(rhs),np.pi-np.arcsin(rhs)]:
            relative_angle=.5*((root-shift+np.pi)%(2*np.pi)-np.pi)
            if abs(relative_angle)>np.pi/4+8*EPS:continue
            theta=(principal+relative_angle)%np.pi
            if np.sign(inner(tau_trial-Qref,geometry(theta)[2]))!=sign:continue
            n,t,m,mp,_,P,Pp,_=geometry(theta,sign)
            xi=(inner(tau_trial-Qref,m)-mu*(p-inner(tau_trial,P)))/(G*(1+1/h))
            if xi<=0:continue
            tau=tau_trial-2*G*xi*m;Q=Qref+2*G*xi/h*m
            answer.append(dict(theta=theta,xi=xi,tau=tau,Q=Q,
                               stationarity=inner(tau-Q,mp)+mu*inner(tau,Pp),
                               global_margin=directions(tau,Q,p,mu)['maximum']))
    return answer


def paper_update(trial,Qref,p,h,relative_sign=False):
    trialmax=directions(trial,Qref,p)
    tol=tolerance(trial,Qref,p)
    if trialmax['maximum']<=tol:
        return dict(tau=trial,Q=Qref,xi=0.,m=np.zeros((3,3)),theta=np.nan,
                    maximum=trialmax['maximum'],stationarity=0.,accepted=True,
                    sign_conflict=False,angle=None)
    angles=paper_angles(trial,Qref,p,h)
    angle=next((a for a in angles if a['in_range']),None)
    if angle is None:return dict(accepted=False,undefined=True,angles=angles)
    _,_,S,*_=geometry(angle['theta'])
    physical=np.sign(inner(trial,S));relative=np.sign(inner(trial-Qref,S))
    sign=relative if relative_sign else physical
    n,t,m,mp,_,P,Pp,_=geometry(angle['theta'],sign)
    f=inner(trial-Qref,m)-MU*(p-inner(trial,P))
    xi=max(0.,f)/(G*(1+1/h))
    tau=trial-2*G*xi*m;Q=Qref+2*G*xi/h*m
    maximum=directions(tau,Q,p)['maximum']
    return dict(tau=tau,Q=Q,xi=xi,m=m,theta=angle['theta'],maximum=maximum,
                stationarity=inner(tau-Q,mp)+MU*inner(tau,Pp),
                accepted=bool(maximum<=tol),sign_conflict=bool(physical!=relative),angle=angle)


def chi_center(phi):
    # AT1 constrained optimal cap: phi(r)=(sqrt(phi0)-|r|/(2 ell))_+².
    # Gamma=3/(8ell)(phi+ell² phi_r²), integral Gamma=phi0^(3/2).
    return 3/(4*ELL*np.sqrt(phi))


def friction(q,Theta=1.,a=A_RSF,b=B_RSF):
    """Fei Eq26 regularization, with no recalibration of mu0."""
    assert q>0 and Theta>0
    if a==0.:return MU,0.,0.
    logz=np.log(q/2)+(MU+b*np.log(Theta))/a
    if logz>=0:
        inv=np.exp(-2*logz);value=logz+np.log1p(np.sqrt(1+inv));fraction=1/np.sqrt(1+inv)
    else:
        z=np.exp(logz);value=np.arcsinh(z);fraction=z/np.hypot(1.,z)
    return a*value,a*fraction/q,b*fraction/Theta


def base_state(phi,Q,stress_direction=None,branch=0,rad=ETA_RAD):
    """Synthetic frozen state on actual kinetic surface; provenance explicit."""
    h=h_of(phi);direction=tensor(xy=1.) if stress_direction is None else stress_direction
    direction=direction/np.sqrt(inner(direction,direction)/2)
    mu=friction(1.)[0]
    amplitude=brentq(lambda s:directions(s*direction,Q,P0,mu)['maximum']-rad*V0,1.,5000.,xtol=1e-11)
    tau=amplitude*direction;r=directions(tau,Q,P0,mu)
    c=sorted(r['candidates'],key=lambda c:c['theta'])[branch]
    assert pmin(tau)>0 and c['curvature']<0
    return dict(phi=phi,h=h,tau=tau,Q=Q,theta=c['theta'],sign=c['sign'],
                w0=chi_center(phi)*V0,chi=chi_center(phi),rad=rad,
                ties=len(r['candidates']),global_error=r['maximum']-rad*V0)


def pmin(tau):return P0-max(np.linalg.eigvalsh(tau))


def pencil(base,wave_angle,a=A_RSF,b=B_RSF,rad=None):
    """Exact mixed reduction; split current/previous pressure contributions.

    y=[tau_mode/G,planar_mean/G,Qxx/G,Qxy/G,Theta], Q omitted if mature.
    z=[delta w/w0,delta plane angle]. ydot=A y+D z,
    Bcur y+Bpressure y+C z=0. Full plane strain retained.
    """
    mature=np.isinf(base['h']);ny=3 if mature else 5;state=ny-1
    w=base['w0'];rad=base['rad'] if rad is None else rad
    wave=np.array([np.cos(2*wave_angle),np.sin(2*wave_angle)])
    stress=[G*tensor(*wave),G*tensor(mean=1.)]
    n,t,m,mp,mpp,P,Pp,Ppp=geometry(base['theta'],base['sign'])
    pn=P0-inner(base['tau'],P);pnp=-inner(base['tau'],Pp)
    mu,mq,mt=friction(1.,1.,a,b)
    curvature=inner(base['tau']-base['Q'],mpp)+mu*inner(base['tau'],Ppp)
    B=np.zeros((2,ny));L=np.zeros_like(B);C=np.zeros((2,2))
    for k,dtau in enumerate(stress):
        B[0,k]=inner(m,dtau)/G;B[1,k]=inner(mp,dtau)/G
        L[0,k]=mu*(inner(P,dtau)-G)/G;L[1,k]=mu*inner(Pp,dtau)/G
    if not mature:
        for k,dQ in enumerate([G*tensor(xx=1.),G*tensor(xy=1.)]):
            B[0,2+k]=-inner(m,dQ)/G;B[1,2+k]=-inner(mp,dQ)/G
    B[:,state]=[-pn*mt/G,-pnp*mt/G]
    C[0,0]=-(pn*mq+rad*V0)/G;C[1,0]=-pnp*mq/G;C[1,1]=curvature/G
    A=-RELAX*np.eye(ny);D=np.zeros((ny,2))
    for k,shape in enumerate([w*m,w*mp]):
        D[0,k]=-2*wave@v2(shape)
        if not mature:D[2:4,k]=2*v2(shape)/base['h']
    A[state,state]=-V0/DC;D[state,0]=-V0/DC
    return A,D,B,L,C


def evolution(base,angle,a=A_RSF,b=B_RSF,rad=None):
    A,D,B,L,C=pencil(base,angle,a,b,rad)
    return A-D@np.linalg.solve(C,B+L)


def amplification(base,angle,dt,lag=False,a=A_RSF,b=B_RSF,rad=None):
    A,D,B,L,C=pencil(base,angle,a,b,rad)
    current=B if lag else B+L;previous=L if lag else np.zeros_like(L)
    I=np.eye(len(A));lhs=I-dt*(A-D@np.linalg.solve(C,current))
    rhs=I-dt*D@np.linalg.solve(C,previous)
    return np.linalg.solve(lhs,rhs)


def full_pencil(base,angle,**kwargs):
    A,D,B,L,C=pencil(base,angle,**kwargs);ny=len(A)
    F=np.block([[A,D],[B+L,C]])
    E=np.zeros_like(F);E[:ny,:ny]=np.eye(ny)
    return F,E


def uneliminated_pencil(base,angle,kell=1.,a=A_RSF,b=B_RSF,rad=None):
    """Full stress/history/state/velocity/pressure/rate/angle mixed pencil.

    Real velocity amplitude represents i*v/(w0*ell). Wave factors kell
    remain explicit; no mechanical equation or normal pressure is frozen.
    """
    mature=np.isinf(base['h']);ny=4 if mature else 6;state=ny-1
    F=np.zeros((ny+5,ny+5));E=np.zeros_like(F);E[:ny,:ny]=np.eye(ny)
    F[:ny,:ny]=-RELAX*np.eye(ny)
    n=np.array([np.cos(angle),np.sin(angle)]);w=base['w0']
    F[0,ny:ny+2]=w*kell*np.array([n[0],-n[1]])
    F[1,ny:ny+2]=w*kell*np.array([n[1],n[0]])
    F[2,ny:ny+2]=w*kell*n/3
    normal,t,m,mp,mpp,P,Pp,Ppp=geometry(base['theta'],base['sign'])
    for j,shape in enumerate([w*m,w*mp]):
        F[:2,ny+3+j]=-2*v2(shape)
        if not mature:F[3:5,ny+3+j]=2*v2(shape)/base['h']
    F[state,state]=-V0/DC;F[state,ny+3]=-V0/DC
    stress=[G*tensor(xx=1.),G*tensor(xy=1.),G*tensor(mean=1.)]
    for j,s in enumerate(stress):F[ny:ny+2,j]=(s[:2,:2]@n)/G
    F[ny:ny+2,ny+2]=-n;F[ny+2,ny:ny+2]=kell*n
    mu,mq,mt=friction(1.,1.,a,b);rad=base['rad'] if rad is None else rad
    pn=P0-inner(base['tau'],P);pnp=-inner(base['tau'],Pp)
    for j,s in enumerate(stress):
        F[ny+3,j]=inner(m+mu*P,s)/G;F[ny+4,j]=inner(mp+mu*Pp,s)/G
    if not mature:
        for j,q in enumerate([G*tensor(xx=1.),G*tensor(xy=1.)]):
            F[ny+3,3+j]=-inner(m,q)/G;F[ny+4,3+j]=-inner(mp,q)/G
    F[ny+3,ny+2]=-mu
    F[ny+3,state]=-pn*mt/G;F[ny+4,state]=-pnp*mt/G
    F[ny+3,ny+3]=-(pn*mq+rad*V0)/G;F[ny+4,ny+3]=-pnp*mq/G
    F[ny+4,ny+4]=(inner(base['tau']-base['Q'],mpp)+mu*inner(base['tau'],Ppp))/G
    # Row equilibration changes no eigenvalue and prevents force/strain units
    # from deciding which QZ roots are identified as infinite constraints.
    scale=np.maximum(np.max(abs(F),axis=1),np.max(abs(E),axis=1))
    F=F/scale[:,None];E=E/scale[:,None]
    # Algebraic rates and velocities carry very different units. Equilibrate
    # BOTH sides of the pencil by identical invertible row/column scalings.
    # This preserves eigenvalues and infinite constraints, not a tolerance fix.
    for _ in range(12):
        columns=np.sqrt(np.maximum(abs(F).max(axis=0),abs(E).max(axis=0)))
        F/=columns;E/=columns
        rows=np.sqrt(np.maximum(abs(F).max(axis=1),abs(E).max(axis=1)))
        F/=rows[:,None];E/=rows[:,None]
    return F,E


def nonlinear_rows(base,angle,y,z,yprev=None,a=A_RSF,b=B_RSF,rad=None):
    """Independent tensor contraction for directional FD of the kinetic rows."""
    rad=base['rad'] if rad is None else rad
    wave=np.array([np.cos(2*angle),np.sin(2*angle)])
    tau=base['tau']+G*(tensor(*(y[0]*wave))+tensor(mean=y[1]))
    Q=base['Q'].copy()
    if len(y)==5:Q+=G*tensor(*y[2:4])
    old=y if yprev is None else yprev
    normalstress=base['tau']+G*(tensor(*(old[0]*wave))+tensor(mean=old[1]))
    pressure=P0+G*(old[0]+old[1])
    n,t,m,mp,_,P,Pp,_=geometry(base['theta']+z[1],base['sign'])
    mu=friction(1+z[0],1+y[-1],a,b)[0]
    pn=pressure-inner(normalstress,P)
    return np.array([inner(tau-Q,m)-mu*pn-rad*V0*(1+z[0]),
                     inner(tau-Q,mp)+mu*inner(normalstress,Pp)])/G


def kinetic_step(old,de,dt,p,lag=False,phi=1.):
    """Small point BE trial, final global test and history tie test retained.

    Prescribed point pressure is distinct from the solved-pressure symbol.
    No failing trial is published by this routine.
    """
    from feng_direction import select
    beta=1/(1+RELAX*dt);h=h_of(phi);chi=chi_center(phi)
    trial=beta*(old['tau']+2*G*de);qref=beta*old['Q']
    normalstress=old['tau'] if lag else None;normalp=old['p'] if lag else p
    def evaluate(x):
        q=np.exp(x[0]);V=V0*q;Theta=(old['Theta']+(V0/DC)*dt)/(1+(V0/DC)*dt*q)
        mu=friction(q,Theta)[0]
        n,t,m,mp,_,P,Pp,_=geometry(x[1],1.)
        xi=beta*dt*chi*V
        tau=trial-2*G*xi*m;Q=qref+2*G*xi/h*m
        normal=tau if normalstress is None else normalstress
        pn=normalp-inner(normal,P)
        rows=np.array([inner(tau-Q,m)-mu*pn-ETA_RAD*V,
                       inner(tau-Q,mp)+mu*inner(normal,Pp)])
        return rows,dict(q=q,V=V,Theta=Theta,mu=mu,tau=tau,Q=Q,m=m,
                         theta=x[1]%np.pi,p=p,pn=pn,xi=xi,beta=beta,chi=chi)
    start=directions(trial,qref,p)
    candidates=[]
    for candidate in start['candidates']:
        # A negative signed m is represented by theta+pi/2 with positive sign.
        # Canonical normal is recovered by the min-pressure global check.
        theta=candidate['theta']+(np.pi/2 if candidate['sign']<0 else 0.)
        shear=abs(inner(trial-qref,candidate['m']))
        drive=shear/(A_RSF*max(1.,candidate['pn']))
        logq=min(10.,np.log(2.)+np.log(max(np.sinh(min(50.,drive)),1e-30))-(MU+B_RSF*np.log(old['Theta']))/A_RSF)
        def fun(x):
            # Pressure must be the smaller compression for the actual N.
            rows,r=evaluate(x)
            normal=r['tau'] if normalstress is None else normalstress
            n,t,m,mp,_,P,Pp,_=geometry(x[1],1.)
            other=normalp-inner(normal,np.outer(t,t))
            if other<r['pn']:
                mu=r['mu'];rows[0]=inner(r['tau']-r['Q'],m)-mu*other-ETA_RAD*r['V']
                rows[1]=inner(r['tau']-r['Q'],mp)-mu*inner(normal,Pp)
            return rows/1000.
        sol=least_squares(fun,[np.clip(logq,-60,20),theta],bounds=([-60,theta-.7],[20,theta+.7]),
                          xtol=1e-13,ftol=1e-13,gtol=1e-13,max_nfev=150)
        _,r=evaluate(sol.x)
        normal=r['tau'] if normalstress is None else normalstress
        global_result=directions(normal,normal-(r['tau']-r['Q']),normalp,r['mu'])
        preferred,reason=select(global_result,old['normal'])
        error=global_result['maximum']-ETA_RAD*r['V']
        actual=np.sqrt(2)*r['m']
        tol=tolerance(r['tau'],r['Q'],p,r['mu'])
        # Nonlinear residual allowance is the retained 1e-8 Pa point check;
        # global inequality retains the operand-scaled roundoff allowance.
        if np.linalg.norm(fun(sol.x))*1000>1e-8:continue
        r.update(global_error=error,global_allowance=tol,
                 accepted=bool(error<=tol and preferred is not None and np.linalg.norm(preferred['N']-actual)<1e-8),
                 normal=preferred['normal'] if preferred is not None else np.array([np.nan]*3),reason=reason)
        if any(np.linalg.norm(r['tau']-c['tau'])<1e-7 and
               np.linalg.norm(r['m']-c['m'])<1e-10 for c in candidates):continue
        candidates.append(r)
    accepted=[c for c in candidates if c['accepted']]
    return accepted[0] if len(accepted)==1 else dict(accepted=False,candidates=candidates,reason='no unique globally admitted kinetic endpoint')
