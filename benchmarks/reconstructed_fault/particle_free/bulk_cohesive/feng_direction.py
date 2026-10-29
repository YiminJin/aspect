#!/usr/bin/env python3
"""Independent Feng direction/power reference; no production constitutive edits.

The actual normalized increment N and min-pressure rule are implemented from
the published definition. The optional Maxwell map and current h are explicitly
our fixed-phase comparison. A returned point must satisfy GLOBAL maximization.
"""
from pathlib import Path
import numpy as np
from scipy.optimize import brentq

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/feng_direction'
G,ETA,MU,P0=1e6,1e8,.1,10000.
EPS=np.finfo(float).eps


def inner(a,b):return float(np.sum(a*b))
def tensor(xx=0.,xy=0.,mean=0.):
    return np.array([[mean+xx,xy,0.],[xy,mean-xx,0.],[0.,0.,-2*mean]])
def deviator(xx=0.,yy=0.,gamma=0.):
    return np.array([[xx,gamma/2,0.],[gamma/2,yy,0.],[0.,0.,0.]])-np.eye(3)*(xx+yy)/3
def components(t):return np.array([t[0,0],t[1,1],t[2,2],t[0,1]])
def v2(t):return np.array([.5*(t[0,0]-t[1,1]),t[0,1]])
def J(v):return np.array([-v[1],v[0]])
def rotation(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
def h_of(phi):return np.inf if phi==1. else 4*phi*(1+phi)/(1-phi)**2
def tolerance(tau,Q,p,mu=MU):
    return 64*EPS*max(1.,np.linalg.norm(tau),np.linalg.norm(Q),abs(mu*p))


def geometry(theta,sign=1.):
    alpha=np.array([np.cos(theta),np.sin(theta),0.])
    beta=np.array([-np.sin(theta),np.cos(theta),0.])
    P=np.outer(alpha,alpha);B=np.outer(beta,beta)
    S=(np.outer(alpha,beta)+np.outer(beta,alpha))/2
    m=sign*S;mp=sign*(B-P)
    return alpha,beta,m,mp,-4*m,P,2*S,2*(B-P)


def objective(tau,Q,p,theta,mu=MU):
    a,b,S,*_=geometry(theta)
    pa=max(0.,p-inner(tau,np.outer(a,a)))
    pb=max(0.,p-inner(tau,np.outer(b,b)))
    shear=inner(tau-Q,S)
    return abs(shear)-mu*min(pa,pb),pa,pb,shear


def _roots(A,B,c):
    radius=np.hypot(A,B)
    if radius==0. or abs(c)>radius:return []
    phase=np.arctan2(B,A);angle=np.arccos(np.clip(c/radius,-1.,1.))
    return [(.5*(phase+angle))%np.pi,(.5*(phase-angle))%np.pi]


def directions(tau,Q,p,mu=MU):
    """Exact piecewise sinusoidal global maximization, including pressure kinks.

    Enumeration partitions at shear sign, normal-pressure ordering and zero
    pressure. No sampled angle is substituted for a genuine maximizer.
    """
    d,e=v2(tau),v2(tau-Q);pbar=p-.5*(tau[0,0]+tau[1,1])
    if np.linalg.norm(d)==0. and np.linalg.norm(e)==0.:
        return dict(maximum=-mu*max(0.,pbar),candidates=[],continuum=True)
    bounds=[0.,np.pi]+_roots(e[1],-e[0],0.)+_roots(*d,0.)
    bounds+=_roots(*d,pbar)+_roots(*d,-pbar)
    bounds=sorted(set(bounds));points=list(bounds[:-1])
    for left,right in zip(bounds[:-1],bounds[1:]):
        theta=.5*(left+right);v=np.array([np.cos(2*theta),np.sin(2*theta)])
        shear=e@J(v);normal=d@v
        coefficient=np.sign(shear)*(-J(e))
        if pbar-abs(normal)>0.:
            coefficient+=mu*np.sign(normal)*d
        if np.linalg.norm(coefficient)>0.:
            peak=.5*np.arctan2(coefficient[1],coefficient[0])%np.pi
            for angle in [peak,(peak+np.pi/2)%np.pi]:
                if left<=angle<=right:points.append(angle)
    values=[objective(tau,Q,p,a,mu)[0] for a in points];maximum=max(values)
    tol=tolerance(tau,Q,p,mu);candidates=[]
    for angle,value in zip(points,values):
        if maximum-value>tol:continue
        f,pa,pb,shear=objective(tau,Q,p,angle,mu)
        if pb<pa-tol:angle=(angle+np.pi/2)%np.pi
        f,pa,pb,shear=objective(tau,Q,p,angle,mu)
        sign=1. if shear>=0 else -1.
        a,b,m,mp,mpp,P,Pp,Ppp=geometry(angle,sign)
        N=np.sqrt(2)*m
        if any(np.linalg.norm(c['N']-N)<1e-12 and abs(c['normal']@a)>1-1e-12 for c in candidates):continue
        candidates.append(dict(theta=angle,sign=sign,normal=a,N=N,m=m,maximum=f,pn=min(pa,pb),
                               normal_order_tie=abs(pa-pb)<=tol,
                               curvature=inner(tau-Q,mpp)+mu*inner(tau,Ppp)))
    return dict(maximum=maximum,candidates=candidates,continuum=False)


def select(result,history_normal=None):
    """History resolves genuine max ties only; no preference invents a plane."""
    candidates=result['candidates']
    if not candidates:return None,'continuum direction symmetry'
    if len(candidates)==1:return candidates[0],'unique maximizer'
    if history_normal is None:return None,'unresolved conjugate symmetry'
    n=np.array(history_normal);n=n/np.linalg.norm(n)
    scores=np.array([(c['normal']@n)**2 for c in candidates]);best=max(scores)
    admissible=np.flatnonzero(best-scores<=64*EPS)
    if len(admissible)!=1:return None,'unresolved equal-distance history'
    return candidates[int(admissible[0])],'continuity among equal maximizers'


def power(tau,Q,p,W,mu=MU):
    magnitude=np.linalg.norm(W)
    if magnitude==0:return dict(defined=False,friction=0.,conjugate_work=0.,defect=0.)
    # Any symmetric trace-free in-plane W defines an unordered orthogonal pair.
    theta=.5*np.arctan2(-W[0,0],W[0,1])%np.pi
    _,pa,pb,_=objective(tau,Q,p,theta,mu)
    dissipation=np.sqrt(2)*mu*min(pa,pb)*magnitude
    work=inner(tau-Q,W)
    return dict(defined=True,friction=dissipation,conjugate_work=work,defect=work-dissipation)


def legacy_normal(tau,L,mu=MU,literal=False):
    _,eigenvectors=np.linalg.eigh(tau[:2,:2]);a1=eigenvectors[:,0]
    omega=.5*(L[1,0]-L[0,1]);angle=np.pi/4-.5*np.arctan(mu)
    sign=1.
    if not literal and abs(omega)>EPS:sign=np.sign(omega)
    # Literal old 2-D t[0] remains NaN; its is_finite guard always fails.
    n=a1*np.sin(angle)+sign*J(a1)*np.cos(angle)
    return np.r_[n,0.],omega


def point_return(tau_trial,Qref,p,h,beta=1.,mu=MU,history_normal=None):
    """Feng Eq72 plus final global Eq56, with independent angular root solving.

    ξ is the engineering increment of relaxing A, so ΔK=ξ m/beta in the
    explicitly mapped Maxwell comparison. A failed global check is reported.
    """
    assert h>0 and p>max(np.linalg.eigvalsh(tau_trial[:2,:2]))
    trial=directions(tau_trial,Qref,p,mu);tol=tolerance(tau_trial,Qref,p,mu)
    if trial['maximum']<=tol:
        return dict(accepted=True,sliding=False,tau=tau_trial,Q=Qref,xi=0.,candidates=[])
    def at(theta):
        a,b,S,*_=geometry(theta);shear=inner(tau_trial-Qref,S)
        sign=1. if shear>=0 else -1.
        a,b,m,mp,mpp,P,Pp,Ppp=geometry(theta,sign)
        pn=p-inner(tau_trial,P)
        f=inner(tau_trial-Qref,m)-mu*pn
        xi=f/(G*(1+1/h))
        tau=tau_trial-2*G*xi*m;Q=Qref+2*G*xi/h*m
        stationarity=inner(tau-Q,mp)+mu*inner(tau,Pp)
        curvature=inner(tau-Q,mpp)+mu*inner(tau,Ppp)
        return dict(theta=theta%np.pi,sign=sign,normal=a,m=m,xi=xi,tau=tau,Q=Q,
                    stationarity=stationarity,curvature=curvature,pn=pn,beta=beta,h=h)
    grid=np.linspace(0,np.pi,513);sample=[at(t) for t in grid];roots=[]
    for left,right,x,y in zip(grid[:-1],grid[1:],sample[:-1],sample[1:]):
        if x['stationarity']*y['stationarity']<0:
            roots.append(brentq(lambda t:at(t)['stationarity'],left,right,xtol=2e-14))
        if abs(x['stationarity'])<=tol:roots.append(left)
    raw=[];accepted=[]
    for theta in roots:
        r=at(theta)
        if r['xi']<=0 or abs(r['stationarity'])>1e-8 or r['curvature']>=0:continue
        if any(np.linalg.norm(r['m']-x['m'])<1e-10 for x in raw):continue
        maxima=directions(r['tau'],r['Q'],p,mu)
        r['global_margin']=maxima['maximum'];r['global_allowance']=tol
        r['closed']=p>max(np.linalg.eigvalsh(r['tau'][:2,:2]))
        selected,reason=select(maxima,history_normal);r['selection']=reason
        r['continuity_ok']=selected is None or np.linalg.norm(selected['N']-np.sqrt(2)*r['m'])<1e-10
        raw.append(r)
        if r['closed'] and r['global_margin']<=tol and r['continuity_ok']:accepted.append(r)
    if len(accepted)!=1:
        return dict(accepted=False,sliding=True,candidates=raw,admissible_count=len(accepted),
                    reason='no global single-direction return' if not accepted else 'unresolved multiple returns')
    return dict(accepted[0],accepted=True,sliding=True,candidates=raw)


def algorithmic_columns(r,mu=MU):
    theta,sign,xi,h,beta=[r[k] for k in ['theta','sign','xi','h','beta']]
    a,b,m,mp,mpp,P,Pp,Ppp=geometry(theta,sign)
    F=m+mu*P;Fp=mp+mu*Pp
    matrix=np.array([[-2*G*(inner(F,m)+inner(m,m)/h),
                      -2*G*xi*(inner(F,mp)+inner(m,mp)/h)],
                     [-2*G*(inner(Fp,m)+inner(mp,m)/h),
                      r['curvature']-2*G*xi*(inner(Fp,mp)+inner(mp,mp)/h)]])
    columns=[]
    for j,E in enumerate([deviator(xx=1),deviator(yy=1),deviator(gamma=1),np.zeros((3,3))]):
        dtrial=2*G*beta*E;dp=G if j==3 else 0.
        dxi,dtheta=np.linalg.solve(matrix,-np.array([inner(F,dtrial)-mu*dp,inner(Fp,dtrial)]))
        dtau=dtrial-2*G*(m*dxi+xi*mp*dtheta)
        columns.append(components(dtau)/G)
    return np.array(columns).T


def rate_columns(theta,sign,h,mu=MU):
    a,b,m,mp,mpp,P,Pp,Ppp=geometry(theta,sign);F=m+mu*P
    columns=[]
    for j,E in enumerate([deviator(xx=1),deviator(yy=1),deviator(gamma=1),np.zeros((3,3))]):
        if h==0.:d=2*E
        else:
            factor=(inner(F,E)-mu*.5*(j==3))/(inner(F,m)+inner(m,m)/h)
            d=2*(E-factor*m)
        columns.append(components(d))
    return np.array(columns).T


def symbol(C,theta):
    n=np.array([np.cos(theta),np.sin(theta)]);a=J(n)
    def traction(v):
        e=np.array([v[0]*n[0],v[1]*n[1],v[0]*n[1]+v[1]*n[0]])
        t=C[:,:3]@e
        return np.array([t[0]*n[0]+t[3]*n[1],t[3]*n[0]+t[1]*n[1]])
    A=np.column_stack([traction(np.array([1.,0.])),traction(np.array([0.,1.]))])
    P=np.array([[C[0,3],C[3,3]],[C[3,3],C[1,3]]])-np.eye(2)
    b=P@n;v=A@a
    mixed=np.zeros((3,3));mixed[:2,:2]=A;mixed[:2,2]=b;mixed[2,:2]=n
    pressure=-n@v/(n@b);coefficient=a@(v+pressure*b)
    return dict(mixed=coefficient,fixed_pressure=a@v,determinant=np.linalg.det(mixed),
                matrix=mixed,null_vector=np.r_[a,pressure],pressure_over_G=pressure)
