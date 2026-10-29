#!/usr/bin/env python3
"""Independent weak Gauss admission reference; no production dependency."""
import sys,json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
import analyze_cohesive_interface_native as previous
ref=previous.ref

def initialize(n):
    Q,w,x,P=previous.surface_rule(n)
    return dict(wall=0.,mem=np.zeros(n),jump=np.zeros(3*n),a=((x>.375)&(x<.625)).astype(float),age=np.ones(3*n))

def front_layout(K,N,*,length=1.,height=.125,movable=None):
    """Cache the fixed Q1/Gauss3 layout; K/N retain lumped-row force units."""
    n=len(K);Q,w,x,P=previous.surface_rule(n);w=w*length
    return dict(Q=csr_matrix(Q),R=csr_matrix(Q.T*w[None,:]/(length/n)),
                w=w,x=x*length,P=P,PN=P@N,height=height,
                movable=np.ones(n,bool) if movable is None else np.asarray(movable,bool))

def solve_front(K,N,old,dt,du,*,layout=None,predictor=None):
    """Positive formation predictor, certified by unrestricted weak multipliers.

    The predictor is not a constitutive constraint: acceptance requires a full
    Gauss box-feasibility certificate, including negative/alternating directions.
    """
    n=len(K)
    if layout is None:layout=front_layout(K,N)
    Q,R,w,PN=(layout[k] for k in ['Q','R','w','PN']);movable=layout['movable']
    beta=1/(1+ref.RELAX*dt);wall=beta*(old['wall']+du);virgin=old['a']==0
    bound=np.asarray(Q[virgin].sum(axis=0)).ravel()>0
    inc=old.get('rate',np.zeros(n)).copy()*dt if predictor is None else np.asarray(predictor).copy()
    inc[bound]=np.maximum(0,inc[bound]);inc[~movable]=0.
    def evaluate(inc):
        z=beta*(old['mem']+inc);stress=ref.G/layout['height']*wall-K@z;pn=ref.P0+PN@z;dq=Q@inc;V=dq/dt
        mu,muv,theta=previous.state_friction(abs(V),old['age'],dt);a,q,dqde,da,storage=ref.bond(old['jump']+dq,old['a'],0.,np.ones(3*n));friction=np.sign(V)*(pn*mu+ref.ETA_RAD*abs(V))
        tied=virgin&(dq==0);q[tied]=ref.TAUP;dqde[tied]=-ref.TAUP/ref.DF;da[tied]=1/ref.DF
        if min(pn)<=0:raise RuntimeError('Independent front lost closed contact')
        res=q+a*friction
        C=Q.multiply((dqde+da*friction+a*(pn*muv+ref.ETA_RAD)/dt)[:,None]).toarray()+beta*(a*np.sign(V)*mu)[:,None]*PN
        f=stress-R@res;J=-beta*K-R@C
        return f,J,(z,stress,pn,dq,V,a,theta,storage,res,tied,q,mu)
    for it in range(100):
        f,J,data=evaluate(inc);free=movable&(~bound|(inc>0)|(f>0));error=max(abs(f[free]),default=0.)
        if error<1e-8 and max(f[movable&~free],default=0.)<1e-8:break
        step=np.zeros(n);step[free]=np.linalg.solve(J[np.ix_(free,free)],-f[free]);scale=1.
        blocking=bound&(step<0)
        if np.any(blocking):scale=min(scale,float(np.min(-inc[blocking]/step[blocking])))
        found=False
        for ls in range(30):
            cand=inc+scale*step;cand[bound]=np.maximum(cand[bound],0);ff,_,_=evaluate(cand)
            projected=np.where(bound&(cand==0),np.maximum(ff,0),ff)[movable]
            if np.linalg.norm(projected)<np.linalg.norm(np.where(bound&(inc==0),np.maximum(f,0),f)[movable]) or max(abs(projected))<1e-8:
                inc=cand;found=True;break
            scale*=.5
        if not found:raise RuntimeError(('positive linesearch',it,error))
    else:raise RuntimeError(('positive iterations',error))
    z,stress,pn,dq,V,a,theta,storage,res,tied,q,mu=data
    # Independent exact box feasibility; no pointwise projected-stress shortcut.
    residual=stress-R[:,~tied]@res[~tied]
    if np.any(tied):
        lp=linprog(np.zeros(sum(tied)),A_eq=R[movable][:,tied],b_eq=residual[movable]/ref.TAUP,bounds=(-1,1),method='highs',options={'dual_feasibility_tolerance':1e-9,'primal_feasibility_tolerance':1e-9})
        if not lp.success:raise RuntimeError(('weak admission certificate',lp.message))
        res[tied]=lp.x*ref.TAUP
    weak_error=max(abs((stress-R@res)[movable]))
    assert weak_error<=1e-8+64*np.finfo(float).eps*(max(abs(stress))+max(abs(res))),weak_error
    state=dict(wall=wall,mem=z,jump=old['jump']+dq,a=a,age=theta,rate=inc/dt)
    # For Q1 with interior positive Gauss points, one tied cell sets both
    # adjacent nodal increments to zero. This is its exact nullspace, not a
    # positive-cone stability test. Retain the complete signed box certificate.
    admissible=movable&~(np.asarray(Q[tied].sum(axis=0)).ravel()>0)
    A=J[np.ix_(admissible,admissible)]
    curvature=np.linalg.eigvalsh(-(A+A.T)/2).min() if len(A) else 0.
    return state,dict(iterations=it,residual_Pa=error,weak_error_Pa=weak_error,tied=tied,traction=stress,pn=pn,point_resistance=res,V=V,curvature=curvature,cohesive_q=np.where(tied,res,q),storage=storage,mu=mu,increment=inc,jacobian=J)
