#!/usr/bin/env python3
"""Independent full mixed frozen-coefficient audit of ONE scalar-state model.

Full 3-D tensor contractions; no production return, damping or model edit.
The explicit Pranger-inspired scalar-rate operator enters the state row only.
"""
from pathlib import Path
import csv
import json
import numpy as np
import scipy.linalg as la

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT/'build-particle-free/pranger-feasibility'
OUT = ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/pranger_feasibility'
G, ETA, P0, MU0 = 1e6, 1e8, 10000., .1
A_RSF, B_RSF, GAMMA_REF, R0, LENGTH = .005, .01, 1e-4, .005, .003
RELAX = G/ETA


def tensor(xx=0., xy=0., mean=0.):
    return np.array([[mean+xx, xy, 0.], [xy, mean-xx, 0.], [0., 0., -2*mean]])


def vector(t):
    return np.array([.5*(t[0, 0]-t[1, 1]), t[0, 1]])


def inner(x, y):
    return np.sum(x*y)


def geometry(angle, sign):
    tangent = np.array([np.cos(angle), np.sin(angle), 0.])
    normal = np.array([-np.sin(angle), np.cos(angle), 0.])
    T, N = np.outer(tangent, tangent), np.outer(normal, normal)
    m = sign*(np.outer(tangent, normal)+np.outer(normal, tangent))/2
    first = sign*(N-T)
    return m, first, -4*m, N, -2*sign*m, -2*sign*first


def angles(tau, Q, pressure=P0, mu=MU0):
    d, f = vector(tau), vector(tau-Q)
    Jf = np.array([-f[1], f[0]])
    return [.5*np.arctan2(*(-sign*Jf-mu*d)[::-1]) for sign in [1, -1]]


def rsf(q, theta=1., a=A_RSF, b=B_RSF):
    assert q > 0 and theta > 0 and a > 0
    reference = MU0+a*np.log1p(-np.exp(-2*MU0/a))
    log_z = np.log(q/2)+(reference+b*np.log(theta))/a
    if log_z >= 0:
        inverse=np.exp(-2*log_z)
        value=log_z+np.log1p(np.sqrt(1+inverse))
        fraction=1/np.sqrt(1+inverse)
    else:
        z=np.exp(log_z);value=np.arcsinh(z);fraction=z/np.hypot(1.,z)
    return a*value, a*fraction/q, b*fraction/theta


def make_double(phi, elastic_memory, imbalance=0., mature=False):
    h = np.inf if mature else 4*phi*(1+phi)/(1-phi)**2
    qxy = 0. if mature else 2*G*elastic_memory/h
    shear = (qxy+np.sqrt((1+MU0**2)*(MU0*P0)**2-MU0**2*qxy*qxy))/(1+MU0**2)
    tau, Q = tensor(xy=shear), tensor(xy=qxy)
    psi = angles(tau, Q)
    m = [geometry(p, s)[0] for p, s in zip(psi, [1, -1])]
    speeds = np.array([1+imbalance, 1-imbalance])
    W = sum((s*x for s, x in zip(speeds, m)), np.zeros((3, 3)))
    speeds *= GAMMA_REF/np.sqrt(2*inner(W, W))
    return dict(phi=1. if mature else phi, h=h, tau=tau, Q=Q, psi=psi, speeds=speeds, active=[0, 1],
                pressure=P0, mature=mature, theta=1.,
                origin=('Analytic exact mature specialization Q=0, not a finite-h API call or native checkpoint.' if mature else
                        'Synthetic admissible instantaneous state, not a native accepted checkpoint.'),
                elastic_memory=elastic_memory)


def archived_single():
    path=OUT/'archived-single-input.csv'
    with path.open() as f:
        row=next(r for r in csv.DictReader(f) if r['kind']=='path' and float(r['phi'])==.8
                 and int(r['step'])==9 and int(r['active']) in [1, 2])
    tau=np.array([[float(row['tauxx']),float(row['tauxy']),0.],
                  [float(row['tauxy']),float(row['tauyy']),0.],[0.,0.,float(row['tauzz'])]])
    Q=tensor(float(row['Qxx']),float(row['Qxy']))
    active=[0 if int(row['active'])==1 else 1]
    return dict(phi=.8,h=float(row['h']),tau=tau,Q=Q,psi=angles(tau,Q),
                speeds=np.array([GAMMA_REF]),active=active,pressure=float(row['P']),mature=False,theta=1.,
                origin='Archived phi=.8 step9 physical tau/Q/pressure; flow magnitude rescaled to q=1. Synthetic frozen-state control, not a replay.')


def verify_base(base):
    mu,_,_=rsf(1.)
    margins=[];curvatures=[]
    W=np.zeros((3,3))
    for i in [0,1]:
        m,mp,mpp,N,Np,Npp=geometry(base['psi'][i],1 if i==0 else -1)
        pressure=base['pressure']-inner(N,base['tau'])
        margin=inner(base['tau']-base['Q'],m)-mu*pressure
        stationarity=inner(base['tau']-base['Q'],mp)+mu*inner(Np,base['tau'])
        curvature=inner(base['tau']-base['Q'],mpp)+mu*inner(Npp,base['tau'])
        assert pressure>0 and abs(stationarity)<1e-9 and curvature<0
        assert margin<=1e-9
        if i in base['active']:
            assert abs(margin)<1e-9
            W+=base['speeds'][base['active'].index(i)]*m
        margins.append(margin);curvatures.append(curvature)
    equivalent=np.sqrt(2*inner(W,W))
    assert abs(equivalent/GAMMA_REF-1)<1e-12
    eigen=la.eigvalsh(base['pressure']*np.eye(3)-base['tau'])
    assert min(eigen)>0 and min(base['speeds'])>0
    return dict(margins_Pa=margins,plane_curvatures_Pa=curvatures,
                min_physical_compression_Pa=float(min(eigen)),q=equivalent/GAMMA_REF,
                active_rates_s1=base['speeds'].tolist(),W=W.tolist(),origin=base['origin'])


def pencil(base, normal_angle, kell, b=B_RSF, a=A_RSF, interaction=True):
    """Reduced EXACT mixed plane-wave pencil; mean stress/zz retained.

    Dynamic y=[normal-mode deviator/G, planar mean/G, Qxx/G,Qxy/G,Theta].
    In explicit mature mode Q variables are absent. Pressure p/G=mean+normal.
    Normal traction/shear equilibrium and incompressibility eliminate velocity
    and p exactly, not by fixing p. Algebraic z=[delta w_i/gamma_ref,delta psi_i].
    """
    mature=base['mature'];ny=3 if mature else 5;r=len(base['active'])
    state=ny-1
    wave=np.array([np.cos(2*normal_angle),np.sin(2*normal_angle)])
    stress=[G*tensor(*wave),G*tensor(mean=1.)]
    shape=np.zeros((2*r,3,3))
    planes=[];W=np.zeros((3,3))
    for j,i in enumerate(base['active']):
        plane=geometry(base['psi'][i],1 if i==0 else -1);planes.append(plane)
        shape[j]=GAMMA_REF*plane[0];shape[r+j]=base['speeds'][j]*plane[1]
        W+=base['speeds'][j]*plane[0]
    q=np.sqrt(2*inner(W,W))/GAMMA_REF
    dq=np.array([2*inner(W,s)/(GAMMA_REF**2*q) for s in shape])
    mu,muq,mutheta=rsf(q,base['theta'],a,b)
    B=np.zeros((2*r,ny));C=np.zeros((2*r,2*r))
    for j,plane in enumerate(planes):
        m,mp,mpp,N,Np,Npp=plane
        pn=base['pressure']-inner(N,base['tau'])
        pnprime=-inner(Np,base['tau'])
        curvature=inner(base['tau']-base['Q'],mpp)+mu*inner(Npp,base['tau'])
        for k,dtau in enumerate(stress):
            # Both stress variations also have delta p=G. Keeping the physical
            # normal traction derivative is essential for oblique waves.
            B[j,k]=(inner(dtau,m)+mu*inner(N,dtau)-mu*G)/G
            B[r+j,k]=(inner(dtau,mp)+mu*inner(Np,dtau))/G
        if not mature:
            for c,dQ in enumerate([G*tensor(xx=1.),G*tensor(xy=1.)]):
                B[j,2+c]=-inner(dQ,m)/G;B[r+j,2+c]=-inner(dQ,mp)/G
        B[j,state]=-pn*mutheta/G
        B[r+j,state]=-pnprime*mutheta/G
        C[j]=-pn*muq*dq/G
        C[r+j]=-pnprime*muq*dq/G
        C[r+j,r+j]+=curvature/G
    A=-RELAX*np.eye(ny);D=np.zeros((ny,2*r))
    for j,shape_j in enumerate(shape):
        D[0,j]=-2*wave@vector(shape_j)
        if not mature:
            D[2,j]=2*shape_j[0,0]/base['h'];D[3,j]=2*shape_j[0,1]/base['h']
    multiplier=1-kell*kell if interaction else 1.
    A[state,state]=-R0*q
    D[state]=-R0*base['theta']*multiplier*dq
    # A y + D z = ydot; B y + C z = 0.
    L=np.block([[A,D],[B,C]])
    M=np.diag(np.r_[np.ones(ny),np.zeros(2*r)])
    return dict(L=L,M=M,A=A,D=D,B=B,C=C,dq=dq,shape=shape,state=state)


def reduction(p):
    A,D,B,C=[p[k] for k in ['A','D','B','C']]
    u,s,vh=la.svd(C)
    rank=int(sum(s>1e-11*s[0]))
    r=C.shape[0]//2
    if rank==2*r:
        operator=A-D@la.solve(C,B)
        return dict(operator=operator,basis=np.eye(len(A)),rank=rank,denominator=None)
    assert rank==2*r-1,(rank,s)
    null=vh[-1];left=u[:,-1];constraint=left@B
    scale=la.norm(B)
    if la.norm(constraint)<1e-10*scale:
        return dict(operator=None,rank=rank,singular=True,null=null,constraint=constraint,
                    free_dynamic_norm=float(la.norm(D@null)))
    constraint/=la.norm(constraint)
    particular=-la.pinv(C,rtol=1e-11)@B
    unconstrained=A+D@particular
    denominator=constraint@D@null
    basis=la.null_space(constraint[None,:])
    operator=unconstrained-np.outer(D@null,constraint@unconstrained)/denominator
    return dict(operator=basis.T@operator@basis,basis=basis,rank=rank,denominator=float(denominator),
                null=null,constraint=constraint,free_dynamic=D@null)


def spectrum(base, angle, kell, **kwargs):
    p=pencil(base,angle,kell,**kwargs);reduced=reduction(p)
    if reduced.get('singular',False):return None,p,reduced
    values=la.eigvals(reduced['operator'])
    return values,p,reduced
