"""Manufactured moving material interface and independent characteristic reference.

The physical direct branch equation supplies the reference; residual histories
are integrated from their own sources. No reconstruction defines a residual.
"""
from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss
from stage_b_fixed import norm3_squared, rotate, spin_source

G, ELL, HALF, END = 3e6, .06, .4, .8
SHEAR, CY, OMEGA, ALPHA_LIMIT = .4, .12, -.2, 1.2
D = np.array([0.,0.,SHEAR/2])
T0=np.array([np.cos(np.deg2rad(37.)),np.sin(np.deg2rad(37.))])
N0=np.array([-T0[1],T0[0]])
L=np.array([[0.,SHEAR],[0.,0.]])


@lru_cache(maxsize=16)
def gauss(order):
    return leggauss(order)


def rate(y):
    return .5+.3*np.tanh(y/.08)


def accumulated_rate(Y,start,end):
    logcosh=lambda x:np.logaddexp(x,-x)-np.log(2.)
    return .5*(end-start)+.3*.08/CY*(logcosh((Y+CY*end)/.08)-logcosh((Y+CY*start)/.08))


def centre(time,cx):
    return np.array([cx*time+.5*SHEAR*CY*time*time,CY*time])


def position(material,time,cx):
    result=np.array(material,copy=True)
    result[...,0]+=SHEAR*time*material[...,1]
    return result+centre(time,cx)


def material_coordinates(points,time,cx):
    result=points-centre(time,cx)
    result=np.array(result,copy=True);result[...,0]-=SHEAR*time*result[...,1]
    return result


def compact_symmetric(a,b):
    return np.array([a[0]*b[0],a[1]*b[1],.5*(a[0]*b[1]+a[1]*b[0])])


@lru_cache(maxsize=16384)
def geometry(time):
    tangent=np.array([T0[0]+SHEAR*time*T0[1],T0[1]])
    g=np.linalg.norm(tangent);tangent/=g
    normal=np.array([-tangent[1],tangent[0]])
    stretch=float(tangent@L@tangent);turn=float(normal@L@tangent)
    S=compact_symmetric(tangent,normal)
    Sdot=turn*(compact_symmetric(normal,normal)-compact_symmetric(tangent,tangent))
    return tangent,normal,float(g),stretch,turn,S,Sdot


@lru_cache(maxsize=16384)
def surface_coefficients(time,order=96):
    p,w=gauss(order)
    # L2 physical ds=g*dalpha; common g cancels in the two-mode mass solve.
    y=CY*time+ALPHA_LIMIT*p*(T0[1])
    values=rate(y)
    return float(.5*np.dot(w,values)),float(1.5*np.dot(w,p*values))


def kinematics(points,time,cx):
    tangent,normal,g,stretch,turn,S,Sdot=geometry(float(time))
    relative=points-centre(time,cx)
    z=relative@normal;alpha=(relative@tangent)/g
    z0=g*z
    chi=g*np.exp(-(z0/ELL)**2)/(ELL*np.sqrt(np.pi))
    alpha_dot=z*(tangent@L@normal+normal@L@tangent)/g
    c0,c1=surface_coefficients(float(time))
    rg=c0+c1*alpha/ALPHA_LIMIT
    amplitude=.025*(-np.expm1(-.35*time))/.35
    modulation=1+.4*np.sin(6*alpha)
    s=amplitude*modulation
    sa=amplitude*2.4*np.cos(6*alpha)
    sdot=.025*np.exp(-.35*time)*modulation
    V=sdot+rg*s
    return dict(alpha=alpha,z=z,z0=z0,chi=chi,alpha_dot=alpha_dot,rg=rg,s=s,sa=sa,V=V,
                rb=rate(points[...,1]),S=S,Sdot=Sdot,stretch=stretch,t=tangent,n=normal,g=g)


def sources(points,time,cx):
    f=kinematics(points,time,cx)
    chi,s,S=f['chi'],f['s'],f['S']
    geometry_part=-2*G*(chi*s)[...,None]*(f['stretch']*S+f['Sdot']-spin_source(OMEGA,S))
    extension_part=-2*G*(chi*f['alpha_dot']*f['sa'])[...,None]*S
    rate_part=-2*G*(chi*s*(f['rb']-f['rg']))[...,None]*S
    direct=2*G*(D-(chi*f['V'])[...,None]*S)
    trial=np.broadcast_to(2*G*D,direct.shape)
    residual=geometry_part+extension_part+rate_part
    return np.concatenate([direct,trial,residual],axis=-1),(geometry_part,extension_part,rate_part)


def initial(material):
    X,Y=material[...,0],material[...,1]
    return G*np.stack([.06+.02*np.cos(4*X)+.01*np.sin(5*Y),
                       .02+.015*np.sin(3*X)*np.cos(4*Y),.025+.01*np.cos(4*(X+Y))],axis=-1)


def propagate(values,dt,attenuation):
    shape=values.shape
    values=values.reshape(shape[0],-1,3)
    return (np.exp(-attenuation)[:,None,None]*rotate(values,OMEGA,dt)).reshape(shape)


def reference(points,time,cx,order=32,parts=False):
    """Independent exact-characteristic physical and residual source integrals.

    Gaussian quadrature only integrates smooth temporal forcing; advection,
    total spin and accumulated local relaxation have independent exact maps.
    """
    points=np.atleast_2d(points);material=material_coordinates(points,time,cx)
    before=initial(material)
    base=np.concatenate([before,before,np.zeros_like(before)],axis=1)
    if time==0.:
        return (base,np.zeros((len(points),9))) if parts else base
    answer=propagate(base,time,accumulated_rate(material[:,1],0.,time))
    terms=np.zeros((len(points),9))
    nodes,weights=gauss(order)
    for u,w in zip(.5*time*(nodes+1),.5*time*weights):
        at=position(material,float(u),cx)
        force,pieces=sources(at,float(u),cx)
        attenuation=accumulated_rate(material[:,1],float(u),time)
        answer+=w*propagate(force,time-u,attenuation)
        if parts:terms+=w*propagate(np.concatenate(pieces,axis=1),time-u,attenuation)
    return (answer,terms) if parts else answer


def reconstruct(histories,points,time,cx):
    f=kinematics(points,time,cx)
    B=2*G*(f['chi']*f['s'])[...,None]*f['S']
    return histories[...,3:6]-B-histories[...,6:9]


def tractions(stress,time):
    t,n,*_=geometry(float(time))
    shear=t[0]*n[0]*stress[...,0]+t[1]*n[1]*stress[...,1]+(t[0]*n[1]+t[1]*n[0])*stress[...,2]
    normal=n[0]**2*stress[...,0]+n[1]**2*stress[...,1]+2*n[0]*n[1]*stress[...,2]
    return shear,normal
