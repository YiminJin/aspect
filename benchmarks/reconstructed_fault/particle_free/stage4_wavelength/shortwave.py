#!/usr/bin/env python3
"""Compatible Fourier cross-section and high-frequency limits of the actual specimen.

Independent variational reference, not a replacement native solver. No phase,
stress, loading, quadrature or history of an ASPECT trajectory is changed.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,csv,time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.optimize import root
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
OUT=HERE.parent/'evidence/stage4_shortwave';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(HERE.parent/'stage4_rsf'))
from stability import friction
OLD=HERE.parent/'evidence/stage4_wavelength/native-128'
B=np.load(OLD/'phase-coefficients.npz')['coefficients'].reshape(128,128,3,3)
sections=np.genfromtxt(OLD/'restoring-sections.csv',delimiter=',',names=True)
operating=np.genfromtxt(OLD/'operating.csv',delimiter=',',names=True)
G=1e6;DC=3.2211126126299625e-7;R=.01

def phase(x,y):
    ix=min(int(x*16),127);u=x*16-ix
    iy=np.minimum((y*32).astype(int),127);v=y*32-iy
    bx=np.array([(1-u)**2,2*u*(1-u),u*u])
    by=np.stack([(1-v)**2,2*v*(1-v),v*v],axis=-1)
    return np.einsum('...i,...ij,j->...',by,B[iy,ix],bx)

def coefficients(x,ny,order):
    z,weight=np.polynomial.legendre.leggauss(order);t=(z+1)/2;dy=4/ny
    y=(np.arange(ny)[:,None]+t)*dy;w=np.broadcast_to(weight*dy/2,y.shape)
    phi=phase(x,y);h=4*phi*(1+phi)/(1-phi)**2;H=h/(1+h)
    angle=np.interp(x,sections['x'],sections['angle']);c=np.cos(2*angle);s=np.sin(2*angle)
    return t,dy,w,h,H,angle,c,s

def assemble(local,load,indices,ndofs):
    rows=np.broadcast_to(indices[:,:,None],local.shape).ravel()
    cols=np.broadcast_to(indices[:,None,:],local.shape).ravel()
    matrix=coo_matrix((local.ravel(),(rows,cols)),shape=(ndofs,ndofs)).tocsc()
    rhs=np.zeros(ndofs,dtype=load.dtype);np.add.at(rhs,indices.ravel(),load.ravel())
    return matrix,rhs

def compatible(x,wavelength,ny=512,order=10):
    """C1 cubic streamfunction: div u=0 exactly, u=0 at y=0,4.

    w=k^2 psi is scaled for a well-conditioned k->infinity mass limit.
    D=cos(2a)*(w+w''/k^2)-2i*sin(2a)*w'/k = 2 E:S.
    """
    t,dy,q,h,H,angle,c,s=coefficients(x,ny,order);k=2*np.pi/wavelength
    N=np.array([1-3*t*t+2*t**3,t-2*t*t+t**3,3*t*t-2*t**3,-t*t+t**3]).T
    dN=np.array([-6*t+6*t*t,1-4*t+3*t*t,6*t-6*t*t,-2*t+3*t*t]).T/dy
    ddN=np.array([-6+12*t,-4+6*t,6-12*t,-2+6*t]).T/dy**2
    Q=N+ddN/k**2;D=c*Q-2j*s*dN/k
    local=np.einsum('cq,qi,qj->cij',q,4*dN/k**2,dN)+np.einsum('cq,qi,qj->cij',q,Q,Q)-np.einsum('cq,qi,qj->cij',q*H,D.conj(),D)
    load=np.einsum('cq,qi->ci',q*H,D.conj())
    ids=2*np.arange(ny)[:,None]+np.arange(4)[None,:];ndofs=2*ny+2
    A,b=assemble(local,load,ids,ndofs);free=np.arange(2,ndofs-2)
    u=np.zeros(ndofs,dtype=complex);u[free]=spsolve(A[free][:,free],b[free])
    residual=np.linalg.norm((A@u-b)[free])/np.linalg.norm(b[free])
    assert residual<2e-7,(ny,wavelength,residual)
    addition=np.vdot(b,u).real;C=np.cos(angle)/G*(np.sum(q*H)+addition)
    # Independent energy at unit normalized resistance f/G, compatible strains.
    uq=u[ids];strain=-uq@Q.T;diag=-uq@dN.T/k
    d=-uq@D.T;e=H*(d-1)
    # E:E - e*(2E:S) + e^2/2 + e^2/(2h).
    cohesive=np.divide(abs(e)**2,h,out=np.zeros_like(h),where=h>0)
    density=2*abs(diag)**2+.5*abs(strain)**2-np.real(e.conj()*d)+.5*abs(e)**2+.5*cohesive
    energy=float(np.sum(q*density));work=.5*(np.sum(q*H)+addition)
    err=abs(energy-work)/work;assert err<2e-7,(ny,wavelength,err)
    return dict(x=float(x),wavelength=wavelength,ny=ny,quadrature=order,compliance=float(C),stiffness=float(1/C),residual=float(residual),work_error=float(err))

def limits(x,ny=128,order=10):
    """Continuum and fixed-normal-Q2 compatible mass limits, including angle."""
    t,dy,q,h,H,angle,c,s=coefficients(x,ny,order)
    N=np.array([2*t*t-3*t+1,4*t*(1-t),2*t*t-t]).T
    local=np.einsum('cq,qi,qj->cij',q*(1-H*c*c),N,N)
    load=np.einsum('cq,qi->ci',q*H*c,N)
    ids=2*np.arange(ny)[:,None]+np.arange(3)[None,:];ndofs=2*ny+1
    A,b=assemble(local,load,ids,ndofs);u=np.zeros(ndofs);u[1:-1]=spsolve(A[1:-1,1:-1],b[1:-1])
    c0=np.cos(angle)/G*np.sum(q*H);cm=np.cos(angle)/G*(b@u)
    continuum=np.cos(angle)/G*np.sum(q*h/(1+h*s*s))
    assert c0+cm<=continuum*(1+1e-10)
    return float(c0),float(cm),float(continuum)

def local_root(K,V,theta):
    _,fv,ft=friction(np.asarray(V),np.asarray(theta),DC)
    q=abs(V)/DC;c=ft*theta*np.sign(V)/DC
    linear=fv*(R+q)+K-c;constant=q*(fv*R+K)-R*c
    return (-linear+np.sqrt(linear**2-4*fv*constant+0j))/(2*fv)

def carrier_root(x,w,Cwork,Csquare):
    """Rapid fixed-patch carrier dispersion, actual V(x), theta(x), measures.

    Cwork = <P C P>, Csquare = <(C P)^2> in the fast coordinate;
    both retain the complete physical x-dependence of the normal profile.
    """
    V=np.interp(x,operating['x'],operating['V']);theta=np.interp(x,operating['x'],operating['theta'])
    _,fv,ft=friction(V,theta,DC);q=abs(V)/DC;c=ft*theta*np.sign(V)/DC
    I=np.sum(w*Cwork);J=np.sum(w*Csquare*fv)
    def equation(z):
        lam=z[0]+1j*z[1]
        value=(I/(lam+R)+np.sum(w*Csquare*(fv-c/(lam+q))))/J
        return [value.real,value.imag]
    roots=[]
    for seed in ([20,140],[40,100],[60,60],[80,5],[5,200]):
        answer=root(equation,seed,tol=1e-10)
        if answer.success and np.linalg.norm(equation(answer.x))<1e-9:
            lam=complex(answer.x[0],abs(answer.x[1]))
            if all(abs(lam-a)>1e-5 for a in roots):roots.append(lam)
    assert roots
    lam=max(roots,key=lambda z:z.real)
    return dict(growth=lam.real,imag=lam.imag,stiffness=float(I/np.sum(w*Csquare)),root_residual=float(np.linalg.norm(equation([lam.real,lam.imag]))))

def main():
    started=time.monotonic();peak=int(np.argmax(operating['V']));xc=float(operating['x'][peak])
    rows=[]
    for ny in (128,256,512):
        for wave in (.125,.0625,.03125,.015625,.0078125,.00390625):
            r=compatible(xc,wave,ny);z=local_root(r['stiffness'],operating['V'][peak],operating['theta'][peak]);r.update(local_growth=float(z.real),local_imag=float(z.imag));rows.append(r)
    for order in (6,16):rows.append(compatible(xc,.03125,512,order))
    # Patch integrals use the physical original cells, separately converged in x.
    carrier=[];profiles=[]
    for order in (6,10):
        t,w=np.polynomial.legendre.leggauss(order);x=((np.arange(40,88)[:,None]+.5+.5*t)*.0625).ravel();dx=np.tile(w*.0625/2,48)
        angle=np.interp(x,sections['x'],sections['angle']);measure=dx/np.cos(angle)
        values=np.array([limits(v,128,order) for v in x]);c0,cm,cc=values.T
        for name,work,square in [('continuum_resolved',cc,cc**2),('normal_Q2_resolved',c0+cm,(c0+cm)**2),
               ('normal_Q2_two_cell_sine',c0+8/np.pi**2*cm,c0**2+8/np.pi**2*(2*c0*cm+cm**2)),
               ('normal_Q2_two_cell_cosine',c0+96/np.pi**4*cm,c0**2+96/np.pi**4*(2*c0*cm+cm**2))]:
            carrier.append(dict(model=name,quadrature=order,**carrier_root(x,measure,work,square)))
        V=np.interp(x,operating['x'],operating['V']);theta=np.interp(x,operating['x'],operating['theta']);local=local_root(1/cc,V,theta)
        profiles=np.c_[x,V,theta,1/cc,1/(c0+cm),local.real,local.imag]
    # Finite-wavelength compatible responses across the patch, keeping the actual
    # transverse profiles. Slow-x/WKB approximation is explicit and checked at
    # progressively shorter waves; this is not the finite-body native spectrum.
    t,w=np.polynomial.legendre.leggauss(3);x=((np.arange(40,88)[:,None]+.5+.5*t)*.0625).ravel();dx=np.tile(w*.0625/2,48)
    angle=np.interp(x,sections['x'],sections['angle']);measure=dx/np.cos(angle)
    for wave in (.125,.0625,.03125,.015625,.0078125,.00390625):
        C=np.array([compatible(v,wave,256)['compliance'] for v in x])
        carrier.append(dict(model='finite_compatible_WKB',quadrature=10,wavelength=wave,ny=256,**carrier_root(x,measure,C,C*C)))
    record=dict(center_x=xc,center_limits=limits(xc),point_rows=rows,carrier_rows=carrier,seconds=time.monotonic()-started,
                assumptions='Fourier x, exact incompressibility, clamped y boundaries; finite-wave patch spectra are a slow-x approximation. No native full-space spectrum is claimed.')
    (OUT/'compatible-limit.json').write_text(json.dumps(record,indent=2)+'\n')
    np.savetxt(OUT/'local-limit-profile.csv',profiles,delimiter=',',header='x,V,theta,K_continuum,K_normal_Q2,local_growth,local_imag',comments='')
    for name,data in [('compatible-point',rows),('carrier-dispersion',carrier)]:
        with (OUT/(name+'.csv')).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in data for k in row)));writer.writeheader();writer.writerows(data)
    print(json.dumps(record,indent=2),flush=True)

if __name__=='__main__':main()
