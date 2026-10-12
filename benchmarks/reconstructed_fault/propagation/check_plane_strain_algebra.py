#!/usr/bin/env python3
"""Explicit 3D energy, work, Maxwell and helper-precondition checks."""
import json
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss

OUT=Path(__file__).resolve().parent/'plane_strain_evidence'
G=1e6; ETA=1e8; LAMBDA=ETA/G
S=np.zeros((3,3));S[0,1]=S[1,0]=.5

def embed_raw(a):
    b=np.zeros((3,3));b[:2,:2]=a
    return b-np.trace(b)*np.eye(3)/3

def embed_block(a):
    b=np.zeros((3,3));b[:2,:2]=a;b[2,2]=-np.trace(a)
    return b

def dot(a,b):return np.sum(a*b)
def metric(a,b):return dot(a,b)+np.trace(a)*np.trace(b)
def h(phi):return 128*phi*(1+phi)/(1-phi)**2

cases=[]
for name,a in [('shear',np.array([[0.,.7],[.7,0.]])),
               ('extension',np.diag([.8,-.8])),
               ('nonzero_trace',np.array([[.8,.3],[.3,.2]]))]:
    full=embed_raw(a);b=full[:2,:2]
    assert np.max(abs(full-embed_block(b)))<1e-15
    assert abs(dot(full,full)-metric(b,b))<1e-15
    assert abs(dot(full,full)-(dot(a,a)-np.trace(a)**2/3))<1e-15
    # Energy derivative with raw strain as independent variable; normal traction
    # uses the in-plane deviator without applying trace removal twice.
    direction=np.array([[.17,.23],[.23,-.09]])
    z=250.;chi=.8;stress=2*G*full-2*z*chi*S
    def energy(raw):return dot(2*G*embed_raw(raw)-2*z*chi*S,2*G*embed_raw(raw)-2*z*chi*S)/(4*G)
    derivative=dot(stress[:2,:2],direction)
    fd=(energy(a+1e-5*direction)-energy(a-1e-5*direction))/(2e-5)
    error=abs(fd-derivative)/max(1.,abs(derivative));assert error<1e-9
    # Phase derivatives at fixed raw FE anchor and fixed Z, including normal I.
    x=np.array([.37,.61]);weights=np.array([.4,.6]);d=np.array([.2,-.1])
    def phase_energy(ph,three):
        hh=h(ph);I=weights@hh
        value=z*z/(2*G*I)
        for w,local in zip(weights,hh/I):
            t=2*G*(full if three else np.pad(a,((0,1),(0,1))))-2*z*local*S
            value+=w*dot(t,t)/(4*G)
        return value
    expected_offset=G*np.trace(a)**2/3
    offsets=[phase_energy(x+k*d,False)-phase_energy(x+k*d,True) for k in [-.001,0,.001]]
    assert max(abs(np.array(offsets)-expected_offset))<1e-8
    # Direct projected-integral chain rule at a nonuniform evolving phase.
    # Subtract the phase-independent anchor energy before differencing, using
    # the exact quadratic expansion, so a large constant does not mask it.
    hh=h(x);I=weights@hh;hp=128*(4/(1-x)**3-3/(1-x)**2)
    dI=weights@(hp*d);dchi=hp*d/I-hh*dI/I**2
    analytic=-z*z*dI/(2*G*I**2)
    for w,local,change in zip(weights,hh/I,dchi):
        analytic-=w*z/G*dot(2*G*full-2*z*local*S,S)*change
    def phase_variable_energy(ph):
        hh=h(ph);I=weights@hh;chi=hh/I;anchor=2*G*full
        return z*z/(2*G*I)+sum(w*(-z/G*dot(anchor,S)*c+z*z/G*dot(S,S)*c*c) for w,c in zip(weights,chi))
    assert abs(phase_energy(x,True)-dot(2*G*full,2*G*full)/(4*G)-phase_variable_energy(x))<1e-8
    fd=(phase_variable_energy(x+1e-5*d)-phase_variable_energy(x-1e-5*d))/(2e-5)
    phase_error=abs(fd-analytic)/max(1.,abs(analytic));assert phase_error<1e-8
    cases.append(dict(case=name,raw_trace=float(np.trace(a)),deviatoric_block_trace=float(np.trace(b)),energy_derivative_relative_error=error,phase_derivative_relative_error=phase_error,raw_minus_3D_storage=expected_offset,normal_traction_yy=float(stress[1,1]),phase_independent_energy_offset_max_error=float(max(abs(np.array(offsets)-expected_offset)))))

# Polynomial manufactured nonuniform velocity/internal strain, not a physical
# compressible flow. Compare exact embedding pointwise, with general beta.
maxwell=[];zq,wq=leggauss(80);tq=(zq+1)/2;wq=wq/2
for ratio in [.00125,1.,4.]:
    dt=ratio*LAMBDA;beta=np.exp(-ratio);kappa=-ETA*np.expm1(-ratio)
    maximum=work_error=derivative_error=0.
    for x,y in [(0.13,.21),(.47,.82),(.89,.61)]:
        anchor=np.array([[2*x*y,.5*(x*x-y*y)],[.5*(x*x-y*y),.6*x*y]])*1e-4
        rate=np.array([[2*x*y,.5*(x*x+.5*y*y)],[.5*(x*x+.5*y*y),x*y]])*1e-5
        phi=.4+.2*y;I=100.;chi=h(phi)/I;Z=70.;V=2e-5*(1+x)
        initial=2*G*embed_raw(anchor)-2*Z*chi*S
        new_anchor=beta*anchor+kappa/G*rate;new_Z=beta*Z+kappa*V
        endpoint=2*G*embed_raw(new_anchor)-2*new_Z*chi*S
        forcing=embed_raw(rate)-chi*V*S
        expected=beta*initial+2*kappa*forcing
        relative=np.max(abs(endpoint-expected))/max(1.,np.max(abs(initial)),np.max(abs(endpoint)))
        maximum=max(maximum,relative);assert relative<2e-14
        W=Db=0.
        for u,w in zip(tq,wq):
            b=np.exp(-u*ratio);t=b*initial+2*ETA*(1-b)*forcing
            W+=dt*w*dot(t,forcing);Db+=dt*w*dot(t,t)/(2*ETA)
        delta=(dot(endpoint,endpoint)-dot(initial,initial))/(4*G)
        remainder=W-Db-delta
        energy_scale=max(1.,dot(endpoint,endpoint)/(4*G)+dot(initial,initial)/(4*G)+abs(W)+abs(Db))
        error=abs(remainder)/energy_scale;work_error=max(work_error,error)
        assert abs(remainder)<=256*np.finfo(float).eps*energy_scale
    maxwell.append(dict(dt_over_lambda=ratio,FE_recurrence_relative_error=maximum,finite_step_bulk_work_relative_error=work_error))
result=dict(cases=cases,general_Maxwell=maxwell,pressure_redefined=False,
            note='Nonzero-trace tests are operator diagnostics; no compressible physical benchmark',
            helpers_checked_natively='plane-strain-algebra.csv in before/after native fixtures')
(OUT/'algebra.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
