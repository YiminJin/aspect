#!/usr/bin/env python3
"""Independent continuum stationary control and discrete derivative checks."""
import json
from pathlib import Path
import numpy as np
from localization_1d import Reference,native_profile,G,ETA,A,M,ELL,h,dh
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent/'localization_evidence'
results=[]
for n in [64,128,256]:
    r=Reference(n,1.);phi=native_profile(r.y);I,J=r.direct(phi)
    q=np.sqrt(2*G*A*.6/h(.6));T=q+1000.;Z=q*I
    # Analytic partial integrals of the sampled Q1 profile, then explicit Q2
    # nodal interpolation of d_R. This is a declared spatial approximation.
    def ray(l,rr,length):
        u,v=1-l,1-rr;z=(v-u)/u
        ratio=np.log1p(z)/z if abs(z)>1e-8 else sum((-1)**k*z**k/(k+1) for k in range(8))
        return M*length*(2/(u*v)-3*ratio/u+1)
    integral=np.zeros(2*n+1)
    for i in range(n):
        mid=(phi[i]+phi[i+1])/2
        integral[2*i+1]=integral[2*i]+ray(phi[i],mid,r.dy/2)
        integral[2*i+2]=integral[2*i]+ray(phi[i],phi[i+1],r.dy)
    yv=np.linspace(r.y[0],r.y[-1],2*n+1)
    r.r=T/G*(yv-r.y[0])+Z/G*integral/I;r.r-=r.r[-1]/2
    r.Z=Z;r.phi=phi.copy()
    objective=r.objective(phi);e,g,terms=objective
    stress=terms[-1];stress_error=max(abs(stress-T))
    # Phase partial derivative holds FE reference/internal variables fixed.
    direction=np.sin(2.3*np.arange(n+1))*.5+.6
    trial=phi+.001;direction/=np.linalg.norm(direction)
    value,gradient,_=r.objective(trial);H=r.phase_hessian(trial)
    fd=[]
    for eps in [1e-4,1e-5,1e-6]:
        ep,gp,_=r.objective(trial+eps*direction);em,gm,_=r.objective(trial-eps*direction)
        fd.append(dict(eps=eps,gradient_error=abs((ep-em)/(2*eps)-gradient@direction)/max(1.,abs(gradient@direction)),hessian_error=np.linalg.norm((gp-gm)/(2*eps)-H@direction)/max(1.,np.linalg.norm(H@direction))))
    assert min(x['gradient_error'] for x in fd)<1e-8
    assert min(x['hessian_error'] for x in fd)<1e-8
    stationary,defect=r.phase(phi)
    stationary_difference=max(abs(stationary-phi))
    # Held-phase Maxwell steady sliding; no phase/history mutation in this test.
    step=r.mechanical(1.,.125,imposed_rate=(T+Z)/ETA)
    Vexpected=Z/ETA
    result=dict(cells=n,q_stationary=q,traction=T,slip_rate_expected=Vexpected,slip_rate_error=abs(step['V']-Vexpected),q_error=abs(step['q']-q),point_stress_error=stress_error,stationary_profile_difference=stationary_difference,stationary_KKT=defect,bulk_weak_drive=max(abs(terms[3]/r.mass)),energy_derivative=fd,mechanical_error=step['mechanical_error'])
    results.append(result)
    if n==256:
        fig,ax=plt.subplots(1,3,figsize=(13,3.6))
        ax[0].plot(r.y,phi,label='analytic stationary, core .6');ax[0].plot(r.y,stationary,'--',label='discrete constrained stationary');ax[0].legend(fontsize=8);ax[0].set_ylabel('phase')
        ax[1].plot(r.y,h(phi)/I);ax[1].set_ylabel('chi [1/m]')
        for k,name in [(3,'bulk'),(4,'cohesive'),(5,'fracture')]:ax[2].plot(r.y,-terms[k]/r.mass,label=name)
        ax[2].set_ylabel('weak driving / nodal measure [Pa]');ax[2].legend()
        for a in ax:a.set_xlabel('y [m]')
        fig.tight_layout();fig.savefig(HERE/'stationary-control.png',dpi=150)
        np.savez_compressed(HERE/'stationary-control.npz',y=r.y,analytic=phi,stationary=stationary,chi=h(phi)/I,bulk=-terms[3]/r.mass,cohesive=-terms[4]/r.mass,fracture=-terms[5]/r.mass)
(HERE/'reference-verification.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
