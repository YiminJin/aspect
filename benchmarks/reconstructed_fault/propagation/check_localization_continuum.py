#!/usr/bin/env python3
"""Stationary continuum control; frozen-field quadrature and space assessment."""
import json
from pathlib import Path
import numpy as np
from localization_1d import Reference,native_profile,G,ETA,A,M,h
HERE=Path(__file__).resolve().parent/'localization_evidence'
controls=[];quadrature=[]
for n in [128,256]:
    r=Reference(n,1.,exact_bulk=True);phi=native_profile(r.y);I,_=r.direct(phi)
    q=np.sqrt(2*G*A*.6/h(.6));T=q+1000.;Z=q*I
    r.phi=phi;r.Z=Z;r.rq=(T+Z*h(r.B@phi)/I)/G
    value,gradient,terms=r.objective(phi)
    actual=max(abs(terms[-1]-T));assert actual<1e-8
    equilibrium,defect=r.phase(phi)
    # Fixed-profile hold at its exact Maxwell-maintenance loading.
    mech=r.mechanical(1.,.125,imposed_rate=(T+Z)/ETA)
    assert abs(mech['q']-q)<1e-8 and abs(mech['V']-Z/ETA)<1e-10
    controls.append(dict(cells=n,stationary_phase_error=float(max(abs(equilibrium-phi))),KKT=defect,stress_error=float(actual),q_error=abs(mech['q']-q),V_error=abs(mech['V']-Z/ETA),mechanical_error=mech['mechanical_error'],bulk_weak_driving=float(max(abs(terms[3]/r.mass)))))
    path=HERE/f'one-d-h1-n{n}-dt0.015625-exact-bulk';d=np.load(path/'fields.npz');hist=json.loads((path/'history.json').read_text());last=hist[-1]
    values=[]
    for count in [6,12]:
        ref=Reference(n,1.,quadrature=count,exact_bulk=True);ref.phi=d['phase'][-1];ref.Z=float(d['Z']);I,_=ref.direct(ref.phi)
        T=(last['stress_min']+last['stress_max'])/2
        ref.rq=(T+ref.Z*h(ref.B@ref.phi)/I)/G
        # Fixed accepted continuous anchor, smooth nonzero phase trial.
        trial=ref.phi+.001*np.exp(-((ref.y-.5)/.2)**2)
        value,grad,terms=ref.objective(trial);values.append((value,grad,terms))
    e6,g6,t6=values[0];e12,g12,t12=values[1]
    quadrature.append(dict(cells=n,relative_energy=abs(e6-e12)/max(1.,abs(e12)),absolute_force_error_Pa_m=float(max(abs(g6-g12))*A),relative_force_error=float(max(abs(g6-g12))/max(1.,max(abs(g12))))))
    assert quadrature[-1]['relative_energy']<1e-10 and quadrature[-1]['relative_force_error']<1e-8
    fields=d['phase'];assert min(np.diff(np.r_[d['initial'][None,:],fields],axis=0).ravel())>=-4.922e-12
    assert min(fields.ravel())>=0 and max(fields.ravel())<1
    assert max(row['kkt'] for row in hist)<=1e-11
one=np.load(HERE/'one-d-h1-n128-dt0.015625-exact-bulk/fields.npz');two=np.load(HERE/'one-d-h1-n256-dt0.015625-exact-bulk/fields.npz')
h1=json.loads((HERE/'one-d-h1-n128-dt0.015625-exact-bulk/history.json').read_text());h2=json.loads((HERE/'one-d-h1-n256-dt0.015625-exact-bulk/history.json').read_text())
phase_error=max(float(max(abs(np.interp(two['y'],one['y'],a)-b))) for a,b in zip(one['phase'],two['phase']))
result=dict(stationary_controls=controls,frozen_bulk_quadrature=quadrature,common_time_refinement=dict(phi=phase_error,q=max(abs(a['q']-b['q']) for a,b in zip(h1,h2)),V=max(abs(a['V']-b['V']) for a,b in zip(h1,h2)),slip=max(abs(a['slip']-b['slip']) for a,b in zip(h1,h2))))
(HERE/'continuum-verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
