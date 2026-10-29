#!/usr/bin/env python3
"""Independent Q1 measures, unloaded forces and intact Maxwell background."""
import json,hashlib
import numpy as np
from continuation_analysis import ROOT,OUT,OLD,read,initial_forces,fracture
from continuation_temporal_sampling import norm2
nodes=read(OLD/'phase-0000.csv');nodes.sort(order='dof');order=np.lexsort((nodes['x'],nodes['y']));x,y=np.unique(nodes['x']),np.unique(nodes['y']);p=nodes['phi'][order].reshape(len(y),len(x));dx=x[1]-x[0];dy=y[1]-y[0];X,Y=np.meshgrid(x,y)
norms={}
for label,field,exact in [('constant',np.ones_like(X),1.5),('x',X,1.125),('y',Y,.5),('xy',X*Y,.375)]:
    error=abs(norm2(field,dx,dy)-exact);assert error<1e-14;norms[label]=error
f=initial_forces(p,dx,dy);native=read(OLD/'phase-forces-0001.csv');native.sort(order='dof');ng=native['fracture_gradient'][order].reshape(p.shape);nm=native['mass'][order].reshape(p.shape)
force_error=float(np.max(abs(f[...,1]-ng)));mass_error=float(np.max(abs(f[...,0]-nm)));assert force_error<1e-14 and mass_error<1e-14
first=read(OLD/'bulk-accepted.csv')[0];energy_error=abs(fracture(p,dx,dy,8.,.046875)-first['fracture']);assert energy_error<1e-13
tau=0.;maximum=0.;dt=.00125;G=1e6;eta=1e8;volume=1.5
for n in range(1,144):
    time=n*dt;rate=.018 if .05+1e-12<time<=.175+1e-12 else 0.;old=tau;tau=(old+G*dt*rate)/(1+dt*G/eta)
    work=volume*tau*dt*rate;delta=volume*(tau*tau-old*old)/(2*G);visc=volume*dt*tau*tau/eta;loss=volume*(tau-old)**2/(2*G);maximum=max(maximum,abs(work-delta-visc-loss))
assert maximum<1e-14
previous=json.load(open(ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/bulk_fracture/entry.json'))
protected={}
for name in ['source/material_model/phase_field_fault/bulk_cohesive.cc','source/material_model/phase_field_fault/bulk_cohesive_history.cc']:
    value=previous['files'][name];oldsha=value['sha256'] if isinstance(value,dict) else value
    digest=hashlib.sha256((ROOT/name).read_bytes()).hexdigest();assert digest==oldsha;protected[name]=digest
result=dict(Q1_mass_norm_errors=norms,independent_unloaded_force_error_J_m=force_error,mass_error_m2=mass_error,initial_fracture_error_J_m=float(energy_error),initial_min_scaled_obstacle_gap=float(np.min(f[...,1]/(8*f[...,0]))),homogeneous_Maxwell_max_ledger_error_J_m=maximum,constitutive_and_history_sources_unchanged=protected,strict_reference_match_passed=False)
(OUT/'independent-reference-checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
