#!/usr/bin/env python3
"""Reduce native prescribed-transport boundary and energy-isolation checks."""
import csv
import hashlib
import json
import os
from pathlib import Path
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "DEAL_II_NUM_THREADS"):
    os.environ[_key]="1"
import numpy as np
from assess_material_native import ROOT, RUNS

OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/boundary_profile'


def main():
    rows=[];inputs={};commands={}
    for n in [8,16,32]:
        p=RUNS/f'boundary-profile-n{n}'
        meta=json.loads((p/'command.json').read_text());assert meta['exit_code']==0
        commands[n]=meta
        f=p/'output/boundary-profile-transport.csv'
        with f.open() as stream: rows.extend(dict(n=n,**{k:float(v) for k,v in row.items()}) for row in csv.DictReader(stream))
        for f in [f,p/'command.json',p/'input.prm']:
            inputs[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
    polynomial=[r for r in rows if r['kind']==0]
    assert max(r['relative_L2_consistent'] for r in polynomial)<1e-10
    assert max(r['max_semidiscrete_identity_defect'] for r in rows)<1e-10
    spatial={}
    for direction in [-1,1]:
        r=sorted([r for r in rows if r['kind']==2 and r['direction']==direction and r['dt']==.00625],key=lambda r:r['n'])
        errors=[a['relative_L2_consistent'] for a in r]
        spatial[direction]=dict(errors=errors,orders=[float(np.log2(a/b)) for a,b in zip(errors,errors[1:])])
    time={}
    for kind in [0,1,2]:
        for direction in [-1,1]:
            r=sorted([r for r in rows if r['n']==16 and r['kind']==kind and r['direction']==direction],key=lambda r:-r['dt'])
            remainders=[a['energy_trapezoid_remainder'] for a in r]
            time[f'{kind},{direction}']=dict(remainders=remainders,absolute_orders=[float(np.log2(abs(a/b))) for a,b in zip(remainders,remainders[1:])])
    report=dict(rows=rows,spatial=spatial,time=time,commands=commands,inputs_sha256=inputs,
       energy_identity='dE_h/dt + outward boundary energy flux + nonnegative upwind jump loss = 0; full3D norm, G=10 Pa.',
       cumulative_remainder='E_final-E_initial + trapezoidal integral(flux+upwind loss). This includes time-quadrature and SSPRK3 errors; it is not physical dissipation.',
       interpretation='Exact polynomial inflow and deliberately zero inflow are different boundary-value problems. Compact fields isolate interior DG error. This nonpublishing operator experiment does not solve moving-fault mechanics or qualify its total energy balance.')
    (OUT/'boundary-transport.json').write_text(json.dumps(report,indent=2)+'\n')
    os.environ.setdefault('MPLCONFIGDIR',str(OUT/'matplotlib-cache'))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for kind,label in [(0,'Exact inflow, polynomial'),(1,'Zero inflow, continued reference'),(2,'Compact, zero inflow')]:
        r=sorted([r for r in rows if r['kind']==kind and r['direction']==1 and r['dt']==.00625],key=lambda r:r['n'])
        axs[0,0].loglog([4/r['n'] for r in r],[r['relative_L2_extended'] for r in r],'.-',label=label)
    r=sorted([r for r in rows if r['kind']==1 and r['direction']==1 and r['dt']==.00625],key=lambda r:r['n'])
    axs[0,1].loglog([4/a['n'] for a in r],[a['relative_L2_consistent'] for a in r],'.-',label='Actual zero-inflow problem')
    axs[0,1].loglog([4/a['n'] for a in r],[a['relative_L2_extended'] for a in r],'.-',label='Continued-field comparison')
    for kind,label in [(1,'Zero inflow, polynomial'),(2,'Compact')]:
        r=sorted([r for r in rows if r['kind']==kind and r['direction']==1 and r['n']==16],key=lambda r:r['dt'])
        axs[1,0].loglog([a['dt'] for a in r],[abs(a['energy_trapezoid_remainder']) for a in r],'.-',label=label)
    labels=['Exact inflow','Zero inflow','Compact'];selected=[next(r for r in rows if r['kind']==k and r['direction']==1 and r['n']==16 and r['dt']==.00625) for k in range(3)]
    x=np.arange(3)
    axs[1,1].bar(x-.2,[r['boundary_flux_integral'] for r in selected],.4,label='Outward physical flux')
    axs[1,1].bar(x+.2,[r['upwind_dissipation_integral'] for r in selected],.4,label='Numerical upwind loss')
    axs[1,1].set_xticks(x,labels);axs[1,1].set_yscale('symlog',linthresh=1e-9)
    axs[0,0].set(xlabel='Cell width (m)',ylabel='Relative L2',title='Boundary condition versus interior discretization')
    axs[0,1].set(xlabel='Cell width (m)',ylabel='Relative L2',title='Two distinct boundary-value references')
    axs[1,0].set(xlabel='Timestep (s)',ylabel='Absolute remainder (J/m)',title='Energy integration error, fixed mesh')
    axs[1,1].set(ylabel='Integrated energy (J/m)',title='Flux and upwind dissipation are separate')
    for ax in axs.flat:ax.grid(alpha=.25);ax.legend(fontsize=7)
    fig.savefig(OUT/'boundary-transport.png',dpi=160)
    print(json.dumps(dict(polynomial_error=max(r['relative_L2_consistent'] for r in polynomial),
         max_energy_identity_defect=max(r['max_semidiscrete_identity_defect'] for r in rows),spatial=spatial,time=time)))


if __name__=='__main__':main()
