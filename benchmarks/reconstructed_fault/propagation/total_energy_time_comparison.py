#!/usr/bin/env python3
"""Common-path refinement of the isolated energy/inelastic-strain candidate.

Reuses the existing energy, normalization adjoint, bound solve and exact
Maxwell step. No native histories, constitutive parameters or projectors change.
"""
import argparse
import csv
import hashlib
import json
import time

import numpy as np

from total_energy_step import (Energy, G, ETA, GC, ELL, C0, M, N, HERE, ROOT, basis, surface_basis,
                               state, step, fingerprint, kkt)


def initial_state(energy):
    source = ROOT/'build-propagation-integration/output-irreversibility_dt03125/material-phase-accepted-0.csv'
    with source.open() as stream:
        lookup = {(float(r['x']), float(r['y'])): float(r['phi']) for r in csv.DictReader(stream)}
    phi = np.array([lookup[(i/N, j/N)] for i in range(N+1) for j in range(N+1)])
    return state(energy, phi, np.zeros_like(energy.x), np.zeros(energy.ns+1)), source


def make_energy(aligned_bulk=False):
    energy = Energy()
    if aligned_bulk:
        # Integrate on the intersection of bulk and surface x partitions.
        # Q1 surface knots are genuine breaks in the mechanical source map.
        # Same Q3 rule, physical energy and phase mesh; no projector change.
        y=np.unique(energy.y);wy=energy.wb.reshape(len(y),len(y)).sum(axis=0)
        xgrid,ygrid=np.meshgrid(energy.xs,y,indexing='ij')
        energy.x,energy.y=xgrid.ravel(),ygrid.ravel()
        energy.wb=np.outer(energy.ws,wy).ravel()
        assert abs(sum(energy.wb)-1)<1e-13
        energy.B=basis(energy.x,energy.y)[0]
        energy.T=surface_basis(energy.x,energy.ns)
    return energy


def run(out, count, scheme, evolve, amplitude, final_time, aligned_bulk=False):
    energy=make_energy(aligned_bulk)
    accepted, source = initial_state(energy)
    original = fingerprint(accepted)
    _, gradient, *_ = energy.objective(accepted.phi, accepted)
    assert kkt(gradient, accepted.phi, accepted.phi) == 0
    # Initial zero-input stick equilibrium; both free mechanical blocks vanish.
    zero, control = step(energy, accepted, final_time/count, ETA, 0.,
                         free_bulk_mode=True, evolve_phase=evolve,
                         endpoint_control=scheme=='endpoint')
    assert fingerprint(accepted) == original and control['phase_release'] == 0
    for key in ['phi','tau','Z','slip','strain','inelastic','history']:
        assert np.array_equal(getattr(zero,key),getattr(accepted,key))
    records = []
    fields = dict(phi=[accepted.phi], tau=[accepted.tau], Z=[accepted.Z],
                  slip=[accepted.slip], I=[accepted.profile['I']],
                  strain=[accepted.strain], inelastic=[accepted.inelastic],
                  history=[accepted.history])
    start = time.monotonic()
    for j in range(count):
        ta, tb = final_time*j/count, final_time*(j+1)/count
        # Physical smooth boundary path gamma(t)=a*t. Its exact interval mean
        # preserves imposed displacement at every common time in both schemes.
        rate = amplitude*(ta+tb)/2
        try:
            new, record = step(energy, accepted, final_time/count, ETA, rate,
                               free_bulk_mode=True, evolve_phase=evolve,
                               endpoint_control=scheme=='endpoint')
        except Exception as exc:
            np.savez_compressed(out/'failed-accepted-state.npz', **{k: getattr(accepted,k)
                for k in ['phi','tau','Z','slip','strain','inelastic','history']})
            (out/'failure.json').write_text(json.dumps(dict(step=j+1, time=ta,
                message=repr(exc), accepted_fingerprint=fingerprint(accepted)),indent=2)+'\n')
            raise
        record.update(time=tb, phase_anchor_time=ta,
                      imposed_displacement=.5*amplitude*tb**2,
                      phase_anchor_displacement=.5*amplitude*ta**2,
                      prescribed_rate=rate,
                      local_q=(new.Z/new.profile['I'])[[0,energy.ns//2,-1]].tolist(),
                      local_slip=new.slip[[0,energy.ns//2,-1]].tolist(),
                      local_phase=new.phi.reshape(N+1,N+1)[[0,N//2,-1],N//2].tolist(),
                      phase_max_location=[int(np.argmax(new.phi))//(N+1)/N,
                                          int(np.argmax(new.phi))%(N+1)/N])
        records.append(record)
        for key in fields:
            fields[key].append(new.profile['I'] if key=='I' else getattr(new,key))
        accepted = new
        if (j+1)%max(1,count//8)==0:
            print(f'{scheme} evolving={evolve} n={count} step={j+1} '
                  f'phi={max(new.phi):.8f} dphi={record["phase"]["max_increment"]:.3g} '
                  f'release={record["phase_release"]:.3g}',flush=True)
    sums = {key: sum(r[key] for r in records) for key in
            ['phase_release','fracture_expenditure','bulk_viscous','fault_viscous',
             'friction','prescribed_external_work','mechanical_work_error',
             'exchange_defect','free_bulk_work','bulk_storage_change','fault_storage_change']}
    onset = next((r for r,phi in zip(records,fields['phi'][1:])
                  if max(phi-fields['phi'][0])>1e-6),None)
    summary = dict(scheme=scheme, phase_evolves=evolve, steps=count, dt=final_time/count,
        aligned_bulk_quadrature=aligned_bulk, bulk_samples=len(energy.x),
        final_time=final_time, amplitude=amplitude, material=dict(G_Pa=G,eta_Pa_s=ETA,
            Gc_J_m2=GC,ell_m=ELL,c0=C0,m=M), input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        initial_obstacle_kkt=0., initial_phi_max=float(max(fields['phi'][0])),
        final_phi_max=float(max(accepted.phi)), max_phase_increment=max(r['phase']['max_increment'] for r in records),
        largest_phase_release=max(r['phase_release'] for r in records),
        final_local_q=records[-1]['local_q'], final_local_slip=records[-1]['local_slip'],
        final_stress_min=float(min(accepted.tau)),final_stress_max=float(max(accepted.tau)),
        endpoint_bulk_imbalance_max=max(abs(r['free_bulk_endpoint_residual']) for r in records),
        mean_bulk_imbalance_max=max(abs(r['free_bulk_residual']) for r in records),
        endpoint_slip_condition_error_max=max(r['endpoint_slip_condition_error'] for r in records),
        mean_slip_condition_error_max=max(r['mean_slip_condition_error'] for r in records),
        mechanical_kkt_max=max(r['mechanical_kkt'] for r in records),
        phase_kkt_max=max((r['phase'].get('kkt',0.) for r in records)),
        min_I=min(r['min_I'] for r in records), healing_max=max(r['phase']['healing'] for r in records),
        phase_path_error_max=max(r['phase_path_error'] for r in records),
        temporal_quadrature_error_max=max(r['temporal_quadrature_error'] for r in records),
        onset=None if onset is None else dict(publication_time=onset['time'],
            anchor_time=onset['phase_anchor_time'], anchor_displacement=onset['phase_anchor_displacement'],
            publication_displacement=onset['imposed_displacement']),
        sums=sums,elapsed_seconds=time.monotonic()-start)
    np.savez_compressed(out/'fields.npz', **{k:np.asarray(v) for k,v in fields.items()},
                        time=np.linspace(0,final_time,count+1))
    (out/'records.json').write_text(json.dumps(records,indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--label',required=True)
    parser.add_argument('--steps',type=int,required=True)
    parser.add_argument('--scheme',choices=['mean','endpoint'],default='mean')
    parser.add_argument('--evolve',action='store_true')
    parser.add_argument('--amplitude',type=float,default=.256)
    parser.add_argument('--final-time',type=float,default=1.)
    parser.add_argument('--aligned-bulk-quadrature',action='store_true')
    args=parser.parse_args()
    out=HERE/'total_energy_time_evidence'/args.label
    out.mkdir(parents=True,exist_ok=False)
    run(out,args.steps,args.scheme,args.evolve,args.amplitude,args.final_time,args.aligned_bulk_quadrature)


if __name__=='__main__':
    main()
