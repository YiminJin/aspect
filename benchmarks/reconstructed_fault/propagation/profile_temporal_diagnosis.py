#!/usr/bin/env python3
"""Fixed-history phase lag, profile scaling, and external FE reaction work."""
import argparse
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from time_resolution import BASE, Reference, G, ETA, A, h, load

HERE = Path(__file__).resolve().parent
E = HERE / 'profile_temporal_evidence'
OLD = HERE / 'time_resolution_evidence'


def directory(d):
    if d == 64:
        return BASE
    return (OLD if d <= 1024 else E) / f'dt-1-{d}'


def fields(path):
    f = np.load(path / 'fields.npz')
    return (np.vstack([f['initial'], f['phase']]),
            np.vstack([f['incoming_r'], f['final_r']]),
            np.r_[f['incoming_Z'], f['final_Z']])


def projected(g, phi, lower):
    # No saturation in these controls. Retain the solver's contact threshold.
    assert max(phi) < 1
    return np.where(phi <= lower + 1e-13, np.minimum(g, 0.), g)


def step(r, lower, phi, Ra, Za, Rb, Zb, V, t, dt):
    """Read-only evaluations, no constitutive publication or phase update."""
    r.r = Ra.copy()
    r.Z = Za
    gin = r.objective(phi)[1]
    r.r = Rb.copy()
    r.Z = Zb
    gend = r.objective(phi)[1]
    pin = projected(gin, phi, lower)
    pend = projected(gend, phi, lower)
    I = r.direct(phi)[0]
    Ia = r.direct(lower)[0]
    chi = h(r.B @ phi) / I
    chia = h(r.B @ lower) / Ia
    dc = chi - chia
    tau_a = G * (r.D @ Ra) - Za * chia
    tau_star = G * (r.D @ Ra) - Za * chi
    beta = np.exp(-G*dt/ETA)
    kappa = -ETA*np.expm1(-G*dt/ETA)
    u = (Rb - beta*Ra)*G/kappa
    x = np.linspace(0., 1., len(u))
    lift = .256*(t-dt/2)*(x-.5)
    relative = u - lift
    force = r.D.T @ (r.w*tau_star)
    pair = relative*force
    z, w = leggauss(12)
    ts = (z+1)*dt/2
    wt = w*dt/2
    b = np.exp(-G*ts/ETA)
    kk = -ETA*np.expm1(-G*ts/ETA)
    tau = b[:,None]*tau_star + kk[:,None]*(r.D@u-chi*V)
    integral_tau = wt @ tau
    integrated_force = r.D.T @ (r.w*integral_tau)
    # The constrained FE boundary residual gives reaction forces. It is
    # independent of the arbitrary interior extension of boundary velocity.
    Wreaction = u[[0,-1]] @ integrated_force[[0,-1]]
    Wbulk = u @ integrated_force
    Waffine = lift @ integrated_force
    # Two explicit smooth zero-boundary extensions; diagnostic only.
    bump = .256*(t-dt/2)*np.sin(np.pi*x)
    asymmetric = .256*(t-dt/2)*np.sin(2*np.pi*x)
    Wother = (lift+bump) @ integrated_force
    Wother2 = (lift+asymmetric) @ integrated_force
    factor = ETA/G-beta*dt/(-np.expm1(-G*dt/ETA))
    endpoint_force = r.D.T@(r.w*(G*r.D@Rb-Zb*chi))
    qpair = -Za*r.w*dc*(r.D@relative)
    rate = dc/dt
    dominant_q = int(np.argmax(abs(qpair)))
    node = int(np.argmax(abs(pend)))
    result = dict(time=t,dt=dt,incoming_phase_KKT=float(max(abs(pin))),
                  endpoint_phase_KKT=float(max(abs(pend))),
                  endpoint_phase_mass_scaled_Pa=float(max(abs(A*pend/r.mass))),
                  endpoint_phase_residual_y=float(r.y[node]),
                  delta_chi_rate_max=float(max(abs(rate))),
                  delta_chi_rate_RMS=float(np.sqrt(r.w@(rate*rate))),
                  delta_chi_rate_y=float(r.yq[np.argmax(abs(rate))]),
                  redistributed_interior_force_max=float(max(abs(force[1:-1]))),
                  redistributed_force_y=float(x[1:-1][np.argmax(abs(force[1:-1]))]),
                  relative_velocity_max=float(max(abs(relative[1:-1]))),
                  relative_velocity_y=float(x[np.argmax(abs(relative))]),
                  pairing_signed=float(sum(pair)),pairing_absolute=float(sum(abs(pair))),
                  predicted_work_difference=float(factor*sum(pair)),
                  actual_work_difference=float(Wbulk-Waffine),
                  work_identity_error=float(abs(Wbulk-Waffine-factor*sum(pair))),
                  bulk_work=float(Wbulk),affine_work=float(Waffine),
                  reaction_work=float(Wreaction),alternative_lifting_work=float(Wother),
                  asymmetric_lifting_work=float(Wother2),
                  lifting_reaction_difference=float(Waffine-Wreaction),
                  bulk_reaction_difference=float(Wbulk-Wreaction),
                  endpoint_interior_force_max=float(max(abs(endpoint_force[1:-1]))),
                  dominant_work_y=float(r.yq[dominant_q]),
                  maximum_phase_increment=float(max(phi-lower)))
    vectors = dict(phase=phi,lower=lower,incoming_phase_gradient=gin,
                   endpoint_phase_gradient=gend,endpoint_projected_gradient=pend,
                   delta_chi_rate=rate,redistributed_force=force,
                   relative_velocity=relative,nodal_pairing=pair,
                   profile_pairing_q=qpair,chi=chi,old_chi=chia)
    return result, vectors, factor*qpair


def audit(d, path=None, label=None):
    path = directory(d) if path is None else path
    out = E / ('diagnosis-'+str(d) if label is None else label)
    out.mkdir(exist_ok=False)
    phi, R, Z = fields(path)
    hist = load(path / 'history.json')
    energy = load((OLD/'baseline-diagnostics' if d==64 and label is None else path)/'energy.json')
    r = Reference(256,1.,6)
    rows = []
    accumulated = np.zeros(len(r.w))
    accumulated_abs = np.zeros(len(r.w))
    snapshots = {}
    maxima = {key:-1. for key in ['endpoint_phase_KKT','delta_chi_rate_max',
                                'redistributed_interior_force_max','pairing_absolute']}
    for j, data in enumerate(hist,1):
        row, vectors, qwork = step(r,phi[j-1],phi[j],R[j-1],Z[j-1],R[j],Z[j],
                                  data['V'],j/d,1/d)
        assert abs(row['bulk_work']-energy[j-1]['bulk_work'])<1e-9
        assert abs(row['affine_work']-energy[j-1]['boundary_work'])<1e-9
        expenditure = energy[j-1]['bulk_work']-energy[j-1]['bulk_work_error']
        row['reaction_work_residual'] = row['reaction_work']-expenditure
        row['alternative_work_residual'] = row['alternative_lifting_work']-expenditure
        row['asymmetric_work_residual'] = row['asymmetric_lifting_work']-expenditure
        rows.append(row)
        accumulated += qwork
        accumulated_abs += abs(qwork)
        for key in maxima:
            if row[key]>maxima[key]:
                maxima[key] = row[key]
                snapshots[key] = (j, {k:v.copy() for k,v in vectors.items()})
        if j==len(hist):snapshots['final']=(j,{k:v.copy() for k,v in vectors.items()})
    stored = dict(y=r.y,yq=r.yq,velocity_y=np.linspace(0,1,len(R[0])),
                  accumulated_profile_work=accumulated,
                  accumulated_absolute_profile_work=accumulated_abs)
    for key,(j,vectors) in snapshots.items():
        stored[key+'_step']=j
        for name,value in vectors.items():stored[key+'_'+name]=value
    np.savez_compressed(out/'diagnostic-fields.npz',**stored)
    totals = {}
    for key in ['actual_work_difference','bulk_work','affine_work','reaction_work',
                'alternative_lifting_work','asymmetric_lifting_work',
                'reaction_work_residual','alternative_work_residual','asymmetric_work_residual',
                'lifting_reaction_difference','bulk_reaction_difference']:
        totals[key] = dict(signed=sum(v[key] for v in rows),absolute=sum(abs(v[key]) for v in rows),
                           maximum=max(abs(v[key]) for v in rows))
    dominant = sorted(range(len(rows)),key=lambda j:abs(rows[j]['actual_work_difference']),reverse=True)[:5]
    summary = dict(denominator=d,steps=len(rows),coupling_diagnostic_not_solver_failure=True,
                   maxima=maxima,maximum_incoming_phase_KKT=max(v['incoming_phase_KKT'] for v in rows),
                   maximum_work_identity_error=max(v['work_identity_error'] for v in rows),
                   endpoint_phase_final=rows[-1]['endpoint_phase_KKT'],
                   relative_velocity_max=max(v['relative_velocity_max'] for v in rows),
                   dominant_work_steps=[rows[j] for j in dominant],totals=totals)
    for name,value in [('history',rows),('summary',summary)]:
        (out/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--denominator',type=int,required=True)
    args=parser.parse_args()
    audit(args.denominator)
