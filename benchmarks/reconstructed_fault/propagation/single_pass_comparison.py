#!/usr/bin/env python3
"""Phase-first P2 comparison using the existing energy and cached FE mechanics.

Only the driver/order and diagnostics differ from the endpoint reference.
Accepted histories are immutable until both subproblems have succeeded.
"""
import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

import native_endpoint as endpoint
import asymmetric_branch as old
from time_resolution import energy

HERE = Path(__file__).resolve().parent
E = HERE / 'single_pass_evidence'
TARGET = old.TARGET


def split_ledger(r, pa, pb, Ra, Za, Rb, Zb, u, V, t, dt, order=16):
    """Actual phase jump, then fixed-profile general-beta Maxwell path.

    D_relax is a measured phase-energy release, not constitutive dissipation.
    The unadjusted external remainder retains it. Interior force and friction
    residual works independently explain the subsequent mechanical remainder.
    """
    result = energy(r, pa, pb, Ra, Za, Rb, Zb, dt, V, t)
    I, _ = r.direct(pb)
    chi = old.core.h(r.B @ pb) / I
    tau0 = old.G * (r.D @ Ra) - Za * chi
    du = r.D @ u
    z, w = leggauss(order)
    times = (z + 1) * dt / 2
    weights = w * dt / 2
    lift = np.linspace(u[0], u[-1], len(u))
    terms = {k: 0. for k in ['external_work', 'lifting_work', 'mechanical_power',
                             'friction_residual_power', 'mechanical_power_absolute',
                             'friction_residual_power_absolute']}
    for s, weight in zip(times, weights):
        beta = np.exp(-old.G * s / old.ETA)
        kappa = -old.ETA * np.expm1(-old.G * s / old.ETA)
        tau = beta * tau0 + kappa * (du - chi * V)
        Z = beta * Za + kappa * V
        force = r.D.T @ (r.w * tau)
        friction_residual = old.core.MU * old.core.PRESSURE + Z / I - r.w @ (chi * tau)
        mechanical_power = u[1:-1] @ force[1:-1]
        friction_power = friction_residual * V
        values = dict(external_work=u[[0, -1]] @ force[[0, -1]], lifting_work=lift @ force,
                      mechanical_power=mechanical_power, friction_residual_power=friction_power,
                      mechanical_power_absolute=abs(mechanical_power),
                      friction_residual_power_absolute=abs(friction_power))
        for key in terms:
            terms[key] += weight * values[key]
    change = sum(result[k] for k in ['bulk_change', 'cohesive_change', 'fracture_change'])
    losses = sum(result[k] for k in ['bulk_viscous', 'cohesive_viscous', 'friction'])
    raw = terms['external_work'] - change - losses
    release = result['D_relax']
    # Neither term is inserted into the state update or hidden in viscosity.
    mechanical_remainder = raw - release
    result.update(terms)
    result.update(time=t, phase_energy_change=-release,
                  external_remainder_unadjusted=raw,
                  fixed_phase_mechanical_remainder=mechanical_remainder,
                  split_identity_error=abs(mechanical_remainder + terms['mechanical_power']
                                           + terms['friction_residual_power']),
                  bulk_external_difference=result['bulk_work'] - terms['external_work'],
                  lifting_external_difference=terms['lifting_work'] - terms['external_work'])
    return {k: float(v) for k, v in result.items()}


def run(denominator, output=None):
    out = E / f'dt-1-{denominator}' if output is None else output
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    cpu = time.process_time()
    p = endpoint.SharedFE()
    r = p.r
    phi, R = r.initial.copy(), np.zeros_like(r.r)
    Z = slip = 0.
    reference = endpoint.E / f'dt-1-{denominator}'
    initial = dict(np.load(reference / 'initialization.npz'))
    mine = dict(phase=phi, R=R, Z=Z, particle_y=r.initial_particle_y, H=r.initial_particle_H)
    assert all(np.array_equal(mine[k], initial[k]) for k in mine), 'Use the same physical initialization'
    np.savez_compressed(out / 'initialization.npz', **mine)
    initial_gradient = r.objective(phi)[1]
    assert r.defect(initial_gradient, phi, phi) <= TARGET
    times = np.arange(1, round(.625 * denominator) + 1) / denominator
    phases, strains, zs, velocities = [phi.copy()], [R.copy()], [Z], []
    history, ledgers, continuous, failures = [], [], [], []
    timing = dict(phase_seconds=0., mechanics_seconds=0., diagnostics_seconds=0.,
                  phase_objective_calls=0, phase_hessian_calls=0, mechanical_calls=0)
    objective, hessian = r.objective, r.phase_hessian

    def counted_objective(x):
        timing['phase_objective_calls'] += 1
        return objective(x)

    def counted_hessian(x):
        timing['phase_hessian_calls'] += 1
        return hessian(x)

    for k, t in enumerate(times):
        dt = 1 / denominator
        x = phi.copy()
        p.incoming(phi, R, Z, slip, t, dt)
        r.r, r.Z, r.phi = R.copy(), Z, phi.copy()
        try:
            r.objective, r.phase_hessian = counted_objective, counted_hessian
            phase_started = time.perf_counter()
            try:
                x, incoming_kkt = r.phase(p.lower)
            finally:
                timing['phase_seconds'] += time.perf_counter() - phase_started
                r.objective, r.phase_hessian = objective, hessian
            assert np.array_equal(p.Ra, R) and p.Za == Z and np.array_equal(p.lower, phi)
            assert np.array_equal(r.r, R) and r.Z == Z, 'Phase trials advanced physical history'
            assert not p.lower.flags.writeable and not p.Ra.flags.writeable
            mechanics_started = time.perf_counter()
            before_calls = p.evaluations
            _, gend = p.evaluate(x)
            timing['mechanics_seconds'] += time.perf_counter() - mechanics_started
            timing['mechanical_calls'] += p.evaluations - before_calls
            data, Rb, Zb = p.last[3].copy(), p.last[4].copy(), p.last[5]
            u = r.velocity.copy()
            endpoint_kkt = float(r.defect(gend, x, p.lower))
            assert incoming_kkt <= TARGET and data['force'] < 1e-7 and data['contact'] < 1e-7
            assert min(x - phi) >= -4.922e-12 and min(x) >= 0 and max(x) < 1 and data['I'] > 0
            # Completed-state phase defect is a splitting diagnostic, not an
            # acceptance test for the incoming-history phase subproblem.
            diagnostic_started = time.perf_counter()
            ledger = split_ledger(r, phi, x, R, Z, Rb, Zb, u, data['V'], t, dt)
            audit = old.core.path_ledger(r, phi, x, R, Z, Rb, Zb, u, data['V'], dt)
            audit.update(time=float(t))
            assert ledger['split_identity_error'] < 1e-9
            assert audit['identity_error'] < 1e-9
            initial_energy = sum(ledger[key] - ledger[key+'_change']
                                 for key in ['bulk', 'cohesive', 'fracture'])
            assert ledger['D_relax'] >= -32*np.finfo(float).eps*max(1., abs(initial_energy))
            timing['diagnostics_seconds'] += time.perf_counter() - diagnostic_started
            history.append(dict(time=float(t), dt=dt, incoming_KKT=float(incoming_kkt),
                endpoint_phase_defect=endpoint_kkt,
                endpoint_mass_scaled_defect_Pa=float(max(abs(old.A*old.core.projected(gend,x,p.lower)/r.mass))),
                phase_max=float(max(x)), phase_increment=float(max(x-phi)), healing=float(max(phi-x)),
                centroid=old.centroid(r,x), **data))
            ledgers.append(ledger); continuous.append(audit)
            # The only publication into the driver's accepted physical state.
            phi, R, Z, slip = x.copy(), Rb.copy(), Zb, data['slip']
            r.phi, r.r, r.Z, r.slip = phi.copy(), R.copy(), Z, slip
            phases.append(phi.copy()); strains.append(R.copy()); zs.append(Z); velocities.append(u)
        except Exception as error:
            failures.append(dict(time=float(t), error=repr(error)))
            np.savez_compressed(out/'failed-entry.npz', lower=phi, Ra=R, Za=Z, slip=slip,
                                trial=x if 'x' in locals() else phi)
            break
        if k % 64 == 0:
            old.write(out/'history.json', history)
            print(json.dumps(dict(time=float(t), phase=max(phi), incoming_KKT=incoming_kkt,
                                  endpoint_phase_defect=endpoint_kkt)), flush=True)
    np.savez_compressed(out/'fields.npz', phase=phases, R=strains, Z=zs, velocity=velocities,
                        y=r.y, yq=r.yq, initial=r.initial, time=np.r_[0.,times[:len(history)]])
    for name, values in [('history',history),('split-energy',ledgers),('energy',continuous)]:
        old.write(out/(name+'.json'),values)
    old.write(out/'summary.json',dict(n=512,dt=1/denominator,completed=not failures,
        accepted=len(history),last_accepted=history[-1]['time'] if history else 0.,failures=failures,
        runtime_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu,
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,timing=timing,
        physical_history='Fresh shared initialization; incoming phase first, one endpoint mechanical solve, commit once.',
        endpoint_defect_is_splitting_error=True,initial_fields_identical=True))
    if failures:
        raise RuntimeError(failures)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--denominator',type=int,choices=[1024,2048],required=True)
    parser.add_argument('--output',type=Path,help='Fresh destination for a repeat; existing outputs remain protected.')
    args=parser.parse_args()
    run(args.denominator,args.output)
