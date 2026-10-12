#!/usr/bin/env python3
"""Matching FE temporal reference for optional native consistent-endpoint P2.

Uses the retained continuum initialization and existing equations/minimizer.
Evidence is separate from all earlier accepted and failed trajectories.
"""
import argparse
import resource
import time

import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor

import asymmetric_branch as old
import matched_initialization as shared

HERE = old.HERE
E = HERE / 'native_endpoint_evidence'
TARGET = old.TARGET


class SharedFE(old.MeshCoupled):
    def __init__(self, n=512):
        self.r = shared.SharedReference(n, 1., 6)
        self.D = self.r.D[:, 1:-1]
        self.gram = self.D.T @ (self.r.w[:, None] * self.D)
        self.factor = cho_factor(self.gram)
        self.r.D = sparse.csr_matrix(self.r.D)
        self.D = self.r.D[:, 1:-1]
        self.evaluations = 0
        self.last = None


def run(denominator, prefix=None, onset_start=None, schedule=None, output=None):
    E.mkdir(exist_ok=True)
    out = E / (f'dt-1-{denominator}' if prefix is None else f'prefix-{prefix}-fine-{denominator}' + ('-onset' if onset_start is not None else ''))
    if output is not None:
        out=output
    out.mkdir(exist_ok=False)
    started = time.perf_counter()
    p = SharedFE()
    r = p.r
    phi = r.initial.copy()
    R = np.zeros_like(r.r)
    Z = slip = 0.
    times = (np.arange(1, round(.625*denominator)+1)/denominator if prefix is None else
             np.r_[np.arange(1, round(.5*prefix)+1)/prefix,
                   .5+np.arange(1, round(.125*denominator)+1)/denominator])
    if onset_start is not None:
        times=np.r_[times[times<=.5],onset_start,
                    onset_start+np.arange(1,round((.625-onset_start)*denominator)+1)/denominator]
    if schedule is not None:
        times=np.array(schedule,dtype=float)
        assert min(np.diff(np.r_[0.,times]))>0 and times[-1]==.625
    phases, strains, zs = [phi.copy()], [R.copy()], [Z]
    velocities, rows, energies, failures = [], [], [], []
    previous_increment = np.zeros_like(phi)
    first_saddle = None
    np.savez_compressed(out / 'initialization.npz', phase=phi, R=R, Z=Z,
                        particle_y=r.initial_particle_y, H=r.initial_particle_H)
    for k, t in enumerate(times):
        dt = t-(times[k-1] if k else 0.)
        p.incoming(phi, R, Z, slip, t, dt)
        x = r.refine(phi + previous_increment, p.lower, p.evaluate, p.hessian)
        if r.defect(p.evaluate(x)[1], x, p.lower) > TARGET:
            x, _ = old.minimize_phase(p, x, 1)
        status, v = old.spectrum(p, x)
        if status['admissible_negative']:
            if first_saddle is None:
                first_saddle = t
            x, _ = old.minimize_phase(p, x + .5 * status['amplitude_two_sided'] * v, 1)
            status, _ = old.spectrum(p, x)
        p.evaluate(x)
        data = p.last[3].copy()
        Rb, Zb = p.last[4].copy(), p.last[5]
        u = r.velocity.copy()
        kkt = float(r.defect(p.evaluate(x)[1], x, p.lower))
        if kkt > TARGET or data['force'] > 1e-7 or data['contact'] > 1e-7 or not status['stable']:
            failures.append(dict(time=t, KKT=kkt, stability=status, mechanical=data))
            np.savez_compressed(out / 'failed-trial.npz', phase=x, lower=phi,
                                Ra=R, Za=Z, R=Rb, Z=Zb)
            break
        assert np.array_equal(phi, p.lower) and np.array_equal(R, p.Ra) and Z == p.Za
        assert min(x - phi) >= -4.922e-12 and max(x) < 1
        ledger = old.core.path_ledger(r, phi, x, R, Z, Rb, Zb, u, data['V'], dt)
        ledger.update(time=t, phase_increment=float(max(x - phi)))
        status.pop('lower_y', None)
        rows.append(dict(time=float(t), dt=float(dt), KKT=kkt, phase_max=float(max(x)),
                         phase_increment=ledger['phase_increment'],
                         centroid=old.centroid(r, x), stability=status, **data))
        energies.append(ledger)
        previous_increment = x - phi
        phi, R, Z, slip = x.copy(), Rb.copy(), Zb, data['slip']
        phases.append(phi.copy()); strains.append(R.copy()); zs.append(Z)
        velocities.append(u)
        if k % 64 == 0:
            old.write(out / 'history.json', rows)
            old.write(out / 'energy.json', energies)
            print(old.json.dumps(dict(dt=dt, time=t, phase=max(phi),
                                      centroid=rows[-1]['centroid'])), flush=True)
    np.savez_compressed(out / 'fields.npz', phase=np.array(phases), R=np.array(strains),
                        Z=np.array(zs), velocity=np.array(velocities), y=r.y,
                        yq=r.yq, initial=r.initial, time=np.r_[0., times[:len(rows)]])
    old.write(out / 'history.json', rows)
    old.write(out / 'energy.json', energies)
    old.write(out / 'summary.json', dict(n=512, dt=1/denominator, prefix_denominator=prefix, onset_start=onset_start,
              schedule=times.tolist() if schedule is not None else None, exact_bulk=False,
              first_saddle=first_saddle, completed=not failures, accepted=len(rows),
              last_accepted=rows[-1]['time'] if rows else 0., failures=failures,
              runtime_seconds=time.perf_counter()-started,
              peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--denominator', type=int, choices=[1024, 2048], required=True)
    parser.add_argument('--prefix', type=int, choices=[2,4,64,128])
    parser.add_argument('--onset-start', type=float, choices=[.52734375])
    args=parser.parse_args()
    run(args.denominator,args.prefix,args.onset_start)
