#!/usr/bin/env python3
"""Search-only limited-memory quasi-Newton versus retained endpoint equations.

Every objective call recomputes endpoint mechanics from the same incoming
accepted FE d_R and Z. The obstacle is the previous physical phase throughout.
The sparse AT1-plus-mass inverse is a search metric, not a physical Hessian.
Final KKT and mass-normalized admissible curvature use the existing operators.
No production source, accepted reference, or history is modified by this test.
"""
import argparse
import contextlib
import json
import resource
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

import native_endpoint as native
import coupled_branch as core
import asymmetric_branch as old
import localization_1d as localization

HERE = Path(__file__).resolve().parent
E = HERE / 'cheaper_coupling_evidence'
TARGET = native.TARGET


class Measured(native.SharedFE):
    def __init__(self):
        super().__init__()
        self.counts = dict(mechanical_evaluations=0, objective_calls=0,
                           reduced_hessians=0, bulk_factor_calls=0,
                           bulk_rhs=0, phase_linear_solves=0,
                           metric_factors=0, metric_solves=0, line_trials=0)
        self.seconds = dict(mechanics=0., hessian=0., phase_linear=0.,
                            metric=0., stability=0.)

    def evaluate(self, x):
        start = time.perf_counter(); before = self.evaluations
        value = super().evaluate(x)
        self.counts['objective_calls'] += 1
        self.counts['mechanical_evaluations'] += self.evaluations-before
        self.seconds['mechanics'] += time.perf_counter()-start
        return value

    def hessian(self, x):
        start = time.perf_counter()
        value = super().hessian(x)
        self.counts['reduced_hessians'] += 1
        self.seconds['hessian'] += time.perf_counter()-start
        return value

    @contextlib.contextmanager
    def count_solves(self):
        """Count actual existing factor applications and free phase solves.

        A bulk call with many right-hand sides is not one cheap native action.
        Report calls and RHS counts separately. This single-process benchmark
        temporarily wraps functions only, never changes their numerical result.
        """
        bulk, phase = core.cho_solve, localization.solve

        def counted_bulk(factor, rhs, *args, **kwargs):
            self.counts['bulk_factor_calls'] += 1
            self.counts['bulk_rhs'] += 1 if rhs.ndim == 1 else rhs.shape[1]
            return bulk(factor, rhs, *args, **kwargs)

        def counted_phase(*args, **kwargs):
            start = time.perf_counter()
            self.counts['phase_linear_solves'] += 1
            answer = phase(*args, **kwargs)
            self.seconds['phase_linear'] += time.perf_counter()-start
            return answer

        core.cho_solve, localization.solve = counted_bulk, counted_phase
        try:
            yield
        finally:
            core.cho_solve, localization.solve = bulk, phase


def entries():
    """Exact retained anchors, starts and accepted reference fields."""
    path = HERE/'native_solver_evidence/core-all/core-4.txt'
    a = np.loadtxt(path).ravel(); n = int(a[0]); offset = 9
    lower = a[offset:offset+n+1]; offset += n+1
    Ra = a[offset:offset+2*n+1]; offset += 2*n+1
    start = a[offset:offset+n+1]; offset += n+1
    reference = a[offset:offset+n+1]
    early = dict(name='early', source=str(path.relative_to(HERE)),
                 lower=lower, Ra=Ra, Za=a[3], slip=a[4], time=a[1], dt=a[2],
                 start=start, reference=reference)
    a = dict(np.load(HERE/'native_solver_evidence/saddle-entry.npz'))
    saddle = dict(name='saddle', source='native_solver_evidence/saddle-entry.npz',
                  lower=a['lower'], Ra=a['Ra'], Za=float(a['Za']),
                  slip=float(a['slip_a']), time=float(a['time']), dt=float(a['dt']),
                  start=a['symmetric'], reference=a['stable'])
    path = native.E/'dt-1-1024'
    f = dict(np.load(path/'fields.npz'))
    hist = json.loads((path/'history.json').read_text())
    j = 640
    late = dict(name='late', source='native_endpoint_evidence/dt-1-1024',
                lower=f['phase'][j-1], Ra=f['R'][j-1], Za=float(f['Z'][j-1]),
                slip=hist[j-2]['slip'], time=.625, dt=1/1024,
                start=f['phase'][j-1], reference=f['phase'][j])
    return [early, saddle, late]


def metric(p, free):
    """Same positive sparse fracture-plus-mass search operator as native."""
    start = time.perf_counter()
    K = sparse.csc_matrix(p.r.F/core.A)+sparse.diags(p.r.mass, format='csc')
    factor = splu(K[free][:, free].tocsc())
    p.counts['metric_factors'] += 1
    p.seconds['metric'] += time.perf_counter()-start
    return factor


def quasi_newton(p, start, maxiter=600, memory=30):
    """Projected L-BFGS, exact reduced gradients, unchanged true-energy search.

    Recompute the restricted SPD base inverse when the search set changes.
    Restrict stored secants to that set and discard nonpositive pairings.
    One existing exact Newton polish is allowed at a small defect; it is counted
    separately. Saddle escape still uses the actual reduced tangent.
    """
    x = np.maximum(p.lower, start.copy())
    pairs = []; previous_free = None; factor = None; trace = []
    for iteration in range(maxiter):
        value, g = p.evaluate(x); defect = float(p.r.defect(g, x, p.lower))
        if defect <= TARGET:
            break
        free = np.flatnonzero((x > p.lower+1e-13) | (g < 0.))
        if not np.array_equal(previous_free, free):
            factor = metric(p, free); previous_free = free.copy()
        restricted = []
        for s, y in pairs:
            sf, yf = s[free], y[free]; sy = float(sf@yf)
            if sy > 1e-12*np.linalg.norm(sf)*np.linalg.norm(yf):
                restricted.append((sf, yf, 1/sy))
        q = g[free].copy(); alphas = []
        for s, y, rho in reversed(restricted):
            alpha = rho*float(s@q); alphas.append(alpha); q -= alpha*y
        tm = time.perf_counter(); z = factor.solve(q)
        p.counts['metric_solves'] += 1
        p.seconds['metric'] += time.perf_counter()-tm
        for (s, y, rho), alpha in zip(restricted, reversed(alphas)):
            z += s*(alpha-rho*float(y@z))
        direction = np.zeros_like(x); direction[free] = -z
        accepted = False
        for ls in range(50):
            trial = np.maximum(p.lower, np.minimum(1., x+2.**(-ls)*direction))
            if max(trial) >= 1 or np.array_equal(trial, x):
                continue
            pairing = float(g@(trial-x))
            if pairing >= 0:
                continue
            vt, gt = p.evaluate(trial); p.counts['line_trials'] += 1
            allowance = 32*np.finfo(float).eps*max(1., abs(value))
            if vt <= value+1e-4*pairing+allowance:
                s, y = trial-x, gt-g
                if float(s@y) > 1e-12*np.linalg.norm(s)*np.linalg.norm(y):
                    pairs.append((s, y)); pairs = pairs[-memory:]
                x = trial; accepted = True; break
        trace.append(dict(iteration=iteration, KKT=defect, free=len(free),
                          line_search=ls, potential=float(value)))
        if not accepted:
            break
    before = float(p.r.defect(p.evaluate(x)[1], x, p.lower))
    if TARGET < before < 1e-8:
        x = p.r.refine(x, p.lower, p.evaluate, p.hessian)
    return x, dict(iterations=len(trace), before_polish=before, trace=trace)


def endpoint_search(p, start):
    x = p.r.refine(start.copy(), p.lower, p.evaluate, p.hessian)
    if p.r.defect(p.evaluate(x)[1], x, p.lower) > TARGET:
        x, _ = old.minimize_phase(p, x, 1)
    return x


def solve(p, start, method):
    anchor = p.Ra.copy(); lower = p.lower.copy(); Za = p.Za
    started = time.perf_counter(); trace = []
    if method.startswith('staggered'):
        x = start.copy()
        for _ in range(int(method.split('-')[1])):
            p.evaluate(x)  # restores the trial mechanical endpoint
            x, defect = p.r.phase(p.lower)  # lower remains previous physical phase
            p.evaluate(x)  # reconstruct from Ra,Za, never from the trial endpoint
            trace.append(dict(KKT=float(p.r.defect(p.evaluate(x)[1], x, p.lower)),
                              subproblem_KKT=float(defect)))
    elif method == 'endpoint':
        x = endpoint_search(p, start)
    else:
        x, details = quasi_newton(p, start)
        trace.append(details)
    # All methods receive the same authoritative stability check. Fixed-budget
    # staggered output is a diagnostic, never published on residual alone.
    stamp = time.perf_counter(); status, v = old.spectrum(p, x)
    p.seconds['stability'] += time.perf_counter()-stamp
    escapes = 0
    if not method.startswith('staggered'):
        while status['admissible_negative'] and status['KKT'] <= TARGET and escapes < 2:
            guess = x+.5*status['amplitude_two_sided']*v
            # The retained endpoint uses this same positive mode orientation.
            if method == 'endpoint':
                x, details = old.minimize_phase(p, guess, 1)
            else:
                x, details = quasi_newton(p, guess)
            trace.append(details); escapes += 1
            stamp = time.perf_counter(); status, v = old.spectrum(p, x)
            p.seconds['stability'] += time.perf_counter()-stamp
    p.evaluate(x)
    assert np.array_equal(anchor, p.Ra) and np.array_equal(lower, p.lower) and Za == p.Za
    assert not p.Ra.flags.writeable and not p.lower.flags.writeable
    status.pop('lower_y', None)
    elapsed = time.perf_counter()-started
    result = dict(method=method, seconds=elapsed, counts=p.counts, timing=p.seconds,
                  KKT=float(p.r.defect(p.evaluate(x)[1], x, p.lower)),
                  stability=status, escapes=escapes, trace=trace,
                  minimum_increment=float(min(x-p.lower)), maximum=float(max(x)),
                  potential_J_per_m=float(core.A*p.evaluate(x)[0]),
                  **p.last[3])
    result['accepted'] = bool(result['KKT'] <= TARGET and status['stable'] and
        result['force'] < 1e-7 and result['contact'] < 1e-7 and
        result['minimum_increment'] >= -4.922e-12 and result['maximum'] < 1.)
    return x, result


def screen():
    out = E/'screen'; out.mkdir(exist_ok=False)
    for entry in entries():
        for method in ['endpoint', 'staggered-4', 'quasi-newton']:
            p = Measured()
            p.incoming(entry['lower'], entry['Ra'], entry['Za'], entry['slip'], entry['time'], entry['dt'])
            with p.count_solves():
                x, result = solve(p, entry['start'], method)
            result.update(entry=entry['name'], source=entry['source'], time=entry['time'],
                          dt=entry['dt'], phase_reference_error=float(max(abs(x-entry['reference']))))
            core.write(out/f'{entry["name"]}-{method}.json', result)
            np.savez_compressed(out/f'{entry["name"]}-{method}.npz', phase=x, R=p.last[4],
                                Z=p.last[5], velocity=p.r.velocity, lower=p.lower)
            print(json.dumps({k:result[k] for k in ['entry','method','accepted','seconds',
                             'KKT','phase_reference_error','counts']}), flush=True)


def trajectory(denominator):
    out = E/f'dt-1-{denominator}'; out.mkdir(exist_ok=False)
    started = time.perf_counter(); p = Measured(); r = p.r
    phi = r.initial.copy(); R = np.zeros_like(r.r); Z = slip = 0.
    np.savez_compressed(out/'initialization.npz', phase=phi, R=R, Z=Z,
                        particle_y=r.initial_particle_y, H=r.initial_particle_H)
    phases, strains, zs, velocities = [phi.copy()], [R.copy()], [Z], []
    rows, energies, failures = [], [], []
    previous_increment = np.zeros_like(phi); first_escape = None
    dt = 1/denominator
    for step in range(1, round(.625*denominator)+1):
        t = step*dt; p.incoming(phi, R, Z, slip, t, dt)
        before_counts = p.counts.copy()
        with p.count_solves():
            x, result = solve(p, phi+previous_increment, 'quasi-newton')
        counts = {key:p.counts[key]-before_counts[key] for key in p.counts}
        if not result['accepted']:
            failures.append(dict(time=t, **{k:result[k] for k in ['KKT','stability','force','contact']}))
            np.savez_compressed(out/'failed-trial.npz', phase=x, lower=phi, Ra=R, Za=Z,
                                R=p.last[4], Z=p.last[5])
            break
        if result['escapes'] and first_escape is None:
            first_escape = t
        Rb, Zb = p.last[4].copy(), p.last[5]; u = r.velocity.copy()
        # The same continuous reconstructed diagnostic path as the reference.
        # These residual works are reported, never added as compensation.
        stamp = time.perf_counter()
        ledger = core.path_ledger(r, phi, x, R, Z, Rb, Zb, u, result['V'], dt)
        ledger.update(time=t, phase_increment=float(max(x-phi)))
        energies.append(ledger)
        rows.append(dict(time=t, dt=dt, KKT=result['KKT'], phase_max=float(max(x)),
                         phase_increment=ledger['phase_increment'], centroid=old.centroid(r,x),
                         stability=result['stability'], seconds=result['seconds'], counts=counts,
                         diagnostic_seconds=time.perf_counter()-stamp,
                         **{k:result[k] for k in p.last[3]}))
        # Publication in this isolated driver is once, after full endpoint checks.
        assert np.array_equal(phi, p.lower) and np.array_equal(R, p.Ra) and Z == p.Za
        previous_increment = x-phi
        phi, R, Z, slip = x.copy(), Rb.copy(), Zb, result['slip']
        phases.append(phi.copy()); strains.append(R.copy()); zs.append(Z); velocities.append(u)
        if step % 64 == 0:
            core.write(out/'history.json', rows); core.write(out/'energy.json', energies)
            print(json.dumps(dict(step=step,time=t,phase=max(phi),elapsed=time.perf_counter()-started)),flush=True)
    np.savez_compressed(out/'fields.npz',phase=np.array(phases),R=np.array(strains),
                        Z=np.array(zs),velocity=np.array(velocities),y=r.y,yq=r.yq,
                        initial=r.initial,time=np.r_[0.,[row['time'] for row in rows]])
    core.write(out/'history.json', rows); core.write(out/'energy.json', energies)
    core.write(out/'summary.json',dict(n=512,dt=dt,last_accepted=rows[-1]['time'] if rows else 0.,
               completed=not failures,accepted=len(rows),failures=failures,first_escape=first_escape,
               runtime_seconds=time.perf_counter()-started,counts=p.counts,timing=p.seconds,
               peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--screen', action='store_true')
    mode.add_argument('--denominator', type=int, choices=[1024,2048])
    args = parser.parse_args()
    screen() if args.screen else trajectory(args.denominator)
