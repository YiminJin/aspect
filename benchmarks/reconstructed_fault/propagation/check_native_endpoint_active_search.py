#!/usr/bin/env python3
"""Frozen-entry 1D shadow of native projected Newton; no history publication.

Uses exact reduced normal Hessians to isolate bound/search behavior from native
Krylov, pressure and tangential modes. It is a cost diagnostic, not a replacement
solver or a full native equivalence test.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve

import native_endpoint as native

E = Path(__file__).resolve().parent / 'native_tangent_evidence'


def check(all_iteration_obstacle=False):
    f = dict(np.load(native.E / 'prefix-2-fine-1024-onset/fields.npz'))
    history = json.loads((native.E / 'prefix-2-fine-1024-onset/history.json').read_text())
    results = []
    times=[.5283203125, .529296875, .5302734375, .53125] if all_iteration_obstacle else [.5302734375, .53125]
    for time in times:
        j = int(np.flatnonzero(f['time'] == time)[0])
        p = native.SharedFE()
        p.incoming(f['phase'][j-1], f['R'][j-1], f['Z'][j-1],
                   history[j-2]['slip'], time, time-f['time'][j-1])
        lower = p.lower.copy()
        anchor = p.Ra.copy()
        za = p.Za
        x = lower.copy()
        metric = p.r.F/native.old.A + np.diag(p.r.mass)
        diagonal = np.abs(np.diag(p.r.F))/native.old.A + p.r.mass
        trace = []
        for iteration in range(100):
            value, gradient = p.evaluate(x)
            projected = gradient.copy()
            projected[(x == lower) & (projected > 0)] = 0
            defect = float(max(abs(projected)))
            if defect <= native.TARGET:
                break
            contact = (x == lower) & (gradient >= 0)
            predicted = iteration == 0
            if predicted or all_iteration_obstacle:
                contact = gradient >= 0
                bound_step = lower-x
                for _ in range(1000):
                    free = np.flatnonzero(~contact)
                    step = np.where(contact, bound_step, 0.)
                    load = -gradient-metric@step
                    step[free] = solve(metric[np.ix_(free, free)], load[free], assume_a='pos')
                    multiplier = gradient + metric @ step
                    next_contact = contact.copy()
                    next_contact[contact & (multiplier < -native.TARGET)] = False
                    next_contact[(~contact) & (step < bound_step)] = True
                    if np.array_equal(next_contact, contact):
                        break
                    contact = next_contact
                else:
                    raise RuntimeError('Shadow obstacle search did not converge')
            if not predicted:
                free = np.flatnonzero(~contact)
                hessian = p.hessian(x)
                active_step = np.where(contact, lower-x, 0.)
                load = -gradient-hessian@active_step
                step = active_step.copy()
                step[free] = solve(hessian[np.ix_(free, free)], load[free], assume_a='sym')
            if gradient @ step >= 0 or not np.any(step):
                step = -projected/diagonal
            for line_search in range(35):
                trial = np.maximum(lower, np.minimum(1., x + 2.**(-line_search)*step))
                if max(trial) >= 1:
                    continue
                vt, _ = p.evaluate(trial)
                allowance = 32*np.finfo(float).eps*max(1., abs(value))
                if vt <= value + 1e-4*min(0., gradient @ (trial-x)) + allowance:
                    break
            else:
                raise RuntimeError('Shadow energy line search failed')
            trace.append(dict(iteration=iteration, KKT=defect, free=int(sum(~contact)),
                              line_search=line_search, value=float(value),
                              newly_contacting=int(sum((trial == lower) & (x > lower))),
                              increment=float(max(trial-x))))
            x = trial
        assert np.array_equal(p.lower, lower) and np.array_equal(p.Ra, anchor) and p.Za == za
        _, g = p.evaluate(x)
        assert p.r.defect(g, x, lower) <= native.TARGET
        assert min(x-lower) >= 0 and max(x) < 1
        results.append(dict(time=time, KKT=float(p.r.defect(g, x, lower)),
                            iterations=len(trace), phase_error=float(max(abs(x-f['phase'][j]))),
                            trace=trace, published=False))
    name='active-search-obstacle-shadow.json' if all_iteration_obstacle else 'active-search-shadow.json'
    (E/name).write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps([{k:v for k,v in row.items() if k != 'trace'} for row in results]))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--all-iteration-obstacle', action='store_true')
    check(parser.parse_args().all_iteration_obstacle)
