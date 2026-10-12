#!/usr/bin/env python3
"""Fixed-entry normal shadow; never publishes physical histories.

Exact dense normal operators are used only to study native search accuracy and
one recycled coarse direction. Full native angular/mechanical tests are separate.
"""
import json
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh

import native_endpoint as native

E = Path(__file__).resolve().parent / 'native_solver_evidence'


def run(mode, time, fields, history):
    j = int(np.flatnonzero(fields['time'] == time)[0])
    p = native.SharedFE()
    p.incoming(fields['phase'][j-1], fields['R'][j-1], fields['Z'][j-1],
               history[j-2]['slip'], time, time-fields['time'][j-1])
    lower, anchor, za = p.lower.copy(), p.Ra.copy(), p.Za
    x = lower.copy()
    metric = p.r.F/native.old.A + np.diag(p.r.mass)
    diagonal = np.diag(metric)
    target = native.TARGET
    initial = p.r.defect(p.evaluate(x)[1], x, lower)
    trace = []
    recycled = None
    for iteration in range(100):
        value, gradient = p.evaluate(x)
        projected = gradient.copy()
        projected[(x == lower) & (gradient > 0)] = 0
        defect = float(max(abs(projected)))
        if defect <= target:
            break
        contact = gradient >= 0
        bound = lower-x
        for _ in range(1000):
            free = np.flatnonzero(~contact)
            factor = cho_factor(metric[np.ix_(free, free)])
            obstacle = np.where(contact, bound, 0.)
            load = -gradient-metric@obstacle
            obstacle[free] = cho_solve(factor, load[free])
            multiplier = gradient+metric@obstacle
            next_contact = contact.copy()
            next_contact[contact & (multiplier < -target)] = False
            next_contact[(~contact) & (obstacle < bound)] = True
            if np.array_equal(next_contact, contact):
                break
            contact = next_contact
        else:
            raise RuntimeError('Obstacle prediction failed')
        calls, eta, stop, overlap = 0, 2e-7, 0., None
        if iteration == 0:
            step = obstacle.copy()
        else:
            hessian = p.hessian(x)
            hf = hessian[np.ix_(free, free)]
            step = np.where(contact, bound, 0.)
            rhs = (-gradient-hessian@step)[free]
            residual = rhs.copy()
            if mode != 'strict':
                eta = min(.01 if mode == 'adaptive-tight' else .25,
                          max(2e-7, .1*np.sqrt(defect/max(initial, target))))
            stop = eta*np.linalg.norm(rhs)
            if mode != 'strict':
                stop = max(stop, .1*target)
            coarse = None
            if mode in ['adaptive-coarse', 'adaptive-translation'] and recycled is not None:
                v = (recycled if mode == 'adaptive-coarse' else -np.gradient(x, p.r.y))[free].copy()
                if np.linalg.norm(v):
                    v /= np.linalg.norm(v)
                    hv = hf@v
                    calls += 1
                    curvature = v@hv
                    if curvature > 0:
                        w = cho_solve(factor, hv)
                        coarse = (v, hv, curvature, w, hv@w)
                        ev, vectors = eigh(hf, metric[np.ix_(free, free)], subset_by_index=[0, 0])
                        slow = vectors[:, 0]
                        overlap = float(abs(v@metric[np.ix_(free, free)]@slow)/
                                        np.sqrt(v@metric[np.ix_(free, free)]@v))
            def precondition(r):
                z = cho_solve(factor, r)
                if coarse is not None:
                    v, hv, curvature, w, pairing = coarse
                    a = v@r/curvature
                    z = z-w*a-v*(hv@z/curvature)+v*((1+pairing/curvature)*a)
                return z
            z = precondition(residual)
            direction, rz = z.copy(), residual@z
            for k in range(1000):
                if np.linalg.norm(residual) <= stop:
                    break
                hd = hf@direction
                calls += 1
                curvature = direction@hd
                if curvature <= 0:
                    raise RuntimeError('Unexpected negative curvature in retained growth control')
                alpha = rz/curvature
                step[free] += alpha*direction
                residual -= alpha*hd
                z = precondition(residual)
                next_rz = residual@z
                direction = z+next_rz/rz*direction
                rz = next_rz
            else:
                raise RuntimeError('Linear iteration budget exhausted')
        if gradient@step >= 0 or not np.any(step):
            step = obstacle.copy() if gradient@obstacle < 0 else -projected/diagonal
        for ls in range(35):
            trial = np.maximum(lower, np.minimum(1., x+2.**(-ls)*step))
            if max(trial) >= 1:
                continue
            vt, _ = p.evaluate(trial)
            allowance = 32*np.finfo(float).eps*max(1., abs(value))
            if vt <= value+1e-4*min(0., gradient@(trial-x))+allowance:
                break
        else:
            raise RuntimeError('True-energy line search failed')
        trace.append(dict(iteration=iteration, KKT=defect, eta=eta,
                          linear_target=stop, tangent_calls=calls, line_search=ls,
                          recycled_slow_overlap=overlap))
        recycled = trial-x
        x = trial
    else:
        raise RuntimeError('Nonlinear budget exhausted')
    assert np.array_equal(p.lower, lower) and np.array_equal(p.Ra, anchor) and p.Za == za
    assert min(x-lower) >= 0 and max(x) < 1
    final = float(p.r.defect(p.evaluate(x)[1], x, lower))
    assert final <= target
    return dict(mode=mode, time=time, KKT=final, phase_error=float(max(abs(x-fields['phase'][j]))),
                iterations=len(trace), tangent_calls=sum(t['tangent_calls'] for t in trace),
                trace=trace, published=False)


if __name__ == '__main__':
    E.mkdir(exist_ok=True)
    fields = dict(np.load(native.E/'prefix-2-fine-1024-onset/fields.npz'))
    history = json.loads((native.E/'prefix-2-fine-1024-onset/history.json').read_text())
    results = [run(mode, t, fields, history)
               for mode in ['strict', 'adaptive', 'adaptive-coarse']
               for t in [.529296875, .5302734375, .53125]]
    (E/'normal-search-comparison.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k != 'trace'} for r in results]))
