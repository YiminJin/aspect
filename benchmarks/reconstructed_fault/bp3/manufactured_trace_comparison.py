"""Independent finite-width trace compatibility control for the kink test.

An analytic piecewise-smooth rate is prescribed, with a jump at the end of
the free interval. Compare the current continuous Q1 trace and Q1 with an
independent free-side trace, using the same continuum operator and load.
This is a diagnostic discretization alternative, NOT a BP3 topology change.
"""
import csv
import json
import argparse
from pathlib import Path
import time

import numpy as np
from scipy.linalg import solve

from manufactured_gradient_kink import LX, D, HERE, elastic_symbol, quadrature


def element_transforms(a, h, k):
    z = k*h/2
    derivative = np.zeros_like(z)
    ordinary = abs(z) > 1e-5
    derivative[ordinary] = (z[ordinary]*np.cos(z[ordinary])-np.sin(z[ordinary]))/z[ordinary]**2
    derivative[~ordinary] = -z[~ordinary]/3+z[~ordinary]**3/30
    center = np.exp(-1j*k*(a+h/2))*h/LX
    midpoint = center*np.sinc(z/np.pi)/2
    linear = .5j*center*derivative
    return midpoint-linear, midpoint+linear


def interval_transform(k):
    # Integral on [0,4], normalized by the full periodic length.
    return .5*np.exp(-2j*k)*np.sinc(2*k/np.pi)


def target_values(x):
    return -.5*(1-np.cos(np.pi*x/4))


def comparison(nf, cutoff, out, nonlinear=False):
    started = time.monotonic()
    n = nf//2
    h = LX/nf
    k = 2*np.pi*np.arange(-cutoff, cutoff+1)/LX
    target_hat = (-.5*interval_transform(k)+
                  .25*interval_transform(k-np.pi/4)+
                  .25*interval_transform(k+np.pi/4))
    elastic = elastic_symbol(k, cutoff)
    # The endpoint basis functions are half hats restricted to the free
    # physical interval, not added copies with overlapping integration area.
    fourier = np.zeros((n+1, len(k)), dtype=complex)
    mass = np.zeros((n+1, n+1))
    local_load = np.zeros(n+1)
    for e in range(n):
        left, right = element_transforms(e*h, h, k)
        fourier[e] += left
        fourier[e+1] += right
        mass[e:e+2, e:e+2] += h/6*np.array([[2., 1.], [1., 2.]])
        x, w = quadrature(np.array([e*h, (e+1)*h]), 12)
        local_load[e] += np.sum(w*(1-(x-e*h)/h)*target_values(x))
        local_load[e+1] += np.sum(w*((x-e*h)/h)*target_values(x))
    matrix = LX*np.real((fourier.conj()*elastic)@fourier.T)+D*mass
    load = LX*np.real(fourier.conj()@(elastic*target_hat))+D*local_load
    x = np.arange(n+1)*h
    if nonlinear:
        from check_gradient_kink_nonlinear import friction
        from manufactured_gradient_kink import product
        from scipy import sparse
        q, wq = quadrature(np.arange(n+1)*h, 12)
        cell = np.floor(q/h).astype(int)
        z = q/h-cell
        Q = sparse.coo_matrix((np.array([1-z, z]).T.ravel(),
                               (np.repeat(np.arange(len(q)), 2),
                                np.array([cell, cell+1]).T.ravel())), shape=(len(q), n+1)).tocsr()
        matrix -= D*mass
        load -= D*local_load
        load += Q.T@(wq*friction(target_values(q))[0])
    # s=0 is compatible with the zero prescribed interval and kept continuous.
    # At s=4, continuity prescribes zero; the alternative leaves that free trace
    # unknown. Prescribed s>4 stays identically zero in both alternatives.
    values = {}
    convergence = {}
    for name, stop in [('continuous', n), ('independent_trace', n+1)]:
        free = np.arange(1, stop)
        v = np.zeros(n+1)
        for iteration in range(12 if nonlinear else 2):
            residual = matrix@v-load
            tangent = matrix.copy()
            if nonlinear:
                f, df = friction(Q@v)
                residual += Q.T@(wq*f)
                tangent += product(Q, wq*df, Q).toarray()
            relative = np.linalg.norm(residual[free])/np.linalg.norm(load[free])
            if relative < 1e-12:
                break
            v[free] -= solve(tangent[np.ix_(free, free)], residual[free], assume_a='sym')
        else:
            raise AssertionError('Trace control failed fresh residual check')
        values[name] = v
        convergence[name] = dict(iterations=iteration, fresh_relative_residual=float(relative))
    target = target_values(x)
    result = dict(nf=nf, h=h, cutoff=cutoff, nonlinear=nonlinear, convergence=convergence,
                  continuous_last_free=float(values['continuous'][-2]),
                  continuous_last_free_exact=float(target[-2]),
                  continuous_undershoot=float(max(0., -1-min(values['continuous']))),
                  independent_trace=float(values['independent_trace'][-1]),
                  independent_trace_max_error=float(max(abs(values['independent_trace']-target))),
                  continuous_L2=float(np.sqrt((values['continuous']-target)@mass@(values['continuous']-target)/4)),
                  independent_trace_L2=float(np.sqrt((values['independent_trace']-target)@mass@(values['independent_trace']-target)/4)),
                  seconds=time.monotonic()-started)
    with (out/f'trace_{nf}_{cutoff}.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['s', 'target_free_trace', 'continuous', 'independent_trace'])
        writer.writerows(zip(x, target, values['continuous'], values['independent_trace']))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nonlinear', action='store_true')
    args = parser.parse_args()
    output = HERE/'gradient-kink'/('trace-nonlinear' if args.nonlinear else 'trace-comparison')
    output.mkdir(parents=True, exist_ok=False)
    results = [comparison(n, 2048, output, args.nonlinear) for n in (32, 64, 128, 256)]
    finer = comparison(128, 4096, output, args.nonlinear)
    assert abs(results[2]['independent_trace']-finer['independent_trace']) < 2e-7
    assert abs(results[2]['continuous_last_free']-finer['continuous_last_free']) < 2e-7
    with (output/'summary.json').open('w') as stream:
        json.dump(dict(cases=results, reference_check=finer), stream, indent=2)
    print(json.dumps(dict(cases=results, reference_check=finer), indent=2))
