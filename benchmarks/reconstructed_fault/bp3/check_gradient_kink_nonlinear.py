"""Reuse saved bulk eliminations with exact frozen-state BP3 friction.

This cheap check removes the infinitesimal-friction approximation; it does
not change or advance Theta. Normal feedback still cancels by strip symmetry.
"""
import csv
import json
import time

import numpy as np
from scipy.linalg import solve, circulant

from manufactured_gradient_kink import HERE, LX, D, basis, quadrature, product


A, B, MU0 = .025, .015, .6
EPSILON = .01
# V/Vp is dimensionless; Dc/Vp=8e6 s and Vref/Vp=1000, as in deep BP3.
ARG0 = np.exp((MU0+B*np.log(1000.))/A)/2000.
SIGMA = D/A


def friction(w):
    rate = 1+EPSILON*w
    assert np.all(rate > 0)
    arg = ARG0*rate
    value = SIGMA*A*(np.arcsinh(arg)-np.arcsinh(ARG0))/EPSILON
    derivative = SIGMA*A*arg/np.sqrt(1+arg*arg)/rate
    return value, derivative


def run(label):
    saved = np.load(HERE/'gradient-kink'/label/'matrices.npz')
    nf = len(saved['mass'])
    h = LX/nf
    mass = np.zeros(nf)
    mass[0], mass[1], mass[-1] = 2*h/3, h/6, h/6
    exact_mass = circulant(mass)
    e = saved['surface']-D*saved['mass']
    er = saved['reference']-D*exact_mass
    x = np.arange(nf)*h
    target = -np.maximum(1-abs(x-3), 0.)
    q, wq = quadrature(np.linspace(0, LX, nf+1), 12)
    Q = basis(q, nf, 1, LX)
    load = er@target + np.asarray(Q.T@(wq*friction(Q@target)[0]))
    free = x < 4
    v = np.zeros(nf)
    reference = np.zeros(nf)
    results = []
    for name, elastic, values in [('FE', e, v), ('reference', er, reference)]:
        initial = np.linalg.norm(load[free])
        for iteration in range(12):
            f, df = friction(Q@values)
            residual = elastic@values + Q.T@(wq*f) - load
            if np.linalg.norm(residual[free]) < 1e-12*initial:
                break
            tangent = elastic + product(Q, wq*df, Q).toarray()
            values[free] -= solve(tangent[np.ix_(free, free)], residual[free], assume_a='sym')
        else:
            raise AssertionError('Frozen-state Newton did not converge')
        results.append(dict(name=name, iterations=iteration,
                            fresh_relative_residual=float(np.linalg.norm(residual[free])/initial),
                            max_error=float(max(abs(values-target)))))
    np.testing.assert_allclose(reference, target, rtol=2e-10, atol=2e-11)
    direction = np.random.default_rng(4893).normal(size=nf)
    fplus = friction(Q@(v+1e-5*direction))[0]
    fminus = friction(Q@(v-1e-5*direction))[0]
    analytic = Q.T@(wq*friction(Q@v)[1]*(Q@direction))
    numeric = Q.T@(wq*(fplus-fminus)/(2e-5))
    fd_error = np.linalg.norm(analytic-numeric)/np.linalg.norm(analytic)
    assert fd_error < 1e-7
    return dict(label=label, amplitude=EPSILON, results=results, friction_tangent_FD_error=float(fd_error))


if __name__ == '__main__':
    started = time.monotonic()
    output = HERE/'gradient-kink'/'nonlinear-friction'
    output.mkdir(parents=True, exist_ok=False)
    results = [run(label) for label in ('bulk32', 'bulk128-normal64')]
    summary = dict(cases=results, seconds=time.monotonic()-started)
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
