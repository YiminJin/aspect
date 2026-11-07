#!/usr/bin/env python3
"""Independent integration screens, regional work and in-rule energy audit."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
from compare_frozen_finite_end import (OUT, GRIDS, Frozen, triangles, batches,
                                     index_basis, mass, METRIC, dot)
from check_true_stress_projection import check


def audit_geometry(tri):
    # Independent analytic moments on a unit triangle test the Duffy Jacobian.
    unit = np.array([[[0., 0.], [1., 0.], [0., 1.]]])
    x, w = next(batches(unit, 6))
    x, w = x.reshape(-1, 2), w.ravel()
    errors = []
    for a in range(5):
        for b in range(5):
            exact = math.factorial(a)*math.factorial(b)/math.factorial(a+b+2)
            errors.append(abs(w@(x[:, 0]**a*x[:, 1]**b)-exact))
    # Full nine-by-nine DGQ2 mass, every finest cell, including clipped cells.
    n = GRIDS[-1]
    gram = np.zeros((n*n, 9, 9))
    for x, w in batches(tri, 6):
        nt, nq, _ = x.shape
        cell, B = index_basis(x.reshape(-1, 2), n)
        B = B.reshape(nt, nq, 9)
        np.add.at(gram, cell.reshape(nt, nq)[:, 0], np.einsum('tqk,tqm,tq->tkm', B, B, w))
    gram_error = float(abs(gram-np.diag(mass(n))).max())
    assert max(errors) < 2e-11 and gram_error < 2e-11
    return dict(triangle_moment_max_error=max(errors), full_cell_mass_max_error=gram_error)


def regional_work(f, tri, coeff):
    M = {k: np.zeros((len(f.X), len(f.X))) for k in ['all', 'central', 'ends']}
    rhs = {k: np.zeros((len(f.X), 2)) for k in M}
    projected = {key: np.zeros((len(f.X), 2)) for key in coeff}
    for x, w in batches(tri, 10):
        x, w = x.reshape(-1, 2), w.ravel()
        src, chi, j, u = f.fields(x)
        s = (x-f.X[0])@f.t
        masks = dict(all=np.ones(len(x), bool), central=(s > .3) & (s < f.length-.3),
                     ends=(s < .3) | (s > f.length-.3))
        shear = src@(METRIC@f.S)
        for window, mask in masks.items():
            weight = w*chi*mask
            for a, ha in [(j, 1-u), (j+1, u)]:
                np.add.at(rhs[window], a, (weight*ha)[:, None]*shear)
                for b, hb in [(j, 1-u), (j+1, u)]:
                    np.add.at(M[window], (a, b), weight*ha*hb)
        for n in GRIDS:
            ids, B = index_basis(x, n)
            for tag in ['cut', 'q8']:
                key = f'n{n}_{tag}'
                tau = np.einsum('pk,pklc->plc', B, coeff[key][ids])
                shear = tau@(METRIC@f.S)
                for a, ha in [(j, 1-u), (j+1, u)]:
                    np.add.at(projected[key], a, (w*chi*ha)[:, None]*shear)
    local = np.linalg.solve(M['all'], rhs['all'])
    result = {}
    for key, load in projected.items():
        delta = np.linalg.solve(M['all'], load)-local
        result[key] = {name: dict(
            relative_L2=np.sqrt(np.einsum('il,ij,jl->l', delta, mat, delta)/
                                np.einsum('il,ij,jl->l', local, mat, local)).tolist(),
            squared_error=np.einsum('il,ij,jl->l', delta, mat, delta).tolist())
            for name, mat in M.items()}
    return result


def in_rule(f, n, coeff):
    q = 8
    g, w = np.polynomial.legendre.leggauss(q)
    u = np.stack(np.meshgrid(g, g, indexing='ij'), -1).reshape(-1, 2)
    weight = np.outer(w, w).ravel()*(4/n)**2/4
    M = np.zeros((len(f.X), len(f.X)))
    rhs, rhs_local = np.zeros((len(f.X), 2)), np.zeros((len(f.X), 2))
    loss, remainder = np.zeros(2), np.zeros(2)
    for start in range(0, n*n, 64):
        ids = np.arange(start, min(start+64, n*n))
        ij = np.stack([ids//n, ids % n], axis=1)
        x = (-2+4/n*(ij[:, None]+.5*(u+1))).reshape(-1, 2)
        w = np.tile(weight, len(ids))
        src, chi, j, xi = f.fields(x)
        cell, B = index_basis(x, n)
        tau = np.einsum('pk,pklc->plc', B, coeff[cell])
        loss += np.einsum('pl,p->l', dot(src, src)-dot(tau, tau), w)/(4*f.G)
        remainder += np.einsum('pl,p->l', dot(src-tau, src-tau), w)/(4*f.G)
        for a, ha in [(j, 1-xi), (j+1, xi)]:
            np.add.at(rhs, a, (w*chi*ha)[:, None]*(tau@(METRIC@f.S)))
            np.add.at(rhs_local, a, (w*chi*ha)[:, None]*(src@(METRIC@f.S)))
            for b, hb in [(j, 1-xi), (j+1, xi)]:
                np.add.at(M, (a, b), w*chi*ha*hb)
    assert np.max(abs(loss-remainder)) < 1e-15
    return dict(energy_local_minus_projected=loss.tolist(), error_energy=remainder.tolist(),
                energy_identity_error=abs(loss-remainder).tolist(),
                normalization_relative_error=float(M.sum()/f.length-1),
                work_traction=np.linalg.solve(M, rhs).tolist(),
                local_work_traction=np.linalg.solve(M, rhs_local).tolist(),
                slip_power=(f.V@rhs).tolist(), local_slip_power=(f.V@rhs_local).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    target = args.output/'audit.json'
    if target.exists():
        raise SystemExit('Refusing to overwrite completed audit.')
    start = time.perf_counter()
    f = Frozen()
    coeff = dict(np.load(args.output/'projection-coefficients.npz'))
    selected = {k: v for k, v in coeff.items() if k.endswith(('_cut', '_q8'))}
    tri = triangles(f, GRIDS[-1])
    geometry = audit_geometry(tri)
    print('Analytic triangle / full cell mass identities pass', flush=True)
    native = {case: check(case) for case in ['true-native-final', 'true-native-layered']}
    print('Native frozen q8 norm bridges pass', flush=True)
    regional = regional_work(f, tri, selected)
    rules = {n: in_rule(f, n, coeff[f'n{n}_q8']) for n in GRIDS}
    # Positive nonzero one-sided source jumps at the two finite endpoints.
    jumps = []
    for node in [0, len(f.X)-1]:
        x = f.X[node]
        # This is the one-sided limit, not a floating-point test for ownership
        # of the measure-zero endpoint. Dot-product roundoff can put the exact
        # last vertex just outside the finite strip.
        cell, B = index_basis(x[None, :], f.state.n)
        phi = float(np.einsum('pk,pk->p', B, f.phi[cell])[0])
        chi = f.state.m*phi*(1+phi)/(1-phi)**2/f.I[node]
        eta = np.array([10., 10*(1+.5*np.tanh(4*x[0]))])
        k = -eta*np.expm1(-f.G*f.dt/eta)
        jump = -2*k[:, None]*chi*f.V[node]*f.S
        assert np.all(abs(jump@(METRIC@f.S)) > 0)
        jumps.append(dict(position=x.tolist(), tensor_jump=jump.tolist(),
                          shear_jump=(jump@(METRIC@f.S)).tolist()))
    result = dict(geometry=geometry, native_bridges=native, regional_work=regional,
                  in_rule_q8=rules, endpoint_jumps=jumps, elapsed_s=time.perf_counter()-start)
    target.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(geometry=geometry, elapsed_s=result['elapsed_s'])), flush=True)


if __name__ == '__main__':
    main()
