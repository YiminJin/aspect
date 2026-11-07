#!/usr/bin/env python3
"""Frozen finite-end DGQ2 approximation versus quadrature, without native runs.

One retained Q1 profile and surface I are held fixed on every tensor grid.
Polygon clipping and triangle quadrature resolve the finite-end discontinuity;
ordinary cell Gauss rules deliberately do not. See FROZEN_FINITE_END_PROTOCOL.md.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from assess_material_native import State, ROOT, RUNS, basis, read

OUT = ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/frozen_finite_end'
GRIDS = (16, 32, 64, 128)
ORDERS = (8, 12, 16, 24)
METRIC = np.array([[2., 1., 0.], [1., 2., 0.], [0., 0., 2.]])


def dot(a, b):
    return np.einsum('...i,ij,...j->...', a, METRIC, b)


def index_basis(x, n):
    z = (x+2)*(n/4)
    ij = np.minimum(np.floor(z).astype(int), n-1)
    return ij[:, 0]*n+ij[:, 1], basis(2*(z-ij)-1)


def mass(n):
    return (4/n)**2/np.outer([1, 3, 5], [1, 3, 5]).ravel()


def norm_coeff(a, n):
    return np.sqrt(np.einsum('ckl,k->l', dot(a, a), mass(n)))


class Frozen:
    def __init__(self, case='true-c-p-ref'):
        folder = RUNS/case
        command = json.loads((folder/'command.json').read_text())
        model = {k: float(v) for k, v in read(folder/'output/material-model.csv')[0].items()}
        self.state = State(folder/'output', 0, command['arguments']['level'], model['m'])
        s = self.state
        self.phi = s.coeff[:, :, :, 0].reshape(-1, 9)
        self.X = s.surface[:, 1:3].copy()
        self.t = self.X[-1]-self.X[0]
        self.length = np.linalg.norm(self.t)
        self.t /= self.length
        self.normal = np.array([-self.t[1], self.t[0]])
        self.labels = (self.X-self.X[0])@self.t
        assert np.max(abs((self.X-self.X[0])@self.normal)) < 1e-14
        self.S = np.array([self.t[0]*self.normal[0], self.t[1]*self.normal[1],
                           .5*(self.t[0]*self.normal[1]+self.t[1]*self.normal[0])])
        self.I = s.surface[:, 5].copy()
        self.V = .002*(1+.2*np.cos(self.X[:, 0]))
        assert np.all(self.I > 0)
        corners = np.array([[-2, -2], [-2, 2], [2, -2], [2, 2]])
        assert np.max(abs((corners-self.X[0])@self.normal)) <= s.surface[:, 7].min()+1e-13
        self.dt = command['arguments']['initial_dt']
        self.G = model['G']

    def fields(self, x):
        cell, B = index_basis(x, self.state.n)
        phi = np.einsum('pk,pk->p', B, self.phi[cell])
        along = (x-self.X[0])@self.t
        j = np.clip(np.searchsorted(self.labels, along, side='right')-1, 0, len(self.I)-2)
        u = (along-self.labels[j])/(self.labels[j+1]-self.labels[j])
        valid = (along >= 0) & (along <= self.length)
        I = (1-u)*self.I[j]+u*self.I[j+1]
        chi = np.zeros(len(x))
        chi[valid] = self.state.m*phi[valid]*(1+phi[valid])/(1-phi[valid])**2/I[valid]
        V = (1-u)*self.V[j]+u*self.V[j+1]
        eta = np.stack([np.full(len(x), 10.), 10*(1+.5*np.tanh(4*x[:, 0]))], axis=1)
        kappa = -eta*np.expm1(-self.G*self.dt/eta)
        src = 2*kappa[:, :, None]*(np.array([.03, -.03, .01])- (chi*V)[:, None, None]*self.S)
        return src, chi, j, u


def clip(poly, normal, offset):
    """Sutherland-Hodgman half-plane clipping, normal.x <= offset."""
    if len(poly) == 0:
        return poly
    result = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        da, db = a@normal-offset, b@normal-offset
        if da <= 0:
            result.append(a)
        if (da < 0 < db) or (db < 0 < da):
            result.append(a+(b-a)*da/(da-db))
    return np.array(result).reshape(-1, 2)


def triangles(f, n):
    """Partition fixed Q1 cells and nested DG cells, labels, and norm windows."""
    assert n % f.state.n == 0
    cuts = [(f.t, a+f.X[0]@f.t) for a in sorted(set([
        *f.labels, -.3, .3, f.length-.3, f.length+.3]))]
    cuts += [(f.normal, a+f.X[0]@f.normal) for a in (-.6, -.4, .4, .6)]
    result = []
    for i in range(n):
        for j in range(n):
            square = -2+4/n*(np.array([[0, 0], [1, 0], [1, 1], [0, 1]])+[i, j])
            pieces = [square]
            for normal, offset in cuts:
                d = square@normal-offset
                if d.min() >= 0 or d.max() <= 0:
                    continue
                split = []
                for p in pieces:
                    d = p@normal-offset
                    if d.min() >= 0 or d.max() <= 0:
                        split.append(p)
                    else:
                        split.extend([clip(p, normal, offset), clip(p, -normal, -offset)])
                pieces = split
            for p in pieces:
                for k in range(1, len(p)-1):
                    tri = np.array([p[0], p[k], p[k+1]])
                    a, b = tri[1]-tri[0], tri[2]-tri[0]
                    if abs(a[0]*b[1]-a[1]*b[0]) > 1e-25:
                        result.append(tri)
    return np.array(result)


def batches(tri, q):
    g, w = np.polynomial.legendre.leggauss(q)
    g, w = (g+1)/2, w/2
    r, s = np.meshgrid(g, g, indexing='ij')
    r, s = r.ravel(), s.ravel()
    weight = np.outer(w, w).ravel()*(1-r)
    for start in range(0, len(tri), 128):
        a = tri[start:start+128]
        v, z = a[:, 1]-a[:, 0], a[:, 2]-a[:, 0]
        det = abs(v[:, 0]*z[:, 1]-v[:, 1]*z[:, 0])
        x = a[:, None, 0]+r[None, :, None]*v[:, None]+((1-r)*s)[None, :, None]*z[:, None]
        yield x, det[:, None]*weight


def ordinary(f, n, q):
    g, w = np.polynomial.legendre.leggauss(q)
    u = np.stack(np.meshgrid(g, g, indexing='ij'), -1).reshape(-1, 2)
    B = basis(u)
    weight = np.outer(w, w).ravel()*(4/n)**2/4
    coeff = np.zeros((n*n, 9, 2, 3))
    for start in range(0, n*n, 64):
        ids = np.arange(start, min(start+64, n*n))
        ij = np.stack([ids//n, ids % n], axis=1)
        x = -2+4/n*(ij[:, None]+.5*(u+1))
        src = f.fields(x.reshape(-1, 2))[0].reshape(len(ids), -1, 2, 3)
        coeff[ids] = np.einsum('qk,tqlc,q->tklc', B, src, weight)/mass(n)[None, :, None, None]
    return coeff


def reference(f, tri, q):
    moments = {n: np.zeros((n*n, 9, 2, 3)) for n in GRIDS}
    work = {n: np.zeros((n*n, 9, len(f.X))) for n in GRIDS}
    M = np.zeros((len(f.X), len(f.X)))
    rhs = np.zeros((len(f.X), 2))
    total = np.zeros(2)
    slip_norm = np.zeros(2)
    area = 0.
    cell_area = np.zeros(GRIDS[-1]**2)
    for x, w in batches(tri, q):
        nt, nq, _ = x.shape
        src, chi, j, u = f.fields(x.reshape(-1, 2))
        source = src.reshape(nt, nq, 2, 3)
        total += np.einsum('tql,tq->l', dot(source, source), w)
        eta = np.stack([np.full(nt*nq, 10.), 10*(1+.5*np.tanh(4*x.reshape(-1, 2)[:, 0]))], axis=1)
        base = -2*eta[:, :, None]*np.expm1(-f.G*f.dt/eta)[:, :, None]*np.array([.03, -.03, .01])
        slip_norm += np.einsum('pl,p->l', dot(src-base, src-base), w.ravel())
        weight = w.ravel()*chi
        shear = np.einsum('plc,c->pl', src, METRIC@f.S)
        for a, ha in [(j, 1-u), (j+1, u)]:
            np.add.at(rhs, a, (weight*ha)[:, None]*shear)
            for b, hb in [(j, 1-u), (j+1, u)]:
                np.add.at(M, (a, b), weight*ha*hb)
        for n in GRIDS:
            ids, B = index_basis(x.reshape(-1, 2), n)
            ids = ids.reshape(nt, nq)
            assert np.all(ids == ids[:, :1])
            B = B.reshape(nt, nq, 9)
            np.add.at(moments[n], ids[:, 0], np.einsum('tqk,tqlc,tq->tklc', B, source, w))
            # Every triangle lies within one surface-label strip. Exterior
            # triangles have chi=0 and contribute no surface mass or load.
            jj = j.reshape(nt, nq)
            assert np.all(jj == jj[:, :1])
            for node, hat in [(jj[:, 0], 1-u), (jj[:, 0]+1, u)]:
                values = np.einsum('tqk,tq->tk', B, (weight*hat).reshape(nt, nq))
                np.add.at(work[n], (ids[:, 0], slice(None), node), values)
        area += w.sum()
        np.add.at(cell_area, ids[:, 0], w.sum(axis=1))
    coeff = {n: moments[n]/mass(n)[None, :, None, None] for n in GRIDS}
    assert abs(area-16) < 2e-11
    assert np.max(abs(cell_area-(4/GRIDS[-1])**2)) < 2e-11
    return dict(coeff=coeff, work=work, mass=M, rhs=rhs, norm2=total,
                slip_norm2=slip_norm, area=area, cell_area_error=float(np.max(abs(cell_area-(4/GRIDS[-1])**2))))


def work_traction(f, ref, n, coeff):
    shear = np.einsum('aklc,c->akl', coeff, METRIC@f.S)
    return np.linalg.solve(ref['mass'], np.einsum('akj,akl->jl', ref['work'][n], shear))


def point_traction(f, n, coeff):
    # Split the line at every Q1/tensor face, label and diagnostic boundary.
    cuts = [0., f.length, .3, f.length-.3, *f.labels]
    for axis in (0, 1):
        if abs(f.t[axis]) > 1e-14:
            cuts.extend(((np.linspace(-2, 2, max(n, f.state.n)+1)-f.X[0, axis])/f.t[axis]).tolist())
    cuts = np.unique(np.clip(cuts, 0, f.length))
    g, w = np.polynomial.legendre.leggauss(10)
    s = (.5*(cuts[:-1, None]+cuts[1:, None])+.5*np.diff(cuts)[:, None]*g).ravel()
    weight = (.5*np.diff(cuts)[:, None]*w).ravel()
    x = f.X[0]+s[:, None]*f.t
    ids, B = index_basis(x, n)
    projected = np.einsum('pk,pklc->plc', B, coeff[ids])
    src = f.fields(x)[0]
    a, b = projected@(METRIC@f.S), src@(METRIC@f.S)
    result = {}
    for name, mask in [('all', np.ones(len(s), bool)), ('central', (s > .3) & (s < f.length-.3)),
                       ('ends', (s < .3) | (s > f.length-.3))]:
        err2 = np.einsum('pl,p->l', (a-b)**2, weight*mask)
        norm2 = np.einsum('pl,p->l', b*b, weight*mask)
        result[name] = dict(L2=np.sqrt(err2).tolist(), relative_L2=np.sqrt(err2/norm2).tolist())
    return result


def windows(f, tri, ref, ordinary_coeff):
    # Stable direct residual integration also independently checks the global
    # orthogonal-error decomposition (avoid subtracting two large energies).
    out = {n: {kind: {window: np.zeros((2, 5)) for window in ['all', 'central', 'ends']}
               for kind in ['reference', 'q8']} for n in GRIDS}
    for x, w in batches(tri, 10):
        points = x.reshape(-1, 2)
        source = f.fields(points)[0]
        s, r = (points-f.X[0])@f.t, (points-f.X[0])@f.normal
        masks = dict(all=np.ones(len(points), bool), central=(s > .3) & (s < f.length-.3) & (abs(r) < .4),
                     ends=(np.minimum(abs(s), abs(s-f.length)) < .3) & (abs(r) < .6))
        for n in GRIDS:
            ids, B = index_basis(points, n)
            for kind, coeff in [('reference', ref['coeff'][n]), ('q8', ordinary_coeff[n, 8])]:
                error = np.einsum('pk,pklc->plc', B, coeff[ids])-source
                values = np.concatenate([error**2, (error[:, :, 0]+error[:, :, 1])[:, :, None]**2,
                                         dot(error, error)[:, :, None]], axis=2)
                for window, mask in masks.items():
                    out[n][kind][window] += np.einsum('plc,p->lc', values, w.ravel()*mask)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output/'comparison.json'
    if result_path.exists():
        raise SystemExit('Refusing to overwrite completed evidence; choose --output.')
    started = time.perf_counter()
    f = Frozen()
    # Retain a compact complete frozen fixture, without copying old trajectories.
    np.savez_compressed(args.output/'frozen-input.npz', phase_coeff=f.phi, surface=f.state.surface,
                        prescribed_V=f.V, G=f.G, dt=f.dt, m=f.state.m, profile_n=f.state.n)
    rng = np.random.default_rng(2804)
    x = rng.uniform(-2, 2, (12000, 2))
    f.state.surface[:, 6] = f.V
    src = f.fields(x)[0][:, 0]
    k = -10*np.expm1(-f.G*f.dt/10)
    old = 2*k*np.array([.03, -.03, .01])-k*f.state.memory(x, f.state.at(x)[:, 0], 6)
    bridge = float(np.max(abs(src-old)))
    assert bridge < 2e-13, bridge
    print(f'Frozen source bridge {bridge:.3e}', flush=True)
    timings = {}
    tri = triangles(f, GRIDS[-1])
    timings['partition_s'] = time.perf_counter()-started
    ref = {}
    for q in (6, 10):
        tick = time.perf_counter()
        ref[q] = reference(f, tri, q)
        timings[f'cut_reference_q{q}_s'] = time.perf_counter()-tick
        print(f'Cut reference q{q}: {timings[f"cut_reference_q{q}_s"]:.2f}s', flush=True)
    hi, lo = ref[10], ref[6]
    convergence = {}
    for n in GRIDS:
        diff = norm_coeff(hi['coeff'][n]-lo['coeff'][n], n)
        workdiff = abs(work_traction(f, hi, n, hi['coeff'][n])-work_traction(f, lo, n, lo['coeff'][n])).max()
        convergence[n] = dict(coefficient_L2=diff.tolist(), work_traction_Linf=float(workdiff))
        assert np.all(diff < 1e-9*np.sqrt(hi['norm2'])+1e-12), (n, diff)
        assert workdiff < 1e-9, (n, workdiff)
    assert abs(np.linalg.solve(hi['mass'], hi['rhs'])-np.linalg.solve(lo['mass'], lo['rhs'])).max() < 1e-9
    ordinary_coeff = {}
    for n in GRIDS:
        for q in ORDERS:
            tick = time.perf_counter()
            ordinary_coeff[n, q] = ordinary(f, n, q)
            timings[f'uncut_n{n}_q{q}_s'] = time.perf_counter()-tick
        print(f'Uncut projections {n}x{n} complete', flush=True)
    tick = time.perf_counter()
    window = windows(f, tri, hi, ordinary_coeff)
    timings['residual_integration_s'] = time.perf_counter()-tick
    local_work = np.linalg.solve(hi['mass'], hi['rhs'])
    local_power = f.V@hi['rhs']
    cases = []
    for n in GRIDS:
        best = window[n]['reference']['all'][:, -1]
        for q, coeff in [('cut', hi['coeff'][n]), *[(q, ordinary_coeff[n, q]) for q in ORDERS]]:
            alias = norm_coeff(coeff-hi['coeff'][n], n)**2
            total = best+alias
            work = work_traction(f, hi, n, coeff)
            delta = work-local_work
            work_error2 = np.einsum('il,ij,jl->l', delta, hi['mass'], delta)
            work_norm2 = np.einsum('il,ij,jl->l', local_work, hi['mass'], local_work)
            work_power = f.V@hi['mass']@work
            energy_loss = (hi['norm2']-norm_coeff(coeff, n)**2)/(4*f.G)
            row = dict(n=n, q=q, h=4/n, best_L2=np.sqrt(best).tolist(), quadrature_L2=np.sqrt(alias).tolist(),
                       total_L2=np.sqrt(total).tolist(), relative_L2=np.sqrt(total/hi['norm2']).tolist(),
                       localized_source_relative_L2=np.sqrt(total/hi['slip_norm2']).tolist(),
                       energy_local_minus_projected=energy_loss.tolist(),
                       work_traction_relative_L2=np.sqrt(work_error2/work_norm2).tolist(),
                       work_traction_Linf=np.max(abs(delta), axis=0).tolist(),
                       slip_power_error=(work_power-local_power).tolist(),
                       slip_power_relative_error=((work_power-local_power)/local_power).tolist(),
                       point_traction=point_traction(f, n, coeff))
            if q in ['cut', 8]:
                kind = 'reference' if q == 'cut' else 'q8'
                row['windows_squared_errors'] = {name: value.tolist() for name, value in window[n][kind].items()}
                identity = abs(window[n][kind]['all'][:, -1]-total)
                row['orthogonal_decomposition_absolute_error'] = identity.tolist()
                assert np.all(identity < 1e-10*total+1e-16), (n, q, identity)
            cases.append(row)
    timings['total_s'] = time.perf_counter()-started
    result = dict(frozen_case='true-c-p-ref', profile_grid=f.state.n, laws=['constant', 'layered'],
                  scope='Offline common frozen source; no evolution or production qualification.',
                  grids=GRIDS, orders=ORDERS, reference_triangle_count=len(tri),
                  source_bridge_Linf=bridge, reference_convergence=convergence,
                  reference=dict(source_L2=np.sqrt(hi['norm2']).tolist(),
                                 localized_source_L2=np.sqrt(hi['slip_norm2']).tolist(),
                                 area=hi['area'], cell_area_error=hi['cell_area_error'],
                                 normalization_measure=float(hi['mass'].sum()), length=f.length,
                                 normalization_relative_error=float(hi['mass'].sum()/f.length-1),
                                 local_slip_power=local_power.tolist(), local_work_traction=local_work.tolist()),
                  cases=cases, timings=timings,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    result_path.write_text(json.dumps(result, indent=2)+'\n')
    np.savez_compressed(args.output/'projection-coefficients.npz', **{
        f'n{n}_{tag}': c for n in GRIDS for tag, c in [('cut', hi['coeff'][n]),
                                                    *[(f'q{q}', ordinary_coeff[n, q]) for q in ORDERS]]})
    print(json.dumps(dict(cases=len(cases), timings=timings)), flush=True)


if __name__ == '__main__':
    main()
