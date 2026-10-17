#!/usr/bin/env python3
"""Independent physical-point and quadrature checks of fixed-reference transfers."""
import argparse
import csv
import json
from pathlib import Path
import re
import numpy as np


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def norm2(t):
    return t[:, 0]**2+t[:, 1]**2+(t[:, 0]+t[:, 1])**2+2*t[:, 2]**2


def lagrange(x):
    return np.stack([2*x*x-3*x+1, 4*x-4*x*x, 2*x*x-x], axis=-1)


class State:
    def __init__(self, folder, label):
        self.folder, self.label = folder, label
        g = rows(folder/(label+'-cells.csv'))
        self.ids = [r['CellId'] for r in g]
        self.box = np.array([[float(r[k]) for k in ['x0', 'y0', 'x1', 'y1']] for r in g])
        self.xedges = np.unique(self.box[:, [0, 2]])
        self.yedges = np.unique(self.box[:, [1, 3]])
        self.grid = np.zeros((len(self.xedges)-1, len(self.yedges)-1), dtype=int)
        for i, b in enumerate(self.box):
            self.grid[np.searchsorted(self.xedges, b[0]), np.searchsorted(self.yedges, b[1])] = i
        coeff = {r['CellId']: np.array([float(r['c'+str(i)]) for i in range(27)]).reshape(3, 9)
                 for r in rows(folder/(label+'-tensor.csv'))}
        self.tensor = np.array([coeff[k] for k in self.ids])
        phase = {(float(r['x']), float(r['y'])): [float(r[k]) for k in ['phase', 'lower', 'old', 'older']]
                 for r in rows(folder/(label+'-phase.csv'))}
        self.phase = np.array([[phase[(b[2] if v & 1 else b[0], b[3] if v & 2 else b[1])]
                               for v in range(4)] for b in self.box])
        nodes = rows(folder/'dg-support.csv')
        self.nodes = np.array([[round(2*float(r['r'])), round(2*float(r['s']))] for r in nodes])
        surf = rows(folder/(label+'-surface.csv'))
        self.Z = np.array([float(r['Z']) for r in surf])
        self.I = np.array([float(r['I']) for r in surf])
        # Recorded actual native constants; no continuum/constitutive replacement.
        log = (folder.parent/'run.log').read_text()
        self.G, self.m = map(float, re.search(r'Accepted FE strain: G=([\d.e+-]+).*? m=([\d.e+-]+)', log).groups())
        self.A, self.ell = 64.00000000008, .15625

    def evaluate(self, p):
        ix = np.clip(np.searchsorted(self.xedges, p[:, 0], side='right')-1, 0, len(self.xedges)-2)
        iy = np.clip(np.searchsorted(self.yedges, p[:, 1], side='right')-1, 0, len(self.yedges)-2)
        cell = self.grid[ix, iy]
        b = self.box[cell]
        h = b[:, 2:]-b[:, :2]
        r = (p-b[:, :2])/h
        lx, ly = lagrange(r[:, 0]), lagrange(r[:, 1])
        basis = lx[:, self.nodes[:, 0]]*ly[:, self.nodes[:, 1]]
        T = np.einsum('nci,ni->nc', self.tensor[cell], basis)
        phi = np.zeros((len(p), 4))
        grad = np.zeros((len(p), 2))
        for v in range(4):
            a = r[:, 0] if v & 1 else 1-r[:, 0]
            c = r[:, 1] if v & 2 else 1-r[:, 1]
            phi += (a*c)[:, None]*self.phase[cell, v]
            grad[:, 0] += (1 if v & 1 else -1)*c/h[:, 0]*self.phase[cell, v, 0]
            grad[:, 1] += (1 if v & 2 else -1)*a/h[:, 1]*self.phase[cell, v, 0]
        Z = np.interp(p[:, 0], np.linspace(0, 1, len(self.Z)), self.Z)
        I = np.interp(p[:, 0], np.linspace(0, 1, len(self.I)), self.I)
        chi = self.m*phi[:, 0]*(1+phi[:, 0])/(1-phi[:, 0])**2/I
        tau = T.copy()
        tau[:, 2] -= chi*Z
        return dict(T=T, tau=tau, phase=phi, gradient=grad, chi=chi)


def quadrature(states, q):
    x = np.unique(np.concatenate([s.xedges for s in states]+[np.linspace(0, 1, len(states[0].Z))]))
    y = np.unique(np.concatenate([s.yedges for s in states]))
    z, w = np.polynomial.legendre.leggauss(q)
    z, w = (z+1)/2, w/2
    xp = (x[:-1, None]+np.diff(x)[:, None]*z).ravel()
    yp = (y[:-1, None]+np.diff(y)[:, None]*z).ravel()
    wx = (np.diff(x)[:, None]*w).ravel()
    wy = (np.diff(y)[:, None]*w).ravel()
    xx, yy = np.meshgrid(xp, yp, indexing='ij')
    ww = np.outer(wx, wy)
    return np.column_stack([xx.ravel(), yy.ravel()]), ww.ravel()


def functional(s, p, w):
    a = s.evaluate(p)
    ns = len(s.Z)-1
    t = p[:, 0]*ns
    j = np.minimum(t.astype(int), ns-1)
    t -= j
    shear = np.zeros(ns+1)
    for k in range(ns+1):
        shear[k] = np.sum(w*a['chi']*a['tau'][:, 2]*np.where(j == k, 1-t, np.where(j+1 == k, t, 0)))
    z, v = np.polynomial.legendre.leggauss(32)
    x = np.concatenate([(i+(z+1)/2)/ns for i in range(ns)])
    v = np.tile(v/(2*ns), ns)
    Z = np.interp(x, np.linspace(0, 1, ns+1), s.Z)
    I = np.interp(x, np.linspace(0, 1, ns+1), s.I)
    coh = np.sum(v*Z**2/I)/(2*s.G)
    E = np.array([np.sum(w*norm2(a['tau']))/(4*s.G), coh,
                  s.A*np.sum(w*(a['phase'][:, 0]+s.ell**2*np.sum(a['gradient']**2, axis=1)))])
    return a, E, shear


def comparison(first, second, q=24):
    p, w = quadrature([first, second], q)
    a, ea, sa = functional(first, p, w)
    b, eb, sb = functional(second, p, w)
    d = a['tau']-b['tau']
    scale = np.sqrt(np.sum(w*norm2(a['tau'])))
    whole = np.sqrt(np.sum(w*norm2(d)))
    localized = np.sqrt(np.sum(w*a['chi']*norm2(d)))
    moments = []
    for i in range(3):
        for j in range(3):
            moments.append(np.sum((w*p[:, 0]**i*p[:, 1]**j)[:, None]*(a['T']-b['T']), axis=0))
    tensor_error = np.sqrt(np.sum(w*norm2(a['T']-b['T'])))
    tensor_scale = np.sqrt(np.sum(w*norm2(a['T'])))
    return dict(tensor_relative_L2=float(tensor_error/max(1., tensor_scale)), stress_L2_Pa=float(whole), stress_relative_L2=float(whole/max(1., scale)),
                localized_stress_L2=float(localized), phase_Linf=float(np.max(np.abs(a['phase']-b['phase']))),
                gradient_Linf=float(np.max(np.abs(a['gradient']-b['gradient']))),
                maximum_moment_defect=float(np.max(np.abs(moments))),
                energy_before=ea.tolist(), energy_after=eb.tolist(), energy_defect=(eb-ea).tolist(),
                quadratic_norm_change=float(np.sum(w*(norm2(b['T'])-norm2(a['T'])))),
                weak_shear_defect=(sb-sa).tolist())


def quadrature_separation(folder):
    result = {}
    minimum_rate = min(float(r['V']) for r in rows(folder/'state-0-surface.csv'))
    for label in ['global-complete-Q2-before', 'global-complete-Q2-refined',
                  'accepted-step-1-before', 'accepted-step-1-refined', 'state-2']:
        state = State(folder, label)
        p, w = quadrature([state], 24)
        _, energy, shear = functional(state, p, w)
        p32, w32 = quadrature([state], 32)
        _, e32, s32 = functional(state, p32, w32)
        native = rows(folder/(label+'-energy.csv'))[0]
        en = np.array([float(native[k]) for k in ['bulk', 'cohesive', 'fracture']])
        loads = rows(folder/(label+'-weak-loads.csv'))
        sn = np.array([float(r['shear']) for r in loads])
        cn = np.array([float(r['cohesive']) for r in loads])
        mass = np.ones(len(state.Z))/(len(state.Z)-1)
        mass[[0, -1]] *= .5
        rates = np.array([float(r['V']) for r in rows(folder/(label+'-surface.csv'))])
        rn, rr = sn-cn-1000*mass, shear-cn-1000*mass
        kn = np.where(rates > minimum_rate, rn, np.maximum(rn, 0))
        kr = np.where(rates > minimum_rate, rr, np.maximum(rr, 0))
        result[label] = dict(native_minus_q24_energy=(en-energy).tolist(),
                             native_minus_q24_shear=(sn-shear).tolist(),
                             energy_scaled_error=float(np.max(np.abs(en-energy)/np.maximum(1., np.abs(energy)))),
                             shear_scaled_error=float(np.max(np.abs(sn-shear))/max(1., np.max(np.abs(shear)))),
                             q24_q32_energy_uncertainty=float(np.max(np.abs(energy-e32))),
                             q24_q32_shear_uncertainty=float(np.max(np.abs(shear-s32))),
                             native_fault_KKT_raw=float(np.max(np.abs(kn))),
                             high_rule_fault_KKT_raw=float(np.max(np.abs(kr))))
    result['native_rule_vs_high_rule_screen_passed'] = all(
        r['energy_scaled_error'] <= 1e-9 and r['shear_scaled_error'] <= 1e-9
        for r in result.values())
    return result


def fracture_reference(state):
    actual = {(float(r['x']), float(r['y'])): r for r in rows(state.folder/(state.label+'-fracture-operator.csv'))}
    expected = {key: np.zeros(2) for key in actual}
    z, w = np.polynomial.legendre.leggauss(5)
    z, w = (z+1)/2, w/2
    xx, yy = np.meshgrid(z, z, indexing='ij')
    u = np.column_stack([xx.ravel(), yy.ravel()])
    weight = np.outer(w, w).ravel()
    for k, box in enumerate(state.box):
        h = box[2:]-box[:2]
        vertex = [(box[2] if v & 1 else box[0], box[3] if v & 2 else box[1]) for v in range(4)]
        d = np.array([np.exp(-17*((p[1]-.5)**2+.3*(p[0]-.4)**2)) for p in vertex])
        shape, grad = [], []
        for v in range(4):
            a, b = (u[:, 0] if v & 1 else 1-u[:, 0]), (u[:, 1] if v & 2 else 1-u[:, 1])
            shape.append(a*b)
            grad.append(np.column_stack([(1 if v & 1 else -1)*b/h[0], (1 if v & 2 else -1)*a/h[1]]))
        gradient = np.einsum('i,ind->nd', state.phase[k, :, 0], np.array(grad))
        dd = np.einsum('i,ind->nd', d, np.array(grad))
        for v in range(4):
            expected[vertex[v]] += state.A*np.prod(h)*np.array([
                np.sum(weight*(shape[v]+2*state.ell**2*np.sum(gradient*grad[v], axis=1))),
                np.sum(weight*2*state.ell**2*np.sum(dd*grad[v], axis=1))])
    error = max(np.max(np.abs(values-np.array([float(actual[key]['gradient']), float(actual[key]['Hd'])])))
                for key, values in expected.items())
    constant = max(abs(float(r['Hones'])) for r in actual.values())
    assert error < 1e-10 and constant < 1e-10
    return dict(independent_gradient_action_error=float(error), constant_action_error=constant)


def lossy_projection(folder):
    fine, coarse = State(folder, 'lossy-fine'), State(folder, 'lossy-coarse')
    z, w = np.polynomial.legendre.leggauss(5)
    z, w = (z+1)/2, w/2
    xx, yy = np.meshgrid(z, z, indexing='ij')
    unit = np.column_stack([xx.ravel(), yy.ravel()])
    weight = np.outer(w, w).ravel()
    def basis(p):
        return lagrange(p[:, 0])[:, coarse.nodes[:, 0]]*lagrange(p[:, 1])[:, coarse.nodes[:, 1]]
    B = basis(unit)
    mass = B.T @ (weight[:, None]*B)
    maximum_error, maximum_mean = 0., 0.
    for k, box in enumerate(coarse.box):
        rhs = np.zeros((9, 3))
        mean = np.zeros(3)
        area = np.prod(box[2:]-box[:2])
        for child in fine.box:
            center = .5*(child[:2]+child[2:])
            if not np.all((center > box[:2]) & (center < box[2:])):
                continue
            p = child[:2]+unit*(child[2:]-child[:2])
            values = fine.evaluate(p)['T']
            target = basis((p-box[:2])/(box[2:]-box[:2]))
            ratio = np.prod(child[2:]-child[:2])/area
            rhs += target.T @ ((weight*ratio)[:, None]*values)
            mean += np.sum((weight*ratio)[:, None]*values, axis=0)
        expected = np.linalg.solve(mass, rhs).T
        maximum_error = max(maximum_error, np.max(np.abs(expected-coarse.tensor[k])))
        maximum_mean = max(maximum_mean, np.max(np.abs(mean-np.sum(weight[:, None]*(B @ coarse.tensor[k].T), axis=0))))
    c = comparison(fine, coarse)
    c.update(independent_projection_coefficient_error=float(maximum_error), target_cell_mean_defect=float(maximum_mean))
    assert maximum_error < 1e-11 and maximum_mean < 1e-11
    assert c['stress_L2_Pa'] > 0 and c['quadratic_norm_change'] < 0
    assert c['phase_Linf'] < 1e-13
    c['orthogonality_defect'] = abs(c['stress_L2_Pa']**2+c['quadratic_norm_change'])
    assert c['orthogonality_defect'] < 1e-11
    return c


def compare_runs(first, second):
    result = {}
    labels = [x for x in ['state-0', 'state-1', 'state-2', 'before-step-2']
              if (first/(x+'-cells.csv')).exists() and (second/(x+'-cells.csv')).exists()]
    for label in labels:
        c = comparison(State(first, label), State(second, label))
        assert c['stress_relative_L2'] < 1e-9 and c['phase_Linf'] < 1e-9
        for suffix, keys in [('tensor', ['CellId']), ('surface', ['i']), ('native-fields', ['CellId', 'node', 'component']),
                             ('particles', ['id']), ('weak-loads', ['i']), ('fracture-operator', ['x', 'y'])]:
            left = {tuple(r[k] for k in keys): r for r in rows(first/(label+'-'+suffix+'.csv'))}
            right = {tuple(r[k] for k in keys): r for r in rows(second/(label+'-'+suffix+'.csv'))}
            assert left.keys() == right.keys()
            error, scale, mismatched_cells = 0., 1., 0
            for key in left:
                for field in left[key]:
                    if field in keys or field == 'rank':
                        continue
                    if field == 'CellId':
                        mismatched_cells += left[key][field] != right[key][field]
                        continue
                    a, b = float(left[key][field]), float(right[key][field])
                    if np.isnan(a) and np.isnan(b):
                        continue
                    assert np.isfinite(a) and np.isfinite(b)
                    error = max(error, abs(a-b))
                    scale = max(scale, abs(a))
            c[suffix+'_max_error'] = error
            c[suffix+'_relative_max_error'] = error/scale
            if suffix == 'particles':
                c['particle_CellId_mismatches'] = mismatched_cells
                assert mismatched_cells == 0
            assert error/scale < 1e-9
        r1 = sorted(first.glob(label+'-rng-rank-*.txt'))
        r2 = sorted(second.glob(label+'-rng-rank-*.txt'))
        c['rng_rank_counts'] = [len(r1), len(r2)]
        c['rng_equal_per_rank'] = len(r1) == len(r2) and all(a.read_bytes() == b.read_bytes() for a, b in zip(r1, r2))
        if len(r1) == len(r2):
            assert c['rng_equal_per_rank']
        result[label] = c
    assert labels
    return result


def assess(folder):
    g = rows(folder/'transfer-gates.csv')
    assert g and all(r['passed'] == '1' for r in g)
    checks = {}
    for label in ['global-complete-Q2', 'accepted-step-1']:
        a = State(folder, label+'-before')
        checks[label+'-fracture-reference'] = fracture_reference(a)
        for suffix in ['refined', 'coarsened']:
            b = State(folder, label+'-'+suffix)
            c = comparison(a, b)
            assert c['stress_relative_L2'] < 1e-11
            assert c['phase_Linf'] < 1e-13
            checks[label+'-'+suffix] = c
        c32 = comparison(a, State(folder, label+'-refined'), 32)
        checks[label+'-q24-q32'] = max(abs(np.array(c32['energy_before'])-checks[label+'-refined']['energy_before']))
    checks['lossy_projection'] = lossy_projection(folder)
    initial = {r['id']: (r['x'], r['y']) for r in rows(folder/'initial-particles.csv')}
    particle_checks = {}
    for f in folder.glob('*-particles.csv'):
        current = {r['id']: (r['x'], r['y']) for r in rows(f)}
        assert current == initial
        particle_checks[f.stem] = dict(count=len(current), fixed=True,
                                      on_face=sum(any(float(r[k]) in (0, 1) for k in ['r', 's']) for r in rows(f)))
    return dict(gates=len(g), checks=checks, particles=particle_checks, quadrature=quadrature_separation(folder))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--compare', type=Path)
    a = p.parse_args()
    result = compare_runs(a.run/'run', a.compare/'run') if a.compare else assess(a.run/'run')
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(comparisons=len(result)) if a.compare else dict(gates=result['gates'], transfer_checks=len(result['checks']))))


if __name__ == '__main__':
    main()
