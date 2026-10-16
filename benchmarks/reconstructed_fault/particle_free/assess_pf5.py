#!/usr/bin/env python3
"""Independent PF5 semidiscrete, composition, physical-load and restart checks.

The operator is assembled afresh from analytic velocity and Q2 polynomials with
Gauss5, not read from C++ for the reference. The exact affine ODE is evaluated
by a matrix exponential. Only mesh/index metadata and accepted outputs are read.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from assess_methods import Data, gauss, lagrange, derivative, norm2, velocity


def flow(x, y):
    return np.stack([.4*x*(1-x)*(1-2*y), -.4*(1-2*x)*y*(1-y)], axis=-1)


def strain(x, y):
    a = .4*(1-2*x)*(1-2*y)
    return np.stack([a, -a, .4*(y*(1-y)-x*(1-x))], axis=-1)


def operator(d):
    n = int(d.indices.max())+1
    matrix = np.zeros((n, n))
    z, w = gauss(5)
    r, s = np.repeat(z, 5), np.tile(z, 5)
    N = d.shape(r, s)
    lookup = {(round(c['x0']*d.nx), round(c['y0']*d.ny)): k for k, c in enumerate(d.cells)}
    for k, c in enumerate(d.cells):
        x, y = c['x0']+c['dx']*r, c['y0']+c['dy']*s
        v = flow(x, y)
        Nx = derivative(r)[:, d.ri]*lagrange(s)[:, d.si]/c['dx']
        Ny = lagrange(r)[:, d.ri]*derivative(s)[:, d.si]/c['dy']
        W = c['dx']*c['dy']*np.repeat(w, 5)*np.tile(w, 5)
        blocks = {k: (W[:, None]*(Nx*v[:, :1]+Ny*v[:, 1:])).T@N}
        ix, iy = round(c['x0']*d.nx), round(c['y0']*d.ny)
        for axis, side, neighbor in [(0, 0, (ix-1, iy)), (0, 1, (ix+1, iy)),
                                     (1, 0, (ix, iy-1)), (1, 1, (ix, iy+1))]:
            rr, ss = (np.full(5, side), z) if axis == 0 else (z, np.full(5, side))
            xx, yy = c['x0']+c['dx']*rr, c['y0']+c['dy']*ss
            vn = (2*side-1)*flow(xx, yy)[:, axis]
            tests = d.shape(rr, ss)
            weights = w*(c['dy'] if axis == 0 else c['dx'])
            for incoming in [False, True]:
                mask = vn < 0 if incoming else vn >= 0
                if not np.any(mask & (vn != 0)):
                    continue
                target = lookup[neighbor] if incoming else k
                nc = d.cells[target]
                trial = d.shape((xx-nc['x0'])/nc['dx'], (yy-nc['y0'])/nc['dy'])
                block = tests.T@((-weights*vn*mask)[:, None]*trial)
                blocks[target] = blocks.get(target, 0)+block
        for target, block in blocks.items():
            matrix[np.ix_(d.indices[k], d.indices[target])] += np.linalg.solve(d.mass(), block)
    return matrix


def preload(d, G, eta, start):
    result = np.zeros((len(d.cells), 9, 3))
    # Cell-local derivatives are essential at DG support points on cell faces.
    for k, c in enumerate(d.cells):
        grad = np.zeros((9, 2, 2))
        r, s = d.basis['r'], d.basis['s']
        l, m = lagrange(r), lagrange(s)
        dl, dm = derivative(r)/c['dx'], derivative(s)/c['dy']
        for i in range(3):
            for j in range(3):
                u = velocity(c['x0']+.5*i*c['dx'], c['y0']+.5*j*c['dy'])
                grad[:, :, 0] += u*dl[:, i, None]*m[:, j, None]
                grad[:, :, 1] += u*l[:, i, None]*dm[:, j, None]
        tr = (grad[:, 0, 0]+grad[:, 1, 1])/3
        eps = np.stack([grad[:, 0, 0]-tr, grad[:, 1, 1]-tr,
                        .5*(grad[:, 0, 1]+grad[:, 1, 0])], axis=-1)
        result[k] = -2*eta*np.expm1(-G/eta*start)*eps
    return result


def flatten(d, a):
    b = np.empty((int(d.indices.max())+1, 3))
    b[d.indices] = a
    return b


def exact(L, initial, forcing, rate, duration):
    n = len(L)
    augmented = np.zeros((n+3, n+3))
    augmented[:n, :n] = L-rate*np.eye(n)
    augmented[:n, n:] = forcing
    return (expm(duration*augmented)@np.vstack([initial, np.eye(3)]))[:n]


def reconstructed(d, values, points, surface):
    T = d.evaluate(values, points['x'], points['y'])
    seg = points['segment'].astype(int)
    Z = points['N0']*surface['Z'][seg]+points['N1']*surface['Z'][seg+1]
    stress = T.copy()
    stress[:, 2] -= points['chi']*Z  # 2 S_xy = 1 in the native horizontal fixture.
    w = points['JxW']*points['chi']
    loads = np.zeros((len(surface), 2))
    for side in [0, 1]:
        weights = w*points[f'N{side}']
        np.add.at(loads, seg+side, weights[:, None]*stress[:, [2, 1]]*[1, -1])
    return stress, loads


def assess(path):
    d = Data(path)
    c = np.genfromtxt(d.path/'constants.csv', delimiter=',', names=True)
    G, eta, rate, duration, start = (float(c[x]) for x in ['G', 'eta', 'relaxation_rate', 'time', 'start'])
    steps = int(c['steps'])
    L = operator(d)
    exported = np.loadtxt(d.path/'operator.csv', delimiter=',', skiprows=1)
    native_L = np.zeros_like(L)
    native_L[exported[:, 0].astype(int), exported[:, 1].astype(int)] = exported[:, 2]
    independent_initial = preload(d, G, eta, start)
    initial = d.load('field-initial')
    x = d.cells['x0'][:, None]+d.cells['dx'][:, None]*d.basis['r']
    y = d.cells['y0'][:, None]+d.cells['dy'][:, None]*d.basis['s']
    F = flatten(d, 2*G*strain(x, y))
    a0 = flatten(d, independent_initial)
    ref = exact(L, a0, F, rate, duration)[d.indices]
    values = d.load('field-final')
    dt = duration/steps
    a = a0.copy()
    beta = np.exp(-rate*dt/2)
    source_half = -np.expm1(-rate*dt/2)/rate*F
    for _ in range(steps):
        a = beta*a+source_half
        s1 = a+dt*(L@a)
        s2 = .75*a+.25*(s1+dt*(L@s1))
        a = (a+2*(s2+dt*(L@s2)))/3
        a = beta*a+source_half
    reference_norm = d.difference(ref, np.zeros_like(ref))
    surface = np.genfromtxt(d.path/'surface-final.csv', delimiter=',', names=True)
    points = np.genfromtxt(d.path/'stress-final.csv', delimiter=',', names=True)
    native_stress = np.stack([points[x] for x in ['xx', 'yy', 'xy']], axis=-1)
    physical, loads = reconstructed(d, values, points, surface)
    refphysical, refloads = reconstructed(d, ref, points, surface)
    native_loads = np.loadtxt(d.path/'loads-final.csv', delimiter=',', skiprows=1)[:, 1:]
    I = points['N0']*surface['I'][points['segment'].astype(int)]+points['N1']*surface['I'][points['segment'].astype(int)+1]
    independent_chi = float(c['m'])*points['phi']*(1+points['phi'])/(1-points['phi'])**2/I
    Zexact = -eta*np.expm1(-rate*(start+duration))*.001
    # Fit the fixed Q1 phase locally. This also quantifies the retained profile
    # derivative term, rather than silently asserting Q_geo=0 at nonzero Z.
    gradphi = np.zeros((len(points), 2))
    for k in range(len(d.cells)):
        mask = points['cell'].astype(int) == k
        xx, yy = points['x'][mask], points['y'][mask]
        p = np.linalg.lstsq(np.stack([xx*0+1, xx, yy, xx*yy], axis=-1), points['phi'][mask], rcond=None)[0]
        gradphi[mask] = np.stack([p[1]+p[3]*yy, p[2]+p[3]*xx], axis=-1)
    gradchi = float(c['m'])*(1+3*points['phi'])[:, None]/(1-points['phi'])[:, None]**3*gradphi/I[:, None]
    seg = points['segment'].astype(int)
    gradchi[:, 0] -= points['chi']*(surface['I'][seg+1]-surface['I'][seg])*16/I
    derivative_chi = np.sum(flow(points['x'], points['y'])*gradchi, axis=-1)
    geo_norm = np.sqrt(np.sum(points['JxW']*2*(Zexact*derivative_chi)**2))
    singular = np.linalg.svd(a0, compute_uv=False)
    record = json.loads((Path(path)/'record.json').read_text())
    result = dict(label=Path(path).name, steps=steps, dt=dt,
                  independent_operator_max_error=float(np.max(abs(L-native_L))),
                  preload_relative_L2=d.difference(initial, independent_initial)/d.difference(independent_initial, 0*initial),
                  composition_relative_L2=d.difference(values, a[d.indices])/reference_norm,
                  trial_stress_absolute_L2=d.difference(values, ref),
                  trial_stress_relative_L2=d.difference(values, ref)/reference_norm,
                  physical_stress_relative_L2=float(np.sqrt(np.sum(points['JxW']*norm2(physical-refphysical))/np.sum(points['JxW']*norm2(refphysical)))),
                  physical_reconstruction_max_error=float(np.max(abs(physical-native_stress))),
                  localization_max_relative_error=float(np.max(abs(independent_chi-points['chi']))/np.max(abs(independent_chi))),
                  weak_load_reconstruction_max_error=float(np.max(abs(loads-native_loads))),
                  weak_load_relative_l2=float(np.linalg.norm(loads-refloads)/np.linalg.norm(refloads)),
                  Z_exact=Zexact, Z_max_error=float(np.max(abs(surface['Z']-Zexact))),
                  elastic_slip_max_error=float(np.max(abs(surface['elastic_slip']-Zexact/G))),
                  cohesive_max_error=float(np.max(abs(surface['cohesive']-Zexact/surface['I']))),
                  initial_tensor_singular_values=singular.tolist(), profile_geometric_term_L2=geo_norm,
                  DoFs_per_tensor_component=len(L), seconds=record['seconds'], max_rss_KiB=record['max_rss_KiB'])
    result['counting_negative_controls'] = {}
    for name, ll, rr, ff in [('double_transport', 2*L, rate, F), ('double_relaxation', L, 2*rate, F),
                             ('double_forcing', L, rate, 2*F), ('omit_forcing', L, rate, 0*F)]:
        wrong = exact(ll, a0, ff, rr, duration)[d.indices]
        result['counting_negative_controls'][name] = d.difference(values, wrong)/reference_norm
    gates = np.genfromtxt(d.path/'gates.csv', delimiter=',', names=True, dtype=None, encoding='utf-8')
    result['gates'] = len(gates)
    assert np.all(gates['passed'] == 1)
    assert result['independent_operator_max_error'] < 2e-11
    assert result['preload_relative_L2'] < 1e-11
    assert result['composition_relative_L2'] < 1e-11
    assert result['physical_reconstruction_max_error'] < 1e-7
    assert result['weak_load_reconstruction_max_error'] < 1e-7
    assert result['Z_max_error'] < 1e-9 and result['profile_geometric_term_L2'] > 0
    assert singular[-1] > .001*singular[0]
    assert min(result['counting_negative_controls'].values()) > 100*result['trial_stress_relative_L2']
    return result


def restart_compare(fresh, restarted):
    a, b = Path(fresh)/'run', Path(restarted)/'run'
    result = {}
    for src, dst in [('mid', 'restored'), ('final', 'final')]:
        checks = {}
        for stem in ['field', 'surface', 'properties', 'shared', 'particles', 'clock', 'stress', 'loads']:
            x, y = (np.loadtxt(p/f'{stem}-{label}.csv', delimiter=',', skiprows=1) for p, label in [(a, src), (b, dst)])
            assert x.shape == y.shape
            assert not np.any(np.isinf(x)) and not np.any(np.isinf(y))
            finite = np.isfinite(x) & np.isfinite(y)
            assert np.array_equal(np.isnan(x), np.isnan(y))
            checks[stem] = float(np.max(abs(x[finite]-y[finite])))
        for stem in ['persistent', 'rng', 'manager']:
            checks[stem+'_bitwise'] = (a/f'{stem}-{src}.bin').read_bytes() == (b/f'{stem}-{dst}.bin').read_bytes()
        result[src+'_'+dst] = checks
        assert all(v is True if isinstance(v, bool) else v == 0 for v in checks.values()), checks
        raw_a = (a/f'manager-raw-{src}.bin').read_bytes()
        raw_b = (b/f'manager-raw-{dst}.bin').read_bytes()
        result[src+'_'+dst+'_raw_manager'] = dict(bitwise=raw_a == raw_b, size_difference=len(raw_b)-len(raw_a),
                                                   reason='Native load makes omitted default shear sense explicitly +1; canonical archives above include every persistent manager field.')
        assert len(raw_b)-len(raw_a) == 13
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='+', type=Path)
    p.add_argument('--restart', type=Path)
    p.add_argument('--restart-parent', type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    rows = sorted([assess(path) for path in args.runs], key=lambda row: row['steps'])
    for prev, row in zip(rows, rows[1:]):
        row['orders'] = {key: float(np.log(prev[key]/row[key])/np.log(prev['dt']/row['dt']))
                         for key in ['trial_stress_relative_L2', 'physical_stress_relative_L2', 'weak_load_relative_l2']}
        assert all(1.95 < value < 2.05 for value in row['orders'].values())
    result = dict(scope='fixed-mesh fixed-geometry closed-boundary prescribed PF5', runs=rows)
    if args.restart:
        result['restart'] = restart_compare(args.restart_parent, args.restart)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
