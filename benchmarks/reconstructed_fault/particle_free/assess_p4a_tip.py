#!/usr/bin/env python3
"""Independent finite-strip energy, normalization adjoint and phase gradients.

Integrates the exported physical Q1 phase and all nine native DGQ2 modes.
Every bulk-cell edge, surface knot and physical strip endpoint is a panel edge.
The represented compact support is resolved by the phase cell partition.
No native energy/normalization routine is called by this calculation.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from assess_p3_extension import csv, gauss, lagrange, inner, seed_table

G, ELL, GC = 1e6, .15625, 26.6666666667
M = 128.00000000016001
A = GC / ((8/3)*ELL)
NX, NY = 32, 64
SEED_R, SEED_PHI = seed_table()


def h(p):
    return M*p*(1+p)/(1-p)**2


def hp(p):
    return M*(1+3*p)/(1-p)**3


def nodal_data(path):
    nodes = csv(path/'phase-gradient.csv')
    nodes.sort(order=['y', 'x'])
    assert len(nodes) == (NX+1)*(NY+1)
    return nodes


def normalization(s, phi, order):
    """Consistent L2 projection of independent normal Gauss integrals and DI."""
    z, w = gauss(order)
    mass = np.zeros((len(s), len(s)))
    rhs = np.zeros(len(s))
    drhs = np.zeros((len(s), phi.size))
    v = np.tile(z, NY)
    yw = np.tile(w/NY, NY)
    row = np.repeat(np.arange(NY), order)
    for seg, ds in enumerate(np.diff(s['x'])):
        mass[seg:seg+2, seg:seg+2] += ds/6*np.array([[2., 1.], [1., 2.]])
        for u, wx in zip(z, w*ds):
            x = s['x'][seg]+u*ds
            i = min(int(x*NX), NX-1)
            t = x*NX-i
            ids = np.array([row*(NX+1)+i, row*(NX+1)+i+1,
                            (row+1)*(NX+1)+i, (row+1)*(NX+1)+i+1]).T
            shape = np.array([(1-t)*(1-v), t*(1-v), (1-t)*v, t*v]).T
            p = np.sum(phi[ids]*shape, axis=1)
            integral = np.sum(yw*h(p))
            for local, n in enumerate([1-u, u]):
                rhs[seg+local] += wx*n*integral
                np.add.at(drhs[seg+local], ids.ravel(),
                          (wx*n*yw[:, None]*hp(p)[:, None]*shape).ravel())
    return np.linalg.solve(mass, rhs), np.linalg.solve(mass, drhs)


def integrate(path, order):
    s = csv(path/'old-surface.csv')
    cells = csv(path/'old-cells-0.csv')
    nodes = nodal_data(path)
    phi = nodes['phi']
    I, DI = normalization(s, phi, order)
    basis = csv(path/'basis-0.csv')
    iu, iv = np.rint(2*basis['u']).astype(int), np.rint(2*basis['v']).astype(int)
    z, w = gauss(order)
    bulk = 0.
    fracture = 0.
    representation_error, continuum_norm = 0., 0.
    gradient = np.zeros(len(nodes))
    gf = np.zeros(len(nodes))
    forceI = np.zeros(len(s))
    shear = np.zeros(len(s))
    cohesive_load = np.zeros(len(s))
    friction_load = np.zeros(len(s))
    for cell in cells:
        x0, y0, dx, dy = (cell[k] for k in ['x0', 'y0', 'hx', 'hy'])
        ix, iy = round(x0*NX), round(y0*NY)
        ids = np.array([iy*(NX+1)+ix, iy*(NX+1)+ix+1,
                        (iy+1)*(NX+1)+ix, (iy+1)*(NX+1)+ix+1])
        cuts = np.unique(np.r_[x0, s['x'][(s['x'] > x0) & (s['x'] < x0+dx)], x0+dx])
        coeff = np.array([[cell[f'c{c}_{i}'] for c in range(3)] for i in range(9)])
        for left, right in zip(cuts[:-1], cuts[1:]):
            x = np.repeat(left+(right-left)*z, order)
            u = (x-x0)/dx
            v = np.tile(z, order)
            J = np.repeat((right-left)*w, order)*np.tile(dy*w, order)
            N = np.array([(1-u)*(1-v), u*(1-v), (1-u)*v, u*v]).T
            Nx = np.array([-(1-v), 1-v, -v, v]).T/dx
            Ny = np.array([-(1-u), -u, 1-u, u]).T/dy
            p = N@phi[ids]
            radial = np.hypot(np.maximum(x-s['x'][-1], 0.), y0+dy*v-.5)
            prescribed = np.interp(radial, SEED_R, SEED_PHI, right=0.)
            representation_error += np.sum(J*(p-prescribed)**2)
            continuum_norm += np.sum(J*prescribed**2)
            px, py = Nx@phi[ids], Ny@phi[ids]
            fracture += np.sum(J*A*(p+ELL**2*(px**2+py**2)))
            gf[ids] += (J*A)@(N+2*ELL**2*(px[:, None]*Nx+py[:, None]*Ny))
            shape = lagrange(u)[:, iu]*lagrange(v)[:, iv]
            tau = shape@coeff
            active = left >= s['x'][0] and right <= s['x'][-1]
            if active:
                seg = np.clip(np.searchsorted(s['x'], x, side='right')-1, 0, len(s)-2)
                t = (x-s['x'][seg])/(s['x'][seg+1]-s['x'][seg])
                qI = (1-t)*I[seg]+t*I[seg+1]
                qZ = (1-t)*s['Z'][seg]+t*s['Z'][seg+1]
                chi = h(p)/qI
                tau[:, 2] -= qZ*chi
                c = -qZ/G*tau[:, 2]
                gradient[ids] += (J*c*hp(p)/qI)@N
                for index, n in [(seg, 1-t), (seg+1, t)]:
                    np.add.at(forceI, index, -J*n*c*h(p)/qI**2)
                    np.add.at(shear, index, J*n*chi*tau[:, 2])
            bulk += np.sum(J*inner(tau))/(4*G)
    cohesive = 0.
    for seg, ds in enumerate(np.diff(s['x'])):
        qZ = (1-z)*s['Z'][seg]+z*s['Z'][seg+1]
        qI = (1-z)*I[seg]+z*I[seg+1]
        J = ds*w
        cohesive += np.sum(J*qZ**2/qI)/(2*G)
        for i, n in enumerate([1-z, z]):
            forceI[seg+i] -= np.sum(J*n*qZ**2/qI**2)/(2*G)
            cohesive_load[seg+i] += np.sum(J*n*qZ/qI)
            friction_load[seg+i] += np.sum(J*n*1000.)
    gradient += DI.T@forceI
    return dict(bulk=bulk, cohesive=cohesive, fracture=fracture, I=I,
                gradient=gradient, fracture_gradient=gf, shear=shear,
                cohesive_load=cohesive_load, friction_load=friction_load,
                initial_phase_relative_L2=np.sqrt(representation_error/continuum_norm))


def relative(a, b):
    return float(np.max(np.abs(a-b))/max(1., np.max(np.abs(b))))


def assess(path):
    refs = [integrate(path, order) for order in [24, 32]]
    ref = refs[-1]
    nodes = nodal_data(path)
    ledger = csv(path/'old-ledger.csv')[0]
    s = csv(path/'old-surface.csv')
    weak = csv(path/'old-native-loads.csv')
    fd = csv(path/'directional-checks.csv')
    dirs = csv(path/'phase-directions.csv')
    q = csv(path/'bulk-quadrature.csv')
    tip = s['x'][-1]
    errors = {k: relative(ledger[k], ref[k]) for k in ['bulk', 'cohesive', 'fracture']}
    errors['I'] = relative(s['I'], ref['I'])
    errors['full_mechanical_gradient'] = relative(nodes['mechanical'], ref['gradient'])
    errors['full_fracture_gradient'] = relative(nodes['fracture'], ref['fracture_gradient'])
    for native, key in [('shear', 'shear'), ('cohesive', 'cohesive_load'), ('friction', 'friction_load')]:
        errors['weak_'+native] = relative(weak[native], ref[key])
    # The continuum seed table has its own fine interpolation breaks; its
    # representation estimate is reported separately from discrete integration.
    qerror = {k: relative(refs[0][k], ref[k]) for k in ref if k != 'initial_phase_relative_L2'}
    direction_results = []
    for k in range(3):
        d = np.zeros(len(nodes))
        for row in dirs[dirs['direction'] == k]:
            i = np.flatnonzero(nodes['dof'] == row['dof'])[0]
            d[i] = row['value']
        part = fd[fd['direction'] == k]
        gm, gf = float(d@ref['gradient']), float(d@ref['fracture_gradient'])
        err = max(relative(part['mechanical_derivative'], gm), relative(part['fracture_derivative'], gf))
        fd_error = max(relative(part['mechanical_FD'], gm), relative(part['fracture_FD'], gf))
        direction_results.append(dict(name=['behind', 'straddling', 'strictly_ahead'][k],
                                      mechanical=gm, fracture=gf,
                                      native_error=err, finite_difference_error=fd_error))
        assert err <= 1e-8 and fd_error <= 1e-7
    ahead = nodes['x'] > tip+1/NX
    q_ahead = q['x'] > tip
    negative = nodes['fracture'] < -1e-12
    straddling = (nodes['x'] > tip) & (nodes['x'] < tip+1/NX)
    result = dict(tip=float(tip), scope='frozen loaded native snapshot, not equilibrated; no time/phase advancement',
                  native_energy={k: float(ledger[k]) for k in ['bulk', 'cohesive', 'fracture']},
                  native_weak_integrals={k: float(np.sum(weak[k])) for k in ['shear', 'cohesive', 'friction', 'residual']},
                  native_vs_independent32=errors, independent24_vs32=qerror,
                  directional_checks=direction_results,
                  ahead_local_mechanical_derivative_max=float(np.max(np.abs(q['local_mechanical_derivative'][q_ahead]))),
                  strictly_ahead_nodal_mechanical_gradient_max=float(np.max(np.abs(nodes['mechanical'][ahead]))),
                  ahead_nodes_with_straddling_support_max=float(np.max(np.abs(nodes['mechanical'][straddling]))) if np.any(straddling) else 0.,
                  tip_window_stress_Pa=dict(auxiliary_Txy_min=float(np.min(q['Txy'])),
                                            auxiliary_Txy_max=float(np.max(q['Txy'])),
                                            physical_tauxy_min=float(np.min(q['tauxy'])),
                                            physical_tauxy_max=float(np.max(q['tauxy']))),
                  unloaded_obstacle_KKT=dict(min_fracture_gradient=float(np.min(nodes['fracture'])),
                                             negative_nodes=int(np.sum(negative)),
                                             negative_nodes_ahead=int(np.sum(negative & (nodes['x'] > tip))),
                                             is_equilibrated=bool(not np.any(negative))),
                  quadrature=dict(zip(csv(path/'quadrature-measures.csv').dtype.names,
                                      [float(v) for v in csv(path/'quadrature-measures.csv')[0]])),
                  initial_phase_equals_obstacle=bool(np.array_equal(csv(path/'old-phase.csv')['phi'], csv(path/'old-phase.csv')['obstacle'])),
                  phase_range=[float(np.min(nodes['phi'])), float(np.max(nodes['phi']))],
                  initial_phase_relative_L2=float(ref['initial_phase_relative_L2']),
                  representation_estimate_24_vs32_absolute_difference=float(abs(refs[0]['initial_phase_relative_L2']-ref['initial_phase_relative_L2'])),
                  minimum_I=float(np.min(s['I'])),
                  physical_publications=int(ledger['physical_publications']),
                  strict_reference_match_passed=False, finite_tip_evolution_enabled=False)
    assert max(errors.values()) <= 1e-10
    assert max(qerror.values()) <= 1e-10
    assert result['ahead_local_mechanical_derivative_max'] == 0.
    assert result['strictly_ahead_nodal_mechanical_gradient_max'] == 0.
    assert direction_results[2]['mechanical'] == 0.
    assert result['initial_phase_equals_obstacle']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = assess(args.path)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
