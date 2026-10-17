#!/usr/bin/env python3
"""Independent checks of the homogeneous mesh-data configuration and deep remap."""
import argparse
import json
from pathlib import Path
import numpy as np
from assess_pf6_transfer import assess, compare_runs, comparison, quadrature, rows
from assess_pf6_local import LocalState, quadrature_control


def initialization(run):
    s = LocalState(run/'run', 'prescribed-initial')
    table = np.loadtxt(Path(__file__).with_name('pf6a-initial-phase.txt'))
    error = 0.
    for k, box in enumerate(s.box):
        for v in range(4):
            y = box[3] if v & 2 else box[1]
            error = max(error, abs(s.phase[k, v, 0]-np.interp(y, table[:, 0], table[:, 1])))
    assert error < 1e-14 and np.max(abs(s.tensor)) == 0 and np.max(abs(s.Z)) == 0
    # Integrate the actual piecewise-linear supplied profile on every table
    # interval, rather than replacing it by an unrelated continuum seed.
    z, w = np.polynomial.legendre.leggauss(8)
    y = (table[:-1, 0, None]+np.diff(table[:, 0])[:, None]*(z+1)/2).ravel()
    weight = (np.diff(table[:, 0])[:, None]*w/2).ravel()
    prescribed = np.interp(y, table[:, 0], table[:, 1])
    physical = s.evaluate(np.column_stack([np.full_like(y, .37), y]))['phase'][:, 0]
    return dict(seed_nodal_Linf=float(error), initial_T_max=float(np.max(abs(s.tensor))),
                initial_Z_max=float(np.max(abs(s.Z))),
                seed_representation_relative_L2=float(np.sqrt(weight @ ((physical-prescribed)**2)/(weight @ (prescribed**2)))),
                seed_representation_Linf_at_reference_points=float(max(abs(physical-prescribed))))


def normalization_reference(s, q):
    p, w = quadrature([s], q)
    phi = s.evaluate(p)['phase'][:, 0]
    ix = np.clip(np.searchsorted(s.xedges, p[:, 0], side='right')-1, 0, len(s.xedges)-2)
    iy = np.clip(np.searchsorted(s.yedges, p[:, 1], side='right')-1, 0, len(s.yedges)-2)
    cell = s.grid[ix, iy]
    boxes = s.box[cell]
    u = (p-boxes[:, :2])/(boxes[:, 2:]-boxes[:, :2])
    dphi = np.zeros(len(p))
    for v in range(4):
        a = u[:, 0] if v & 1 else 1-u[:, 0]
        b = u[:, 1] if v & 2 else 1-u[:, 1]
        vertex_x = boxes[:, 2] if v & 1 else boxes[:, 0]
        value = s.phase[cell, v, 0]
        dphi += a*b*value*(1-value)*(1+.2*vertex_x)
    h = s.m*phi*(1+phi)/(1-phi)**2
    first = s.m*(1+3*phi)/(1-phi)**3*dphi
    second = 6*s.m*(1+phi)/(1-phi)**4*dphi**2
    ns = len(s.I)-1
    t = p[:, 0]*ns
    j = np.minimum(t.astype(int), ns-1)
    t -= j
    rhs = np.zeros((ns+1, 3))
    for k in range(ns+1):
        shape = np.where(j == k, 1-t, np.where(j+1 == k, t, 0))
        for c, integrand in enumerate([h, first, second]):
            rhs[k, c] = np.sum(w.astype(np.longdouble)*shape*integrand, dtype=np.longdouble)
    mass = np.zeros((ns+1, ns+1))
    for k in range(ns):
        mass[k:k+2, k:k+2] += np.array([[2., 1.], [1., 2.]])/(6*ns)
    return np.linalg.solve(mass, rhs)


def normalization_derivatives(run):
    s = LocalState(run/'run', 'state-1')
    data = rows(run/'run/normalization-operator.csv')
    native = np.array([[float(r[k]) for k in ['I', 'dI', 'd2I']] for r in data])
    q24, q32 = normalization_reference(s, 24), normalization_reference(s, 32)
    error = np.max(abs(native-q32), axis=0)
    scaled = error/np.maximum(1., np.max(abs(q32), axis=0))
    assert max(scaled) < 1e-9
    return dict(native_reference_max_error=error.tolist(), native_reference_scaled_error=scaled.tolist(),
                Gauss24_32_uncertainty=np.max(abs(q24-q32), axis=0).tolist())


def domain_absence(run):
    reports = list((run/'run').glob('*-domain-free.csv'))
    assert reports
    fixed = None
    for path in reports:
        info = rows(path)[0]
        assert info['domains_requested'] == '0' and info['domain_handler_present'] == '0'
        assert info['point_associations_only'] == '1' and info['particles'] == '144'
        particles = rows(path.with_name(path.name.replace('-domain-free.csv', '-particles.csv')))
        assert len(particles) == 144 and 'domain_volume' not in particles[0]
        current = {r['id']: (r['x'], r['y']) for r in particles}
        if fixed is None:
            fixed = current
        assert current == fixed
    return dict(snapshots=len(reports), domain_objects_present=0, fixed_particles=144,
                point_associations_have_no_integration_measure=True,
                successful_domain_reads_impossible_without_object=True)


def legacy_comparison(first, second):
    result = {}
    for label in ['prescribed-initial', 'state-0', 'state-1', 'before-step-2', 'state-2']:
        a, b = LocalState(first/'run', label), LocalState(second/'run', label)
        c = comparison(a, b)
        assert c['stress_relative_L2'] < 1e-9 and c['phase_Linf'] < 1e-9
        surface_a, surface_b = rows(a.folder/(label+'-surface.csv')), rows(b.folder/(label+'-surface.csv'))
        for field in ['Z', 'I', 'V', 'committed_V']:
            left, right = np.array([float(r[field]) for r in surface_a]), np.array([float(r[field]) for r in surface_b])
            c[field+'_Linf'] = float(max(abs(left-right)))
            c[field+'_scaled_Linf'] = c[field+'_Linf']/max(1., max(abs(left)))
            assert c[field+'_scaled_Linf'] < 1e-9
        native_a = {(r['CellId'], r['node'], r['component']): r for r in rows(a.folder/(label+'-native-fields.csv'))}
        native_b = {(r['CellId'], r['node'], r['component']): r for r in rows(b.folder/(label+'-native-fields.csv'))}
        assert native_a.keys() == native_b.keys()
        physical, diagnostic = 0., 0.
        scale = 1.
        for key in native_a:
            for field in ['solution', 'old', 'older', 'anchor']:
                left, right = float(native_a[key][field]), float(native_b[key][field])
                if int(key[2]) in [0, 1, 2, 3, 8]:
                    physical = max(physical, abs(left-right))
                    scale = max(scale, abs(left))
                else:
                    diagnostic = max(diagnostic, abs(left-right))
        c['native_velocity_pressure_temperature_phase_scaled_error'] = physical/scale
        c['unmatched_non_authoritative_stress_chemical_slot_max_difference'] = diagnostic
        assert physical/scale < 1e-9
        result[label] = c
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--compare', type=Path)
    p.add_argument('--legacy-comparison', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = dict(domain_absence=domain_absence(a.run))
    if a.compare:
        result['comparisons'] = legacy_comparison(a.run, a.compare) if a.legacy_comparison else compare_runs(a.run/'run', a.compare/'run')
        if not a.legacy_comparison:
            result['compared_domain_absence'] = domain_absence(a.compare)
    else:
        result.update(initialization=initialization(a.run), transfers=assess(a.run/'run'),
                      normalization_derivatives=normalization_derivatives(a.run),
                      physical_and_normalization=quadrature_control(a.run, ['prescribed-initial', 'state-0', 'state-1', 'state-2']))
        assert result['transfers']['quadrature']['native_rule_vs_high_rule_screen_passed']
        assert all(r['physical_rule_passed'] and r['normalization_passed'] for r in result['physical_and_normalization'].values())
        costs = rows(a.run/'run/transfer-cost.csv')
        deep = [r for r in costs if r['before'] == '64' and r['after'] == '256']
        assert deep
        result['deep_uniform_remap'] = dict(refinements=len(deep), costs=costs, passed=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, domain_free_snapshots=result['domain_absence']['snapshots'])))


if __name__ == '__main__':
    main()
