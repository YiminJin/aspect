#!/usr/bin/env python3
"""Separate fixed-rule physical integration, normalization and frozen remap errors."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from assess_pf6_transfer import State, quadrature, functional, rows, comparison, fracture_reference, lagrange


class LocalState(State):
    def __init__(self, folder, label):
        super().__init__(folder, label)
        # The union grid subdivides coarse cells as well as fine cells. Fill
        # every subrectangle; a single lower-left entry suffices only on a
        # tensor-product mesh and would silently evaluate the wrong coarse cell.
        self.grid.fill(-1)
        for i, b in enumerate(self.box):
            x0, x1 = np.searchsorted(self.xedges, b[[0, 2]])
            y0, y1 = np.searchsorted(self.yedges, b[[1, 3]])
            assert np.all(self.grid[x0:x1, y0:y1] == -1)
            self.grid[x0:x1, y0:y1] = i
        assert np.all(self.grid >= 0)


def independent_normalization(state, q, projection_diagnostic=False):
    p, w = quadrature([state], q)
    phi = state.evaluate(p)['phase'][:, 0]
    h = state.m*phi*(1+phi)/(1-phi)**2
    n = len(state.I)-1
    t = p[:, 0]*n
    j = np.minimum(t.astype(int), n-1)
    t -= j
    # Accumulate in extended precision so reference summation does not obscure
    # the much smaller error of the native analytic ray integral.
    rhs = np.zeros(n+1)
    for k in range(n+1):
        shape = np.where(j == k, 1-t, np.where(j+1 == k, t, 0))
        rhs[k] = np.sum(w.astype(np.longdouble)*h*shape, dtype=np.longdouble)
    mass = np.zeros((n+1, n+1))
    for k in range(n):
        mass[k:k+2, k:k+2] += np.array([[2., 1.], [1., 2.]])/(6*n)
    nodes = np.linalg.solve(mass, rhs)
    if not projection_diagnostic:
        return nodes
    xs = np.unique(p[:, 0])
    ny = len(p)//len(xs)
    weights = w.reshape(len(xs), ny)
    wx, wy = weights.sum(axis=1), weights.sum(axis=0)
    raw = np.sum(h.reshape(len(xs), ny).astype(np.longdouble)*wy, axis=1, dtype=np.longdouble).astype(float)
    projected = np.interp(xs, np.linspace(0, 1, n+1), nodes)
    native = np.interp(xs, np.linspace(0, 1, n+1), state.I)
    return nodes, dict(surface_projection_RMS=float(np.sqrt(wx @ ((projected-raw)**2))),
                       surface_projection_Linf=float(max(abs(projected-raw))),
                       native_I_vs_unprojected_ray_Linf=float(max(abs(native-raw))))


def quadrature_control(run, labels):
    result = {}
    for label in labels:
        s = LocalState(run/'run', label)
        p24, w24 = quadrature([s], 24)
        _, e24, l24 = functional(s, p24, w24)
        p32, w32 = quadrature([s], 32)
        _, e32, l32 = functional(s, p32, w32)
        native = rows(s.folder/(label+'-energy.csv'))[0]
        en = np.array([float(native[k]) for k in ['bulk', 'cohesive', 'fracture']])
        ln = np.array([float(r['shear']) for r in rows(s.folder/(label+'-weak-loads.csv'))])
        i24 = independent_normalization(s, 24)
        i32, projection = independent_normalization(s, 32, True)
        exact_i = copy.copy(s)
        exact_i.I = i32
        _, ei, li = functional(exact_i, p32, w32)
        de, dl = en-e32, ln-l32
        r = dict(native_minus_reference_energy=de.tolist(), native_minus_reference_load=dl.tolist(),
                 energy_scaled_error=float(max(abs(de)/np.maximum(1., abs(e32)))),
                 load_scaled_error=float(max(abs(dl))/max(1., max(abs(l32)))),
                 q24_q32_energy_uncertainty=float(max(abs(e24-e32))),
                 q24_q32_load_uncertainty=float(max(abs(l24-l32))),
                 native_I_error=float(max(abs(s.I-i32))),
                 native_I_scaled_error=float(max(abs(s.I-i32))/max(1., max(abs(i32)))),
                 q24_q32_I_uncertainty=float(max(abs(i24-i32))),
                 normalization_only_energy_change=(ei-e32).tolist(),
                 normalization_only_load_change=(li-l32).tolist())
        r.update(projection)
        r['physical_rule_passed'] = r['energy_scaled_error'] <= 1e-9 and r['load_scaled_error'] <= 1e-9
        r['normalization_passed'] = r['native_I_scaled_error'] <= 1e-11
        result[label] = r
    return result


def compare_exports(first, second, left_label, right_label):
    c = comparison(LocalState(first/'run', left_label), LocalState(second/'run', right_label))
    assert c['stress_relative_L2'] < 1e-9 and c['phase_Linf'] < 1e-11
    for suffix, keys in [('tensor', ['CellId']), ('surface', ['i']),
                         ('native-fields', ['CellId', 'node', 'component']),
                         ('particles', ['id']), ('weak-loads', ['i']), ('energy', []),
                         ('fracture-operator', ['x', 'y'])]:
        left = {tuple(r[k] for k in keys): r for r in rows(first/'run'/(left_label+'-'+suffix+'.csv'))}
        right = {tuple(r[k] for k in keys): r for r in rows(second/'run'/(right_label+'-'+suffix+'.csv'))}
        assert left.keys() == right.keys()
        error, scale = 0., 1.
        for key in left:
            for field in left[key]:
                if field in keys or field == 'rank':
                    continue
                if field == 'CellId':
                    assert left[key][field] == right[key][field]
                    continue
                u, v = float(left[key][field]), float(right[key][field])
                if np.isnan(u) and np.isnan(v):
                    continue
                assert np.isfinite(u) and np.isfinite(v)
                error, scale = max(error, abs(u-v)), max(scale, abs(u))
        c[suffix+'_error'], c[suffix+'_scaled_error'] = error, error/scale
        assert error/scale < 1e-9
    a = sorted((first/'run').glob(left_label+'-rng-rank-*.txt'))
    b = sorted((second/'run').glob(right_label+'-rng-rank-*.txt'))
    c['rng_counts'] = [len(a), len(b)]
    c['rng_equal_per_rank'] = len(a) == len(b) and all(x.read_bytes() == y.read_bytes() for x, y in zip(a, b))
    if len(a) == len(b):
        assert c['rng_equal_per_rank']
    return c


def hanging_trace_check(s):
    data = rows(s.folder/(s.label+'-native-fields.csv'))
    nc = 1+max(int(r['component']) for r in data)
    coeff = np.zeros((len(s.ids), 9, nc, 4))
    lookup = {k: i for i, k in enumerate(s.ids)}
    for r in data:
        coeff[lookup[r['CellId']], int(r['node']), int(r['component'])] = [float(r[k]) for k in ['solution', 'old', 'older', 'anchor']]
    def value(k, p):
        b = s.box[k]
        u = (p-b[:2])/(b[2:]-b[:2])
        lx, ly = lagrange(u[:, 0]), lagrange(u[:, 1])
        B = lx[:, s.nodes[:, 0]]*ly[:, s.nodes[:, 1]]
        return np.einsum('qi,icf->qcf', B, coeff[k])
    z, _ = np.polynomial.legendre.leggauss(3)
    z = (z+1)/2
    error = np.zeros(nc)
    faces = 0
    slaves = set()
    for i, a in enumerate(s.box):
        for j, b in enumerate(s.box):
            if not np.prod(a[2:]-a[:2]) > np.prod(b[2:]-b[:2]):
                continue
            for d in [0, 1]:
                other = 1-d
                for face in [a[d], a[d+2]]:
                    if face not in [b[d], b[d+2]]:
                        continue
                    lo, hi = max(a[other], b[other]), min(a[other+2], b[other+2])
                    if hi <= lo:
                        continue
                    p = np.zeros((3, 2))
                    p[:, d], p[:, other] = face, lo+z*(hi-lo)
                    error = np.maximum(error, np.max(abs(value(i, p)-value(j, p)), axis=(0, 2)))
                    faces += 1
                    for v in [lo, hi]:
                        if a[other] < v < a[other+2]:
                            x = [0., 0.]
                            x[d], x[other] = face, v
                            slaves.add(tuple(x))
    scale = np.maximum(1., np.max(abs(coeff), axis=(0, 1, 3)))
    assert faces > 0 and slaves and max(error/scale) < 1e-11
    return dict(coarse_fine_face_pairs=faces, physical_phase_hanging_nodes=len(slaves),
                native_component_trace_errors=error.tolist(), native_component_scaled_trace_errors=(error/scale).tolist())


def domain_check(run):
    before, after = {}, {}
    generations = []
    stale = zero = outside = 0
    for f in sorted((run/'run').glob('first-local-before-rank-*.csv')):
        a = rows(f)
        b = rows(f.with_name(f.name.replace('-before-', '-after-')))
        rank = int(f.stem.rsplit('-', 1)[1])
        generations.append([int(a[0]['generation']), int(b[0]['generation'])])
        for r in a:
            assert r['id'] not in before
            before[r['id']] = r | {'owner_rank': rank}
        for r in b:
            assert r['id'] not in after
            after[r['id']] = r | {'owner_rank': rank}
            stale += r['id'] != r['previous_slot_id']
            zero += float(r['raw_slot_volume']) == 0
            outside += not np.isfinite(float(r['raw_slot_volume']))
    assert before and before.keys() == after.keys()
    assert all((r['x'], r['y']) == (after[k]['x'], after[k]['y']) for k, r in before.items())
    xs = np.unique([float(r['x']) for r in before.values()])
    ys = np.unique([float(r['y']) for r in before.values()])
    assert len(xs)*len(ys) == len(before)
    # An independent Voronoi oracle for this Cartesian-product particle set:
    # bisectors are the midpoint lines, clipped to the unit physical box.
    edges = [np.r_[0., .5*(x[1:]+x[:-1]), 1.] for x in [xs, ys]]
    widths = [np.diff(x) for x in edges]
    error, total, box_error = 0., 0., 0.
    for r in after.values():
        expected = widths[0][np.searchsorted(xs, float(r['x']))]*widths[1][np.searchsorted(ys, float(r['y']))]
        volume = float(r['raw_slot_volume'])
        error = max(error, abs(volume-expected)) if np.isfinite(volume) else float('inf')
        total += volume
        if 'xmin' in r:
            ix, iy = np.searchsorted(xs, float(r['x'])), np.searchsorted(ys, float(r['y']))
            actual = np.array([float(r[k]) for k in ['xmin', 'ymin', 'xmax', 'ymax']])
            expected_box = np.array([edges[0][ix], edges[1][iy], edges[0][ix+1], edges[1][iy+1]])
            box_error = max(box_error, max(abs(actual-expected_box))) if np.all(np.isfinite(actual)) else float('inf')
    rebuilt = all(b > a for a, b in generations)
    return dict(particles=len(after), fixed_global_ids_coordinates=True, generations=generations,
                rank_migrations=sum(r['owner_rank'] != after[k]['owner_rank'] for k, r in before.items()),
                CellId_changes=sum(r['CellId'] != after[k]['CellId'] for k, r in before.items()),
                changed_or_missing_old_slot_owners=stale, zero_slots=zero, out_of_old_storage=outside,
                domains_rebuilt=rebuilt, independent_cartesian_Voronoi_volume_error=error if np.isfinite(error) else None,
                independent_cartesian_Voronoi_box_error=box_error if np.isfinite(box_error) else None,
                total_volume=total if np.isfinite(total) else None,
                domain_check_passed=rebuilt and error<1e-12 and box_error<1e-12 and abs(total-1.)<1e-12)


def domain_snapshots(run):
    result = {}
    for f in sorted((run/'run').glob('*-particles.csv')):
        data = rows(f)
        if not data or 'domain_volume' not in data[0]:
            continue
        xs = np.unique([float(r['x']) for r in data])
        ys = np.unique([float(r['y']) for r in data])
        assert len(xs)*len(ys) == len(data) == 144
        edges = [np.r_[0., .5*(x[:-1]+x[1:]), 1.] for x in [xs, ys]]
        volume_error = box_error = total = 0.
        for r in data:
            ix, iy = np.searchsorted(xs, float(r['x'])), np.searchsorted(ys, float(r['y']))
            expected = np.array([edges[0][ix], edges[1][iy], edges[0][ix+1], edges[1][iy+1]])
            actual = np.array([float(r[k]) for k in ['xmin', 'ymin', 'xmax', 'ymax']])
            assert np.all(np.isfinite(actual))
            box_error = max(box_error, max(abs(actual-expected)))
            volume = float(r['domain_volume'])
            volume_error = max(volume_error, abs(volume-np.prod(expected[2:]-expected[:2])))
            total += volume
        assert box_error < 1e-12 and volume_error < 1e-12 and abs(total-1.) < 1e-12
        result[f.stem] = dict(volume_error=volume_error, box_error=box_error, total_volume=total)
    assert result
    return result


def local_transfer(run):
    gates = rows(run/'run/transfer-gates.csv')
    assert gates and all(g['passed'] == '1' for g in gates)
    result = dict(gates=len(gates), transfers={})
    for label in ['local-complete-Q2', 'local-accepted']:
        first = LocalState(run/'run', label+'-before')
        for suffix in ['refined', 'coarsened']:
            s = LocalState(run/'run', label+'-'+suffix)
            c = comparison(first, s)
            z, w = np.polynomial.legendre.leggauss(3)
            xx, yy = np.meshgrid((z+1)/2, (z+1)/2, indexing='ij')
            unit = np.column_stack([xx.ravel(), yy.ravel()])
            weight = np.outer(w/2, w/2).ravel()
            means = []
            for box in s.box:
                p = box[:2]+unit*(box[2:]-box[:2])
                means.append(np.sum(weight[:, None]*(s.evaluate(p)['T']-first.evaluate(p)['T']), axis=0))
            c['maximum_target_cell_T_mean_error'] = float(np.max(abs(np.array(means))))
            c['scaled_target_cell_T_mean_error'] = c['maximum_target_cell_T_mean_error']/max(1., np.max(abs(first.tensor)))
            assert c['tensor_relative_L2'] < 1e-11 and c['stress_relative_L2'] < 1e-11 and c['phase_Linf'] < 1e-13
            assert c['scaled_target_cell_T_mean_error'] < 1e-11
            result['transfers'][label+'-'+suffix] = c
        result['transfers'][label+'-fracture-reference'] = fracture_reference(LocalState(run/'run', label+'-refined'))
    initial = {r['id']: (r['x'], r['y']) for r in rows(run/'run/initial-particles.csv')}
    for f in (run/'run').glob('*-particles.csv'):
        assert {r['id']: (r['x'], r['y']) for r in rows(f)} == initial
    result['particles_fixed'] = len(initial)
    result['independent_hanging_traces'] = hanging_trace_check(LocalState(run/'run', 'local-held'))
    result['particle_domains'] = domain_check(run)
    assert result['particle_domains']['domain_check_passed']
    result['all_snapshot_domains'] = domain_snapshots(run)
    result['quadrature'] = quadrature_control(run, ['local-held', 'local-checkpoint'])
    assert all(v['physical_rule_passed'] and v['normalization_passed'] for v in result['quadrature'].values())
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--frozen', action='store_true')
    p.add_argument('--local', action='store_true')
    p.add_argument('--restart-comparison', action='store_true')
    p.add_argument('--diagnose-domains', action='store_true')
    p.add_argument('--compare', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.diagnose_domains:
        result = domain_check(a.run)
        a.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result))
        return
    if a.local:
        if a.compare:
            result = compare_exports(a.run, a.compare, 'local-checkpoint', 'local-restored' if a.restart_comparison else 'local-checkpoint')
        else:
            result = local_transfer(a.run)
        result['case_label'] = json.loads((a.run/'record.json').read_text())['case']['label']
        if a.compare:
            result['comparison_case_label'] = json.loads((a.compare/'record.json').read_text())['case']['label']
        a.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(dict(local_comparison_passed=True)))
        return
    labels = ['frozen-G'+str(q) for q in [6, 12, 24, 32]] if a.frozen else ['state-0', 'state-1', 'state-2']
    result = dict(physical_and_normalization=quadrature_control(a.run, labels))
    result['snapshot_particle_domains'] = domain_snapshots(a.run)
    required = labels[1:] if a.frozen else labels
    result['G12_gate_passed'] = all(result['physical_and_normalization'][k]['physical_rule_passed']
                                   and result['physical_and_normalization'][k]['normalization_passed'] for k in required)
    assert result['G12_gate_passed'], 'Retain a failed G12 gate; do not proceed to local refinement'
    if a.frozen:
        result['original_G6_failure_retained'] = not result['physical_and_normalization']['frozen-G6']['physical_rule_passed']
        assert result['original_G6_failure_retained']
    if a.compare:
        result['comparison_reference'] = str(a.compare)
        result['trajectory_reference_vs_run'] = {label: comparison(LocalState(a.compare/'run', label), LocalState(a.run/'run', label)) for label in labels}
        result['reference_physical_and_normalization'] = quadrature_control(a.compare, labels)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(G12_gate_passed=result['G12_gate_passed'], snapshots=len(labels))))


if __name__ == '__main__':
    main()
