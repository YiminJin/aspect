#!/usr/bin/env python3
"""Independent Legendre/KKT candidate checks and controlled transport comparison."""
import argparse
import json
import copy
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import Legendre
from assess_methods import Data, gauss, norm2
from assess_transport import Entry
from assess_stabilization import measure, load, full_norm, finite_fronts, finite_scalar, finite_derivative
from assess_limiter import polynomial_matrices, fronts


def array(path):
    return np.atleast_2d(np.loadtxt(path, delimiter=',', skiprows=1))


def legendre(r, s):
    return np.stack([Legendre.basis(i)(2*r-1)*Legendre.basis(j)(2*s-1)
                     for i in range(3) for j in range(3)], axis=-1)


def rows(d, source, other, volume=False):
    a, b = d.cells[source], d.cells[other]
    if volume:
        x = b['x0']+b['dx']*d.basis['r']
        y = b['y0']+b['dy']*d.basis['s']
        return legendre((x-a['x0'])/a['dx'], (y-a['y0'])/a['dy'])
    q, w = gauss(5)
    x = b['x0']+b['dx']*np.repeat(q, 5)
    y = b['y0']+b['dy']*np.tile(q, 5)
    return (np.repeat(w, 5)*np.tile(w, 5))@legendre((x-a['x0'])/a['dx'], (y-a['y0'])/a['dy'])


def kkt(d, patch, j, mask):
    source = patch[j]
    M = np.diag([1/((2*i+1)*(2*k+1)) for i in range(3) for k in range(3)])
    averages = np.stack([rows(d, source, k) for k in patch])
    useful = [l for l in range(1, len(patch)) if mask & (1 << l)]
    C = averages[useful]
    # Unit-square area is the reference area. This is the physical-integral
    # weighting divided by the neighbor-cell measure, not retuned per case.
    rho = d.cells[source]['dx']*d.cells[source]['dy']
    H = M+rho*C.T@C
    K = np.zeros((10, 10)); K[:9, :9] = H
    K[:9, 9] = averages[0]; K[9, :9] = averages[0]
    return M, H, K, averages, useful, rho


def independent_weights(d, patch, candidates, values):
    N, derivatives, w = polynomial_matrices(d)
    amplitude = max(np.sum(w*norm2(N@values[k])) for k in patch)
    beta = np.zeros(len(patch))
    if amplitude:
        for j, a in enumerate(candidates):
            for i, D in enumerate(derivatives):
                beta[j] += np.sum(w*(2 if i == 3 else 1)*norm2(D@a))/amplitude
    linear = np.full(len(patch), .001); linear[0] = 1-.001*(len(patch)-1)
    if np.max(beta):
        linear *= ((min(beta)+1e-6)/(beta+1e-6))**2
    return beta, linear/sum(linear)


def local(path):
    d = Data(path)
    stencils = array(d.path/'cls-stencils.csv').astype(int)
    systems = array(d.path/'cls-systems.csv')
    patches = {}
    for cell, candidate, position, patchcell in stencils:
        patches.setdefault((cell, candidate), {})[position] = patchcell
    patches = {k: [v[i] for i in range(len(v))] for k, v in patches.items()}
    largest_kernel_error = 0.
    smallest_eigenvalue = 1.
    for key, patch in patches.items():
        cell, j = key
        table = systems[(systems[:, 0] == cell) & (systems[:, 1] == j)]
        for mask in np.unique(table[:, 2]).astype(int):
            M, H, K, averages, useful, rho = kkt(d, patch, j, mask)
            smallest_eigenvalue = min(smallest_eigenvalue, float(np.linalg.eigvalsh(H)[0]))
            rhs = np.zeros((10, len(patch))); rhs[9, 0] = 1
            for l in useful: rhs[:9, l] = rho*averages[l]
            expected = rows(d, patch[j], patch[0], True)@np.linalg.solve(K, rhs)[:9]
            actual = np.zeros_like(expected)
            tt = table[table[:, 2] == mask]
            actual[tt[:, 3].astype(int), tt[:, 4].astype(int)] = tt[:, 5]
            largest_kernel_error = max(largest_kernel_error, float(np.max(abs(expected-actual))))
    errors = {}
    metric_block_error = 0.
    N, derivatives, w = polynomial_matrices(d)
    means = w@N
    for name in ['piecewise_constant', 'nonproportional']:
        values = d.load('cls-input-'+name)
        candidates = array(d.path/('cls-candidates-'+name+'.csv'))
        max_coefficient, max_constraint, max_stationarity, max_beta, max_weight = 0., 0., 0., 0., 0.
        for cell in range(len(d.cells)):
            patch = patches[cell, 1]
            target_mean = means@values[cell]
            own_means = np.array([means@values[k] for k in patch])
            result = []
            for j in range(len(patch)):
                lines = candidates[(candidates[:, 0] == cell) & (candidates[:, 1] == j)]
                actual = lines[:, 3:6]
                if j == 0:
                    expected = values[cell]
                else:
                    mask = int(lines[0, 8])
                    distance = norm2(own_means-own_means[0])
                    others = [l for l in range(1, len(patch)) if l != j]
                    farthest = max(distance[others], default=0.)
                    selected = sum(1 << l for l in others if distance[l] < farthest*(1-64*np.finfo(float).eps))
                    assert mask == selected
                    M, H, K, averages, useful, rho = kkt(d, patch, j, mask)
                    native_modes = np.linalg.solve(rows(d, patch[j], patch[j], True), values[patch[j]])
                    rhs = np.zeros((10, 3)); rhs[:9] = M@native_modes; rhs[9] = target_mean
                    for l in useful: rhs[:9] += rho*averages[l, :, None]*own_means[l]
                    solution = np.linalg.solve(K, rhs)
                    if cell == 0 and j == 1:
                        metric = np.array([[2., 1., 0.], [1., 2., 0.], [0., 0., 2.]])
                        block = np.zeros((30, 30)); block[:27, :27] = np.kron(H, metric)
                        constraint = np.kron(averages[0][None, :], np.eye(3))
                        block[:27, 27:] = constraint.T; block[27:, :27] = constraint
                        load = np.r_[(rhs[:9]@metric).ravel(), rhs[9]]
                        block_solution = np.linalg.solve(block, load)[:27].reshape(9, 3)
                        metric_block_error = max(metric_block_error, float(np.max(abs(block_solution-solution[:9])))/1000)
                    B = rows(d, patch[j], cell, True)
                    expected = B@solution[:9]
                    restored_modes = np.linalg.solve(B, actual)
                    stationarity = H@restored_modes-rhs[:9]
                    lam = averages[0]@stationarity/(averages[0]@averages[0])
                    max_stationarity = max(max_stationarity, float(np.max(abs(stationarity-averages[0, :, None]*lam)))/1000)
                result.append(actual)
                max_coefficient = max(max_coefficient, float(np.max(abs(actual-expected)))/1000)
                max_constraint = max(max_constraint, float(np.max(abs(means@actual-target_mean)))/1000)
            beta, weights = independent_weights(d, patch, result, values)
            for j in range(len(patch)):
                line = candidates[(candidates[:, 0] == cell) & (candidates[:, 1] == j)][0]
                max_beta = max(max_beta, abs(beta[j]-line[6])/max(1., beta[j]))
                max_weight = max(max_weight, abs(weights[j]-line[7]))
        errors[name] = dict(coefficient_error=max_coefficient, constraint_error=max_constraint,
                            stationarity_error=max_stationarity, beta_error=max_beta, weight_error=max_weight)
    summary = np.genfromtxt(d.path/'cls-local.csv', names=True, delimiter=',', dtype=None, encoding='utf-8')
    max_mean = float(np.max(summary['mean_change']))
    for key in ['order_error', 'basis_error', 'weight_error', 'rescale_error']:
        assert np.max(summary[key]) < 1e-11, key
    for row in summary:
        if row['case'] not in ['piecewise_constant', 'nonproportional']: assert row['polynomial_change'] < 1e-12
    for row in errors.values(): assert max(row.values()) < 1e-10, row
    assert largest_kernel_error < 1e-10 and max_mean < 1e-9 and smallest_eigenvalue > 0 and metric_block_error < 1e-10
    return dict(independent_Legendre_KKT_max_kernel_error=largest_kernel_error,
                minimum_objective_eigenvalue=smallest_eigenvalue, polynomial_space_dimension=9,
                maximum_physical_mean_change=max_mean, candidates=errors,
                full_tensor_metric_block_KKT_error=metric_block_error,
                native_cases=[{k: (str(row[k]) if k == 'case' else float(row[k])) for k in summary.dtype.names} for row in summary])


def transverse_fronts(entry, values, case):
    ids_by_x = [np.flatnonzero(np.rint(entry.cells['x0']*entry.nx) == i) for i in range(entry.nx)]
    ri = np.rint(entry.basis['r']*2).astype(int)
    local_values = values[entry.indices]
    transverse = 0.
    for ids in ids_by_x:
        bottom = min(ids, key=lambda k: entry.cells['y0'][k])
        for r in range(3):
            a = local_values[ids][:, ri == r]
            reference = local_values[bottom, ri == r][0]
            transverse = max(transverse, float(np.max(np.sqrt(norm2(a-reference)))))
    widths = []
    if case == 'finite_width':
        width = float(entry.constants['width'])
        a, b = .25, .5
        for _ in range(60):
            m = (a+b)/2
            if finite_derivative(m, 0., width) > 0: a = m
            else: b = m
        peak_x = (a+b)/2
        peak = finite_scalar(peak_x, 0., width)
    for row in range(entry.ny):
        mask = np.rint(entry.cells['y0']*entry.ny).astype(int) == row
        view = copy.copy(entry); view.cells = entry.cells[mask]; view.indices = entry.indices[mask]
        if case == 'signed_jump': widths.append(fronts(view, values))
        else: widths.append(finite_fronts(view, values, peak_x, peak)['front_widths'])
    assert all(v is not None for row in widths for v in row)
    widths = np.array(widths)
    return dict(maximum_stored_nodal_full_tensor_transverse_deviation_Pa=transverse,
                middle_row_front_widths=widths[entry.ny//2].tolist(),
                minimum_row_front_widths=np.min(widths, axis=0).tolist(),
                maximum_row_front_widths=np.max(widths, axis=0).tolist(),
                front_width_measure='Physical center slice of each y-cell row; sharp widths use retained 1024 points per x cell, finite widths use complete Q2 roots/jumps.')


def transverse_probe(entry, values):
    from assess_transport import shape
    q, w = gauss(4)
    face, _, _ = shape(np.ones(4), q, entry.basis)
    N, _, W = polynomial_matrices(entry)
    local = values[entry.indices]
    means = np.einsum('i,cik->ck', W@N, local)
    faces = np.einsum('i,cik->ck', w@face, local)
    result = dict(physical_cell_mean_spread_Pa=0., right_face_mean_spread_Pa=0.)
    for column in range(entry.nx):
        ids = np.flatnonzero(np.rint(entry.cells['x0']*entry.nx) == column)
        bottom = min(ids, key=lambda k: entry.cells['y0'][k])
        result['physical_cell_mean_spread_Pa'] = max(result['physical_cell_mean_spread_Pa'], float(np.max(np.sqrt(norm2(means[ids]-means[bottom])))))
        result['right_face_mean_spread_Pa'] = max(result['right_face_mean_spread_Pa'], float(np.max(np.sqrt(norm2(faces[ids]-faces[bottom])))))
    # Exact tensor-product polynomial projection onto fields independent of y.
    q, w = gauss(3)
    samples, _, _ = shape(np.repeat([0., .5, 1.], 3), np.tile(q, 3), entry.basis)
    evaluated = np.einsum('qi,cik->cqk', samples, local).reshape(-1, 3, 3, 3)
    own_y_means = np.einsum('j,crjk->crk', w, evaluated)
    averaged = np.zeros_like(values)
    ri = np.rint(entry.basis['r']*2).astype(int)
    for column in range(entry.nx):
        ids = np.flatnonzero(np.rint(entry.cells['x0']*entry.nx) == column)
        average = np.sum(entry.cells['dy'][ids, None, None]*own_y_means[ids], axis=0)
        averaged[entry.indices[ids]] = average[ri][None, :, :]
    result['transverse_variance_L2'] = float(np.sqrt(full_norm(entry, values-averaged)))
    return result


def traces(entry, label):
    a = array(entry.path/('stage-means-'+label+'.csv'))
    # Sort first by cell and then by call. Every trace row includes all three
    # tensor components of physical means, face averages and the native RHS.
    bycell = a[np.lexsort((a[:, 0], a[:, 2]))]
    cells = np.unique(bycell[:, 2]).astype(int)
    series = np.stack([bycell[bycell[:, 2] == k] for k in cells])
    before, after = series[:, :, 5:8], series[:, :, 8:11]
    rhs = series[:, :, 26:29]
    flux_delta = (series[:, :, 20:23]-series[:, :, 17:20]-(series[:, :, 14:17]-series[:, :, 11:14]))*.4*entry.nx
    native = series[:, :, 29:32]
    stages = series.shape[1]
    dt = .2/(stages//3)
    from assess_methods import Data
    d = Data(entry.path.parent)
    initial = d.load('initial-'+label.rsplit('-', 1)[0])
    N, _, w = polynomial_matrices(entry)
    initial_means = np.einsum('i,cik->ck', w@N, initial)[cells]
    max_rk = 0.
    for call in range(stages):
        stage, step = call % 3, call//3
        start = initial_means if step == 0 else after[:, 3*step-1]
        if stage == 0:
            if step == 0: continue
            expected = start+dt*rhs[:, call-1]
        elif stage == 1:
            expected = .75*start+.25*(after[:, call-1]+dt*rhs[:, call-1])
        else:
            expected = start/3+2*(after[:, call-1]+dt*rhs[:, call-1])/3
        max_rk = max(max_rk, float(np.max(abs(expected-before[:, call]))))
    mean_change = float(np.max(np.sqrt(norm2(after-before))))
    flux_error = float(np.max(abs(flux_delta-native)))
    assert mean_change < 1e-8 and max_rk < 1e-8 and flux_error < 1e-7
    index = np.unravel_index(np.argmax(norm2(flux_delta)), flux_delta.shape[:2])
    selected = series[index]
    return dict(maximum_physical_mean_change=mean_change, maximum_RK_mean_balance_error=max_rk,
                maximum_native_flux_delta_error=flux_error,
                largest_flux_change=dict(cell=int(selected[2]), x=float(selected[3]), call=int(selected[0]),
                                         stage=int(selected[0]) % 3, time=float(selected[1]),
                                         mean_before=selected[5:8].tolist(), mean_after=selected[8:11].tolist(),
                                         rhs_before=selected[23:26].tolist(), rhs_after=selected[26:29].tolist()))


def transport(path):
    entry = Entry(path)
    # The emitted fixture contains one named measured path; the Gaussian
    # unlimited control is allowed separately.
    assert len(entry.metrics) == 1
    label = next(iter(entry.metrics))
    case, detector = label.rsplit('-', 1)
    row = measure(entry, case, detector)
    row.update(label=path.name, case=case, detector=detector,
               divisor=int(entry.constants['time_divisor']), nx=entry.nx)
    text = (path/'input.prm').read_text()
    row['construction'] = 'cls' if 'set Candidate construction = constrained least squares' in text else 'shift'
    record = json.loads((path/'record.json').read_text()); assert record['gate_passed']
    row.update(seconds=record['seconds'], max_rss_KiB=record['max_rss_KiB'])
    if case in ['signed_jump', 'finite_width']:
        row['transverse'] = transverse_fronts(entry, entry.coefficients_for(label), case)
        row['transverse'].update(transverse_probe(entry, entry.coefficients_for(label)))
        row['transverse']['fraction_of_error_squared'] = (row['transverse']['transverse_variance_L2']/row['error_L2'])**2
    cost = np.genfromtxt(entry.path/'costs.csv', delimiter=',', names=True, dtype=None, encoding='utf-8')
    row['costs'] = {k: (str(cost[k]) if k == 'case' else float(cost[k])) for k in cost.dtype.names}
    row['limited_fraction'] = float(cost['limited']/cost['visits']) if cost['visits'] else 0.
    if detector != 'unlimited':
        calls = array(entry.path/('norm-calls-'+label+'.csv'))
        initial = row['initial_norm2']
        row['limiter_calls'] = len(calls)
        row['limiter_raw_norm_change_fraction'] = float(sum(calls[:, 4])/initial)
        row['limiter_RK_weighted_norm_change_fraction'] = float(sum(calls[:, 4]*np.tile([1/6, 2/3, 1.], len(calls)//3))/initial)
        row['positive_norm_calls'] = int(sum(calls[:, 4] > 128*np.finfo(float).eps*initial))
        row['maximum_all_cell_physical_mean_change'] = float(max(calls[:, 9]))
        row['maximum_integral_mean_change'] = float(np.max(np.sqrt(norm2(calls[:, 10:13]))))
        row['trace'] = traces(entry, label)
        held = array(entry.path/('held-'+label+'.csv'))
        final = load(entry.path/('field-'+label+'.csv'))
        repeated = load(entry.path/('held-field-'+label+'-12.csv'))
        row['held'] = dict(norm_change_fraction=float(held[-1, 1]/held[0, 1]-1),
                           relative_L2_change=np.sqrt(full_norm(entry, repeated-final)/row['final_norm2']),
                           relative_to_initial_pulse_norm=np.sqrt(full_norm(entry, repeated-final)/row['initial_norm2']),
                           limited_cells_per_pass=held[1:, 3].astype(int).tolist())
        assert max(calls[:, 9]) < 1e-8
        if int(entry.constants['time_divisor']) == 1:
            row['first_stage_transverse'] = {side: transverse_probe(entry, load(entry.path/f'call-{side}-{label}.csv')) for side in ['input', 'output']}
        details = array(entry.path/('candidate-trace-'+label+'.csv'))
        N, _, quadrature_weights = polynomial_matrices(entry)
        representatives = []
        for call, cell in np.unique(details[:, [0, 2]], axis=0):
            table = details[(details[:, 0] == call) & (details[:, 2] == cell)]
            ids = np.unique(table[:, 4]).astype(int)
            candidate = np.stack([table[table[:, 4] == j, 6:9] for j in ids])
            weights = np.array([table[table[:, 4] == j, 10][0] for j in ids])
            beta = np.array([table[table[:, 4] == j, 9][0] for j in ids])
            assert abs(sum(weights)-1) < 1e-12 and min(weights) >= 0 and np.all(np.isfinite(weights))
            before = N@candidate[0]
            after = N@np.einsum('j,jik->ik', weights, candidate)
            mean = quadrature_weights@before
            q_before = float(np.sum(quadrature_weights*norm2(before-mean)))
            q_after = float(np.sum(quadrature_weights*norm2(after-mean)))
            if q_before > 256*np.finfo(float).eps*np.sum(quadrature_weights*norm2(before)) and weights[0] < .5:
                representatives.append(dict(call=int(call), cell=int(cell), x=float(table[0, 3]), time=float(table[0, 1]),
                                            variance_ratio=q_after/q_before, beta=beta.tolist(), weights=weights.tolist(),
                                            target_nodal_xx=candidate[0, :, 0].tolist(), limited_nodal_xx=(weights@candidate[:, :, 0]).tolist()))
        # This is an analysis selection for explaining a staircase, not a
        # detector, limiter bound or condition that changes the experiment.
        row['representative_flattening'] = sorted(representatives, key=lambda a: a['variance_ratio'])[:3]
    assert bool(np.genfromtxt(entry.path/'accepted.csv', delimiter=',', names=True, dtype=None, encoding='utf-8')['value'])
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='*', type=Path)
    p.add_argument('--local', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = {}
    if args.local: result['local'] = local(args.local)
    if args.runs:
        result['transport'] = [transport(path) for path in args.runs]
        result['initialization'] = {}
        for case in ['signed_jump', 'finite_width']:
            path = next((p for p in args.runs if any(r['label'] == p.name and r['case'] == case
                                                   for r in result['transport'])), None)
            if path is None:
                continue
            d = Data(path)
            initial = d.errors(d.load('initial-'+case), case, 0., 'DG', 24)
            assert initial['orthogonality_defect'] < 1e-12
            result['initialization'][case] = initial
        differences = []
        for first in result['transport']:
            for second in result['transport']:
                if (first['case'], first['detector'], first['construction'], first['nx']) != (second['case'], second['detector'], second['construction'], second['nx']): continue
                if second['divisor'] != 2*first['divisor']: continue
                aa, bb = Entry(Path(next(p for p in args.runs if p.name == first['label']))), Entry(Path(next(p for p in args.runs if p.name == second['label'])))
                field = first['case']+'-'+first['detector']
                v, w = aa.coefficients_for(field), bb.coefficients_for(field)
                change = np.sqrt(full_norm(aa, v-w)/second['final_norm2'])
                differences.append(dict(case=first['case'], detector=first['detector'], construction=first['construction'],
                                        divisors=[first['divisor'], second['divisor']], relative_L2_change=change,
                                        relative_to_analytic_norm=float(np.sqrt(full_norm(aa, v-w))*second['relative_L2']/second['error_L2']),
                                        difference_to_fine_analytic_error=float(np.sqrt(full_norm(aa, v-w))/second['error_L2'])))
        result['time_differences'] = differences
        screen = []
        for fine in differences:
            if fine['divisors'] != [2, 4]: continue
            coarse = next(r for r in differences if (r['case'], r['detector'], r['construction'], r['divisors']) == (fine['case'], fine['detector'], fine['construction'], [1, 2]))
            screen.append(dict(case=fine['case'], detector=fine['detector'], construction=fine['construction'],
                               fine_difference_to_analytic_error=fine['difference_to_fine_analytic_error'],
                               differences_decrease=bool(fine['relative_L2_change'] < coarse['relative_L2_change']),
                               passed=bool(fine['difference_to_fine_analytic_error'] <= .01 and fine['relative_L2_change'] < coarse['relative_L2_change'])))
        result['temporal_screen'] = screen
    result['implementation_checks_passed'] = True
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
