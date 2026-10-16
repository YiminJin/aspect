#!/usr/bin/env python3
"""Independent PF4 timestep, direct limiter norm, scalar-envelope and width checks."""
import argparse
import json
from pathlib import Path
from functools import lru_cache
import numpy as np
from assess_transport import Entry, table, gauss, shape, norm2, exact
from assess_limiter import extrema, fronts, reference, polynomial_matrices

T = np.array([1000., -400., 700.])


@lru_cache(maxsize=128)
def load(path):
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    assert data.shape[1] == 4 and np.array_equal(data[:, 0], np.arange(len(data)))
    return data[:, 1:]


def rk_identity(checks):
    rng = np.random.default_rng(847291)
    defect = 0.
    for _ in range(64):
        w = rng.uniform(.1, 1., size=17)
        w /= sum(w)
        q = lambda a: float(np.sum(w*norm2(a)))
        matrix, source, u = rng.normal(size=(3, 3)), rng.normal(size=(17, 3)), rng.normal(size=(17, 3))
        euler = lambda a: a+.1*(a@matrix+source)
        def limit(a, theta):
            mean = np.einsum('i,ik->k', w, a)
            return mean+theta*(a-mean)
        z1 = euler(u)
        y1 = limit(z1, .7)
        ey1 = euler(y1)
        z2 = .75*u+.25*ey1
        y2 = limit(z2, .8)
        ey2 = euler(y2)
        z3 = u/3+2*ey2/3
        y3 = limit(z3, .9)
        delta = np.array([q(y1)-q(z1), q(y2)-q(z2), q(y3)-q(z3)])
        rhs = delta@[1/6, 2/3, 1.]+(q(z1)-q(u)+q(ey1)-q(y1))/6+2*(q(ey2)-q(y2))/3-q(u-ey1)/8-2*q(u-ey2)/9
        defect = max(defect, abs(q(y3)-q(u)-rhs)/max(1., q(u)))
    checks.append(('independent full-tensor SSPRK3 quadratic identity', defect < 1e-13))
    return defect


def inner(a, b):
    return a[..., 0]*b[..., 0]+a[..., 1]*b[..., 1]+(a[..., 0]+a[..., 1])*(b[..., 0]+b[..., 1])+2*a[..., 2]*b[..., 2]


def full_norm(entry, data):
    N, _, _, w = entry.samples(3)
    v = np.einsum('qi,cik->cqk', N, data[entry.indices])
    return float(np.sum(w*norm2(v)))


def norm_delta(entry, a, b):
    N, _, _, w = entry.samples(3)
    v = np.einsum('qi,cik->cqk', N, a[entry.indices])
    d = np.einsum('qi,cik->cqk', N, (b-a)[entry.indices])
    return float(np.sum(w*(2*inner(v, d)+norm2(d))))


def scalar(a):
    return inner(a, T)/norm2(T)


def scaling(entry, data):
    from numpy.polynomial.legendre import Legendre
    roots = Legendre.basis(8).deriv().roots()
    assert max(abs(roots.imag)) < 1e-13
    q = (np.r_[-1., roots.real, 1.]+1)/2
    L, _, _ = shape(np.repeat(q, 9), np.tile(q, 9), entry.basis)
    N, _, w = polynomial_matrices(entry)
    values = data[entry.indices]
    means = np.einsum('i,cik->ck', w@N, values)
    average = scalar(means)
    samples = scalar(np.einsum('qi,cik->cqk', L, values))
    allowance = 128*np.finfo(float).eps
    assert np.all(abs(average) <= 1+allowance)
    factors = np.ones(len(entry.cells))
    for k, row in enumerate(samples):
        low, high = min(-1, average[k]), max(1, average[k])
        over, under = row > high+allowance, row < low-allowance
        if np.any(over):
            factors[k] = min(factors[k], np.min((high-average[k])/(row[over]-average[k])))
        if np.any(under):
            factors[k] = min(factors[k], np.min((low-average[k])/(row[under]-average[k])))
    out = np.array(data, copy=True)
    out[entry.indices] = means[:, None, :]+factors[:, None, None]*(values-means[:, None, :])
    return out, factors, means


def local(entry, checks):
    rows = table(entry.path/'scaling-local.csv')
    out = {}
    N, _, _, w = entry.samples(3)
    mean = polynomial_matrices(entry)[2]@polynomial_matrices(entry)[0]
    for row in rows:
        name = str(row['case'])
        data = load(entry.path/f'scaling-input-{name}.csv')
        actual = load(entry.path/f'scaling-output-{name}.csv')
        expected, theta, means = scaling(entry, data)
        actual_factors = table(entry.path/f'scaling-theta-{name}.csv')['theta']
        scale = max(1., np.max(abs(data)))
        defect = np.max(abs(actual-expected))/scale
        mean_error = np.max(abs(np.einsum('i,cik->ck', mean, (actual-data)[entry.indices])))/scale
        delta = norm_delta(entry, data, actual)
        deviations = np.einsum('qi,cik->cqk', N, data[entry.indices])-means[:, None, :]
        predicted = float(np.sum(w*norm2(deviations)*(theta**2-1)[:, None]))
        rec = dict(reconstruction_error=float(defect), mean_error=float(mean_error), factor_error=float(max(abs(theta-actual_factors))), direct_norm_delta=delta, contraction_formula_delta=predicted, native_delta=float(row['change']))
        out[name] = rec
        checks.extend([(name+': independent scalar-envelope reconstruction', defect < 1e-11), (name+': independent theta', rec['factor_error'] < 1e-11), (name+': mean preservation', mean_error < 1e-12), (name+': theta in[0,1]', min(theta) >= 0 and max(theta) <= 1), (name+': independent norm delta', abs(delta-row['change']) < 1e-10*max(1., abs(delta))), (name+': exact tensor norm contraction', abs(delta-predicted) < 1e-10*max(1., abs(delta)))])
    checks.extend([('native basis/ordering/idempotence', max(rows['basis_error']) < 1e-12 and max(rows['indicator_error']) < 1e-12 and max(rows['order_error']) < 1e-12 and max(rows['idempotence_error']) < 1e-12), ('inadmissible mean rejection', bool(table(entry.path/'scaling-rejection.csv')['value']))])
    return out


def finite_scalar(x, t, width):
    x = x-.4*t
    a = width/(2*np.arctanh(.8))
    return .5*np.tanh((x-.25)/a)-np.tanh((x-.5)/a)+.5*np.tanh((x-.75)/a)


def finite_derivative(x, t, width):
    x = x-.4*t
    a = width/(2*np.arctanh(.8))
    return (.5/np.cosh((x-.25)/a)**2-1/np.cosh((x-.5)/a)**2+.5/np.cosh((x-.75)/a)**2)/a


def finite_fronts(entry, data, peak_x, amplitude):
    """Exact roots of each Q2 trace and its jumps, using actual pulse levels."""
    time, width = entry.time, float(entry.constants['width'])
    positive, negative = peak_x+.4*time, 1-peak_x+.4*time
    branches = [(0., positive, .1*amplitude, .9*amplitude),
                (positive, negative, .8*amplitude, -.8*amplitude),
                (negative, 1., -.9*amplitude, -.1*amplitude)]
    N, _, _ = shape(np.array([0., .5, 1.]), np.full(3, .5), entry.basis)
    bottom = min(entry.cells['y0'])
    cells = sorted([(c, scalar(N@data[entry.indices[k]])) for k, c in enumerate(entry.cells) if c['y0'] == bottom], key=lambda a: a[0]['x0'])
    widths, references = [], []
    for lo, hi, level1, level2 in branches:
        exact_crossings, numerical_crossings = [], []
        for level in [level1, level2]:
            a, b = lo, hi
            for _ in range(60):
                middle = (a+b)/2
                if (finite_scalar(a, time, width)-level)*(finite_scalar(middle, time, width)-level) <= 0:
                    b = middle
                else:
                    a = middle
            target = (a+b)/2
            exact_crossings.append(target)
            candidates = []
            previous = None
            for c, values in cells:
                aa, bb, cc = 2*(values[2]-2*values[1]+values[0]), 4*values[1]-values[2]-3*values[0], values[0]-level
                coefficients = np.trim_zeros([aa, bb, cc], 'f')
                if len(coefficients) > 1:
                    for root in np.roots(coefficients):
                        if abs(root.imag) < 1e-12 and -1e-12 <= root.real <= 1+1e-12:
                            x = c['x0']+c['dx']*root.real
                            if lo <= x <= hi:
                                candidates.append(float(x))
                if previous is not None and (previous-level)*(values[0]-level) <= 0 and lo <= c['x0'] <= hi:
                    candidates.append(float(c['x0']))
                previous = values[2]
            numerical_crossings.append(min(candidates, key=lambda x: abs(x-target)) if candidates else None)
        references.append(abs(exact_crossings[1]-exact_crossings[0]))
        widths.append(abs(numerical_crossings[1]-numerical_crossings[0]) if None not in numerical_crossings else None)
    return dict(front_widths=widths, reference_front_widths=references,
                front_broadening=[w/r-1 if w is not None else None for w, r in zip(widths, references)])


def field(entry, case, x, y, time):
    if case != 'finite_width':
        return exact(case, x, y, time, 0.)
    return finite_scalar(x, time, float(entry.constants['width']))[..., None]*T


def measure(entry, case, method):
    label = case+'-'+method
    data = entry.coefficients_for(label)
    initial = load(entry.path/f'initial-{case}.csv')
    initial_norm = full_norm(entry, initial)
    if case == 'signed_jump':
        entry.coefficients['signed_jump'] = data
        entry.metrics['signed_jump'] = entry.metrics[label]
        row = entry.jump_resolved(48)
        coarse = entry.jump_resolved(24)
        row.update(front_widths=fronts(entry, data), L1_uncertainty=abs(row['error_L1']-coarse['error_L1']))
        numeric_norm = row['quadratic_norm_ratio']*1.25e6
        exact_initial = exact_final = 1.25e6
    else:
        N, x, y, w = entry.samples(24)
        val = np.einsum('qi,cik->cqk', N, data[entry.indices])
        ref = field(entry, case, x, y, entry.time)
        initial_ref = field(entry, case, x, y, 0.)
        error = float(np.sqrt(np.sum(w*norm2(val-ref))))
        exact_final = float(np.sum(w*norm2(ref)))
        exact_initial = float(np.sum(w*norm2(initial_ref)))
        numeric_norm = full_norm(entry, data)
        row = dict(error_L2=error, error_L1=float(np.sum(w*np.sqrt(norm2(val-ref)))), relative_L2=error/np.sqrt(exact_final), **extrema(entry, data))
        if case == 'finite_width':
            width = float(entry.constants['width'])
            xcenter, ycenter = .5+.4*entry.time, .5
            k = next(k for k, c in enumerate(entry.cells) if c['x0'] <= xcenter < c['x0']+c['dx'] and c['y0'] <= ycenter < c['y0']+c['dy'])
            c = entry.cells[k]
            _, Nr, _ = shape(np.array([(xcenter-c['x0'])/c['dx']]), np.array([(ycenter-c['y0'])/c['dy']]), entry.basis)
            derivative = float(scalar((Nr@data[entry.indices[k]])[0])/c['dx'])
            reference_derivative = float(finite_derivative(xcenter, entry.time, width))
            row.update(central_derivative=derivative, reference_derivative=reference_derivative, central_slope_width=float(2*np.arctanh(.8)/abs(derivative)), reference_slope_width=float(2*np.arctanh(.8)/abs(reference_derivative)))
            lo, hi = .25, .5
            for _ in range(60):
                mid = (lo+hi)/2
                if finite_derivative(mid, 0., width) > 0:
                    lo = mid
                else:
                    hi = mid
            maximum = float(finite_scalar((lo+hi)/2, 0., width))
            row.update(reference_peak=1000*maximum, overshoot=max(0., row['max_xx']/1000-maximum)/maximum, undershoot=max(0., -row['min_xx']/1000-maximum)/maximum)
            row.update(finite_fronts(entry, data, (lo+hi)/2, maximum))
    peak = row.get('reference_peak', 1000.)
    row.update(positive_peak_attenuation=max(0., 1-row['max_xx']/peak), negative_peak_attenuation=max(0., 1+row['min_xx']/peak))
    row.update(initial_norm2=initial_norm, final_norm2=numeric_norm, norm_change_fraction=numeric_norm/initial_norm-1, exact_boundary_norm_change=exact_final-exact_initial, excess_norm_change=(numeric_norm-initial_norm)-(exact_final-exact_initial), initial_extrema=extrema(entry, initial))
    return row


def kx_mask(entry, data, case, time):
    N, _, w = polynomial_matrices(entry)
    fq, fw = gauss(4)
    target, _, _ = shape(np.zeros(4), fq, entry.basis)
    neighbor, _, _ = shape(np.ones(4), fq, entry.basis)
    lookup = {(round(c['x0']*entry.nx), round(c['y0']*entry.ny)): i for i, c in enumerate(entry.cells)}
    decisions = []
    for k, c in enumerate(entry.cells):
        ix, iy = round(c['x0']*entry.nx), round(c['y0']*entry.ny)
        keys = [(ix-1, iy), (ix+1, iy), (ix, iy-1), (ix, iy+1)]
        patch = [k]+[lookup[a] for a in keys if a in lookup]
        M = max(np.sqrt(np.sum(w*norm2(N@data[entry.indices[j]]))) for j in patch)
        inside = target@data[entry.indices[k]]
        if (ix-1, iy) in lookup:
            outside = neighbor@data[entry.indices[lookup[ix-1, iy]]]
        else:
            outside = field(entry, case, np.zeros(4), c['y0']+fq*c['dy'], time)
        jump = np.einsum('q,qk->k', fw, inside-outside)
        numerator = np.sqrt(norm2(jump))
        hpower = (np.hypot(c['dx'], c['dy'])/np.sqrt(2))**1.5
        indicator = numerator/(hpower*M) if M else (np.inf if numerator else 0.)
        decisions.append(indicator > 1)
    return np.array(decisions)


def compare(entry, a, b):
    N, _, _, w = entry.samples(3)
    value = np.einsum('qi,cik->cqk', N, (a-b)[entry.indices])
    return float(np.sqrt(np.sum(w*norm2(value))))


def pulse_extrema(entry, data):
    N, _, _ = shape(np.array([0., .5, 1.]), np.full(3, .5), entry.basis)
    low, high, spread = np.inf, -np.inf, 0.
    values = data[entry.indices, 0]
    for x in np.unique(entry.cells['x0']):
        column = values[entry.cells['x0'] == x]
        for r in [0., .5, 1.]:
            samples = column[:, entry.basis['r'] == r]
            spread = max(spread, float(np.max(samples)-np.min(samples)))
    bottom = min(entry.cells['y0'])
    for k, c in enumerate(entry.cells):
        if c['y0'] != bottom:
            continue
        v = (N@data[entry.indices[k]])[:, 0]
        aa, bb = 2*(v[2]-2*v[1]+v[0]), 4*v[1]-v[2]-3*v[0]
        points = [0., 1.]
        if aa and 0 < -bb/(2*aa) < 1:
            points.append(-bb/(2*aa))
        extrema_values = [aa*r*r+bb*r+v[0] for r in points]
        low, high = min(low, *extrema_values), max(high, *extrema_values)
    return dict(polynomial_min_xx=float(low), polynomial_max_xx=float(high), transverse_spread=spread)


def transport(entries, checks):
    result = {}
    for e in entries:
        key = f'{e.nx}-{int(e.constants["time_divisor"])}'
        rows = {}
        for case in ['pulse_translation', 'signed_jump', 'finite_width']:
            for method in (['unlimited'] if case == 'pulse_translation' else ['unlimited', 'kxrcf', 'fu_shu', 'scaling']):
                label = case+'-'+method
                row = measure(e, case, method)
                if case != 'pulse_translation':
                    p = pulse_extrema(e, e.coefficients_for(label))
                    row.update(p)
                    peak = row.get('reference_peak', 1000.)
                    row.update(sampled_positive_peak_attenuation=row['positive_peak_attenuation'], sampled_negative_peak_attenuation=row['negative_peak_attenuation'], positive_peak_attenuation=max(0., 1-p['polynomial_max_xx']/peak), negative_peak_attenuation=max(0., 1+p['polynomial_min_xx']/peak), polynomial_overshoot=max(0., p['polynomial_max_xx']/peak-1), polynomial_undershoot=max(0., -p['polynomial_min_xx']/peak-1))
                    checks.append((key+' '+label+': transverse symmetry retained', p['transverse_spread']/1000 < 1e-11))
                if method != 'unlimited':
                    calls = np.atleast_1d(table(e.path/f'norm-calls-{label}.csv'))
                    steps = int(e.metrics[label]['steps'])
                    checks.extend([(key+' '+label+': all three stage calls', len(calls) == 3*steps and np.array_equal(calls['call'], np.arange(3*steps))), (key+' '+label+': norm identity', np.max(abs(calls['after']-calls['before']-calls['change'])/np.maximum(1., calls['before'])) < 1e-13)])
                    relative = calls['change']/row['initial_norm2']
                    weighted = float(np.sum(calls['change'].reshape(-1, 3)*[1/6, 2/3, 1.]))
                    denominator = np.sqrt(calls['before'])+np.sqrt(calls['after'])
                    norm_changes = np.divide(calls['change'], denominator, out=np.zeros(len(calls)), where=denominator > 0)
                    positive = calls['change'] > 128*np.finfo(float).eps*np.maximum(1., calls['before'])
                    row['limiter_calls'] = dict(count=len(calls), direct_sum=float(sum(calls['change'])), direct_sum_fraction=float(sum(relative)), rk_weighted_sum=weighted, rk_weighted_sum_fraction=weighted/row['initial_norm2'], positive_calls=int(sum(positive)), max_relative_increase=float(max(relative)), max_relative_decrease=float(min(relative)), positive_cells=int(sum(calls['positive_cells'])), negative_cells=int(sum(calls['negative_cells'])), transport_RK_remainder=float(row['final_norm2']-row['initial_norm2']-weighted))
                    row['limiter_calls']['direct_L2_sum'] = float(sum(norm_changes))
                    if 'maximum_mean_excess' in calls.dtype.names:
                        row['limiter_calls'].update(means_outside_exact_envelope=int(sum(calls['means_outside_envelope'])), maximum_mean_excess=float(max(calls['maximum_mean_excess'])))
                        if method == 'scaling':
                            checks.append((key+' '+case+': mean excess confined to declared roundoff', max(calls['maximum_mean_excess']) <= 128*np.finfo(float).eps))
                    if method == 'scaling':
                        checks.append((key+' '+case+': scaling contracts quadratic norm', not np.any(positive)))
                    snapshot_path = e.path/f'call-input-{label}.csv'
                    if snapshot_path.exists():
                        before = load(snapshot_path)
                        after = load(e.path/f'call-output-{label}.csv')
                        if method == 'scaling':
                            expected, _, _ = scaling(e, before)
                        else:
                            _, expected = reference(e, before, force=method == 'kxrcf')
                            if method == 'kxrcf':
                                mask = kx_mask(e, before, case, float(calls[0]['stage_time']))
                                expected[e.indices[~mask]] = before[e.indices[~mask]]
                        reconstruction_error = np.max(abs(expected-after))/max(1., np.max(abs(before)))
                        delta = norm_delta(e, before, after)
                        row['first_call_oracle'] = dict(reconstruction_error=float(reconstruction_error), before_error=abs(full_norm(e, before)-calls[0]['before']), delta_error=abs(delta-calls[0]['change']))
                        checks.extend([(key+' '+label+': independent first-call reconstruction', reconstruction_error < 1e-11), (key+' '+label+': independent direct norm change', abs(delta-calls[0]['change']) < 1e-10*max(1., abs(delta)))])
                    held_path = e.path/f'held-{label}.csv'
                    if held_path.exists():
                        history = table(held_path)
                        before = e.coefficients_for(label)
                        after = load(e.path/f'held-field-{label}-1.csv')
                        final = load(e.path/f'held-field-{label}-12.csv')
                        if method == 'scaling':
                            expected, _, _ = scaling(e, before)
                        else:
                            _, expected = reference(e, before, force=method == 'kxrcf')
                            if method == 'kxrcf':
                                mask = kx_mask(e, before, case, e.time)
                                expected[e.indices[~mask]] = before[e.indices[~mask]]
                        initial = row['initial_norm2']
                        row['held_map'] = dict(relative_field_change=compare(e, before, final)/np.sqrt(initial), norm_change_fraction=(full_norm(e, final)-full_norm(e, before))/initial, positive_passes=int(sum(history['change'] > 128*np.finfo(float).eps*history['norm2'])))
                        checks.extend([(key+' '+label+': held first map independently reconstructed', np.max(abs(expected-after))/max(1., np.max(abs(before))) < 1e-11), (key+' '+label+': held norm ledger', abs(sum(history['change'])-norm_delta(e, before, final)) < 1e-11*initial)])
                        if method == 'scaling':
                            checks.append((key+' '+case+': native held scaling is idempotent', row['held_map']['relative_field_change'] < 1e-12))
                rows[label] = row
        costs = table(e.path/'costs.csv')
        rows['costs'] = {str(r['case']): {k: float(r[k]) for k in costs.dtype.names if k != 'case'} for r in costs}
        checks.extend([(key+': means unchanged by limiting', max(costs['max_mean_error']) < 1e-12), (key+': nonpublication', bool(table(e.path/'accepted.csv')['value']))])
        result[key] = rows
    fine = sorted([e for e in entries if e.nx == max(a.nx for a in entries)], key=lambda e: int(e.constants['time_divisor']))
    sensitivity = {}
    for case in ['pulse_translation', 'signed_jump', 'finite_width']:
        for method in (['unlimited'] if case == 'pulse_translation' else ['unlimited', 'kxrcf', 'fu_shu', 'scaling']):
            label = case+'-'+method
            comparisons = []
            for a, b in zip(fine[:-1], fine[1:]):
                initial = load(a.path/f'initial-{case}.csv')
                checks.append((label+': identical initial projection across dt', np.array_equal(initial, load(b.path/f'initial-{case}.csv'))))
                denominator = np.sqrt(full_norm(a, initial))
                difference = compare(a, a.coefficients_for(label), b.coefficients_for(label))
                common = {}
                for p in sorted(a.path.glob(f'common-{label}-*.csv')):
                    other = b.path/p.name
                    assert other.exists()
                    common[p.stem.rsplit('-', 1)[1]] = compare(a, load(p), load(other))/denominator
                comparisons.append(dict(divisors=[int(a.constants['time_divisor']), int(b.constants['time_divisor'])], full_L2_difference=difference, relative_difference=difference/denominator, common_time_relative_differences=common))
            sensitivity[label] = comparisons
    return dict(runs=result, timestep_comparison=sensitivity)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs', type=Path, nargs='+')
    p.add_argument('--local', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    checks = []
    identity_defect = rk_identity(checks)
    entries = [Entry(d) for d in args.runs]
    result = dict(local=local(entries[0], checks)) if args.local else transport(entries, checks)
    result['rk_identity_scaled_defect'] = identity_defect
    result.update(passed=all(bool(v) for _, v in checks), checks=[dict(name=n, passed=bool(v)) for n, v in checks])
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=result['passed'], checks=len(checks), failures=[n for n, v in checks if not v]), indent=2))
    raise SystemExit(0 if result['passed'] else 1)

if __name__ == '__main__':
    main()
