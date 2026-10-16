#!/usr/bin/env python3
"""Independent physical-coordinate Q2 tensor detector/WENO and PF4 assessment."""
import argparse
import json
from pathlib import Path
import numpy as np
from assess_transport import Entry, table, gauss, shape, norm2, exact


def load(path):
    a = table(path)
    return np.stack([a['xx'], a['yy'], a['xy']], axis=-1).astype(float)


def polynomial_matrices(entry, order=3):
    a, w = gauss(order)
    r, s = np.repeat(a, order), np.tile(a, order)
    N, Nr, Ns = shape(r, s, entry.basis)
    ri, si = np.rint(entry.basis['r']*2).astype(int), np.rint(entry.basis['s']*2).astype(int)
    lr = np.array([(2*r-1)*(r-1), 4*r*(1-r), r*(2*r-1)]).T
    ls = np.array([(2*s-1)*(s-1), 4*s*(1-s), s*(2*s-1)]).T
    dr = np.array([4*r-3, 4-8*r, 4*r-1]).T
    ds = np.array([4*s-3, 4-8*s, 4*s-1]).T
    dd = np.array([4., -8., 4.])
    derivatives = [Nr, Ns, ls[:, si]*dd[ri], dr[:, ri]*ds[:, si], lr[:, ri]*dd[si]]
    return N, derivatives, np.repeat(w, order)*np.tile(w, order)


def reference(entry, data, force=False):
    # No native coefficients/shape/indicator routines imported. Extrapolate all
    # nine nodal values by physical-coordinate Lagrange polynomials.
    N, D, w = polynomial_matrices(entry)
    mean_row = w@N
    lookup = {(round(c['x0']*entry.nx), round(c['y0']*entry.ny)): i for i, c in enumerate(entry.cells)}
    rows, output = [], np.copy(data)
    for k, cell in enumerate(entry.cells):
        ix, iy = round(cell['x0']*entry.nx), round(cell['y0']*entry.ny)
        neighbors = [lookup[t] for t in [(ix-1, iy), (ix+1, iy), (ix, iy-1), (ix, iy+1)] if t in lookup]
        patch = [k]+neighbors
        scale = np.max(np.abs(data[entry.indices[patch]]))
        if scale == 0:
            own = np.zeros((len(patch), 9, 3))
        else:
            own = data[entry.indices[patch]]/scale
        M = np.sqrt(max(np.sum(w*norm2(N@v)) for v in own))
        candidates = []
        for c, values in zip(patch, own):
            j = entry.cells[c]
            x = cell['x0']+cell['dx']*entry.basis['r']
            y = cell['y0']+cell['dy']*entry.basis['s']
            E, _, _ = shape((x-j['x0'])/j['dx'], (y-j['y0'])/j['dy'], entry.basis)
            candidates.append(values[0]+E@(values-values[0]))
        candidates = np.asarray(candidates)
        means = np.einsum('i,jik->jk', mean_row, candidates)
        imean = sum(np.sqrt(norm2(means[0]-m))/M for m in means[1:]) if M else 0.
        ipoly = max([0.]+[np.sqrt(np.sum(w*norm2(N@(candidates[0]-v))))/M for v in candidates[1:]]) if M else 0.
        beta = np.array([sum(np.sum(w*norm2(d@(v-v[0])/(M if M else 1)))*(2 if i == 3 else 1) for i, d in enumerate(D)) for v in candidates])
        linear = np.full(len(patch), .001)
        linear[0] = 1-.001*len(neighbors)
        weights = linear.copy()
        if np.max(beta) > 0:
            weights *= ((np.min(beta)+1e-6)/(beta+1e-6))**2
        weights /= sum(weights)
        if force or max(imean, ipoly) > .1:
            output[entry.indices[k]] = scale*np.einsum('j,jik->ik', weights, candidates-means[:, None, :]+means[0])
        rows.append((imean, ipoly, weights, beta))
    return rows, output


def local(entry, checks):
    path = entry.path
    results = {}
    for source in sorted(path.glob('local-input-*.csv')):
        name = source.stem[len('local-input-'):]
        data, actual = load(source), load(path/f'local-output-{name}.csv')
        rows, expected = reference(entry, data, force=True)
        native = table(path/f'local-indicators-{name}.csv')
        scale = np.max(np.abs(data)) or 1.
        mismatch = max(np.max(np.abs(np.array([r[0] for r in rows])-native['mean'])), np.max(np.abs(np.array([r[1] for r in rows])-native['poly'])))
        weights_error = max(np.max(np.abs(r[2]-np.array([native[f'w{j}'][i] for j in range(len(r[2]))]))) for i, r in enumerate(rows))
        output_error = np.max(np.abs(expected-actual))/scale
        means_error = np.max(np.abs(np.einsum('i,cik->ck', polynomial_matrices(entry)[2]@polynomial_matrices(entry)[0], (actual-data)[entry.indices])))/scale
        results[name] = dict(indicator_error=float(mismatch), weight_error=float(weights_error), reconstruction_error=float(output_error), physical_mean_error_scaled=float(means_error), max_mean=float(np.max(native['mean'])), max_poly=float(np.max(native['poly'])))
        checks.extend([(f'{name}: independent indicators', mismatch < 1e-11), (f'{name}: independent weights', weights_error < 1e-11), (f'{name}: full Q2 reconstruction', output_error < 1e-11), (f'{name}: physical target means', means_error < 1e-12)])
    # Full embedded matrices, no second deviator operation.
    a = load(path/'local-output-nonproportional.csv')
    embedded = np.zeros((len(a), 3, 3))
    embedded[:, 0, 0], embedded[:, 1, 1], embedded[:, 2, 2] = a[:, 0], a[:, 1], -a[:, 0]-a[:, 1]
    embedded[:, 0, 1] = embedded[:, 1, 0] = a[:, 2]
    checks.extend([('symmetry and trace', np.max(np.abs(embedded-embedded.transpose(0, 2, 1))) == 0 and np.max(np.abs(np.trace(embedded, axis1=1, axis2=2))) < 1e-14), ('full tensor metric', np.max(np.abs(np.sum(embedded**2, axis=(1, 2))-norm2(a))) < 1e-13)])
    checks.append(('mixed xy mean blind spot / polynomial detection', results['mixed_xy']['max_mean'] < 1e-12 and results['mixed_xy']['max_poly'] > .2))
    checks.append(('direction changes detected as tensor differences', results['orientation']['max_poly'] > .1))
    for name in ['nonproportional', 'orientation']:
        a, b = table(path/f'local-kx-{name}.csv'), table(path/f'local-kx-rotated-{name}.csv')
        checks.append((f'{name}: KXRCF tensor basis covariance', np.max(abs(a['kxrcf']-b['kxrcf'])) < 1e-11 and np.array_equal(a['limited'], b['limited'])))
        data = load(path/f'local-input-{name}.csv')
        lookup = {(round(c['x0']*entry.nx), round(c['y0']*entry.ny)): i for i, c in enumerate(entry.cells)}
        N, _, w = polynomial_matrices(entry)
        fq, fw = gauss(4)
        indicator_error = 0.
        for k, c in enumerate(entry.cells):
            ix, iy = round(c['x0']*entry.nx), round(c['y0']*entry.ny)
            keys = [(ix-1, iy), (ix+1, iy), (ix, iy-1), (ix, iy+1)]
            patch = [k]+[lookup[t] for t in keys if t in lookup]
            scale = np.max(abs(data[entry.indices[patch]]))
            amplitude = max(np.sqrt(np.sum(w*norm2(N@(data[entry.indices[j]]/scale)))) for j in patch)
            jump, area = np.zeros(3), 0.
            for key, r, s, length in [(keys[0], np.zeros(4), fq, c['dy']), (keys[2], fq, np.zeros(4), c['dx'])]:
                face, _, _ = shape(r, s, entry.basis)
                inside = face@(data[entry.indices[k]]/scale)
                outside = np.zeros_like(inside)
                if key in lookup:
                    j = lookup[key]
                    other = entry.cells[j]
                    face_other, _, _ = shape((c['x0']+c['dx']*r-other['x0'])/other['dx'], (c['y0']+c['dy']*s-other['y0'])/other['dy'], entry.basis)
                    outside = face_other@(data[entry.indices[j]]/scale)
                jump += length*np.einsum('q,qk->k', fw, inside-outside)
                area += length
            hpower = (np.hypot(c['dx'], c['dy'])/np.sqrt(2))**1.5
            oracle = np.sqrt(norm2(jump))/(area*hpower*amplitude)
            indicator_error = max(indicator_error, abs(oracle-a['kxrcf'][k]))
        checks.append((f'{name}: independent tensor KXRCF', indicator_error < 1e-11))
    checks.append(('constant background normalization sensitivity exposed', results['background']['max_poly'] < results['nonproportional']['max_poly']/1000))
    q = table(path/'local.csv')
    checks.extend([('cell/stencil traversal independence', max(q['order_error']) < 1e-12), ('in-plane tensor basis covariance', max(q['basis_error']) < 1e-11), ('finite normalized nonnegative weights', max(q['weight_sum_error']) < 1e-14 and min(q['min_weight']) >= 0)])
    for name in ['tiny', 'huge']:
        b, c = table(path/f'local-indicators-{name}.csv'), table(path/'local-indicators-nonproportional.csv')
        checks.append((f'{name}: amplitude invariant', max(np.max(abs(b[f]-c[f])) for f in ['mean', 'poly', 'w0', 'w1', 'w2', 'w3', 'w4']) < 1e-11))
    # Polynomial mass and squared integrals: actually inspect rule rank/accuracy.
    N3, _, w3 = polynomial_matrices(entry, 3)
    N5, _, w5 = polynomial_matrices(entry, 5)
    N2, _, w2 = polynomial_matrices(entry, 2)
    results['quadrature'] = dict(gauss3_mass_rank=int(np.linalg.matrix_rank(N3.T@(w3[:, None]*N3))), gauss2_mass_rank=int(np.linalg.matrix_rank(N2.T@(w2[:, None]*N2))), mass_error=float(np.max(abs(N3.T@(w3[:, None]*N3)-N5.T@(w5[:, None]*N5)))))
    checks.append(('Gauss3 full rank and exact Q2 mass', results['quadrature']['gauss3_mass_rank'] == 9 and results['quadrature']['mass_error'] < 1e-14))
    # Boundary and genuinely mixed-sign face: constant inside, zero boundary.
    kx, points = table(path/'boundary-indicators.csv'), table(path/'inflow-points.csv')
    max_error, mixed_faces = 0., 0
    for k, cell in enumerate(entry.cells):
        p = points[points['cell'] == k]
        area = sum(p['weight'])
        boundary_area = sum(p[p['boundary'] == 1]['weight'])
        hpower = (np.hypot(cell['dx'], cell['dy'])/np.sqrt(2))**1.5
        target = boundary_area/(area*hpower) if area else 0.
        max_error = max(max_error, abs(float(kx['kxrcf'][k])-target))
        if cell['y0'] < .4375 < cell['y0']+cell['dy']:
            # Inflow on left upper half and right lower half: whole-face sign
            # would be zero for both and miss both contributions.
            mixed_faces += len(p) > 0
    results['boundary'] = dict(max_error=float(max_error), mixed_cells=int(mixed_faces), no_inflow_cells=int(np.sum(kx['kxrcf'] == 0)))
    checks.append(('pointwise mixed inflow and boundary/no-inflow', max_error < 1e-11 and mixed_faces > 0))
    lifecycle = table(path/'limited-lifecycle.csv')
    checks.append(('limited repeated trials and late rejection preserve incoming', np.all(lifecycle['repeat_error'] == 0) and np.all(lifecycle['late_source_rejected'] == 1) and np.all(lifecycle['incoming_unchanged'] == 1)))
    return results


def calibration(entries, checks):
    out = {}
    for entry in entries:
        rows = {}
        for path in sorted(entry.path.glob('snapshot-indicators-*.csv')):
            name = path.stem[len('snapshot-indicators-'):]
            a = table(path)
            independently, _ = reference(entry, load(entry.path/f'snapshot-{name}.csv'))
            defect = max(np.max(abs(np.array([x[j] for x in independently])-a[f])) for j, f in [(0, 'mean'), (1, 'poly')])
            checks.append((f'{entry.nx} {name} snapshot indicators', defect < 1e-10))
            rows[name] = {str(C): dict(mean_fraction=float(np.mean(a['mean'] > C)), limited_fraction=float(np.mean(np.maximum(a['mean'], a['poly']) > C)), safeguard_only_fraction=float(np.mean((a['poly'] > C) & (a['mean'] <= C)))) for C in [.05, .1, .2]}
        out[entry.nx] = rows
    return out


def extrema(entry, data):
    # Exactly PF4's tensor-product Gauss-Lobatto9 sampling.
    from numpy.polynomial.legendre import Legendre
    roots = Legendre.basis(8).deriv().roots()
    assert np.max(abs(roots.imag)) < 1e-13
    r = (np.r_[-1., roots.real, 1.]+1)/2
    N, _, _ = shape(np.repeat(r, 9), np.tile(r, 9), entry.basis)
    value = np.einsum('qi,cik->cqk', N, data[entry.indices])
    return dict(min_xx=float(np.min(value[..., 0])), max_xx=float(np.max(value[..., 0])))


def fronts(entry, data):
    # Sum physical lengths in the 10--90% range around each separate front;
    # midpoint sampling at 1024 points/cell, errors <=3*h/1024 in each width.
    widths = np.zeros(3)
    locations = np.array([.25, .5, .75])+.4*entry.time
    levels = [(0.1, .9), (-.8, .8), (-.9, -.1)]
    r = (np.arange(1024)+.5)/1024
    N, _, _ = shape(r, np.full_like(r, .5), entry.basis)
    yrow = np.min(entry.cells['y0'])
    for k, c in enumerate(entry.cells):
        if c['y0'] != yrow:
            continue
        x = c['x0']+c['dx']*r
        val = (N@data[entry.indices[k]])[:, 0]/1000
        for j, (lo, hi) in enumerate(levels):
            widths[j] += np.sum((abs(x-locations[j]) < .115) & (val > lo) & (val < hi))*c['dx']/1024
    return widths.tolist()


def validation(entries, checks):
    out = {}
    for entry in entries:
        results = {}
        for case in ['pulse_translation', 'signed_jump']:
            initial = load(entry.path/f'initial-{case}.csv')
            N, _, _, w = entry.samples(3)
            initial_norm2 = np.sum(w*norm2(np.einsum('qi,cik->cqk', N, initial[entry.indices])))
            results[case+'-initial'] = dict(**extrema(entry, initial), norm2=float(initial_norm2))
            for method in ['unlimited', 'kxrcf', 'fu_shu']:
                label = case+'-'+method
                data = entry.coefficients_for(label)
                entry.coefficients[case] = data
                entry.metrics[case] = entry.metrics[label]
                if case == 'signed_jump':
                    row = entry.jump_resolved(48)
                    coarse = entry.jump_resolved(24)
                    row.update(L2_quadrature_difference=abs(coarse['error_L2']-row['error_L2']), L1_quadrature_uncertainty=abs(coarse['error_L1']-row['error_L1']), front_widths=fronts(entry, data))
                    checks.append((f'{entry.nx} {label} split L2 quadrature', row['L2_quadrature_difference'] < 1e-9))
                else:
                    row = entry.measure(case, 16)
                    Ns, x, y, ws = entry.samples(16)
                    values = np.einsum('qi,cik->cqk', Ns, data[entry.indices])
                    row['error_L1'] = float(np.sum(ws*np.sqrt(norm2(values-exact(case, x, y, entry.time, 0.)))))
                    row.update(extrema(entry, data))
                ratio = row['quadratic_norm_ratio']*1.25e6/initial_norm2 if 'quadratic_norm_ratio' in row else row['norm_L2']**2/initial_norm2
                row['quadratic_norm_change_from_projection'] = float(ratio-1.)
                row['initial_norm2'] = float(initial_norm2)
                # Full independently sampled tensor extrema match native output.
                ex = extrema(entry, data)
                native = entry.metrics[label]
                checks.append((f'{entry.nx} {label} extrema', max(abs(ex[f]-native[f]) for f in ['min_xx', 'max_xx']) < 1e-8))
                results[label] = row
        costs = table(entry.path/'costs.csv')
        results['costs'] = {r['case']: {f: float(r[f]) for f in costs.dtype.names if f != 'case'} for r in np.atleast_1d(costs)}
        checks.append((f'{entry.nx} mean preservation each stage', max(costs['max_mean_error']) < 1e-12))
        for r in costs:
            if r['case'].endswith('unlimited'):
                continue
            checks.append((f'{entry.nx} {r["case"]} all three stages', r['visits'] == 3*r['steps']*entry.nx*entry.ny))
        accepted = table(entry.path/'accepted.csv')
        checks.append((f'{entry.nx} nonpublication', bool(accepted['value'])))
        out[entry.nx] = results
    convergence = {}
    for method in ['unlimited', 'kxrcf', 'fu_shu']:
        errors = [out[e.nx]['pulse_translation-'+method]['relative_L2'] for e in entries]
        orders = [float(np.log(errors[i]/errors[i+1])/np.log(entries[i+1].nx/entries[i].nx)) for i in range(len(errors)-1)]
        convergence[method] = dict(errors=errors, orders=orders)
        checks.append((method+' finest smooth order >= 2.5', orders[-1] >= 2.5))
        checks.append((method+' final three smooth errors decrease', all(errors[i+1] < errors[i] for i in range(max(0, len(errors)-3), len(errors)-1))))
    return dict(meshes=out, convergence=convergence)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs', type=Path, nargs='+')
    p.add_argument('--calibration', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    entries = sorted([Entry(d) for d in args.runs], key=lambda e: e.nx)
    checks = []
    if args.calibration:
        result = dict(local=local(entries[0], checks), snapshots=calibration(entries, checks))
    else:
        result = validation(entries, checks)
    result['checks'] = [dict(name=n, passed=bool(v)) for n, v in checks]
    result['passed'] = all(v for _, v in checks)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=result['passed'], checks=len(checks), failures=[n for n, v in checks if not v]), indent=2))
    raise SystemExit(0 if result['passed'] else 1)

if __name__ == '__main__':
    main()
