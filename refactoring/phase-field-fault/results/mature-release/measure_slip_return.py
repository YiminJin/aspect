"""Measure area returned by a unit mature slip speed on bent and straight traces."""
import argparse
import csv
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from numpy.polynomial.legendre import leggauss


CASES = ('horizontal', 'straight-chord', 'public-bend')
COLUMNS = ('line_m', 'Q8_m', 'Q8_independent_m', 'cut_m', 'independent_m',
           'all_eligible_m', 'defect_m', 'overlap_loss_m', 'negative_side_loss_m')
REFERENCE = Path(__file__).with_name('slip-return.csv')


def read_inventory(path):
    """Index the scalar return (metres) by geometry and material vertex ID."""
    rows = {}
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['case', 'id', *COLUMNS]:
            raise ValueError(f'{path}: unexpected station inventory header')
        for row in reader:
            if None in row or None in row.values():
                raise ValueError(f'{path}: incomplete or extra columns on row {reader.line_num}')
            key = (row['case'], int(row['id']))
            if key in rows:
                raise ValueError(f'{path}: duplicate station {key}')
            values = {column: float(row[column]) for column in COLUMNS}
            if not all(map(math.isfinite, values.values())):
                raise ValueError(f'{path}: nonfinite station inventory at {key}')
            rows[key] = values
    return rows


def compare_inventory(candidate_path):
    expected = read_inventory(REFERENCE)
    candidate = read_inventory(candidate_path)
    if candidate.keys() != expected.keys():
        raise ValueError(f'Inventory station identities differ: missing {sorted(expected.keys()-candidate.keys())}, '
                         f'unexpected {sorted(candidate.keys()-expected.keys())}')
    for (case, label), row in expected.items():
        for column, reference in row.items():
            value = candidate[case, label][column]
            # Use the mature cross-build budget, scaled by the frozen value.
            tolerance = 1.e-10 + 1.e-8*abs(reference)
            if abs(value-reference) > tolerance:
                raise ValueError(f'{case} ID {label} {column}: candidate {value:.17g}, '
                                 f'reference {reference:.17g}, difference {value-reference:.9g} m '
                                 f'exceeds {tolerance:.9g} m')
    print(f'Frozen station inventory passed for {len(expected)} (case, ID) rows and {len(COLUMNS)} columns.')


def run_cases(prefix, plugins, destination):
    """Change only the vertex coordinates between controls, keeping the ID properties."""
    case = prefix/'benchmarks/fixed_fault/curved_mature'
    destination.mkdir(parents=True, exist_ok=False)
    env = os.environ | dict(ASPECT_SOURCE_DIR=str(prefix), OMP_NUM_THREADS='1',
                            OPENBLAS_NUM_THREADS='1', DEAL_II_NUM_THREADS='1')
    content = [s.split('#', 1)[0].split() for s in (case/'vertices.txt').read_text().splitlines()]
    header, *rows = [r for r in content if r]
    xcol, ycol = header.index('x_m'), header.index('y_m')
    x0, x1 = [float(r[xcol]) for r in (rows[0], rows[-1])]
    y0, y1 = [float(r[ycol]) for r in (rows[0], rows[-1])]
    for name in CASES:
        directory = destination/name
        directory.mkdir()
        vertices = directory/'vertices.txt'
        records = [r.copy() for r in rows]
        for r in records:
            if name == 'horizontal':
                r[ycol] = '0.5'
            elif name == 'straight-chord':
                t = (float(r[xcol])-x0)/(x1-x0)
                r[ycol] = str((1-t)*y0 + t*y1)
        vertices.write_text('\n'.join(' '.join(r) for r in [header, *records])+'\n')
        prm = f'''include {case}/curved_mature.prm
set Output directory = output
set End time = 0
set Additional shared libraries = {", ".join(map(str, plugins))}
subsection Material model
  subsection Phase field fault
    set Fault vertex file = {vertices}
  end
end
subsection Postprocess
  set List of postprocessors = moving fault samples, integrated slip transfer
end
'''
        (directory/'case.prm').write_text(prm)
        with (directory/'run.log').open('w') as log:
            subprocess.run([prefix/'bin/aspect-release', 'case.prm'], cwd=directory, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)


def ray_rule(origin, tangent, normal, length, planes, n_grid, order, integrate):
    """Integrate in physical (arclength, normal distance), cutting at each plane A.x=b.

    Cell faces bound the smooth FE polynomials. Projected strip ends and equal-distance
    lines bound regions of unique admission. Between consecutive along-ray stations
    where two planes cross, each intersection moves linearly with arclength.
    """
    basis = np.stack([tangent, normal])
    axes = np.repeat(np.eye(2), n_grid+1, axis=0)
    levels = np.tile(np.linspace(0., 1., n_grid+1), 2)
    directions, rhs = map(np.array, zip(*planes))
    a = np.vstack([axes, directions])
    b = np.r_[levels, rhs]
    # Transform every line to a_s s + a_d d = b' in the segment's own frame.
    along, across = basis @ a.T
    offsets = b - a@origin
    transverse = abs(across) > 1.e-13
    valid_vertical = (~transverse) & (abs(along) > 1.e-13)
    crossings = list(offsets[valid_vertical]/along[valid_vertical])
    slope = -along[transverse]/across[transverse]
    intercept = offsets[transverse]/across[transverse]
    for i in range(len(slope)):
        delta = slope[i]-slope[:i]
        ok = abs(delta)>1.e-12
        s = (intercept[:i][ok]-intercept[i])/delta[ok]
        d = intercept[i]+slope[i]*s
        x = origin+s[:, None]*tangent+d[:, None]*normal
        crossings.extend(s[np.all((x>=-1.e-12) & (x<=1.+1.e-12), axis=1)])

    cuts = np.unique(np.clip([0., length, *crossings], 0., length))
    cuts = cuts[np.r_[True, np.diff(cuts) > 1.e-13]]
    z, w = leggauss(order)
    z, w = (1+z)/2, w/2
    # Nodes never lie on a cell face or an admission edge.
    s = (cuts[:-1, None] + np.diff(cuts)[:, None]*z).ravel()
    ws = (np.diff(cuts)[:, None]*w).ravel()
    result = np.zeros((3, 2))
    for first in range(0, len(s), 128):
        st = s[first:first+128]
        center = origin + st[:, None]*tangent
        box = [(np.array([0., 1.])-center[:, d, None])/normal[d] for d in range(2)
               if abs(normal[d])>1.e-13]
        lo = np.maximum.reduce([h.min(axis=1) for h in box])
        hi = np.minimum.reduce([h.max(axis=1) for h in box])
        edges = np.sort(np.clip(intercept+st[:, None]*slope, lo[:, None], hi[:, None]), axis=1)
        widths = np.diff(edges)
        ray, part = np.nonzero(widths > 1.e-13)
        width = widths[ray, part]
        dd = edges[ray, part, None]+width[:, None]*z
        points = center[ray, None, :] + dd[:, :, None]*normal
        midpoint_offset = edges[ray, part]+.5*width
        mid = center[ray]+midpoint_offset[:, None]*normal
        all_weight, admitted = integrate(st[ray]/length, points, mid)
        area_integral = (all_weight@w)*width*ws[first+ray]
        shape = np.array([1-st[ray]/length, st[ray]/length])
        result[0] += shape @ (area_integral * admitted)
        result[1] += shape @ area_integral
        result[2] += shape @ (area_integral * (~admitted) * (midpoint_offset < 0.))
    return result


def independent_area(fields, vertices, integrals, order, association, compliance):
    """Integrate the winning strips, all strips, and the loss on their negative-normal side."""
    edges = np.diff(vertices, axis=0)
    length = np.linalg.norm(edges, axis=1)
    t = edges/length[:, None]
    n = np.column_stack([-t[:, 1], t[:, 0]])
    admitted = np.zeros(len(vertices))
    all_rays = np.zeros(len(vertices))
    negative_loss = np.zeros(len(vertices))
    for j in range(len(length)):
        cuts = [(n[j], n[j]@vertices[j])]
        for k in range(len(length)):
            if k != j:
                cuts.extend([(t[k], t[k]@p) for p in vertices[k:k+2]])
                for sign in (-1, 1):
                    cuts.append((n[j]+sign*n[k], n[j]@vertices[j]+sign*(n[k]@vertices[k])))

        def weights(xi, points, mid):
            phi = fields.at('phi', points.reshape(-1, 2)).reshape(points.shape[:2])
            normalization = (1-xi)*integrals[j] + xi*integrals[j+1]
            winning_segment, _, count = association(mid, vertices)
            if np.any(count == 0):
                raise ValueError('Point on a complete in-box ray must have at least one eligible foot')
            return compliance(phi)/normalization[:, None], winning_segment == j

        selected, complete, lost = ray_rule(vertices[j], t[j], n[j], length[j], cuts, fields.n, order, weights)
        admitted[j:j+2] += selected
        all_rays[j:j+2] += complete
        negative_loss[j:j+2] += lost
    return admitted, all_rays, negative_loss


def native_section_rule(fields, vertices, integrals, association, cell_rule, compliance):
    """Independently integrate the same Q8 measure used in section assembly."""
    xy, cell_weights = cell_rule(8, fields.n)
    segment, xi, count = association(xy, vertices)
    ray_integral = (1-xi)*integrals[segment]+xi*integrals[segment+1]
    chi = np.where(count > 0, compliance(fields.at('phi', xy))/ray_integral, 0.)
    weights = np.tile(cell_weights.ravel(), fields.n**2)/fields.n**2
    n = len(vertices)
    return (np.bincount(segment, weights*chi*(1-xi), minlength=n) +
            np.bincount(segment+1, weights*chi*xi, minlength=n))


def measure(prefix, destination):
    sys.path.insert(0, str(prefix/'benchmarks/fixed_fault/curved_mature'))
    from check_curve import Fields, association, cell_rule, compliance, normal_integrals, table

    candidate_path = destination/'slip-return-candidate.csv'
    if candidate_path.exists() and candidate_path.samefile(REFERENCE):
        raise ValueError(f'Candidate output aliases the frozen station table: {candidate_path}')
    with candidate_path.open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(('case', 'id', *COLUMNS))
        for name in CASES:
            out = destination/name/'output'
            rows = table(out/'slip-transfer-0.csv')
            surface = table(out/'material-0-surface-0.csv')
            np.testing.assert_array_equal([int(r['id']) for r in rows], np.arange(len(rows)))
            vertices = np.array([(float(r['x_m']), float(r['y_m'])) for r in rows])
            np.testing.assert_array_equal(vertices, [(float(r['x']), float(r['y'])) for r in surface])
            q8, cut, integrals = [np.array([float(r[col]) for r in rows]) for col in
                                  ('Q8_m', 'cut_m', 'normal_integral_m')]
            if not all(np.isfinite(array).all() for array in (vertices, q8, cut, integrals)):
                raise ValueError(f'{name}: nonfinite accepted trace or scalar weight')
            expected_vertices = np.genfromtxt(destination/name/'vertices.txt', names=True)
            np.testing.assert_array_equal(vertices, np.column_stack([expected_vertices['x_m'],
                                                                     expected_vertices['y_m']]))
            fields = Fields(out, 0)
            edges = np.diff(vertices, axis=0)
            length = np.linalg.norm(edges, axis=1)
            independent_I = normal_integrals(vertices, edges/length[:, None], length, fields, 24)
            np.testing.assert_allclose(integrals, independent_I, atol=2e-13, rtol=0.)
            discrete = native_section_rule(fields, vertices, independent_I, association, cell_rule, compliance)
            np.testing.assert_allclose(q8, discrete, atol=2e-13, rtol=0.,
                                       err_msg=f'{name}: native Q8 per-station return')
            # The geometric reference is exact: each incident linear hat has
            # a line integral equal to half the length of its segment.
            line = (np.r_[0., length] + np.r_[length, 0.])/2
            coarse = independent_area(fields, vertices, independent_I, 6, association, compliance)
            fine = independent_area(fields, vertices, independent_I, 10, association, compliance)
            np.testing.assert_allclose(coarse, fine, atol=5e-9, rtol=0.)
            selected, complete, negative_loss = fine
            np.testing.assert_allclose(cut, selected, atol=1e-8, rtol=0.)
            if name == 'horizontal':
                np.testing.assert_allclose(q8, line, atol=1e-10, rtol=0.)
            if name != 'public-bend':
                np.testing.assert_allclose(complete, selected, atol=1e-12, rtol=0.)
            for i in range(len(line)):
                writer.writerow((name, i, line[i], q8[i], discrete[i], cut[i], selected[i], complete[i],
                                 cut[i]-line[i], complete[i]-selected[i], negative_loss[i]))
            print(f'{name}: line {line.sum():.12g} m, Q8 {q8.sum():.12g} m, '
                  f'cut {cut.sum():.12g} m, full normal strips {complete.sum():.12g} m; '
                  f'Q8 vs direct {max(abs(q8-discrete)):.3g} m, '
                  f'max station cut vs independent {max(abs(cut-selected)):.3g} m, '
                  f'Q6 vs Q10 {np.max(abs(np.array(coarse)-fine)):.3g} m', flush=True)
    compare_inventory(candidate_path)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('prefix', 'layered_plugin', 'transfer_plugin', 'destination'):
        p.add_argument(arg, type=lambda s: Path(s).resolve())
    p.add_argument('--replay', action='store_true', help='measure saved time-zero accepted outputs')
    args = p.parse_args()
    if not args.replay:
        run_cases(args.prefix, (args.layered_plugin, args.transfer_plugin), args.destination)
    measure(args.prefix, args.destination)
