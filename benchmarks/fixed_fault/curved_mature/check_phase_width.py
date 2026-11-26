"""Measure the initial DG phase against independent finite-segment and Q2 references."""
import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from check_curve import Fields, table


BACKGROUND = .03
EXCESS = .32
WIDTH = .15  # metres
EFOLD = BACKGROUND + EXCESS / np.e


def read_vertices(path):
    records = [line.partition('#')[0].split() for line in path.read_text().splitlines()]
    header, *rows = [r for r in records if r]
    np.testing.assert_array_equal([int(r[header.index('id')]) for r in rows], np.arange(len(rows)))
    return np.array([[float(r[header.index(c)]) for c in ('x_m', 'y_m')] for r in rows])


def feet(points, vertices):
    """Clamp to each closed segment, including its two endpoints."""
    edges = np.diff(vertices, axis=0)
    delta = np.atleast_2d(points)[:, None, :] - vertices[:-1]
    fraction = np.einsum('psi,si->ps', delta, edges) / np.einsum('si,si->s', edges, edges)
    fraction = np.clip(fraction, 0., 1.)
    distances = np.linalg.norm(delta - fraction[:, :, None] * edges, axis=2)
    which = distances.argmin(axis=1)
    rows = np.arange(len(which))
    return which, fraction[rows, which], distances[rows, which]


def gaussian(distance):
    return BACKGROUND + EXCESS * np.exp(-(distance / WIDTH)**2)


def interpolant(points, vertices, n, vertical=False):
    """Tensor-product interpolation of analytic values at the cell's nine DGQ2 nodes.

    The native output is at Gauss points. This reference instead starts with the
    analytic field at corners, edge midpoints and cell center, then interpolates.
    """
    nodes = np.array([0., .5, 1.])
    points = np.atleast_2d(points)
    cell = np.clip(np.floor(points*n).astype(int), 0, n-1)
    local = points*n - cell
    x = (cell[:, 0, None, None] + nodes[None, :, None]) / n
    y = (cell[:, 1, None, None] + nodes[None, None, :]) / n
    coords = np.stack(np.broadcast_arrays(x, y), axis=-1).reshape(-1, 2)
    if vertical:
        d = coords[:, 1] - np.interp(coords[:, 0], vertices[:, 0], vertices[:, 1])
    else:
        _, _, d = feet(coords, vertices)
    values = gaussian(d).reshape(-1, 3, 3)

    def cardinal(t):
        return np.stack(((1-2*t)*(1-t), 4*t*(1-t), t*(2*t-1)), axis=-1)

    return np.einsum('pi,pij,pj->p', cardinal(local[:, 0]), values, cardinal(local[:, 1]))


def output_vertices(output, expected, ranks):
    clocks = np.loadtxt(output/'statistics', usecols=(0, 1), ndmin=2)
    np.testing.assert_array_equal(clocks[0], [0, 0])
    samples_per_rank = []
    for owner in range(ranks):
        rows = table(output/f'material-0-surface-{owner}.csv')
        np.testing.assert_array_equal([int(r['i']) for r in rows], np.arange(len(expected)))
        np.testing.assert_array_equal([[float(r[a]) for a in ('x', 'y')] for r in rows], expected)
        samples_per_rank.append(len(table(output/f'material-0-rank{owner}.csv')))
    if min(samples_per_rank) == 0 or len(list(output.glob('material-0-rank*.csv'))) != ranks:
        raise ValueError(f'Expected volume samples on all {ranks} owners; counts {samples_per_rank}')


def cut_distances(phi, origin, direction, width):
    """Signed location of the peak, and the two positive distances from the trace to EFOLD."""
    def sample(offset):
        return phi(origin + offset*direction)[0]
    peak = minimize_scalar(lambda r: -sample(r), bounds=(-.025, .025), method='bounded',
                           options={'xatol': 1e-12})
    roots = [brentq(lambda r: sample(sign*r) - EFOLD, width-.009, width+.009, xtol=1e-13)
             for sign in (-1., 1.)]
    return [peak.x, *roots]


def check_field(output, v, ranks=1, vertical=False):
    output_vertices(output, v, ranks)
    f = Fields(output, 0)
    # Compare by physical cell and coordinates, independently of MPI shard order.
    ref = interpolant(f.xy, v, f.n, vertical)
    mismatch = np.max(np.abs(f.at('phi', f.xy) - ref))
    if not np.isfinite(mismatch) or mismatch > 2e-13:
        raise ValueError(f'Initialized DGQ2 phase differs from its vertex-table reference by {mismatch:.9g}')
    phi = lambda xy: f.at('phi', xy)
    edges = np.diff(v, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    tangents = edges/lengths[:, None]
    cuts = []
    for j, tangent in enumerate(tangents):
        normal = np.array([-tangent[1], tangent[0]])
        origin = .5*(v[j]+v[j+1])
        width = WIDTH*abs(tangent[0]) if vertical else WIDTH
        offsets = np.linspace(-width-.009, width+.009, 65)
        xy = origin + offsets[:, None]*normal
        if np.any((xy < 0) | (xy > 1)) or np.any(feet(xy, v)[0] != j):
            raise ValueError(f'Normal cut for segment {j} leaves its nearest segment or box')
        np.testing.assert_allclose(phi(xy), interpolant(xy, v, f.n, vertical), rtol=0, atol=2e-13)
        np.testing.assert_allclose(phi(xy), gaussian(offsets*WIDTH/width), rtol=0,
                                   atol=.002 if vertical else .0008)
        measured = cut_distances(phi, origin, normal, width)
        # The full independent nodal polynomial above determines the FE error.
        # On this 16x16 mesh also require physical ridge and width accuracy.
        if abs(measured[0]) > .0007 or max(abs(np.array(measured[1:])-width)) > .0003:
            raise ValueError(f'Segment {j} ridge/widths {measured} m, expected (0,{width},{width})')
        cuts.append(measured)

    specials = {}
    if not vertical:
        # The outward sector of a turn has the vertex itself as its nearest
        # point. Probe it and the exterior cap of the left contact inside the box.
        endpoint = v[0] + [0., .1125]
        points = [('left cap', 0, endpoint)]
        j = len(v)//2
        t0, t1 = tangents[j-1:j+1]
        turn = t0[0]*t1[1] - t0[1]*t1[0]
        if abs(turn) > 1e-8:
            bisector = t0+t1
            outward = -np.sign(turn)*np.array([-bisector[1], bisector[0]])/np.linalg.norm(bisector)
            points.append(('outer bend', j, v[j]+.1125*outward))
        for label, i, point in points:
            k, fraction, d = feet(point, v)
            foot = v[k]+fraction[:, None]*edges[k]
            np.testing.assert_allclose(foot[0], v[i], atol=1e-14, rtol=0)
            actual = phi(point)[0]
            analytic = gaussian(d)[0]
            if abs(actual-analytic) > .0015:
                raise ValueError(f'{label}: sampled phase {actual}, expected {analytic}')
            specials[label] = (point, actual, analytic)
    return f, np.array(cuts), specials, mismatch


def show(name, result):
    f, cuts, specials, mismatch = result
    print(f'{name}: {f.n} x {f.n} cells; maximum independent DGQ2 phase error {mismatch:.3g}')
    for j, (peak, minus, plus) in enumerate(cuts):
        print(f'  segment {j}: ridge offset {peak:+.8f} m; negative/positive widths {minus:.8f} / {plus:.8f} m')
    for label, (point, actual, analytic) in specials.items():
        print(f'  {label} at ({point[0]:.8f}, {point[1]:.8f}) m: phi_FE {actual:.10f}, phi_exact {analytic:.10f}')


def plot(path, runs, vertices):
    import matplotlib.pyplot as plt

    fig, (width, shift) = plt.subplots(1, 2, figsize=(9, 3.6), layout='constrained')
    colors = ['#1c5975', '#af4a21', '#68787c']
    for i, name in enumerate(('original', 'moved', 'vertical')):
        v = vertices[name]
        ds = np.linalg.norm(np.diff(v, axis=0), axis=1)
        stations = np.cumsum(ds) - .5*ds
        widths = runs[name][1][:, 1:]
        for side in range(2):
            width.plot(stations, 1000*widths[:, side], color=colors[i],
                       ls=['-', '--'][side], marker=['o', 'x'][side], ms=4,
                       label=(['Original', 'Moved vertex', 'Vertical fixture'][i] if side == 0 else None))
    width.axhline(1000*WIDTH, lw=.8, color='.35', zorder=-1)
    width.set(xlabel='Segment midpoint arclength (m)', ylabel='Normal e-fold distance (mm)',
              title='Width on both sides at time zero')
    width.legend(fontsize=8)
    ys = np.linspace(.34, .90, 500)
    for i, name in enumerate(('original', 'moved')):
        f = runs[name][0]
        v = vertices[name]
        x, y = v[2]
        p = f.at('phi', np.column_stack([np.full_like(ys, x), ys]))
        shift.plot(ys, p, color=colors[i], label=['Original', 'Moved vertex'][i])
        shift.axvline(y, color=colors[i], lw=1, ls=':')
    shift.set(xlabel='Height at x = 0.50 m (m)', ylabel='Initialized phase',
              title='Ridge follows vertex 2')
    shift.legend(fontsize=8)
    fig.savefig(path, dpi=190)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for k in ('original', 'vertical', 'cases'):
        parser.add_argument(k, type=Path)
    parser.add_argument('--vertices', type=Path, default=Path(__file__).with_name('vertices.txt'))
    parser.add_argument('--figure', type=Path)
    args = parser.parse_args()
    original = read_vertices(args.vertices)
    moved_file = args.cases/'moved/vertices.txt'
    # Include property columns: coordinate-only input changes matter as much
    # as matching ordered IDs when comparing the two solved initial states.
    old_tokens = args.vertices.read_text().split()
    new_tokens = moved_file.read_text().split()
    if (len(old_tokens) != len(new_tokens) or
            sum(a != b for a, b in zip(old_tokens, new_tokens)) != 1):
        raise ValueError('Expected exactly one changed value in the copied vertex table')
    moved = read_vertices(moved_file)
    changed = np.argwhere(original != moved)
    np.testing.assert_array_equal(changed, [[2, 1]])
    np.testing.assert_allclose(moved[2, 1]-original[2, 1], -.12, atol=1e-14, rtol=0)
    straight = read_vertices(args.cases/'straight/vertices.txt')
    np.testing.assert_allclose(straight[:, 1], np.interp(straight[:, 0], original[[0, -1], 0],
                                                        original[[0, -1], 1]), atol=1e-14, rtol=0)
    directories = {'original': args.original, 'vertical': args.vertical,
                   'moved': args.cases/'moved/output', 'straight': args.cases/'straight/output'}
    geometry = {'original': original, 'vertical': original, 'moved': moved, 'straight': straight}
    runs = {}
    for name, output in directories.items():
        runs[name] = check_field(output, geometry[name], vertical=name=='vertical')
        show(name, runs[name])
    for name in ('original', 'moved'):
        v = geometry[name]
        parallel = check_field(args.cases/f'{name}-mpi/output', v, ranks=2)
        serial = runs[name][0]
        # Identical physical Gauss samples on both decompositions, before
        # comparing interpolated features, so rank-dependent omissions fail.
        np.testing.assert_allclose(parallel[0].at('phi', serial.xy), serial.at('phi', serial.xy),
                                   atol=2e-13, rtol=0)
        print(f'{name} serial/2-rank maximum sampled phase difference '
              f'{np.max(np.abs(parallel[0].at("phi", serial.xy)-serial.at("phi", serial.xy))):.3g}')
    # This solver run used the moved mechanical table with a stale copy of
    # the original five-segment phase function. It reached the time-zero
    # output; the independent check must diagnose phase, not reject the trace.
    stale = args.cases/'stale/output'
    old_on_moved = Fields(stale, 0)
    np.testing.assert_allclose(old_on_moved.at('phi', old_on_moved.xy),
                               runs['original'][0].at('phi', old_on_moved.xy), atol=2e-13, rtol=0)
    try:
        check_field(stale, moved)
    except ValueError as err:
        if 'Initialized DGQ2 phase differs' not in str(err):
            raise
        print(f'Stale phase rejected: {err}')
    else:
        raise ValueError('Stale phase passed the moved geometry check')
    peaks = []
    for name in ('original', 'moved'):
        f = runs[name][0]
        x, y = geometry[name][2]
        peak = minimize_scalar(lambda yy: -f.at('phi', [x, yy])[0],
                               bounds=(y-.025, y+.025), method='bounded',
                               options={'xatol': 1e-12})
        if abs(peak.x-y) > .002:
            raise ValueError(f'{name}: FE vertical-cut peak at {peak.x} m, vertex at {y} m')
        peaks.append(peak.x)
    print(f'Vertex-2 x=0.5 m cut peaks: {peaks[0]:.9f} -> {peaks[1]:.9f} m; '
          f'measured displacement {peaks[1]-peaks[0]:.9f} m')
    if args.figure:
        plot(args.figure, runs, geometry)


if __name__ == '__main__':
    main()
