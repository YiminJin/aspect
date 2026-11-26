"""Check contacts, phase support and actual co-rotating velocity for an exact mature seed."""
import argparse
import csv
import math
from pathlib import Path

import numpy as np

from check import bulk, labels


def check_phase_support(output, tangent, normal, mesh_cells, dx):
    # The nine Gauss points of each cell are written by its MPI owner.
    # Match their global identities to the cells in the independent mesh export.
    ranks = sorted(int(path.stem.split('-')[-1]) for path in output.glob('bulk-0-*.txt'))
    if not ranks:
        raise ValueError('Missing initial bulk samples')
    shards = [bulk(output, 0, rank) for rank in ranks]
    data = np.sort(np.concatenate(shards), order=['cell', 'q'])
    cells, counts = np.unique(data['cell'], return_counts=True)
    if (set(cells) != mesh_cells or np.any(counts != 9) or
            not np.array_equal(data['q'], np.tile(np.arange(9), len(cells)))):
        raise ValueError('Bulk (cell, q) identities do not cover every mesh cell exactly once')
    points = np.stack((data['x'],data['y']),axis=1)
    n, s = points@normal, points@tangent
    radius = np.loadtxt(output.parent/'profile.txt')[-1,0]
    if not np.isfinite(points).all() or not np.isfinite(data['phi']).all():
        raise ValueError('Nonfinite bulk coordinates or phase')
    for station in (-.3,0.,.3):
        core = data['phi'][(abs(s-station)<dx) & (abs(n)<dx/4)]
        if not core.size or min(core) < .55:
            raise ValueError(f'No mature ridge at the expected station {station:g} m')
    outside = data['phi'][abs(n) > radius+math.sqrt(2)*dx]
    if not outside.size or max(abs(outside)) > 2e-14:
        raise ValueError('Phase extends past the analytic support and its last cut cell')
    return [len(shard) for shard in shards]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path,help='Output of run.py inclined --angle ... --mesh-plugin ...')
    parser.add_argument('degrees',type=float,help='Expected angle counterclockwise from laboratory +x')
    args = parser.parse_args()
    angle = math.radians(args.degrees)
    tangent = np.array([math.cos(angle),math.sin(angle)])
    normal = np.array([-tangent[1],tangent[0]])
    # Independent centered-square contact: the first component to reach a face
    # fixes the distance. Vertices in this control are neither fitted nor moved.
    end = .5*tangent/max(abs(tangent))
    for step in (0,1):
        vertices = labels(args.output,step)[:,1:3]
        np.testing.assert_allclose(vertices, np.linspace(-end,end,len(vertices)), rtol=0, atol=2e-15)
    nodes = ((0,0),(1,0),(0,1),(1,1),(0,.5),(1,.5),(.5,0),(.5,1),(.5,.5))
    count = [0,0]
    errors = [0.,0.]
    ranges = [[],[]]
    cells = set()
    widths = set()
    for path in args.output.glob('mesh-0001-r*.csv'):
        with path.open() as stream:
            for row in csv.DictReader(stream):
                d, i = int(row['component']), int(row['local'])
                if d > 1:
                    continue
                lo = np.array([float(row['x0']),float(row['y0'])])
                hi = np.array([float(row['x1']),float(row['y1'])])
                if d == 0 and i == 0:
                    cells.add(row['cell'])
                    widths.update(hi - lo)
                x, y = p = lo + (hi-lo)*nodes[i]
                if abs(x) != .5 and y != .5 and not (y == -.5 and d == 0):
                    continue  # bottom y is natural
                value = float(row['value'])
                if not math.isfinite(value):
                    raise ValueError(f'Nonfinite velocity component {d} at {p}')
                expected = .01*(normal@p)*tangent[d]
                errors[d] = max(errors[d],abs(value-expected))
                count[d] += 1
                ranges[d].append(value)
    if not all(count) or max(errors)>2e-12:
        raise ValueError(f'{args.degrees:g} degrees: wall velocity errors {errors}, counts {count}')
    if len(widths) != 1:
        raise ValueError('Expected square cells of uniform size')
    dx, = widths
    if dx <= 0 or not math.isclose(len(cells)*dx**2, 1., abs_tol=1e-14, rel_tol=0):
        raise ValueError('Mesh export does not cover the uniform unit square')
    samples = check_phase_support(args.output, tangent, normal, cells, dx)
    print(f'{args.degrees:g} degrees: contacts {vertices[[0,-1]].tolist()} m; '
          f'phase support verified with {samples} samples by rank; '
          f'wall component ranges {[(min(v),max(v)) for v in ranges]} m/s, '
          f'max errors {errors} m/s ({count} prescribed local entries)')


if __name__ == '__main__':
    main()
