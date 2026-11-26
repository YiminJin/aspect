"""Check contacts, phase support and actual co-rotating velocity for an exact mature seed."""
import argparse
import csv
import math
from pathlib import Path

import numpy as np

from check import bulk, labels


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
    data = bulk(args.output,0)
    points = np.stack((data['x'],data['y']),axis=1)
    n, s = points@normal, points@tangent
    radius = np.loadtxt(args.output.parent/'profile.txt')[-1,0]
    dx = math.sqrt(9/len(n))  # one square metre, nine observed points per cell
    if not np.isfinite(data['phi']).all():
        raise ValueError('Missing or nonfinite phase')
    for station in (-.3,0.,.3):
        core = data['phi'][(abs(s-station)<dx) & (abs(n)<dx/4)]
        if not core.size or min(core) < .55:
            raise ValueError(f'No mature ridge at the expected station {station:g} m')
    outside = data['phi'][abs(n) > radius+math.sqrt(2)*dx]
    if not outside.size or max(abs(outside)) > 2e-14:
        raise ValueError('Phase extends past the analytic support and its last cut cell')

    nodes = ((0,0),(1,0),(0,1),(1,1),(0,.5),(1,.5),(.5,0),(.5,1),(.5,.5))
    count = [0,0]
    errors = [0.,0.]
    ranges = [[],[]]
    for path in args.output.glob('mesh-0001-r*.csv'):
        with path.open() as stream:
            for row in csv.DictReader(stream):
                d, i = int(row['component']), int(row['local'])
                if d > 1:
                    continue
                lo = np.array([float(row['x0']),float(row['y0'])])
                hi = np.array([float(row['x1']),float(row['y1'])])
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
    print(f'{args.degrees:g} degrees: contacts {vertices[[0,-1]].tolist()} m; '
          f'phase support verified; wall component ranges {[(min(v),max(v)) for v in ranges]} m/s, '
          f'max errors {errors} m/s ({count} prescribed local entries)')


if __name__ == '__main__':
    main()
