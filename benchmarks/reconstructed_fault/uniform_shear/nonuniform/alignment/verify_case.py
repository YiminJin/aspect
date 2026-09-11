#!/usr/bin/env python3
"""Check actual staggered geometry before reusing the accepted K2.3 checks."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'true-pressure'))
import verify_pilot


def geometry(case):
    rows = []
    for k in range(3):
        phase = verify_pilot.read(HERE / case / f'phase_{k}.csv')
        surface = verify_pilot.read(HERE / case / f'surface_{k}.csv')
        x, y = np.unique(phase['x']), np.unique(phase['y'])
        assert len(x) == 65 and len(y) == 256
        assert np.allclose(x, np.linspace(0, .25, 65), rtol=0, atol=1e-14)
        assert np.allclose(y, np.linspace(-.5, .5, 256), rtol=0, atol=1e-14)
        assert len(np.unique(phase[['x', 'y']])) == 65*256
        indices = np.searchsorted(y, surface['y'])
        assert np.all((indices > 0) & (indices < len(y)))
        lower, upper = y[indices-1], y[indices]
        xi = (surface['y']-lower)/(upper-lower)
        row = dict(step=k, nx=len(x)-1, ny=len(y)-1,
                   hx_range_m=[float(min(np.diff(x))), float(max(np.diff(x)))],
                   hy_range_m=[float(min(np.diff(y))), float(max(np.diff(y)))],
                   fault_y_range_m=[float(min(surface['y'])), float(max(surface['y']))],
                   normal_reference_coordinate_range=[float(min(xi)), float(max(xi))],
                   containing_row_bounds_m=[float(min(lower)), float(max(upper))],
                   surface_nodes=len(surface))
        rows.append(row)
        # Record the measured geometry even if its intended placement fails.
        (HERE / f'{case}-geometry.json').write_text(json.dumps(rows, indent=2)+'\n')
        assert len(surface) == 33 and np.all(surface['fault'] == 0)
        assert max(abs(surface['y'])) < 1e-12, 'Reconstruction departed from y=0'
        assert np.all((xi > 0) & (xi < 1)), 'Fault is not in normal cell interiors'
        assert np.allclose(surface['x'], np.linspace(0, .25, 33), rtol=0, atol=1e-12)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', choices=('pilot', 'homogeneous'))
    args = parser.parse_args()
    measured = geometry(args.case)
    print('Actual geometry:', json.dumps(measured), flush=True)
    # Only relocate input/output files; retain every established convergence,
    # history, constitutive, containment and actual-normalization assertion.
    verify_pilot.base = HERE
    result = verify_pilot.check(args.case, runtime_limit=300)
    print(json.dumps(result, indent=2))
