"""Report, do not hide, changes from correcting the old BP3 boundary integral.

This script does not reset histories or retime either run. It compares matched
accepted indices and explicitly reports the physical-time mismatch. It does
not redefine old-trajectory equivalence when that comparison fails.
"""
import argparse
import json
from pathlib import Path
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('old', type=Path)
    parser.add_argument('corrected', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {'relative_comparison_coefficient': 1e-8, 'steps': {}}
    qualified = True
    for step in range(4):
        a = np.atleast_1d(np.genfromtxt(args.old/f'fault_{step}.csv', names=True, delimiter=','))
        b = np.atleast_1d(np.genfromtxt(args.corrected/f'fault_{step}.csv', names=True, delimiter=','))
        assert a.shape == b.shape
        assert np.array_equal(a['x'], b['x']) and np.array_equal(a['y'], b['y'])
        fields = {}
        for name in ('time','dt','V','Theta','C','Ih','q','slip','tau_bg','sigma_n_bg'):
            error = np.abs(a[name]-b[name])
            scale = float(np.max(np.abs(a[name])))
            maximum = float(error.max())
            passing = maximum <= 1e-8*scale
            qualified &= passing
            fields[name] = dict(max_absolute=maximum, old_max_scale=scale,
                                scaled_error=maximum/scale if scale else None,
                                worst_vertex=int(error.argmax()), within_numerical_tolerance=bool(passing))
        result['steps'][step] = fields
    result['old_trajectory_equivalent'] = bool(qualified)
    result['interpretation'] = ('Same accepted indices, not common-time errors: the unchanged timestep controller reacts to the corrected initialization.')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
