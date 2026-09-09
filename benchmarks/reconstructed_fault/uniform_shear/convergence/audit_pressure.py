#!/usr/bin/env python3
"""Read-only audit of the configured volume-pressure normalization.

Report raw pressure offsets; mean subtraction is used ONLY to distinguish
a gauge difference between two outputs from a spatial pressure difference.
No ASPECT output is normalized or replaced by this diagnostic.
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('debug',type=Path)
    parser.add_argument('release',type=Path)
    args = parser.parse_args()
    for directory in [args.debug,args.release]:
        parameter = re.search(r'^\s*set Pressure normalization\s*=\s*(\w+)',
                              (directory/'parameters.prm').read_text(), re.MULTILINE)
        if parameter is None or parameter[1] != 'volume':
            raise ValueError('This audit requires the actual configured volume normalization')
    report = []
    for path in sorted(args.debug.glob('bulk_*.csv')):
        suffix = path.stem.removeprefix('bulk_')
        if not suffix.isdecimal():
            continue
        a = np.genfromtxt(path,names=True,delimiter=',')
        b = np.genfromtxt(args.release/path.name,names=True,delimiter=',')
        if a.shape != b.shape or max(abs(a['x']-b['x'])) > 1e-14 or max(abs(a['y']-b['y'])) > 1e-14:
            raise ValueError('Not matching native quadrature-point coordinates')
        mean = lambda p: float(np.average(p,weights=a['weight']))
        difference = b['p']-a['p']
        time = np.atleast_1d(np.genfromtxt(args.debug/f'time_{suffix}.csv',names=True,delimiter=','))[0]['time']
        report.append(dict(time_s=float(time),debug_mean_Pa=mean(a['p']),release_mean_Pa=mean(b['p']),
                           raw_difference_max_Pa=float(max(abs(difference))),
                           constant_difference_Pa=mean(difference),
                           nonconstant_difference_max_Pa=float(max(abs(difference-mean(difference)))),
                           debug_pressure_range_Pa=float(np.ptp(a['p'])),
                           release_pressure_range_Pa=float(np.ptp(b['p']))))
    # Use the already documented traction absolute allowance to make the
    # failure unambiguous; this is not a new physical gauge or solver tolerance.
    passed = all(abs(r['debug_mean_Pa']) <= .015 and abs(r['release_mean_Pa']) <= .015 for r in report)
    print(json.dumps(dict(configured_normalization='volume',required_mean_Pa=0,
                          existing_traction_absolute_allowance_Pa=.015,
                          passed=passed,steps=report),indent=2))
    return 0 if passed else 2


if __name__ == '__main__':
    raise SystemExit(main())
