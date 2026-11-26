"""Compare an ordinary accepted strip-8 trajectory to the independent C1 native-point peak."""
import argparse
import csv
import math
from pathlib import Path
import sys


def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def strip8(path):
    rows = [r for r in read_rows(path) if int(r['section']) == 8]
    if len(rows) != 1:
        raise ValueError(f'Expected one strip-8 record in {path}')
    return rows[0]


def check(case, output):
    c1 = Path(__file__).resolve().parent.parent / 'direction-c1'
    peak, = (float(r['angle_rad']) for r in read_rows(c1/'peak-summary.csv')
             if r['case'] == case and r['method'] == 'independent')
    width, = (float(r['width_m']) for r in read_rows(c1/'peak-convergence.csv') if r['case'] == case)
    parent = strip8(output/'sections-0000.csv')
    if float(parent['ds']) != width or float(parent['beta']) != 0. or int(parent['formed']):
        raise ValueError('Expected the captured undeformed, unformed strip at the initial clock')

    # Compare to the arclength at the direct peak, independently of the angle
    # installed by ASPECT. Also test the geometry exported at the measured angle.
    length = width / math.cos(peak)
    angle_budget = 1e-7 + 1e-7 * abs(peak)
    length_budget = 1e-11 + 1e-7 * length
    writer = csv.writer(sys.stdout, lineterminator='\n')
    writer.writerow('case,time_s,beta_rad,peak_rad,beta_minus_peak_rad,ds_m,expected_ds_m,ds_minus_expected_m'.split(','))
    failures = []
    for step in range(1, 4 if case == 'frozen-2' else 2):
        row = strip8(output/f'sections-{step:04d}.csv')
        balance, = read_rows(output/f'balance-{step:04d}.csv')
        t, beta, ds = map(float, (balance['time'], row['beta'], row['ds']))
        if t != .005 * step or int(row['formed']) != 1:
            raise ValueError(f'Unexpected clock or formation at step {step}')
        if not math.isclose(ds, width / math.cos(beta), rel_tol=0., abs_tol=math.ulp(length)):
            raise ValueError(f'Published strip length disagrees with its accepted angle at {t} s')
        writer.writerow((case, t, beta, peak, beta-peak, ds, length, ds-length))
        if not (abs(beta-peak) <= angle_budget and abs(ds-length) <= length_budget):
            failures.append(f'{t} s: angle error {beta-peak:.12g} rad (budget {angle_budget:.12g}), '
                            f'length error {ds-length:.12g} m (budget {length_budget:.12g})')
    if failures:
        raise ValueError('Independent strip-8 peak comparison failed. ' + ' / '.join(failures))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', choices=['frozen-2', 'affine-1', 'affine-2'])
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    check(args.case, args.output)
