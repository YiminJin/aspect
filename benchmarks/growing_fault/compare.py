"""Compare every accepted target frame with the frozen nine-strip native output.

Point identities are (ASPECT CellId, chord-rule index). tau has units Pa, A
and engineering shear are dimensionless. Section xi, ds and slip are metres,
V is the signed area-averaged rate in m/s and force is the signed RSF reaction
in Pa. Each energy column is work per metre out of plane. No rank ordering
or unit conversion enters these comparisons.

nine-strip-geometry.csv records the accepted geometry at three clocks from the frozen
two-rank C1 candidate run (bffffca07, frozen-arclength.csv in the C1 results).
Only the angle and arclength of strip 8 use that reference. The donor columns
identify the original localized-onecut records. Use --historical-donor to
compare against the supplied donor trajectory in full.
"""
import argparse
import csv
import gzip
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np


def rows(paths, keys):
    table = {}
    for path in paths:
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt') as stream:
            for row in csv.DictReader(stream):
                key = tuple(row[k] if k == 'cell' else int(row[k]) for k in keys)
                if key in table:
                    raise ValueError(f'Duplicate {key} in {path}')
                table[key] = row
    if not table:
        raise ValueError('Missing or empty output')
    return table


def allowance(reference, atol, rtol):
    return atol + rtol * abs(reference)


def compare_values(a, b, column, atol, rtol):
    if a.keys() != b.keys():
        raise ValueError(f'{column} identities differ')
    keys = sorted(a)
    left, right = (np.array([float(table[k][column]) for k in keys]) for table in (a, b))
    tolerance = allowance(left, atol, rtol)
    passing = np.isfinite(left) & np.isfinite(right) & (abs(left-right) <= tolerance)
    if not passing.all():
        i = np.flatnonzero(~passing)[0]
        raise ValueError(f'{column} at {keys[i]}: reference {left[i]:.17g}, target {right[i]:.17g}, '
                         f'difference {right[i]-left[i]:.5g}, tolerance {tolerance[i]:.5g}')
    return max(abs(left-right))


def plane_tolerances(column):
    return (1e-7 if column == 'beta' else 1e-11), 1e-7


def reference_planes(reference, step, time, historical_donor):
    planes = rows([reference/f'tip-planes-{step:04}.csv'], ('section',))
    if historical_donor:
        return planes

    geometry = rows([Path(__file__).with_name('nine-strip-geometry.csv')], ('step', 'section'))
    if set(geometry) != {(1, 8), (2, 8), (3, 8)}:
        raise ValueError('Expected corrected geometry only for strip 8 at steps 1, 2 and 3')
    expected = geometry[step, 8]
    strip = planes[8,]
    if (float(expected['time_s']) != float(time) or
        float(expected['donor_beta_rad']) != float(strip['beta']) or
        float(expected['donor_ds_m']) != float(strip['ds'])):
        raise ValueError(f'step {step}: corrected geometry requires the frozen two-rank donor')
    strip['beta'] = expected['beta_rad']
    strip['ds'] = expected['ds_m']
    return planes


def frozen(reference, target, archive=False, historical_donor=False):
    donor = rows([reference/'bulk-accepted.csv'], ('step',))
    actual = rows(target.glob('balance-000[1-9].csv'), ('step',))
    if set(donor) != {(1,), (2,), (3,)} or donor.keys() != actual.keys():
        raise ValueError('Expected one publication at each of the three physical endpoints')
    # The portable archive retains the last native frame and the full ledger.
    # The replay additionally supplies every intermediate material frame.
    for step in ([3] if archive else range(4)):
        tag = f'{step:04}'
        old = rows(reference.glob(f'tip-points-{tag}-r*.csv*'), ('cell', 'q'))
        new = rows(target.glob(f'points-{tag}-r*.csv'), ('cell', 'q'))
        for name in ('section', 'x', 'y', 'weight', 'phi', 'tauxx', 'tauyy', 'tauzz', 'tauxy',
                     'Axx', 'Ayy', 'Azz', 'Axy', 'signed_shear', 'absolute_shear'):
            atol = (1e-6 if name.startswith('tau') else 1e-9 if name=='phi' else
                    1e-10 if name.startswith(('A', 'signed', 'absolute')) else 1e-14)
            try:
                compare_values(old, new, name, atol, 1e-7 if atol>=1e-10 else 0.)
            except ValueError as exc:
                raise ValueError(f'step {step}, native points: {exc}') from exc
        if not step:
            continue
        old = reference_planes(reference, step, donor[step,]['time'], historical_donor)
        new = rows([target/f'sections-{tag}.csv'], ('section',))
        parent = rows([target/f'sections-{step-1:04}.csv'], ('section',))
        for k in old:
            if old[k]['parent_formed'] != parent[k]['formed'] or old[k]['formed'] != new[k]['formed']:
                raise ValueError(f'Step {step}, strip {k}: formed plane history differs')
        for name in ('xi', 'ds', 'beta', 'age', 'slip'):
            try:
                compare_values(old, new, name, *plane_tolerances(name))
            except ValueError as exc:
                raise ValueError(f'step {step}, strip geometry/history: {exc}') from exc
        with (reference/f'tip-sections-{tag}.csv').open() as stream:
            rates = {(i,): r for i, r in enumerate(csv.DictReader(stream))}
        for name in ('V', 'force'):
            compare_values(rates, new, name, 1e-11 if name=='V' else 1e-6, 1e-7)
        slip = sum(float(row['slip'])*float(row['ds']) for row in new.values())
        print(f'{step}: {len(new)} strips, integrated slip {slip:.12g} m²')
    for key in sorted(donor):
        for name in ('time','bulk','cohesive','Wext','Db','Dc','Df','phase_remainder'):
            compare_values({key:donor[key]}, {key:actual[key]}, name,
                           1e-15 if name=='time' else 1e-11, 0. if name=='time' else 1e-6)
        for name, tol in [('mechanical_residual',1e-9),('phase_KKT',1e-7),('adjoint_error',1e-12)]:
            if not abs(float(actual[key][name])) <= tol:
                raise ValueError(f'{key} {name} outside tolerance')
    old, new = (np.diff([float(t[k]['fracture']) for k in sorted(t)]) for t in (donor, actual))
    np.testing.assert_allclose(new,old,rtol=1e-6,atol=1e-11,equal_nan=False)


def continuation(first, other, steps):
    for step in steps:
        tag=f'{step:04}'
        for pattern,keys in [(f'fields-{tag}-r*.csv',('cell','component','local')),
                             (f'points-{tag}-r*.csv',('cell','q')),
                             (f'sections-{tag}.csv',('section',)),(f'balance-{tag}.csv',('step',))]:
            a,b=(rows(path.glob(pattern),keys) for path in (first,other))
            for column in set(next(iter(a.values())))-set(keys):
                compare_values(a,b,column,1e-12,1e-12)
        print(f'Continuation matched at step {step}')


def check_corruption(reference, target, historical_donor=False):
    # Each trial modifies one value in a fresh copy and must identify its
    # clock, column and (for the revised geometry) material strip.
    frozen(reference, target, historical_donor=historical_donor)
    point_file = sorted(target.glob('points-0001-r*.csv'))[0]
    with point_file.open() as stream:
        tauzz = float(next(csv.DictReader(stream))['tauzz'])
    trials = [(point_file, 'tauzz', tauzz + .001, 'step 1, native points: tauzz')]
    donor = rows([reference/'bulk-accepted.csv'], ('step',))
    for step in (1, 2, 3):
        strip = reference_planes(reference, step, donor[step,]['time'], historical_donor)[8,]
        for column in ('beta', 'ds'):
            expected = float(strip[column])
            outside = expected + 2 * allowance(expected, *plane_tolerances(column))
            trials.append((target/f'sections-{step:04}.csv', column, outside,
                           f'step {step}, strip geometry/history: {column} at (8,)'))

    for damaged, column, replacement, diagnostic in trials:
        with TemporaryDirectory() as directory:
            scratch = Path(directory)
            for path in target.glob('*.csv'):
                if path != damaged:
                    (scratch/path.name).symlink_to(path.resolve())
            with damaged.open() as source, (scratch/damaged.name).open('w') as output:
                reader = csv.DictReader(source)
                records = list(reader)
                row = records[0] if column == 'tauzz' else next(r for r in records if int(r['section']) == 8)
                row[column] = str(replacement)
                writer = csv.DictWriter(output, reader.fieldnames)
                writer.writeheader()
                writer.writerows(records)
            try:
                frozen(reference, scratch, historical_donor=historical_donor)
            except ValueError as exc:
                if diagnostic not in str(exc):
                    raise
                print(f'Rejected corrupted copy: {exc}')
            else:
                raise ValueError(f'Failed to detect corrupted {column} in {damaged.name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('target', type=Path)
    parser.add_argument('--historical-donor', action='store_true',
                        help='use the original donor angle and arclength for strip 8')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--archive', action='store_true', help='use the retained final native frame and three-step ledger')
    mode.add_argument('--check-corruption', action='store_true')
    mode.add_argument('--continuation', type=int, nargs='+')
    args = parser.parse_args()
    if args.continuation:
        continuation(args.reference, args.target, args.continuation)
    elif args.check_corruption:
        check_corruption(args.reference, args.target, args.historical_donor)
    else:
        frozen(args.reference, args.target, args.archive, args.historical_donor)
