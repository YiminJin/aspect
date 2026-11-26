"""Compare complete installed trajectories and resume generation 02 from each old material."""
import argparse
from pathlib import Path
import sys

import numpy as np


def growing_clocks(output, last):
    """ASPECT restores the entire accepted statistics clock, including its initial row."""
    clock = np.loadtxt(output/'statistics', usecols=(0, 1, 2), ndmin=2)
    expected = [[i, .005*i, .005 if i else 0.] for i in range(last+1)]
    np.testing.assert_allclose(clock, expected, atol=1e-15, rtol=0., equal_nan=False)


def check_growing_tables(reference, actual, step, adiabatic=True):
    """Complete the native-point and strip checks with the FE solution, station pressure and work."""
    from compare import rows, compare_values

    tag = f'{step:04}'
    old, new = (rows(d.glob(f'fields-{tag}-r*.csv'), ('cell', 'component', 'local'))
                for d in (reference, actual))
    # The existing same-mesh restart test uses this tolerance for all FE coefficients.
    field_error = compare_values(old, new, 'value', 1e-12, 1e-12)
    if not step:
        print(f'  step 0: {len(old)} FE coefficients, max difference {field_error:g}')
        return
    paths = [d/f'friction-pressure-{tag}.csv' for d in (reference, actual)]
    if any(path.exists() != adiabatic for path in paths):
        raise ValueError('Unexpected set of adiabatic friction pressure stations')
    pressure_result = 'prescribed compression'
    if adiabatic:
        old, new = (rows([p], ('section',)) for p in paths)
        for col in ('x', 'y', 'depth'):
            compare_values(old, new, col, 5e-13, 0.)
        error = compare_values(old, new, 'compression', 5e-13, 0.)
        pressure_result = f'pressure {error:g} Pa'
        compare_values(old, new, 'V', 0., 1e-12)
    old, new = (rows([d/f'balance-{tag}.csv'], ('step',)) for d in (reference, actual))
    compare_values(old, new, 'time', 1e-15, 0.)
    compare_values(old, new, 'iterations', 0., 0.)
    energy_cols = ('bulk', 'cohesive', 'fracture', 'Wext', 'Db', 'Dc', 'Df',
                   'phase_remainder', 'phase_work', 'energy_change')
    energy_error = max(compare_values(old, new, col, 1e-11, 1e-6) for col in energy_cols)
    for col in ('adjoint_error', 'phase_KKT', 'mechanical_residual'):
        compare_values(old, new, col, 1e-12, 1e-12)
    print(f'  step {step}: max differences FE {field_error:g}, {pressure_result}, '
          f'storage/work {energy_error:g} J/m')


def compare_public(args):
    from check_interface import compare_growing_frames
    from check_seed_example import check_layout

    for case in ('seed', 'affine', 'control'):
        ref, out = (root/case/'output' for root in (args.reference, args.candidate))
        names = ['phase_field'] if case != 'control' else ['tau_xx', 'tau_yy', 'tau_xy', 'phase_field']
        for trajectory in (ref, out):
            check_layout(trajectory, names)
            growing_clocks(trajectory, 1)
        errors = compare_growing_frames(ref, out)
        print(f'{case}, max absolute differences at 5 ms: phase {errors["phi"]:g}, '
              f'stress {max(errors[c] for c in ("tauxx", "tauyy", "tauzz", "tauxy")):g} Pa, '
              f'angle {errors["beta"]:g} rad, rate {errors["V"]:g} m/s, '
              f'age {errors["age"]:g} s, slip {errors["slip"]:g} m, force {errors["force"]:g} Pa')
        for step in (0, 1):
            check_growing_tables(ref, out, step, adiabatic=case != 'control')


def continue_parents(args):
    from check_restart import OBSERVE, check_clocks, run
    from check_public_reference import REFERENCE, compare, csv_rows, public_rows
    from compare import continuation, rows, compare_values

    args.destination.mkdir()
    curved = args.prefix/'benchmarks/fixed_fault/curved_mature'
    mature = run('mature', curved/'curved_mature.prm', [args.layered_plugin], settings=OBSERVE,
                 checkpoint=args.starting/'starting-parent-mature/output',
                 root=args.destination, prefix=args.prefix)
    check_clocks(mature, [2])
    for frozen_name, candidate_name, keys in (
            ('material-2-rank0.csv.gz', 'material-2-rank0.csv', ('cell', 'q')),
            ('material-2-surface-0.csv', 'material-2-surface-0.csv', ('i',))):
        compare(csv_rows(REFERENCE/frozen_name, keys), csv_rows(mature/candidate_name, keys),
                f'post-restart {keys}')
    filename = 'fault-labels-00002.txt'
    compare(public_rows(REFERENCE/filename), public_rows(mature/filename), 'post-restart public labels')
    print('Mature generation 02: 12.5 ms to 25 ms, five IDs and 2,304 native samples passed.')

    growing = run('growing', args.prefix/'benchmarks/growing_fault/growing_seed.prm', [],
                  settings='set End time = .015\n', checkpoint=args.starting/'starting-parent-growing/output',
                  root=args.destination, prefix=args.prefix)
    ref = args.starting/'starting-growing-long/output'
    growing_clocks(growing, 3)
    growing_clocks(ref, 3)
    if sorted(p.name for p in growing.glob('sections-*.csv')) != ['sections-0002.csv', 'sections-0003.csv']:
        raise ValueError('Expected only 10 and 15 ms publications after resuming the old 5 ms parent')
    continuation(ref, growing, [2, 3])
    for step in (2, 3):
        old, new = (rows([d/f'sections-{step:04}.csv'], ('section',)) for d in (ref, growing))
        for col in ('V', 'slip', 'force'):
            compare_values(old, new, col, 0., 1e-12)
        check_growing_tables(ref, growing, step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    path = lambda s: Path(s).resolve()
    parser.add_argument('prefix', type=path, help='candidate installation containing the example checkers')
    commands = parser.add_subparsers(dest='operation', required=True)
    frames = commands.add_parser('frames', help='compare old and new two-rank public outputs')
    frames.add_argument('reference', type=path)
    frames.add_argument('candidate', type=path)
    parents = commands.add_parser('continue', help='copy and resume both old checkpoint generations')
    parents.add_argument('starting', type=path, help='directory containing starting-parent-* and starting-growing-long')
    parents.add_argument('layered_plugin', type=path, help='observer built against the candidate SDK')
    parents.add_argument('destination', type=path)
    args = parser.parse_args()
    sys.path[:0] = [str(args.prefix/'benchmarks/growing_fault'),
                    str(args.prefix/'benchmarks/fixed_fault/curved_mature')]
    (compare_public if args.operation == 'frames' else continue_parents)(args)
