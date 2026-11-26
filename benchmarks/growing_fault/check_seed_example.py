"""Check the 5 ms teaching example against the corrected two-rank four-field control."""
import argparse
import json
from pathlib import Path

from check_adiabatic_pressure import check_profile
from check_interface import compare_growing_frames
from compare import compare_values, plane_tolerances, rows


def check_layout(output, expected_fields):
    parameters = json.loads((output/'parameters.json').read_text())
    names = parameters['Compositional fields']['Names of fields']['value']
    if [name.strip() for name in names.split(',')] != expected_fields:
        raise ValueError(f'Unexpected compositional fields in {output}: {names}')
    for tag in ('0000', '0001'):
        shards = sorted(output.glob(f'points-{tag}-r*.csv'))
        if [p.name for p in shards] != [f'points-{tag}-r{rank}.csv' for rank in (0, 1)]:
            raise ValueError(f'Expected a native frame from both MPI ranks in {output} at {tag}')
        counts = [len(rows([shard], ('cell', 'q'))) for shard in shards]
        if counts != [2484, 2484]:
            raise ValueError(f'Expected the two owned halves of the frozen native rule, found {counts}')
    balance = rows(output.glob('balance-*.csv'), ('step',))
    if set(balance) != {(1,)} or float(balance[1,]['time']) != 0.005:
        raise ValueError(f'Expected a single accepted 5 ms interval in {output}')
    return balance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('control', type=Path, help='four-field constant-pressure output')
    parser.add_argument('seed', type=Path, help='one-field constant-pressure output')
    parser.add_argument('affine', type=Path, help='one-field 50 Pa/m output')
    args = parser.parse_args()
    reference = check_layout(args.control, ['tau_xx', 'tau_yy', 'tau_xy', 'phase_field'])
    actual = check_layout(args.seed, ['phase_field'])
    check_layout(args.affine, ['phase_field'])
    errors = compare_growing_frames(args.control, args.seed)
    stress_errors = [float(errors[name]) for name in ('tauxx', 'tauyy', 'tauzz', 'tauxy')]
    print(f'4968 matched native points: max |Δτxx, Δτyy, Δτzz, Δτxy| (Pa) = {stress_errors}')
    print('9 matched strip IDs and formed flags; maximum absolute differences:',
          f"β {errors['beta']:.12g} rad, ds {errors['ds']:.12g} m, V {errors['V']:.12g} m/s,",
          f"force {errors['force']:.12g} Pa, age {errors['age']:.12g} s, slip {errors['slip']:.12g} m")

    # All energy terms are per metre of out-of-plane thickness.
    energy = ('bulk', 'cohesive', 'fracture', 'Wext', 'Db', 'Dc', 'Df',
              'phase_remainder', 'phase_work', 'energy_change')
    differences = [compare_values(reference, actual, col, 1e-11, 1e-6) for col in energy]
    print(f'5 ms max absolute work/energy difference {max(differences):.12g} J/m')

    accepted = rows([args.seed/'sections-0001.csv'], ('section',))[8,]
    corrected = rows([Path(__file__).with_name('nine-strip-geometry.csv')], ('step', 'section'))[1, 8]
    if accepted['formed'] != '1':
        raise ValueError('Expected strip 8 to have selected a direction by 5 ms')
    for name, source, unit in [('beta', 'beta_rad', 'rad'), ('ds', 'ds_m', 'm')]:
        difference = compare_values({(8,): {name: corrected[source]}}, {(8,): accepted},
                                    name, *plane_tolerances(name))
        print(f'Strip 8: {name} = {accepted[name]} {unit}, C2 difference {difference:.12g} {unit}')
    for output, gradient in [(args.seed, 0.), (args.affine, 50.)]:
        print(f'Frictional pressure at clipped strip stations, dp/dz = {gradient:g} Pa/m')
        check_profile(output, 200., gradient)


if __name__ == '__main__':
    main()
