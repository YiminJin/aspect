"""Reverse both wall components in the installed affine nine-strip input at 5 ms."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

import mpmath as mp
import numpy as np

from check_peak import check


def run(executable, inputs, directory, wall_expression=None):
    directory.mkdir()
    shutil.copyfile(inputs / 'onecut.prm', directory / 'onecut.prm')
    affine = (inputs / 'adiabatic_pressure.prm').read_text()
    if wall_expression is not None:
        affine += ('\nsubsection Boundary velocity model\n  subsection Function\n'
                   f'    set Function expression = {wall_expression}\n  end\nend\n')
    (directory / 'input.prm').write_text(affine + '\nset Output directory = output\n')
    with (directory / 'log').open('w') as log:
        subprocess.run([executable, 'input.prm'], cwd=directory, stdout=log, stderr=subprocess.STDOUT,
                       env=dict(os.environ, DEAL_II_NUM_THREADS='1', OMP_NUM_THREADS='1',
                                OPENBLAS_NUM_THREADS='1'), check=True, timeout=180)
    return directory / 'output'


def reversed_wall_expression(forward_output):
    with (forward_output / 'parameters.json').open() as stream:
        function = json.load(stream)['Boundary velocity model']['Function']
    components = [part.strip() for part in function['Function expression']['value'].split(';')]
    if len(components) != 2 or not all(components):
        raise ValueError('Expected two nonempty wall velocity components in the forward parameters')
    return ';'.join(f'-({part})' for part in components)


def check_velocity_reversal(forward, reverse):
    from compare import rows

    fields = []
    velocities = []
    for name, output in [('forward', forward), ('reverse', reverse)]:
        try:
            table = rows(output.glob('fields-0001-r*.csv'), ('cell', 'component', 'local'))
        except ValueError as exc:
            raise ValueError(f'{name} run, velocity components 0 and 1: {exc}') from exc
        fields.append(table)
        keys = sorted(table)
        components = []
        for component in (0, 1):
            values = np.array([float(table[k]['value']) for k in keys if k[1] == component])
            if not values.size:
                raise ValueError(f'{name} run: missing velocity component {component}')
            if not np.isfinite(values).all():
                raise ValueError(f'{name} run: nonfinite velocity component {component}')
            components.append(values)
        velocities.append(components)

    if fields[0].keys() != fields[1].keys():
        raise ValueError('The two field runs have different DoF identities')
    for component, (a, b) in enumerate(zip(*velocities)):
        if max(abs(a)) < 1e-5:
            raise ValueError(f'Inactive forward velocity component {component}')
        np.testing.assert_allclose(a, -b, rtol=1e-7, atol=1e-11, equal_nan=False)
        print(f'Velocity component {component}, max |forward + reverse| = {max(abs(a+b)):.9g} m/s')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable', type=lambda path: Path(path).resolve())
    parser.add_argument('destination', type=lambda path: Path(path).resolve())
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--inputs', type=Path, help='use copies of the two affine example parameter files')
    mode.add_argument('--replay', action='store_true', help='check saved forward and reverse outputs')
    args = parser.parse_args()
    installed = args.executable.parent.parent / 'benchmarks/growing_fault'
    sys.path.insert(0, str(installed))
    from check_adiabatic_pressure import check_profile
    from compare import rows

    if args.replay:
        forward = args.destination / 'forward/output'
        reverse = args.destination / 'reverse/output'
    else:
        args.destination.mkdir(parents=True, exist_ok=False)
        inputs = args.inputs or installed
        forward = run(args.executable, inputs, args.destination / 'forward')
        reverse = run(args.executable, inputs, args.destination / 'reverse', reversed_wall_expression(forward))
    check_velocity_reversal(forward, reverse)
    for out in (forward, reverse):
        check_profile(out, 200., 50.)
    sections = [rows([out / 'sections-0001.csv'], ('section',)) for out in (forward, reverse)]
    for key, positive in sections[0].items():
        negative = sections[1][key]
        np.testing.assert_allclose([float(negative[col]) for col in ('V', 'force')],
                                   [-float(positive[col]) for col in ('V', 'force')], atol=0., rtol=1e-7)
        for row in (positive, negative):
            if float(row['V']) * float(row['force']) <= 0.:
                raise ValueError(f'Expected positive frictional power density on strip {key}')

    # Mirror the archived first-turn field. Its native stresses change sign,
    # while phase, the accepted parent, disk, pressure and full strip stay fixed.
    # Evaluate the signed RSF roots and the direct pointwise score for both loads.
    c1 = Path(__file__).resolve().parent.parent / 'direction-c1'
    sys.path.insert(0, str(c1))
    from peaks import PointScore
    mp.mp.dps = 65
    with tempfile.TemporaryDirectory() as scratch, tarfile.open(c1 / 'turns.tar.gz') as archive:
        for name in ('root-input', 'sections-0000.csv', 'turn-points-0'):
            (Path(scratch) / name).write_bytes(archive.extractfile('affine-1/' + name).read())
        direct = PointScore(Path(scratch))
    mirrored = copy.copy(direct)
    mirrored.integral_u, mirrored.integral_v = -direct.integral_u, -direct.integral_v
    mirrored.score_points = [(w, -u, -v) for w, u, v in direct.score_points]
    reversed_strip = sections[1][8,]
    selected_angle = float(reversed_strip['beta'])
    for angle in (selected_angle, .23):
        v, f, score = direct.value(angle)
        vr, fr, sr = mirrored.value(angle)
        if not (vr == -v and fr == -f and sr == score and score > 0):
            raise ValueError('Mirrored native-point score lost signed symmetry')
        if angle == selected_angle:
            # The post-turn solve in this fixture changes these signed strip
            # quantities by less than the comparison budgets.
            np.testing.assert_allclose(float(reversed_strip['V']), float(vr), rtol=1e-7, atol=0.)
            np.testing.assert_allclose(float(reversed_strip['force']), float(fr), rtol=0., atol=1e-6)
        print(f'Mirrored first-turn field, {angle:.15g} rad: V={float(vr):.12g} m/s, '
              f'f={float(fr):.12g} Pa, score={float(sr):.12g} Pa^2 m^2')
    for out, table in zip((forward, reverse), sections):
        strip = table[8,]
        print(f"{out.parent.name} accepted strip 8: V={strip['V']} m/s, f={strip['force']} Pa")
        check('affine-1', out)


if __name__ == '__main__':
    main()
