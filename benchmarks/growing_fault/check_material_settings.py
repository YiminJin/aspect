"""Check the modulus in the accepted native stress, aliases and route-specific defaults."""
import argparse
import csv
import math
import os
from pathlib import Path
import re
import subprocess

from check_adiabatic_pressure import adiabatic_constants
from parameter_file import legacy_mature_input, subsection, take_material_subsection


def remove_setting(settings, name):
    settings, count = re.subn(r'(?m)^[ \t]*set[ \t]+' + re.escape(name) + r'[ \t]*=[^\n]*$', '', settings)
    if count != 1:
        raise ValueError(f'Expected one setting of {name}, found {count}')
    return settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('aspect', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    aspect = args.aspect.resolve()
    root = args.destination.resolve()
    root.mkdir(parents=True, exist_ok=False)
    example = Path(__file__).with_name('growing_seed.prm').read_text()
    # The reference for this regression uses 200 Pa + (50 Pa/m) times depth.
    example += adiabatic_constants(200, 50)
    public = 'Phase field fault'
    legacy = 'Growing fault'
    outside, settings = take_material_subsection(example, public)
    shared = remove_setting(settings, 'Shear modulus')
    # Keep the geometry and pressure profile fixed when testing route defaults.
    required = ('Friction normal stress', 'Seed tangent angle', 'Forward seed tip', 'Cohesion')
    seed = '\n'.join(line for line in shared.splitlines()
                     if any(line.strip().startswith('set '+key+' =') for key in required))
    omitted = outside + subsection('Material model', subsection(public, seed))
    env = dict(os.environ, DEAL_II_NUM_THREADS='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')

    def run(name, content, diagnostic=()):
        case = root/name
        case.mkdir()
        (case/'input.prm').write_text(content+'\nset Output directory = output\n')
        with (case/'log').open('w') as log:
            result = subprocess.run([aspect, 'input.prm'], cwd=case, env=env,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=120)
        if diagnostic:
            message = ' '.join((case/'log').read_text().split())
            if result.returncode == 0 or any(word not in message for word in diagnostic):
                raise ValueError(f'{name}: expected {diagnostic}, see {case}/log')
            print(name, *diagnostic)
        else:
            result.check_returncode()
        return case/'output'

    def setting(section, key, value):
        return '\n'+subsection('Material model', subsection(section, f'set {key} = {value}'))

    def compare(a, b):
        for f in ('points-0001-r0.csv', 'sections-0001.csv', 'friction-pressure-0001.csv', 'balance-0001.csv'):
            if (a/f).read_bytes() != (b/f).read_bytes():
                raise ValueError(f'{a.name}, {b.name}: different accepted file {f}')

    def modulus_input(section, mpa):
        settings = shared + f'\nset Shear modulus = {mpa}e6'
        return outside + subsection('Material model', subsection(section, settings))

    for n, reference in ((1, 75.978862635792), (2, 151.95405267759796)):
        result = run(f'public-{n}', modulus_input(public, n))
        compare(result, run(f'legacy-{n}', modulus_input(legacy, n)))
        with (result/'balance-0001.csv').open() as stream:
            clock = float(next(csv.DictReader(stream))['time'])
        if not math.isclose(clock, .005, rel_tol=0., abs_tol=1e-15):
            raise ValueError(f'G={n} MPa: expected accepted time 0.005 s, found {clock} s')
        with (result/'points-0001-r0.csv').open() as stream:
            sample, = (row for row in csv.DictReader(stream) if (row['cell'], row['q']) == ('0_0:', '0'))
        stress = float(sample['tauxy'])
        # Use the frozen nine-strip native stress budget (Pa) at this same material point.
        if not abs(stress-reference) <= 1e-6 + 1e-7*abs(reference):
            raise ValueError(f'G={n} MPa: signed tauxy at 5 ms, cell 0_0:, q=0 was {stress} Pa, expected {reference} Pa')
        print(f'G={n} MPa: signed tauxy at 5 ms, cell 0_0:, q=0 = {stress} Pa')

    base = root/'public-1/output'
    compare(base, run('omitted', omitted))
    compare(base, run('equal', example+setting(legacy, 'Shear modulus', '1000000.0')))
    # Both spellings equal a historical default in this negative control.
    run('both-defaults', example+setting(public,'Shear modulus','1e10')+setting(legacy,'Shear modulus','1e6'),
        [public+' / Shear modulus', legacy+' / Shear modulus'])
    run('conflict', example+setting(public,'Shear modulus','2e6')+setting(legacy,'Shear modulus','1e6'),
        [public+' / Shear modulus', legacy+' / Shear modulus'])
    run('rsf-conflict', example+setting(legacy,'Direct effect','.01'),
        [public+' / Direct effect', legacy+' / Direct effect'])
    run('viscosity-conflict', example+setting('Simple model','Viscosity','5e24'),
        [public+' / Viscosity', 'Simple model / Viscosity'])
    run('mature-only-at-default', example+setting(public,'Fit trace to phase','true'),
        ['Fit trace to phase requires mature trace mechanics'])
    fixed = (Path(__file__).resolve().parents[1]/'fixed_fault/fixed_fault.prm').read_text()
    old_trace = run('mature-legacy-viscosity', legacy_mature_input(fixed))
    new_trace = run('mature-public-viscosity', fixed)
    for step in range(3):
        name = f'fault-labels-{step:05}.txt'
        if (old_trace/name).read_bytes() != (new_trace/name).read_bytes():
            raise ValueError(f'Mature viscosity spelling changed {name}')
    print('Mature public and Simple viscosity: identical label histories at all three clocks')
    run('growing-only-at-default', fixed+setting(legacy,'Section quadrature order','6'),
        ['Section quadrature order requires growing strip mechanics'])


if __name__ == '__main__':
    main()
