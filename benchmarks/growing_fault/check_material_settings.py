"""Check the modulus in the accepted native stress, aliases and route-specific defaults."""
import argparse
import csv
import os
from pathlib import Path
import re
import subprocess

from check_repairs import subsection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('aspect', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    aspect = args.aspect.resolve()
    root = args.destination.resolve()
    root.mkdir(parents=True, exist_ok=False)
    example = Path(__file__).with_name('growing_seed.prm').read_text()
    public = 'Phase field fault'
    legacy = 'Growing fault'
    public_block = re.search(r'^ subsection Phase field fault\n(.*?)^ end$', example, re.M | re.S).group()
    legacy_block = public_block.replace('subsection '+public, 'subsection '+legacy)
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

    def first_stress(directory):
        with (directory/'points-0001-r0.csv').open() as stream:
            return float(next(r for r in csv.DictReader(stream) if (r['cell'],r['q'])==('0_0:','0'))['tauxy'])

    stresses = []
    for n in (1, 2):
        modulus = f'{n}e6'
        canonical = example.replace(public_block, public_block.replace('Shear modulus = 1e6', 'Shear modulus = '+modulus))
        old = example.replace(public_block, legacy_block.replace('Shear modulus = 1e6', 'Shear modulus = '+modulus))
        result = run(f'public-{n}', canonical)
        compare(result, run(f'legacy-{n}', old))
        stresses.append(first_stress(result))
    if abs(stresses[0]-stresses[1]) < 1:
        raise ValueError(f'Changing the public G did not change the native shear stress on the Pa scale: {stresses}')
    print('Accepted 5 ms, cell 0_0:, q=0, tau_xy (Pa) at G=1 and 2 MPa:', *stresses)

    base = root/'public-1/output'
    compare(base, run('omitted', example.replace(public_block, '')))
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
    old_trace = run('mature-legacy-viscosity', fixed)
    public_viscosity, n = re.subn(r'(?m)^ +subsection Simple model\n +set Viscosity = 10\n +end$', '', fixed)
    if n != 1:
        raise ValueError('Missing the historical Simple viscosity in the mature control')
    new_trace = run('mature-public-viscosity', public_viscosity+setting(public, 'Viscosity', '10'))
    for step in range(3):
        name = f'fault-labels-{step:05}.txt'
        if (old_trace/name).read_bytes() != (new_trace/name).read_bytes():
            raise ValueError(f'Mature viscosity spelling changed {name}')
    print('Mature public and Simple viscosity: identical label histories at all three clocks')
    run('growing-only-at-default', fixed+setting(legacy,'Section quadrature order','6'),
        ['Section quadrature order requires growing strip mechanics'])


if __name__ == '__main__':
    main()
