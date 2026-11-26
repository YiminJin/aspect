"""Bounded installed pf-rsf probes for a surface-reaching, right-dipping thrust."""
import argparse
from importlib.util import module_from_spec, spec_from_file_location
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefix', type=Path, help='Assignment D installation with benchmark plugins')
    parser.add_argument('destination', type=Path, help='new local directory')
    args = parser.parse_args()
    root, prefix = args.destination.resolve(), args.prefix.resolve()
    root.mkdir(parents=True, exist_ok=False)
    aspect = prefix/'bin/aspect-release'
    examples = prefix/'benchmarks/fixed_fault'
    ridge, probe = examples/'phase_ridge', examples/'transfer_work'
    env = dict(os.environ, ASPECT_SOURCE_DIR=str(prefix), DEAL_II_NUM_THREADS='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with (root/'fresh.log').open('w') as log:
        subprocess.run([sys.executable, '-B', ridge/'run.py', aspect, ridge/'libphase_ridge.release.so',
                        root/'free-32', 'free', '32'], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    with (root/'reference.log').open('w') as log:
        subprocess.run([sys.executable, '-BO', '-c',
                        'from pathlib import Path; from check import check_case; check_case(Path.cwd(),"free",32)'],
                       cwd=root, env=dict(env, PYTHONPATH=str(ridge)), stdout=log, stderr=subprocess.STDOUT, check=True)

    controls = {
        'checkpoint': ('subsection Checkpointing\n set Steps between checkpoint = 1\nend',
                       'requires a fresh run with checkpointing disabled'),
        'surface': ('subsection Material model\n subsection Phase field fault\n'
                    '  set Fault geometry = fixed interior\n  set Fit trace to phase = false\n end\nend',
                    'A seed vertex is outside its supported box region'),
        'friction': ('subsection Material model\n subsection Phase field fault\n'
                     '  set Direct effect = 0.010, 0.025\n end\nend', 'does not match the given pattern')
    }
    for name, (settings, diagnostic) in controls.items():
        case = root/name
        case.mkdir()
        prm = case/'input.prm'
        prm.write_text(f"include {root/'free-32/input.prm'}\nset Output directory = {case/'output'}\n{settings}\n")
        log = case/'log'
        with log.open('w') as stream:
            result = subprocess.run([aspect, prm], cwd=case, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode == 0 or diagnostic not in ' '.join(log.read_text().split()):
            raise ValueError(f'{name}: expected {diagnostic!r}, see {log}')
        print(f'{name}: rejected ({diagnostic})', flush=True)

    # Reuse the diagnostic that perturbs the middle rate, calls FaultStressLoad
    # before constraint scatter and evaluates the reverse section row. It takes
    # the common fixed physical test v=t*(n.x). Reversing both vectors leaves v unchanged.
    spec = spec_from_file_location('paired_work', probe/'run.py')
    driver = module_from_spec(spec)
    spec.loader.exec_module(driver)
    os.environ.update(env)
    tangent = np.array([.5, -math.sqrt(3)/2])
    length = -1/tangent[1]  # one metre of vertical extent
    start = np.array([.5,.5]) - .5*length*tangent
    vertices = [start, np.array([.5,.5]), start + length*tangent]
    for order in ('down-dip', 'up-dip'):
        points = vertices if order == 'down-dip' else vertices[::-1]
        coords = '; '.join(f'{x:.17g},{y:.17g}' for x,y in points)
        prm = ('subsection Material model\n subsection Phase field fault\n'
               f'  set Fault vertices = {coords}\n end\nend\n')
        action = driver.run_pair(aspect, probe/'libtransfer_work.release.so', root/order, prm)
        # Independent Q8 contraction at a common transfer measure. Use the
        # measured normal integrals as inputs so only the shear sense is tested.
        integrals = action['normal_integrals'][::1 if order == 'down-dip' else -1]
        locations, weights = np.polynomial.legendre.leggauss(8)
        weight = .35*(1+.35)/(1-.35)**2  # constant phase, degradation magnitude 1
        expected = 0.
        for x, wx in zip((locations+1)/2, weights/2):
            for y, wy in zip((locations+1)/2, weights/2):
                u = (np.array([x,y])-start) @ tangent / length
                if 0 <= u <= 1:
                    hat = 1-abs(2*u-1)
                    integral = np.interp(u, [0,.5,1], integrals)
                    expected += wx*wy*weight*hat/integral
        if expected <= 0:
            raise ValueError('Inactive middle-label transfer')
        for k in ('stokes', 'section'):
            np.testing.assert_allclose(action[k], expected, atol=2e-13, rtol=0)
        print(f"{order}: actual Stokes / reverse row {action['stokes']:.15g} / {action['section']:.15g} m; "
              f"independent +sense {expected:.15g} m; donor thrust sense at this measure {-expected:.15g} m")


if __name__ == '__main__':
    main()
