"""Run signed strip and positive-speed Q1 controls with a stateless material."""
import argparse
import os
from pathlib import Path
import subprocess

import numpy as np

from check_history import check
from check_repairs import subsection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('aspect', 'plugin', 'destination'):
        parser.add_argument(name, type=lambda p: Path(p).resolve())
    parser.add_argument('--ranks', type=int, choices=[1,2], default=1)
    parser.add_argument('--mpiexec', default='mpirun')
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', DEAL_II_NUM_THREADS='1')

    def run(name, settings, expected=None):
        case = args.destination/name
        case.mkdir()
        output = case/'output'
        prm = case/'input.prm'
        prm.write_text(settings + f'\nset Additional shared libraries = {args.plugin}\nset Output directory = {output}\n')
        command = [args.aspect, prm]
        if args.ranks>1:
            command = [args.mpiexec,'--bind-to','core','--map-by','core','-np',str(args.ranks),*command]
        with (case/'log').open('w') as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=150)
        if expected:
            if result.returncode == 0 or expected not in (case/'log').read_text():
                raise ValueError(f'Missing provider diagnosis in {case}/log')
        else:
            result.check_returncode()
        return output

    base = (here/'onecut.prm').read_text()
    run('incompatible', base.replace('Model name = growing fault', 'Model name = simple'),
        'requires a material providing the Maxwell/AT1 midpoint response')
    settings = base.replace('Model name = growing fault', 'Model name = linear strip resistance')
    run('incompatible-energy',settings.replace('Crack geometry = AT1','Crack geometry = AT2'),
        'requires the configured AT1 surface energy')
    # The RSF-only scales and monotonicity guard must never reach the dashpot.
    settings += subsection('Material model', subsection('Growing fault',
                           'set Initial age = 0\nset Direct effect = 0\nset Radiation damping = 0'))
    settings += subsection('Postprocess', 'set List of postprocessors = midpoint publication')
    settings += subsection('Time stepping', 'set List of model names = convection time step, check returned midpoint')
    settings += subsection('Checkpointing', 'set Steps between checkpoint = 1\nset Number of checkpoints to keep = 4')
    trajectories = []
    for name in ('forward', 'reverse'):
        prm = settings
        if name == 'reverse':
            # Reflect the entire affine, divergence-free loading history.
            walls = prm.split('subsection Boundary velocity model',1)[1]
            line = next(s for s in walls.splitlines() if 'set Function expression' in s)
            prefix, expr = line.split(' = ')
            prm = prm.replace(line, prefix+' = '+';'.join('-('+v+')' for v in expr.split(';')))
        out = run(name, prm)
        trajectories.append(check(out, linear_impedance=1000.))
        print(next(line.strip() for line in (out.parent/'log').read_text().splitlines()
                   if 'Returned midpoint tangent relative error' in line))
    if not np.all(np.sign(trajectories[0][0])*np.sign(trajectories[1][0]) == -1):
        raise ValueError('The matched reversed loading did not reverse signed strip rates')

    trace = (here.parent/'fixed_fault/fixed_fault.prm').read_text()
    trace = trace.replace('Model name = phase field fault', 'Model name = linear trace resistance')
    trace += subsection('Material model',subsection('Phase field fault','set Initial age = 0\nset Direct effect = 0'))
    trace += subsection('Postprocess', 'set List of postprocessors = fixed fault statistics')
    out = run('mature', trace)
    # The interior-seed path supports a native label checkpoint. The provider
    # decodes the saved records and checks the seven geometric/Maxwell values.
    interior = trace.replace('stress, stress, stress, generic','stress, stress, stress, phase field')
    interior += subsection('Material model', subsection('Phase field fault',
                           'set Fault geometry = fixed interior\nset Fault vertices = .5,.05; .5,.95\nset Fit trace to phase = false'))
    interior += subsection('Checkpointing', 'set Steps between checkpoint = 1\nset Number of checkpoints to keep = 3')
    interior_output = run('mature-interior',interior)
    if not (interior_output/'restart/02/resume.z').is_file():
        raise ValueError('Missing state-free mature checkpoint')
    slip = 0.
    for step in range(3):
        text = (out/f'fault-labels-{step:05}.txt').read_text().splitlines()
        if 'age' in text[0] or len(text)!=3:
            raise ValueError('Unexpected mature label table')
        # The measured load is uniform. At physical endpoints the imposed
        # traction is 0.02+t Pa; the trace tangent points upward on the left.
        data = np.loadtxt(text[1:])
        if step:
            rate = (3.85-(.02+.0125*step))/20.
            slip += .0125*rate
            np.testing.assert_allclose(data[:,3:], [[slip,rate]]*2, rtol=0, atol=2e-12,equal_nan=False)
            print(f'Mature endpoint {step}: rate {data[0,-1]:.12g} m/s, slip {data[0,-2]:.12g} m')


if __name__ == '__main__':
    main()
