"""Run the installed curved-phase initial conditions on alternative ordered tables."""
import argparse
import os
from pathlib import Path
import subprocess


def geometry_copy(source, case):
    """Keep IDs, headers, units and property columns unchanged."""
    lines = source.read_text().splitlines(keepends=True)
    coords = []
    for i, line in enumerate(lines):
        words = line.partition('#')[0].split()
        if not words or not words[0].isdigit():
            continue
        row, x, y = int(words[0]), float(words[1]), float(words[2])
        coords.append((x, y))
        if case in ('moved', 'moved-mpi', 'stale') and row == 2:
            # Change one number in the installed default, leaving other bytes intact.
            lines[i] = line.replace(words[2], format(y-.12, '.2f'), 1)
        elif case == 'straight':
            words[2] = format(.82 + (.14-.82)*x, '.15g')
            lines[i] = ' '.join(words)+'\n'
    return ''.join(lines), coords


def stale_phase(coords):
    """A frozen function of the OLD vertices for the deliberately stale run.

    In segment coordinates, perpendicular distance and distance past either
    endpoint give the squared Euclidean distance by Pythagoras. Friction,
    mechanics and the shared geometric table still see the moved vertex.
    """
    distances = []
    for (x0, y0), (x1, y1) in zip(coords, coords[1:]):
        dx, dy = x1-x0, y1-y0
        length2 = dx*dx + dy*dy
        x, y = f'(x-({x0:.17g}))', f'(y-({y0:.17g}))'
        along = f'(({dx:.17g})*{x}+({dy:.17g})*{y})'
        cross = f'(({dx:.17g})*{y}-({dy:.17g})*{x})'
        end = f'max(0,max(-{along},{along}-({length2:.17g})))'
        distances.append(f'({cross}^2+{end}^2)/({length2:.17g})')
    shortest, *others = distances
    for other in others:
        shortest = f'min({shortest},{other})'
    return ('subsection Initial composition model\n  set List of model names = function\n'
            '  subsection Function\n    set Function expression = 0; 0; 0; '
            f'0.03+0.32*exp(-({shortest})/0.0225)\n  end\nend\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prefix', 'observer', 'destination'):
        parser.add_argument(name, type=lambda s: Path(s).resolve())
    parser.add_argument('--mpiexec', default='mpirun')
    parser.add_argument('--group', choices=('all', 'serial', 'mpi'), default='all')
    args = parser.parse_args()
    example = args.prefix/'benchmarks/fixed_fault/curved_mature'
    args.destination.mkdir(parents=True, exist_ok=True)
    env = os.environ | dict(ASPECT_SOURCE_DIR=str(args.prefix), OMP_NUM_THREADS='1',
                            OPENBLAS_NUM_THREADS='1', DEAL_II_NUM_THREADS='1')
    cases = []
    if args.group in ('all', 'serial'):
        cases.extend(('moved', 'straight', 'stale'))
    if args.group in ('all', 'mpi'):
        cases.extend(('original-mpi', 'moved-mpi'))
    for name in cases:
        case = args.destination/name
        case.mkdir()
        vertices, original = geometry_copy(example/'vertices.txt', name)
        (case/'vertices.txt').write_text(vertices)
        prm = (f'include {example}/curved_mature.prm\nset End time = 0\n'
               f'set Additional shared libraries = {args.observer}\nset Output directory = output\n'
               'subsection Material model\n  subsection Phase field fault\n'
               '    set Fault vertex file = vertices.txt\n  end\nend\n'
               'subsection Postprocess\n  set List of postprocessors = moving fault samples\nend\n')
        if name == 'stale':
            prm += stale_phase(original)
        (case/'case.prm').write_text(prm)
        cmd = [args.prefix/'bin/aspect-release', 'case.prm']
        if name.endswith('-mpi'):
            cmd = [args.mpiexec, '-np', '2', *cmd]
        with (case/'run.log').open('w') as log:
            subprocess.run(cmd, cwd=case, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f'{name}: accepted time-zero geometry and FE field in {case}', flush=True)


if __name__ == '__main__':
    main()
