"""Run the installed mature example and its geometric admission controls from time zero."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prefix', 'layered_plugin', 'destination'):
        parser.add_argument(name, type=lambda s: Path(s).resolve())
    parser.add_argument('--vertical-gaussian', action='store_true',
                        help='Replay the manufactured vertical-width regression field')
    args = parser.parse_args()
    case = args.prefix/'benchmarks/fixed_fault/curved_mature'
    aspect = args.prefix/'bin/aspect-release'
    args.destination.mkdir(parents=True, exist_ok=False)
    env = os.environ | dict(ASPECT_SOURCE_DIR=str(args.prefix), OMP_NUM_THREADS='1',
                            OPENBLAS_NUM_THREADS='1', DEAL_II_NUM_THREADS='1')
    materials = 'subsection Material model\n  subsection Phase field fault\n    set '
    source = [line for line in (case/'vertices.txt').read_text().splitlines()
              if line and not line.startswith('#')]
    for name in ('public', 'observed', 'curved-fit', 'repeated', 'crossed'):
        directory = args.destination/name
        directory.mkdir()
        base = 'vertical_gaussian.prm' if args.vertical_gaussian else 'curved_mature.prm'
        entries = [f'include {case / base}', 'set Output directory = output']
        diagnostic = None
        if name == 'observed':
            entries.extend([f'set Additional shared libraries = {args.layered_plugin}',
                            'subsection Postprocess',
                            '  set List of postprocessors = mature fault statistics, moving fault samples',
                            'end'])
        elif name == 'curved-fit':
            entries.append(materials+'Fit trace to phase = true\n  end\nend')
            diagnostic = 'Fit trace to phase = false'
        elif name in ('repeated', 'crossed'):
            # Keep the contact points and consecutive IDs. Change only the three
            # interior coordinates so that the material graph itself is invalid.
            xy = ([('0.25', '0.77'), ('0.25', '0.77'), ('0.75', '0.45')] if name == 'repeated'
                  else [('0.8', '0.2'), ('0.2', '0.2'), ('0.8', '0.8')])
            records = [r.split() for r in source]
            for r, p in zip(records[2:5], xy):
                r[1:3] = p
            (directory/'changed.txt').write_text('\n'.join(' '.join(r) for r in records)+'\n')
            entries.append(materials+f'Fault vertex file = {directory}/changed.txt\n  end\nend')
            diagnostic = ('Degenerate material fault edge' if name == 'repeated'
                          else 'Nonadjacent material fault edges intersect')
        (directory/'case.prm').write_text('\n'.join(entries)+'\n')
        with (directory/'run.log').open('w') as log:
            result = subprocess.run([aspect, 'case.prm'], cwd=directory, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
        if diagnostic:
            if result.returncode == 0 or diagnostic not in ' '.join((directory/'run.log').read_text().split()):
                raise ValueError(f'{name}: expected {diagnostic!r}, see {directory}/run.log')
            print(f'{name}: rejected ({diagnostic})', flush=True)
        else:
            result.check_returncode()
            labels = sorted((directory/'output').glob('fault-labels-*.txt'))
            if len(labels) != 3:
                raise ValueError(f'{name}: expected three accepted frames in {directory}/output')
            if name == 'observed':
                bare = args.destination/'public/output'
                for label in labels:
                    if label.read_bytes() != (bare/label.name).read_bytes():
                        raise ValueError(f'Public and observed labels differ at {label.name}')
                if not args.vertical_gaussian:
                    subprocess.run([sys.executable, '-B', case/'check_public_reference.py',
                                    directory/'output', '--public-output', bare], check=True)
            print(f'{name}: accepted 0, 12.5 and 25 ms', flush=True)


if __name__ == '__main__':
    main()
