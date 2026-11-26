"""Check the three plotted times and material positions using surface output only."""
import argparse
import csv
from decimal import Decimal
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark', type=Path, help='directory containing the installed plotter')
    parser.add_argument('output', type=Path, help='ordinary layered example output')
    parser.add_argument('--steps', type=int, nargs=3, required=True)
    parser.add_argument('--vertices', type=int, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.benchmark.resolve()))
    from plot_layers import accepted_positions

    with TemporaryDirectory() as tmp:
        output = Path(tmp)
        clock = output/'layered-background.csv'
        clock.write_bytes((args.output/clock.name).read_bytes())
        expected = []
        for step in args.steps:
            name = f'material-{step}-surface-0.csv'
            with (args.output/name).open() as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
                coords = {int(r['i']): (Decimal(r['x']), Decimal(r['y'])) for r in rows}
                if sorted(coords) != list(range(args.vertices)) or len(rows) != args.vertices:
                    raise ValueError(f'{name}: wrong material index set')
                expected.append([coords[i] for i in range(args.vertices)])
            # Reversing alternate CSV tables must preserve physical tracks.
            with (output/name).open('w') as stream:
                writer = csv.DictWriter(stream, reader.fieldnames)
                writer.writeheader()
                writer.writerows(rows[::-1] if len(expected) == 2 else rows)

        frames, ids, positions = accepted_positions(output)
        np.testing.assert_array_equal(frames['step'], args.steps)
        np.testing.assert_array_equal(ids, np.arange(args.vertices))
        np.testing.assert_array_equal(positions, np.array(expected, dtype=float))
        mid = (args.vertices-1)//2
        print(f'{args.vertices} material IDs at steps {args.steps}; '
              f'ID {mid} horizontal displacement {(expected[-1][mid][0]-expected[0][mid][0])*1000:.9f} mm')

        def reject(path, data, diagnostic):
            original = path.read_text()
            path.write_text(data)
            try:
                accepted_positions(output)
            except ValueError as error:
                if diagnostic not in str(error):
                    raise
                print(error)
            else:
                raise ValueError(f'Accepted invalid data: {diagnostic}')
            finally:
                path.write_text(original)

        lines = clock.read_text().splitlines(keepends=True)
        endpoint = next(line for line in lines[1:] if line.split(',')[0] == str(args.steps[-1]))
        reject(clock, ''.join(line for line in lines if line != endpoint), '0.025 s, found 0')
        reject(clock, ''.join(lines + [endpoint]), '0.025 s, found 2')
        surface = output/f'material-{args.steps[-1]}-surface-0.csv'
        lines = surface.read_text().splitlines(keepends=True)
        reject(surface, ''.join(lines[:-1]), 'Material vertex IDs changed')
        reject(surface, ''.join(lines + [lines[-1]]), 'Expected distinct material vertex IDs')


if __name__ == '__main__':
    main()
