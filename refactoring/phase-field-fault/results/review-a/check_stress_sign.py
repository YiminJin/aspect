"""Reverse the measured shear in real solver output before the installed checker reads it."""
import argparse
from pathlib import Path
import runpy
import subprocess
import sys
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('checker', 'aspect', 'destination'):
        parser.add_argument(arg, type=lambda p: Path(p).resolve())
    parser.add_argument('--modulus', choices=('1', '2', 'both'), default='both',
                        help='Which accepted modulus case(s) to change, in MPa')
    parser.add_argument('--expect-old-pass', action='store_true')
    args = parser.parse_args()
    # An independent solver process creates each frame before the test changes
    # just the sign of its measured (cell,q,tauxy) entry. Change both spellings.
    first = {'public-1', 'legacy-1', 'omitted', 'equal'}
    second = {'public-2', 'legacy-2'}
    selected = {'1': first, '2': second, 'both': first | second}[args.modulus]
    changed = set()
    run = subprocess.run

    def solve(command, *, cwd, **kwargs):
        result = run(command, cwd=cwd, **kwargs)
        if result.returncode == 0 and cwd.name in selected:
            path = cwd / 'output/points-0001-r0.csv'
            lines = path.read_bytes().splitlines(keepends=True)
            column = lines[0].split(b',').index(b'tauxy')
            matches = [i for i, line in enumerate(lines) if line.startswith(b'0_0:,0,')]
            if len(matches) != 1:
                raise ValueError('Expected one cell 0_0:, q=0')
            i, = matches
            values = lines[i].split(b',')
            if not float(values[column]) > 0:
                raise ValueError(f'Expected a positive reference shear in {path}')
            values[column] = b'-' + values[column]
            lines[i] = b','.join(values)
            path.write_bytes(b''.join(lines))
            changed.add(cwd.name)
        return result

    sys.path.insert(0, str(args.checker.parent))
    with patch.object(sys, 'argv', [str(args.checker), str(args.aspect), str(args.destination)]), \
         patch.object(subprocess, 'run', solve):
        try:
            runpy.run_path(str(args.checker), run_name='__main__')
        except ValueError as exc:
            expected = '2' if args.modulus == '2' else '1'
            if args.expect_old_pass or f'G={expected} MPa: signed tauxy' not in str(exc):
                raise
            if not {f'public-{expected}', f'legacy-{expected}'} <= changed:
                raise ValueError('The checker stopped before comparing both spellings') from exc
            print(exc)
            return
    if not args.expect_old_pass or changed != selected:
        raise ValueError(f'Checker accepted the reversed stresses in {sorted(changed)}')
    print('Reviewed checker accepted reversed stress in', ', '.join(sorted(changed)))


if __name__ == '__main__':
    main()
