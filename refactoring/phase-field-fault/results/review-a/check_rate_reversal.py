"""Feed a captured, single-value mutation through the installed public interface checker."""
import argparse
import csv
import runpy
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


class ComparisonPassed(Exception):
    """The first pair compared successfully and the checker requested the next case."""


def replay(checker, accepted, archived, mutation=None):
    with TemporaryDirectory() as temp:
        def supply_output(command, *, cwd, **kwargs):
            # Replay just the two solver calls. The checker's main function reads
            # and compares the actual CSV files, including all 4,968 native IDs.
            if cwd.name not in ('growth', 'legacy-growth'):
                raise ComparisonPassed
            output = cwd / 'output'
            output.mkdir()
            for tag in ('0000', '0001'):
                for source in accepted.glob(f'points-{tag}-r*.csv'):
                    (output / source.name).symlink_to(source)
                path = output / f'sections-{tag}.csv'
                source = accepted / path.name if tag == '0000' else archived
                mutated_tag = {'reverse': '0001', 'perturb-zero': '0000'}.get(mutation)
                if cwd.name == 'growth' or tag != mutated_tag:
                    path.symlink_to(source)
                    continue
                # Preserve every byte, including the row terminator, outside V.
                with source.open(newline='') as incoming, path.open('w', newline='') as dest:
                    header = incoming.readline()
                    columns = next(csv.reader([header]))
                    dest.write(header)
                    changes = 0
                    for line in incoming:
                        fields = line.split(',')
                        if int(fields[columns.index('section')]) == 8:
                            i = columns.index('V')
                            v = float(fields[i])
                            if mutation == 'reverse' and v > 0:
                                changed = '-' + fields[i]
                            elif mutation == 'perturb-zero' and v == 0:
                                changed = '1e-25'
                            else:
                                raise ValueError(f'Unexpected archived strip-8 rate {v} at {tag}')
                            print(f'Strip 8: {fields[i]} -> {changed} m/s')
                            fields[i] = changed
                            changes += 1
                        dest.write(','.join(fields))
                if changes != 1:
                    raise ValueError(f'Changed {changes} rows')
            return subprocess.CompletedProcess(command, 0)

        sys.path.insert(0, str(checker.parent))
        try:
            args = [str(checker), '/unused/aspect', '/unused/plugin', str(Path(temp) / 'checks')]
            with patch.object(sys, 'argv', args), patch.object(subprocess, 'run', supply_output):
                try:
                    runpy.run_path(str(checker), run_name='__main__')
                except ComparisonPassed:
                    return True
                except ValueError as exc:
                    if not mutation or 'V at (8,)' not in str(exc):
                        raise
                    print(f'Interface comparison rejected {mutation}: {exc}')
                    return False
                raise RuntimeError('The checker did not run its first pair')
        finally:
            sys.path.pop(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('checker', 'accepted_output', 'archived_sections'):
        parser.add_argument(arg, type=lambda p: Path(p).resolve())
    parser.add_argument('--expect-old-pass', action='store_true',
                        help='Reproduce the reviewed checker accepting the wrong sign.')
    args = parser.parse_args()
    inputs = args.checker, args.accepted_output, args.archived_sections
    if not replay(*inputs):
        raise ValueError('The checker rejected the unchanged capture')
    for mutation in ('reverse', 'perturb-zero'):
        if replay(*inputs, mutation) != args.expect_old_pass:
            raise ValueError(f'{mutation} did not give the expected interface verdict')
        if args.expect_old_pass:
            print(f'Reviewed checker accepted {mutation}')
