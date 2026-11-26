"""Check the physical parent and native metadata of the last 3616642 snapshot.

Run with the 3616642 parameter file, the original package fault.txt, and the
restart directory. This reads the native marker and files without touching them.
Only ASPECT can deserialize the distributed solution and validate the coupled
constitutive history; the next eight accepted steps supply that test.
"""
import argparse
import csv
import hashlib
from decimal import Decimal
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_parent(prm, fault, restart):
    for path, expected in [
            (prm, '1c8c078ba8e949787cb5985b35ae1c286516b456bf018cfdbefe9477053c4660'),
            (fault, '5460e34da479ef16e7dcc95f62669250a891ee8c969df5064dd3e4320bea4a24')]:
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                f'{path}: input differs from the archived run or ordered fault')
    slot = int((restart / 'last_good_checkpoint.txt').read_text().strip())
    require(slot == 3, f'Expected marker 3 for the recorded snapshot, got {slot}')
    parent = restart / f'{slot:02d}'
    step, clock = (parent / 'bp3_accepted_state.txt').read_text().split()
    expected = (4229, Decimal('5172072500.6710749'))
    require((int(step), Decimal(clock)) == expected,
            f'{parent}: expected step {expected[0]} at {expected[1]} s, found {step} at {clock} s')

    # deal.II writes shared MPI files for the mesh and its fixed/variable
    # attachments. Determine the attachment payloads from the native header.
    with (parent / 'mesh.info').open() as stream:
        names, values = stream.readline().split(), stream.readline().split()
    require(len(names) == len(values), 'Malformed mesh.info')
    info = dict(zip(names, map(int, values)))
    require(info['nproc'] == 160, f'Unexpected checkpoint writer rank count: {info["nproc"]}')
    payloads = ['mesh', 'resume.z']
    for column, name in [('n_attached_fixed_size_objs', 'mesh_fixed.data'),
                         ('n_attached_variable_size_objs', 'mesh_variable.data')]:
        if info[column] > 0:
            payloads.append(name)
    for name in payloads:
        path = parent / name
        require(path.is_file() and path.stat().st_size > 0, f'Missing or empty native payload: {path}')

    # The live output can lead the snapshot. Compare the checkpoint's own
    # last accepted records; particle/DG data are still needed for the solve.
    metadata = parent / 'bp3_output_metadata'
    for name in ('accepted_steps.csv', 'stations.csv'):
        with (metadata / name).open() as stream:
            last = None
            for row in csv.DictReader(stream):
                last = row
        require(last is not None and (int(last['step']), Decimal(last['time'])) == expected,
                f'{name}: last checkpoint record disagrees with accepted label')
    print(f'{parent}: slot {slot:02d}, accepted {step} at {clock} s, writer ranks {info["nproc"]}')
    print('Native files present:', ', '.join(payloads))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prm', type=Path, help='exact original.prm from job 3616642')
    parser.add_argument('fault', type=Path, help='five-anchor fault.txt used by the donor')
    parser.add_argument('restart', type=Path, help='parent restart/ directory including its marker')
    args = parser.parse_args()
    try:
        check_parent(args.prm, args.fault, args.restart)
    except (OSError, ValueError, KeyError, ArithmeticError) as error:
        parser.exit(1, f'{error}\n')
