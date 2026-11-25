"""Exercise native growing-history files with actual installed one- and two-rank writers."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess

from compare import rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('aspect', type=Path)
    parser.add_argument('work', type=Path, help='fresh directory for this check')
    parser.add_argument('check', choices=('blocked', 'close', 'overwrite'))
    parser.add_argument('--mpiexec', default='mpirun')
    args = parser.parse_args()
    executable = args.aspect.resolve()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=False)
    example = Path(__file__).resolve().parent
    for name in ('growing_seed.prm', 'adiabatic_pressure.prm', 'onecut.prm'):
        shutil.copy(example/name, work/name)
    threads = dict.fromkeys(('DEAL_II_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS'), '1')

    def run(directory, ranks):
        prm = work/(directory.name + '.prm')
        prm.write_text(f'include growing_seed.prm\nset Output directory = {directory}\n')
        command = [executable, prm]
        if ranks > 1:
            command = [args.mpiexec, '-np', str(ranks)] + command
        log = work/f'{directory.name}-r{ranks}.log'
        with log.open('w') as stream:
            process = subprocess.run(command, cwd=work, stdout=stream, stderr=subprocess.STDOUT,
                                     env=os.environ | threads, timeout=120)
        return process.returncode, log

    output = work/'output'
    output.mkdir()
    if args.check != 'overwrite':
        # Point shards belong to individual ranks. The small root ledger stays
        # buffered until close; /dev/full accepts the open and rejects the flush.
        rank = 1 if args.check == 'blocked' else 0
        file = output/('points-0001-r1.csv' if args.check == 'blocked' else 'balance-0001.csv')
        if args.check == 'blocked':
            file.mkdir()
        else:
            file.symlink_to('/dev/full')
        status, log = run(output, 2)
        if status == 0:
            raise RuntimeError(f'Writer reported success with unusable {file.name}')
        message = ' '.join(log.read_text().split())
        if file.name not in message or f'writer rank {rank}' not in message:
            raise RuntimeError(f'Missing failed file and writer rank in {log}')
        print(f'{args.check}: exporter identified {file.name} and writer rank {rank}, exit {status}')
        return

    for directory, ranks in ((output, 2), (work/'clean', 1)):
        status, log = run(directory, ranks)
        if status:
            raise RuntimeError(f'Writer failed; see {log}')
    # The shipped reader rejects any empty, missing or duplicate shard.
    counts = [len(rows([output/f'points-0001-r{rank}.csv'], ('cell', 'q'))) for rank in (0, 1)]
    print(f'Two-rank native frame: {counts} points by owner', flush=True)
    # Keep the 5 ms frame complete. On the zero-clock frame also exercise
    # recovery after the prior writer lost its rank-zero points and sections.
    for name in ('points-0000-r0.csv', 'sections-0000.csv'):
        (output/name).unlink()
    optional = [output/'balance-0000.csv', output/'friction-pressure-0000.csv']
    for path in optional:
        path.write_text('obsolete\n')
    # A different clock, a different suffix and an unrelated file must survive.
    keep = [output/'points-1000-r1.csv', output/'memory-0001-r1.csv', output/'operator-r1.txt']
    for path in keep:
        path.write_text('retain\n')
    status, log = run(output, 1)
    if status:
        raise RuntimeError(f'Serial overwrite failed; see {log}')
    for tag in ('0000', '0001'):
        for prefix, keys in (('points', ('cell', 'q')), ('fields', ('cell', 'component', 'local'))):
            pattern = f'{prefix}-{tag}-r*.csv'
            actual = rows(output.glob(pattern), keys)
            expected = rows((work/'clean').glob(pattern), keys)
            if actual != expected or set(output.glob(pattern)) != {output/f'{prefix}-{tag}-r0.csv'}:
                raise ValueError(f'Overwriting {pattern} changed record identities, columns or quantities')
            print(f'Frame {tag}: all {len(actual)} {prefix} records matched serial exactly')
        if (output/f'memory-{tag}-r1.txt').exists():
            raise ValueError('Obsolete memory shard survived')
    if any(path.read_text() != 'retain\n' for path in keep):
        raise ValueError('Overwriting a frame changed an unrelated file')
    if any(path.exists() for path in optional):
        raise ValueError('The seed frame retained an obsolete ledger or pressure table')
    print('Repeated two-rank frames on one rank; unrelated files preserved')


if __name__ == '__main__':
    main()
