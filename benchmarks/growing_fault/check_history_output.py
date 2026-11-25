"""Exercise native growing-history files with the installed exporter across MPI partition changes."""
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
    parser.add_argument('check', choices=('blocked', 'close', 'overwrite', 'gap', 'retry', 'cleanup'))
    parser.add_argument('--mpiexec', default='mpirun')
    args = parser.parse_args()
    executable = args.aspect.resolve()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=False)
    example = Path(__file__).resolve().parent
    for name in ('growing_seed.prm', 'adiabatic_pressure.prm', 'onecut.prm'):
        shutil.copy(example/name, work/name)
    threads = dict.fromkeys(('DEAL_II_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS'), '1')

    def run(directory, ranks, attempt=''):
        prm = work/(directory.name + '.prm')
        prm.write_text(f'include growing_seed.prm\nset Output directory = {directory}\n')
        command = [executable, prm]
        if ranks > 1:
            command = [args.mpiexec, '-np', str(ranks)] + command
        log = work/f'{directory.name}-r{ranks}{attempt}.log'
        with log.open('w') as stream:
            process = subprocess.run(command, cwd=work, stdout=stream, stderr=subprocess.STDOUT,
                                     env=os.environ | threads, timeout=120)
        return process.returncode, log

    def success(directory, ranks, attempt=''):
        status, log = run(directory, ranks, attempt)
        if status:
            raise RuntimeError(f'Writer failed; see {log}')

    def failure(directory, ranks, message_prefix, file):
        status, log = run(directory, ranks, '-failed')
        if status == 0:
            raise RuntimeError(f'Writer reported success with unusable {file.name}')
        message = ' '.join(log.read_text().split())
        if file.name not in message or message_prefix not in message:
            raise RuntimeError(f'Missing failed file and writer rank in {log}')
        print(f'{args.check}: {message_prefix}, {file.name}, exit {status}')

    output = work/'output'
    output.mkdir()
    if args.check == 'cleanup':
        # Only rank zero enumerates the directory, with every rank observing
        # the outcome before returning from the postprocessor.
        directory = output/'points-0001-r2.csv'
        directory.mkdir()
        keep = directory/'user-data.txt'
        keep.write_text('retain\n')
        failure(output, 2, 'cleanup on rank 0', directory)
        if keep.read_text() != 'retain\n':
            raise ValueError('Cleanup changed unrelated directory contents')
        return

    if args.check in ('blocked', 'close'):
        # Point shards belong to individual ranks. The small root ledger stays
        # buffered until close; /dev/full accepts the open and rejects the flush.
        rank = 1 if args.check == 'blocked' else 0
        file = output/('points-0001-r1.csv' if args.check == 'blocked' else 'balance-0001.csv')
        if args.check == 'blocked':
            file.mkdir()
        else:
            file.symlink_to('/dev/full')
        failure(output, 2, f'writer rank {rank}', file)
        return

    clean = work/'clean'

    def compare_frames(ranks):
        for tag in ('0000', '0001'):
            for prefix, keys in (('points', ('cell', 'q')), ('fields', ('cell', 'component', 'local'))):
                pattern = f'{prefix}-{tag}-r*.csv'
                actual = rows(sorted(output.glob(pattern)), keys)
                expected = rows(sorted(clean.glob(pattern)), keys)
                files = {output/f'{prefix}-{tag}-r{r}.csv' for r in range(ranks)}
                if actual != expected or set(output.glob(pattern)) != files:
                    raise ValueError(f'Overwriting {pattern} changed record identities, columns or quantities')
                print(f'Frame {tag}: all {len(actual)} {prefix} records matched clean {ranks}-rank output exactly')
            memory = {output/f'memory-{tag}-r{r}.txt' for r in range(ranks)}
            if set(output.glob(f'memory-{tag}-r*.txt')) != memory:
                raise ValueError(f'Unexpected memory shards in frame {tag}')
            for prefix in ('sections', 'balance', 'friction-pressure'):
                name = f'{prefix}-{tag}.csv'
                if (clean/name).exists():
                    if (output/name).read_bytes() != (clean/name).read_bytes():
                        raise ValueError(f'Overwriting changed {name}')
                elif (output/name).exists():
                    raise ValueError(f'Unexpected {name}')

    if args.check in ('gap', 'retry'):
        ranks = 1 if args.check == 'gap' else 2
        success(clean, ranks)
        source = clean/f'points-0001-r{ranks-1}.csv'
        unrelated = output/'points-0003-r2.csv'
        shutil.copyfile(source, unrelated)

        if args.check == 'gap':
            # Leave only a genuine point shard at rank 2 at the 5 ms clock.
            shutil.copyfile(source, output/'points-0001-r2.csv')
        else:
            success(output, 3)
            old = [output/f'{prefix}-0001-r2.{extension}'
                   for prefix, extension in (('points', 'csv'), ('fields', 'csv'), ('memory', 'txt'))]
            saved = {path: path.read_bytes() for path in old}
            print('Three-rank native frame:', [len(rows([output/f'points-0001-r{r}.csv'], ('cell', 'q')))
                                               for r in range(3)], 'points by owner', flush=True)
            blocked = output/'points-0001-r1.csv'
            blocked.rename(work/'old-points-r1.csv')
            blocked.mkdir()
            failure(output, 2, 'writer rank 1', blocked)
            lost = [p.name for p, data in saved.items() if not p.is_file() or p.read_bytes() != data]
            print('Old rank-2 shards changed or missing after rank-1 write failure:', lost, flush=True)
            # Readers of an unsuccessful frame can still encounter partial data.
            try:
                rows(sorted(output.glob('points-0001-r*.csv')), ('cell', 'q'))
            except IsADirectoryError as exc:
                if Path(exc.filename) != blocked:
                    raise
                print(f'Failed-frame reader identified the unusable {blocked.name}', flush=True)
            else:
                raise ValueError('Reader accepted the frame with an unusable rank-1 points file')
            if unrelated.read_bytes() != source.read_bytes():
                raise ValueError('Failed output changed the unrelated frame')
            blocked.rmdir()

        success(output, ranks, '-retry' if args.check == 'retry' else '')
        print('5 ms points shards after successful export:',
              ', '.join(sorted(p.name for p in output.glob('points-0001-r*.csv'))), flush=True)
        compare_frames(ranks)
        if unrelated.read_bytes() != source.read_bytes():
            raise ValueError('Successful output changed the unrelated frame')
        if args.check == 'retry' and lost:
            raise ValueError(f'The failed write lost old shards before retry: {lost}')
        return

    success(output, 2)
    success(clean, 1)
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
    success(output, 1)
    compare_frames(1)
    if any(path.read_text() != 'retain\n' for path in keep):
        raise ValueError('Overwriting a frame changed an unrelated file')
    if any(path.exists() for path in optional):
        raise ValueError('The seed frame retained an obsolete ledger or pressure table')
    print('Repeated two-rank frames on one rank; unrelated files preserved')


if __name__ == '__main__':
    main()
