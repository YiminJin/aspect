"""Restore the append-only slip table to a checkpoint's accepted prefix.

The benchmark checkpoints accumulated slip itself. This file is output, not
restart state: stream its prefix once when making a restart branch instead of
copying an ever-growing vertex table into every ordinary checkpoint.
"""
from pathlib import Path

HEADER = 'step,time_s,fault,node,s_m,xd_m,slip_m'


def restore_prefix(path, step):
    path = Path(path)
    temporary = path.with_suffix('.restoring')
    try:
        with path.open() as source, temporary.open('x') as target:
            header = source.readline()
            if header.strip() != HEADER:
                raise ValueError('Incompatible cumulative slip table')
            target.write(header)
            last_step = -1
            for line in source:
                current = int(line.split(',', 1)[0])
                if current > step:
                    break
                if current < last_step:
                    raise ValueError('Nonmonotone cumulative slip steps')
                target.write(line)
                last_step = current
            if last_step != step:
                raise ValueError('Cumulative slip table does not reach checkpoint state')
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
