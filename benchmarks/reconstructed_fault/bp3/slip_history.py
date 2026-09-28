"""Restore the append-only slip table to a checkpoint's accepted prefix.

The benchmark checkpoints accumulated slip itself. This file is output, not
restart state: stream its prefix once when making a restart branch instead of
copying an ever-growing vertex table into every ordinary checkpoint.
"""
from pathlib import Path
import csv
import shutil

HEADER = 'step,time_s,fault,node,s_m,xd_m,slip_m'


def restore_profile_payloads(parent, branch, checkpoint_step):
    """Copy only payloads named by the branch's checkpoint-restored index.

    Call after restoring metadata, into a new branch without a profiles
    directory. The parent remains immutable. No physical history is read here.
    """
    parent, branch = Path(parent), Path(branch)
    index = branch/'profiles.csv'
    if not index.is_file():
        return
    with index.open() as stream:
        entries = list(csv.DictReader(stream))
    selected = {}
    previous = None
    for entry in entries:
        step, time = int(entry['step']), float(entry['time_s'])
        if step > checkpoint_step:
            raise ValueError('Checkpoint profile index contains a newer state')
        if previous is not None and entry != previous and (
                step <= int(previous['step']) or time <= float(previous['time_s'])):
            raise ValueError('Conflicting checkpoint profile index')
        relative = Path(entry['file'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Expected a run-relative profile payload')
        source = parent/relative
        if not source.is_file():
            raise ValueError(f'Missing checkpoint profile payload: {source}')
        selected[relative] = source
        previous = entry
    # All sources must exist before starting; never overwrite branch payloads.
    for relative, source in selected.items():
        target = branch/relative
        if target.exists():
            raise ValueError(f'Refuse to overwrite profile payload: {target}')
    for relative, source in selected.items():
        target = branch/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


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
