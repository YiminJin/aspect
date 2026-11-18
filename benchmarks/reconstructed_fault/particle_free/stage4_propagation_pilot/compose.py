#!/usr/bin/env python3
"""Compose a read-only analysis view of successive, closed pilot segments.

Native data stay in place. Intentional checkpoint stops have exit143; their
accepted rows are included, without describing the process as a successful run.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[4]
BUILD = ROOT / 'build-particle-free/stage4-propagation-pilot'
RUNS = ROOT / 'build-particle-free/bulk-tip/runs'
parser = argparse.ArgumentParser()
parser.add_argument('view')
parser.add_argument('sources', nargs='+', help='Chronological native segment labels')
args = parser.parse_args()
assert all(Path(label).name == label for label in [args.view] + args.sources)
runs = [RUNS / ('stage4-propagation-' + label) for label in args.sources]
commands = [json.loads((run / 'command.json').read_text()) for run in runs]
assert all(command.get('exit_code') in (0, 143) for command in commands), 'Use only closed accepted segments'
assert len({command['binary_sha256'] for command in commands}) == 1
assert len({command['checker_sha256'] for command in commands}) == 1
target = BUILD / 'analysis_views' / args.view
target.mkdir(parents=True, exist_ok=False)
tables = ['pilot-summary.csv', 'bulk-accepted.csv', 'release-regions.csv',
          'release-driving.csv', 'tip-power.csv', 'tip-profile.csv', 'tip-solver.csv']
for name in tables:
    paths = [run / 'output' / name for run in runs if (run / 'output' / name).exists()]
    if not paths:
        continue
    header, merged, previous = None, [], -1
    for path in paths:
        with path.open() as stream:
            reader = csv.DictReader(stream)
            if header is None:
                header = reader.fieldnames
            assert reader.fieldnames == header, name
            rows = list(reader)
        assert 'step' in header, name
        if rows:
            steps = [int(float(row['step'])) for row in rows]
            assert steps[0] > previous and steps == sorted(steps), (name, path, previous)
            previous = steps[-1]
        merged.extend(rows)
    with (target / name).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader(); writer.writerows(merged)
patterns = ['release-phase-*.csv*', 'pilot-sections-*.csv*', 'tip-sections-*.csv*',
            'tip-planes-*.csv*', 'preload-scan-*.csv*', 'tip-balance-*.csv*']
for run in runs:
    for pattern in patterns:
        for path in (run / 'output').glob(pattern):
            destination = target / path.name
            assert not destination.exists(), ('Duplicate native frame', path.name)
            destination.symlink_to(path.resolve())
with (target / 'pilot-summary.csv').open() as stream:
    summary = list(csv.DictReader(stream))
steps = [int(row['step']) for row in summary]
assert steps == list(range(steps[-1] + 1)), steps
times = [float(row['time']) for row in summary]
assert all(a < b for a, b in zip(times, times[1:]))
for step in steps:
    assert len(list(target.glob(f'release-phase-{step:04d}-r*.csv*'))) == 4
    assert len(list(target.glob(f'pilot-sections-{step:04d}.csv*'))) == 1
record = dict(analysis_view=True, accepted_step=steps[-1], accepted_time=times[-1],
              sources=[dict(path=str(run), process_exit=command['exit_code'],
                            command_sha256=hashlib.sha256((run/'command.json').read_bytes()).hexdigest(),
                            process_seconds=command['seconds'])
                       for run, command in zip(runs, commands)])
(target / 'provenance.json').write_text(json.dumps(record, indent=2)+'\n')
print(target)
