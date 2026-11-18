#!/usr/bin/env python3
"""Make an explicitly marked analysis view of successive native restart runs.

Native run directories, commands, scalar files and checkpoints remain intact.
Only merged scalar CSVs and symlinks are written under analysis_views/.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BUILD = ROOT / 'build-particle-free/stage4-preparation'
RUNS = ROOT / 'build-particle-free/bulk-tip/runs'


def compose(family, sources):
    view = BUILD / 'analysis_views' / family
    view.mkdir(parents=True, exist_ok=False)
    records = []
    for law in ('weakening', 'neutral'):
        law_sources = sources[law] if isinstance(sources, dict) else sources
        assert law_sources
        inputs = [RUNS / f'stage4-preparation-{source}-{law}' for source in law_sources]
        commands = [json.loads((run / 'command.json').read_text()) for run in inputs]
        assert all(command['exit_code'] == 0 for command in commands)
        target = view / f'stage4-preparation-{family}-{law}'
        output = target / 'output'
        output.mkdir(parents=True)
        patterns = ['pilot-*.csv', 'bulk-accepted.csv', 'release-regions.csv',
                    'release-driving.csv', 'preload-scan-*.csv*',
                    'release-phase-*.csv*', 'tip-balance-*.csv*',
                    'tip-seed-fracture-*.csv*', 'prepared-history-*',
                    'preparation-iterations.csv', 'preload-checks.csv',
                    'tip-sections-*.csv', 'tip-planes-*.csv', 'tip-power.csv',
                    'tip-profile.csv', 'tip-solver.csv', 'restore-*.csv*']
        names = sorted({p.name for run in inputs for pattern in patterns
                        for p in (run / 'output').glob(pattern) if p.is_file()})
        for name in names:
            files = [run / 'output' / name for run in inputs if (run / 'output' / name).is_file()]
            destination = output / name
            if len(files) == 1:
                destination.symlink_to(files[0])
                continue
            # Per-frame observations must never silently override a native frame.
            if any(character.isdigit() for character in name) or not name.endswith('.csv'):
                assert all(p.read_bytes() == files[0].read_bytes() for p in files), name
                destination.symlink_to(files[0])
                continue
            tables = []
            for path in files:
                with path.open() as stream:
                    reader = csv.DictReader(stream)
                    columns = reader.fieldnames
                    rows = list(reader)
                tables.append((columns, rows))
            assert all(columns == tables[0][0] for columns, _ in tables), name
            if 'step' not in tables[0][0]:
                assert all(rows == tables[0][1] for _, rows in tables), name
                destination.symlink_to(files[0])
                continue
            merged = []
            previous = -1
            for _, rows in tables:
                if rows:
                    assert int(float(rows[0]['step'])) > previous, (name, previous, rows[0]['step'])
                    previous = int(float(rows[-1]['step']))
                merged.extend(rows)
            with destination.open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=tables[0][0])
                writer.writeheader()
                writer.writerows(merged)
        # The common accepted sequence is the authoritative join check.
        with (output / 'pilot-summary.csv').open() as stream:
            sequence = list(csv.DictReader(stream))
        steps = [int(row['step']) for row in sequence]
        assert steps == list(range(steps[0], steps[-1] + 1)), steps
        times = [float(row['time']) for row in sequence]
        assert all(right > left for left, right in zip(times, times[1:]))
        provenance = dict(
            analysis_view=True,
            source_runs=[dict(path=str(run), command_sha256=hashlib.sha256((run / 'command.json').read_bytes()).hexdigest()) for run in inputs],
            seconds=sum(command['seconds'] for command in commands),
            exit_code=0,
            accepted_initial_step=steps[0],
            accepted_final_step=steps[-1],
            accepted_final_time=times[-1])
        (target / 'command.json').write_text(json.dumps(provenance, indent=2) + '\n')
        records.append(provenance)
    (view / f'{family}-finished.json').write_text(json.dumps(dict(complete=True, analysis_view=True, cases=records), indent=2) + '\n')
    print(view)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('family')
    parser.add_argument('sources', nargs='*', help='Native paired labels, in chronological order')
    parser.add_argument('--source-map', type=Path, help='JSON law-to-source-list map for unequal restart segmentation')
    args = parser.parse_args()
    assert args.family.isidentifier() and bool(args.sources)!=bool(args.source_map)
    sources=json.loads(args.source_map.read_text()) if args.source_map else args.sources
    labels=[label for group in sources.values() for label in group] if isinstance(sources,dict) else sources
    assert all(source.isidentifier() for source in labels)
    compose(args.family, sources)
