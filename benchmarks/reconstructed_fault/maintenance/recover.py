#!/usr/bin/env python3
"""Preview/recover closeout-retired tracked files from the exact local baseline."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

here = Path(__file__).resolve().parent
repo = here.parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('prefix', help='original repository-relative file/directory prefix')
parser.add_argument('--destination', type=Path, required=True,
                    help='destination root (a new empty directory is recommended)')
parser.add_argument('--write', action='store_true', help='restore; otherwise preview only')
args = parser.parse_args()
records = list(csv.DictReader((here/'classification.tsv').open(), delimiter='\t'))
prefix = args.prefix.rstrip('/')
selected = [row for row in records if row['action'] == 'untrack-keep-local'
            and (row['path'] == prefix or row['path'].startswith(prefix+'/'))]
if not selected:
    parser.error('prefix contains no retired tracked paths')
revision = json.loads((here/'baseline.json').read_text())['baseline']
print(f'{len(selected)} files, {sum(int(row["bytes"]) for row in selected)} bytes from {revision}')
for row in selected:
    relative = Path(row['path'])
    assert not relative.is_absolute() and '..' not in relative.parts
    path = args.destination/relative
    if not args.write:
        print(path)
        continue
    data = subprocess.check_output(['git', 'show', revision+':'+row['path']], cwd=repo)
    assert len(data) == int(row['bytes'])
    assert hashlib.sha256(data).hexdigest() == row['sha256'], row['path']
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise SystemExit(f'Refusing to overwrite different content: {path}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(data)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
print('Verified restoration complete.' if args.write else 'Preview only; add --write to restore.')
