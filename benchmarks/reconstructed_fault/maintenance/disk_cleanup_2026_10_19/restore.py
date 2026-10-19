#!/usr/bin/env python3
"""Preview or restore archived fields/package copies without overwriting changes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
from cleanup import HERE, ROOT, digest, read_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefix', help='Original repository-relative file or directory')
    parser.add_argument('--destination', type=Path, default=ROOT,
                        help='Destination root; defaults to this worktree')
    parser.add_argument('--write', action='store_true', help='Restore; otherwise preview only')
    args = parser.parse_args()
    prefix = args.prefix.rstrip('/')
    selected = []
    for kind, manifest in [('fields', 'field-archives.json'), ('package', 'package-copies.json')]:
        for record in json.loads((HERE / manifest).read_text()):
            rows = {name: row for name, row in record['files'].items()
                    if name == prefix or name.startswith(prefix + '/')}
            if rows:
                selected.append((kind, record, rows))
    if not selected:
        parser.error('Prefix has no archived files in this cleanup.')
    count = sum(len(rows) for _, _, rows in selected)
    size = sum(row['bytes'] for _, _, rows in selected for row in rows.values())
    print(f'{count} files, {size / 2**20:.2f} MiB; destination root: {args.destination}')
    destination = args.destination.resolve()
    # Check every destination first, before writing any file.
    for _, _, rows in selected:
        for name, row in rows.items():
            relative = Path(name)
            assert not relative.is_absolute() and '..' not in relative.parts
            target = destination / relative
            assert target.resolve().is_relative_to(destination), target
            if target.exists():
                if not target.is_file() or target.is_symlink() or digest(target) != row['sha256']:
                    raise SystemExit('Refusing to overwrite different content: ' + str(target))
    if not args.write:
        for kind, record, rows in selected:
            print(kind, record.get('path', record.get('archive')), len(rows), 'files')
        print('Preview only; add --write to restore verified bytes.')
        return

    restored = []
    for kind, record, rows in selected:
        archive = ROOT / (record['path'] if kind == 'fields' else record['archive'])
        checksum = record['sha256'] if kind == 'fields' else record['archive_sha256']
        assert digest(archive) == checksum, 'Archive checksum mismatch: ' + str(archive)
        seen = set()
        relative_map = ({str(Path(name).relative_to(record['folder'])): name for name in rows}
                        if kind == 'package' else None)

        def extract(bundle, member):
            assert member.isfile()
            if kind == 'fields':
                name = member.name
            else:
                relative = str(Path(*Path(member.name).parts[1:]))
                name = relative_map.get(relative)
            if name not in rows:
                return
            assert name not in seen
            row = rows[name]
            target = destination / name
            with bundle.extractfile(member) as stream:
                if target.exists():
                    assert hashlib.file_digest(stream, 'sha256').hexdigest() == row['sha256']
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_name(target.name + '.restore-partial')
                    assert not temporary.exists()
                    with temporary.open('xb') as output:
                        shutil.copyfileobj(stream, output, length=2**20)
                    assert digest(temporary) == row['sha256']
                    temporary.chmod(row['mode'])
                    os.utime(temporary, ns=(row['mtime_ns'], row['mtime_ns']))
                    temporary.rename(target)
                    restored.append(name)
            assert digest(target) == row['sha256']
            seen.add(name)

        if kind == 'fields':
            read_archive(archive, extract)
        else:
            with tarfile.open(archive, 'r:gz') as bundle:
                for member in bundle:
                    extract(bundle, member)
        assert seen == set(rows)
    print(f'Restored {len(restored)} new files; all {count} selected payloads verified.')


if __name__ == '__main__':
    main()
