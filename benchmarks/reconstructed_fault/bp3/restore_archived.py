"""Restore selected historical BP3 artifacts without overwriting changed files.

The archive and its manifest are local, not a remote backup. See CLEANUP.md.
Listing/preview is read-only; --restore explicitly moves files back in place.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('prefix', nargs='?', default='', help='BP3-relative file or directory')
    parser.add_argument('--restore', action='store_true', help='restore listed files; default is preview')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if Path(manifest['root']) != ROOT:
        raise SystemExit('Manifest belongs to another directory; refusing restoration.')
    prefix = args.prefix.rstrip('/')
    selected = [r for r in manifest['files'] if not prefix or r['path'] == prefix
                or r['path'].startswith(prefix+'/')]
    if not selected:
        raise SystemExit('No archived files match this prefix.')
    groups = {}
    for r in selected:
        key = r['path'].split('/')[0]
        count, size = groups.get(key, (0, 0))
        groups[key] = count+1, size+r['bytes']
    for key, (count, size) in sorted(groups.items()):
        print(f'{key}: {count} files, {size:,} bytes')
    if not args.restore:
        print('Preview only. Add --restore to move these files back into BP3.')
        return

    # Validate every target and payload before the first move. Equal existing
    # files are left untouched, making a partially restored group resumable.
    pending = []
    for r in selected:
        rel = Path(r['path'])
        if rel.is_absolute() or '..' in rel.parts or r['link'] is not None:
            raise SystemExit(f'Unsupported archive entry: {rel}')
        source = args.manifest.resolve().parent/'payload'/rel
        target = ROOT/rel
        if not target.resolve().is_relative_to(ROOT):
            raise SystemExit(f'Target escapes BP3 through a symlink: {target}')
        if target.exists():
            if not target.is_file() or digest(target) != r['sha256']:
                raise SystemExit(f'Existing target differs; refusing overwrite: {target}')
            continue
        if not source.is_file() or digest(source) != r['sha256']:
            raise SystemExit(f'Missing or changed archive payload: {source}')
        pending.append((source, target, r['sha256']))
    for source, target, expected in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        if digest(target) != expected:
            raise SystemExit(f'Restored checksum mismatch: {target}')
    print(f'Restored {len(pending)} files; no differing existing file was overwritten.')


if __name__ == '__main__':
    main()
