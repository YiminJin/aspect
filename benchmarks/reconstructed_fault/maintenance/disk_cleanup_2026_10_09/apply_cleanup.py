#!/usr/bin/env python3
"""Apply the explicitly requested local disk cleanup, preserving evidence bytes."""
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import time

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
BUILD = ROOT / 'build-propagation-integration'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def main():
    assert not (HERE / 'cleanup.json').exists(), 'Cleanup already recorded.'
    before = json.loads((HERE / 'before.json').read_text())
    tracked = set(subprocess.check_output(['git', 'ls-files'], cwd=ROOT, text=True).splitlines())
    pinned = set()
    for parent in ['benchmarks/reconstructed_fault', 'doc/reconstructed_fault']:
        for path in (ROOT / parent).rglob('*.json'):
            try:
                data = json.loads(path.read_text())
            except (ValueError, UnicodeError):
                continue
            for name in strings(data):
                if name.startswith(str(ROOT) + '/'):
                    pinned.add(name[len(str(ROOT)) + 1:])
                elif name.startswith(('build-', 'benchmarks/', 'doc/', 'source/', 'include/', 'tests/')):
                    pinned.add(name)
    remove = []
    for build in ROOT.glob('build-*'):
        if not build.is_dir():
            continue
        for path in build.rglob('*'):
            if path.is_symlink() or not path.is_file():
                continue
            if 'CMakeFiles' in path.parts and path.suffix in ['.o', '.gch']:
                remove.append((path, 'regenerable compiler object/precompiled header'))
    for path in (ROOT / 'benchmarks/reconstructed_fault').rglob('*.pyc'):
        if '__pycache__' in path.parts and not path.is_symlink():
            remove.append((path, 'regenerable Python bytecode'))
    install = ROOT / 'build-propagation-p1/cmake_install.cmake'
    if install.stat().st_size > 100 * 1024**2:
        with install.open() as stream:
            assert stream.readline().startswith('# Install script for directory:')
        remove.append((install, 'oversized generated CMake installation script; reconfigure before install'))
    deletions = []
    for path, reason in remove:
        name = str(path.relative_to(ROOT))
        if name in pinned or name in tracked:
            continue
        assert time.time() - path.stat().st_mtime > 300, 'Recent build product: ' + name
        deletions.append(dict(path=name, bytes=path.stat().st_size, sha256=digest(path), reason=reason))
    (HERE / 'deletions.json').write_text(json.dumps(deletions, indent=2) + '\n')
    for row in deletions:
        (ROOT / row['path']).unlink()

    # Share only immutable snapshot copies, never a writable source or live binary.
    archives = [p for p in BUILD.iterdir() if p.is_dir() and not p.is_symlink()
                and (p.name.endswith('-baseline') or p.name in
                     ['native-fe-uncached', 'native-fe-first-curvature', 'native-fe-pdas'])]
    groups = defaultdict(list)
    for folder in archives:
        for path in folder.rglob('*'):
            if path.is_file() and not path.is_symlink():
                s = path.stat()
                mode = stat.S_IMODE(s.st_mode) & ~0o222
                groups[(s.st_size, mode)].append(path)
    duplicates = []
    for (size, mode), paths in groups.items():
        hashes = defaultdict(list)
        for path in paths:
            hashes[digest(path)].append(path)
        for checksum, copies in hashes.items():
            for path in copies:
                path.chmod(mode)
            if len(copies) < 2:
                continue
            anchor = copies[0]
            linked = []
            for path in copies[1:]:
                if os.path.samefile(anchor, path):
                    continue
                assert anchor.read_bytes() == path.read_bytes()
                temporary = path.with_name(path.name + '.cleanup-link')
                assert not temporary.exists()
                os.link(anchor, temporary)
                try:
                    os.replace(temporary, path)
                finally:
                    if temporary.exists():
                        temporary.unlink()
                assert digest(path) == checksum
                linked.append(str(path.relative_to(ROOT)))
            if linked:
                duplicates.append(dict(anchor=str(anchor.relative_to(ROOT)), sha256=checksum,
                                       bytes_each=size, copies=linked))
    (HERE / 'shared_snapshots.json').write_text(json.dumps(duplicates, indent=2) + '\n')

    # Group artifacts physically; compatibility links keep all old paths usable.
    moves = []
    folders = [p for p in BUILD.iterdir() if p.is_dir() and not p.is_symlink()
               and (p in archives or p.name.startswith('output-'))]
    for path in sorted(folders):
        assert not any(p.is_symlink() for p in path.rglob('*')), 'Review internal links before moving ' + str(path)
        group = 'runs' if path.name.startswith('output-') else ('archive/baselines' if path.name.endswith('-baseline') else 'archive/executables')
        destination = BUILD / group / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        assert not destination.exists()
        path.rename(destination)
        try:
            path.symlink_to(os.path.relpath(destination, path.parent), target_is_directory=True)
        except Exception:
            destination.rename(path)
            raise
        moves.append(dict(original=str(path.relative_to(ROOT)), physical=str(destination.relative_to(ROOT))))
    (HERE / 'relocations.json').write_text(json.dumps(moves, indent=2) + '\n')

    for name, checksum in before['protected'].items():
        assert digest(ROOT / name) == checksum, name
    for name, checksum in before['executables'].items():
        assert digest(ROOT / name) == checksum, name
    current_size = int(subprocess.check_output(['du', '-scB1', *map(str, ROOT.glob('build-*'))], text=True).splitlines()[-1].split()[0])
    result = dict(deleted_files=len(deletions), deleted_bytes=sum(x['bytes'] for x in deletions),
                  duplicate_groups=len(duplicates), duplicate_bytes_shared=sum(x['bytes_each'] * len(x['copies']) for x in duplicates),
                  relocated_directories=len(moves), protected_files_verified=len(before['protected']),
                  executables_verified=len(before['executables']), build_bytes_before=before['scope_build_allocated_bytes'],
                  build_bytes_after=current_size, build_allocated_bytes_freed=before['scope_build_allocated_bytes'] - current_size,
                  disk_free_bytes_before=before['disk_free_bytes'],
                  disk_free_bytes_after=os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize)
    (HERE / 'cleanup.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
