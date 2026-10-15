#!/usr/bin/env python3
"""One-shot, manifest-based disk cleanup requested on 2026-10-15."""

import gzip
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tarfile
import time


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
BUILD = ROOT / 'build-propagation-integration'
ARCHIVE = BUILD / 'archive/cleanup-20261015/P2-bulk-fields.tar.zst'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def files_under(base):
    for current, directories, names in os.walk(base, followlinks=False):
        for name in list(directories):
            path = Path(current) / name
            if path.is_symlink():
                directories.remove(name)
                yield path
        for name in names:
            yield Path(current) / name


def main():
    assert not (HERE / 'before.json.gz').exists(), 'Do not repeat a recorded cleanup.'
    assert not ARCHIVE.exists()
    busy = subprocess.run(['ps', '-C', 'aspect,aspect-release,aspect.exe,cc1plus,cmake,make',
                           '-o', 'pid,etime,comm', '--no-headers'], capture_output=True, text=True)
    assert busy.returncode == 1 and not busy.stdout.strip(), busy.stdout
    builds = sorted(p for p in ROOT.glob('build-*') if p.is_dir() and not p.is_symlink())
    scope = [ROOT / 'benchmarks/reconstructed_fault', ROOT / 'doc/reconstructed_fault', *builds]
    tracked = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0'))
    status = subprocess.check_output(['git', 'status', '--porcelain=v1', '-uall'], cwd=ROOT)
    with gzip.open(HERE / 'incoming-status.txt.gz', 'wb') as stream:
        stream.write(status)
    with gzip.open(HERE / 'incoming.diff.gz', 'wb') as stream:
        stream.write(subprocess.check_output(['git', 'diff', '--binary', 'HEAD'], cwd=ROOT))
    modified = subprocess.check_output(['git', 'diff', '--name-only', 'HEAD', '-z'], cwd=ROOT)
    incoming = {name: digest(ROOT / name) for name in modified.decode().split('\0') if name}
    before = dict(root=str(ROOT), head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
                  free_bytes=os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize,
                  incoming_modified=incoming, files={})
    for base in scope:
        for path in files_under(base):
            if path.is_relative_to(HERE):
                continue
            info = path.lstat()
            row = dict(mode=stat.S_IMODE(info.st_mode), mtime_ns=info.st_mtime_ns,
                       bytes=info.st_size, allocated_bytes=info.st_blocks * 512)
            if stat.S_ISLNK(info.st_mode):
                row['link'] = os.readlink(path)
            else:
                assert stat.S_ISREG(info.st_mode), str(path)
                row['sha256'] = digest(path)
            before['files'][str(path.relative_to(ROOT))] = row
    with gzip.open(HERE / 'before.json.gz', 'wt') as stream:
        json.dump(before, stream)
    print('Recorded pre-cleanup hashes for', len(before['files']), 'paths.', flush=True)

    deleted = []
    archived = []
    for name, row in before['files'].items():
        if 'sha256' not in row or name in tracked:
            continue
        path = ROOT / name
        reason = None
        if path.is_relative_to(ROOT / 'benchmarks/reconstructed_fault'):
            if path.suffix == '.pyc' and '__pycache__' in path.parts:
                reason = 'regenerable Python bytecode'
        elif name.startswith('build-') and 'CMakeFiles' in path.parts and path.suffix in {'.o', '.gch'}:
            reason = 'regenerable compiler object/precompiled header'
        elif path == BUILD / 'tests/cmake_install.cmake' and row['bytes'] > 100 * 1024**2:
            with path.open() as stream:
                assert stream.readline().startswith('# Install script for directory:')
            reason = 'oversized generated CMake install script; reconfigure before installation'
        if reason:
            assert time.time_ns() - row['mtime_ns'] > 300 * 10**9, name
            deleted.append(dict(path=name, reason=reason, **row))
        if (path.is_relative_to(BUILD / 'runs') and path.name.startswith('fe-bulk-')
                and path.suffix == '.csv' and row['bytes'] >= 8 * 1024**2):
            assert path.parent.name in {'output-normal-resolution-q2-n256',
                                       'output-normal-resolution-q2-n256-probe-failed'}, name
            archived.append(dict(path=name, **row))
    assert len(archived) == 8
    write_json(HERE / 'deletions.json', deleted)
    write_json(HERE / 'archived-files.json', archived)

    # Verify every compressed member before removing any scientific data.
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    with ARCHIVE.open('xb') as output:
        compressor = subprocess.Popen(['zstd', '-T2', '-6', '-q', '-c'], stdin=subprocess.PIPE, stdout=output)
        try:
            with tarfile.open(fileobj=compressor.stdin, mode='w|', format=tarfile.PAX_FORMAT) as bundle:
                for row in archived:
                    bundle.add(ROOT / row['path'], arcname=row['path'], recursive=False)
        finally:
            compressor.stdin.close()
        assert compressor.wait() == 0
    expected = {row['path']: row for row in archived}
    seen = set()
    decompressor = subprocess.Popen(['zstd', '-q', '-d', '-c', str(ARCHIVE)], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=decompressor.stdout, mode='r|') as bundle:
            for member in bundle:
                assert member.isfile() and member.name in expected and member.name not in seen
                with bundle.extractfile(member) as stream:
                    assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected[member.name]['sha256']
                seen.add(member.name)
        # Consume the tar end padding so the decompressor can finish normally.
        while decompressor.stdout.read(1024 * 1024):
            pass
    finally:
        decompressor.stdout.close()
    assert decompressor.wait() == 0 and seen == set(expected)
    write_json(HERE / 'archive.json', dict(path=str(ARCHIVE.relative_to(ROOT)), sha256=digest(ARCHIVE),
                                         bytes=ARCHIVE.stat().st_size, verified_members=len(seen)))
    for row in [*archived, *deleted]:
        path = ROOT / row['path']
        assert not path.is_symlink() and digest(path) == row['sha256'], row['path']
        path.unlink()
    print('Removed only verified archive members and listed regenerable files.', flush=True)

    moves = []
    for name in ['localization-baseline', 'normal-resolution-baseline', 'plane-strain-baseline', 'quadrature-baseline']:
        original = BUILD / name
        destination = BUILD / 'archive/baselines' / name
        assert original.is_dir() and not original.is_symlink() and not destination.exists()
        assert not any(path.is_symlink() for path in files_under(original))
        original.rename(destination)
        try:
            original.symlink_to(os.path.relpath(destination, original.parent), target_is_directory=True)
        except Exception:
            destination.rename(original)
            raise
        moves.append(dict(original=str(original.relative_to(ROOT)), physical=str(destination.relative_to(ROOT))))
    write_json(HERE / 'relocations.json', moves)

    removed = {row['path'] for row in [*archived, *deleted]}
    verified = 0
    existing_broken = []
    for name, row in before['files'].items():
        path = ROOT / name
        if name in removed:
            assert not path.exists()
        elif 'link' in row:
            assert path.is_symlink() and os.readlink(path) == row['link'], name
            if not path.exists():
                existing_broken.append(name)
        else:
            assert digest(path) == row['sha256'], name
            verified += 1
    for name, checksum in incoming.items():
        assert digest(ROOT / name) == checksum, name
    result = dict(retained_files_sha256_verified=verified, symlinks_verified=sum('link' in r for r in before['files'].values()),
                  existing_broken_symlinks=existing_broken, incoming_modified_files_verified=len(incoming),
                  deleted_files=len(deleted), deleted_allocated_bytes=sum(row['allocated_bytes'] for row in deleted),
                  archived_files=len(archived), archived_original_bytes=sum(row['bytes'] for row in archived),
                  archive_bytes=ARCHIVE.stat().st_size, relocated_directories=len(moves),
                  free_bytes_before=before['free_bytes'],
                  free_bytes_after=os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize)
    write_json(HERE / 'verification.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
