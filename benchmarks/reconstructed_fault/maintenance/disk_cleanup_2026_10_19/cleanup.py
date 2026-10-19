#!/usr/bin/env python3
"""Manifest-based cleanup of this particle-free worktree only; explicit phases."""
from collections import defaultdict
import argparse
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
BUILD = ROOT / 'build-particle-free'
ARCHIVES = BUILD / 'archive/disk-cleanup-20261019'
HISTORICAL = {'runs', 'mature-pilot', 'mature-resolution', 'mature-dg', 'publication-replay'}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def load_before():
    with gzip.open(HERE / 'before.json.gz', 'rt') as stream:
        return json.load(stream)


def free_bytes():
    info = os.statvfs(ROOT)
    return info.f_bavail * info.f_frsize


def paths_under(base):
    for current, directories, files in os.walk(base, followlinks=False):
        for name in list(directories):
            path = Path(current) / name
            if path == HERE or path.name in {'.git', '.aws', '.codex', '.agents'}:
                directories.remove(name)
            elif path.is_symlink():
                directories.remove(name)
                yield path
        for name in files:
            yield Path(current) / name


def inventory():
    assert ROOT.name == 'aspect-pf-rsf-particle-free'
    assert subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip() == 'pf-rsf-particle-free'
    assert not (HERE / 'before.json.gz').exists(), 'Inventory already exists.'
    busy = subprocess.run(['ps', '-C', 'aspect,aspect-release,aspect.exe.release,cc1plus,cmake,make,mpirun',
                           '-o', 'pid,etime,comm', '--no-headers'], capture_output=True, text=True)
    assert busy.returncode == 1 and not busy.stdout.strip(), busy.stdout
    before = {'root': str(ROOT), 'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'free_bytes': free_bytes(), 'files': {}}
    for command, filename in [(['git', 'status', '--porcelain=v1', '-uall'], 'incoming-status.txt.gz'),
                              (['git', 'diff', '--binary', 'HEAD'], 'incoming.diff.gz')]:
        with gzip.open(HERE / filename, 'wb') as stream:
            stream.write(subprocess.check_output(command, cwd=ROOT))
    total = 0
    started = time.monotonic()
    for i, path in enumerate(paths_under(ROOT)):
        info = path.lstat()
        if not (stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)):
            continue
        row = {'mode': stat.S_IMODE(info.st_mode), 'mtime_ns': info.st_mtime_ns,
               'bytes': info.st_size, 'allocated_bytes': info.st_blocks * 512,
               'inode': info.st_ino, 'nlink': info.st_nlink}
        if path.is_symlink():
            row['link'] = os.readlink(path)
        else:
            row['sha256'] = digest(path)
            total += info.st_size
        before['files'][str(path.relative_to(ROOT))] = row
        if i % 10000 == 0:
            print(f'Inventory {i} paths, {total / 2**30:.2f} GiB hashed, {time.monotonic()-started:.1f} s', flush=True)
    temporary = HERE / 'before.json.gz.partial'
    with gzip.open(temporary, 'wt') as stream:
        json.dump(before, stream)
    temporary.replace(HERE / 'before.json.gz')
    print(f"Inventory complete: {len(before['files'])} paths, {total/2**30:.2f} GiB.", flush=True)


def deduplicate():
    before = load_before()
    assert not (HERE / 'shared-executables.json').exists()
    groups = defaultdict(list)
    for name, row in before['files'].items():
        path = Path(name)
        if (not name.startswith('build-particle-free/') or 'sha256' not in row
                or not path.name.startswith('aspect-') or row['bytes'] < 10 * 2**20
                or row['nlink'] != 1 or any(x in path.parts for x in ('restart', 'checkpoint', 'checkpoints'))):
            continue
        # Never share a live build target or anything in a CMake build tree.
        if any((parent / 'CMakeCache.txt').exists() for parent in (ROOT / path).parents if parent.is_relative_to(BUILD)):
            continue
        with (ROOT / path).open('rb') as stream:
            if stream.read(4) != b'\x7fELF':
                continue
        groups[(row['sha256'], row['mode'] & ~0o222)].append(name)
    records = []
    for (checksum, mode), names in sorted(groups.items()):
        if len(names) < 2:
            continue
        # The immutable store is another hard link. Original loader paths and
        # $ORIGIN are preserved; no live binary/source is used as an anchor.
        store = BUILD / 'archive/executables' / checksum / f'aspect-{mode:o}'
        store.parent.mkdir(parents=True, exist_ok=True)
        assert not store.exists()
        assert all(digest(ROOT / name) == checksum for name in names)
        assert all(time.time_ns() - (ROOT / name).stat().st_mtime_ns > 600 * 10**9 for name in names)
        os.link(ROOT / names[0], store)
        store.chmod(mode)
        for name in names[1:]:
            path = ROOT / name
            temporary = path.with_name(path.name + '.cleanup-link')
            assert not temporary.exists()
            os.link(store, temporary)
            os.replace(temporary, path)
        records.append({'sha256': checksum, 'store': str(store.relative_to(ROOT)), 'mode': mode,
                        'bytes_each': before['files'][names[0]]['bytes'], 'paths': names,
                        'allocated_bytes_reclaimed': sum(before['files'][n]['allocated_bytes'] for n in names[1:])})
        save(HERE / 'shared-executables.json', records)
        print(f'Shared {len(names)} frozen executables, reclaimed {records[-1]["allocated_bytes_reclaimed"]/2**30:.2f} GiB', flush=True)
    if not records:
        save(HERE / 'shared-executables.json', records)


def historical_group(name, row):
    path = Path(name)
    if (not name.startswith('build-particle-free/') or 'sha256' not in row
            or path.suffix != '.csv' or row['bytes'] < 2**20):
        return None
    parts = path.parts
    if parts[1] not in HISTORICAL or any(x in parts for x in ('archive', 'restart', 'checkpoint', 'checkpoints', 'distribution', 'package')):
        return None
    # Keep compact surface/acceptance/weak-load and profile evidence extracted.
    if any(x in path.name for x in ('surface', 'weak', 'profile', 'accepted', 'checks', 'summary')):
        return None
    if parts[1] != 'runs':
        return parts[1]
    label = parts[2]
    if label.startswith('pf6a-publication'):
        return 'pf6a-publication'
    for stage in ('pf1', 'pf2', 'pf3', 'pf4', 'pf5', 'pf6a', 'weno'):
        if label.startswith(stage):
            return stage
    return 'other-historical-runs'


def read_archive(archive, visit):
    process = subprocess.Popen(['zstd', '-q', '-d', '-c', str(archive)], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdout, mode='r|') as bundle:
            for member in bundle:
                visit(bundle, member)
        while process.stdout.read(2**20):
            pass
    finally:
        process.stdout.close()
        code = process.wait()
    assert code == 0


def archive_fields():
    before = load_before()
    assert not (HERE / 'field-archives.json').exists()
    groups = defaultdict(list)
    for name, row in before['files'].items():
        group = historical_group(name, row)
        if group:
            groups[group].append(name)
    records = []
    ARCHIVES.mkdir(parents=True, exist_ok=True)
    for group, names in sorted(groups.items()):
        archive = ARCHIVES / f'{group}-dense-fields.tar.zst'
        assert not archive.exists()
        partial = archive.with_name(archive.name + '.partial')
        with partial.open('xb') as output:
            process = subprocess.Popen(['zstd', '-T2', '-3', '-q', '-c'], stdin=subprocess.PIPE, stdout=output)
            try:
                with tarfile.open(fileobj=process.stdin, mode='w|', format=tarfile.PAX_FORMAT) as bundle:
                    for name in names:
                        path = ROOT / name
                        assert digest(path) == before['files'][name]['sha256']
                        assert time.time_ns() - path.stat().st_mtime_ns > 600 * 10**9
                        bundle.add(path, arcname=name, recursive=False)
            finally:
                process.stdin.close()
            assert process.wait() == 0
        partial.replace(archive)
        seen = set()
        def check(bundle, member):
            assert member.isfile() and member.name in names and member.name not in seen
            with bundle.extractfile(member) as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == before['files'][member.name]['sha256']
            seen.add(member.name)
        read_archive(archive, check)
        assert seen == set(names)
        record = {'group': group, 'path': str(archive.relative_to(ROOT)), 'sha256': digest(archive),
                  'bytes': archive.stat().st_size, 'original_bytes': sum(before['files'][n]['bytes'] for n in names),
                  'files': {name: before['files'][name] for name in names}}
        records.append(record)
        # Persist restoration metadata BEFORE deleting a verified source file.
        save(HERE / 'field-archives.json', records)
        for name in names:
            path = ROOT / name
            assert digest(path) == before['files'][name]['sha256']
            path.unlink()
        for parent in sorted({str(Path(name).parent) for name in names}):
            note = ROOT / parent / 'ARCHIVED_FIELDS.md'
            note.write_text('Dense historical fields are preserved losslessly in '+str(archive.relative_to(ROOT))+
                            '.\nSee benchmarks/reconstructed_fault/maintenance/disk_cleanup_2026_10_19/README.md '
                            'for verified restoration before raw-field analysis.\n')
        print(f'Archived {group}: {len(names)} files, {record["original_bytes"]/2**30:.2f} -> {record["bytes"]/2**30:.2f} GiB', flush=True)
    if not records:
        save(HERE / 'field-archives.json', records)


def packages():
    before = load_before()
    assert not (HERE / 'package-copies.json').exists()
    parents = [BUILD / 'first-event-package', BUILD / 'bp3-startup-review/package',
               BUILD / 'bp3-optimization/package', BUILD / 'bp3-optimization/package-before-final-report-check',
               BUILD / 'bp3-homogeneous-coefficients/package']
    records, skipped = [], []
    for parent in parents:
        folder = parent / 'distribution'
        bundles = list(parent.glob('*.tar.gz'))
        if not folder.is_dir() or len(bundles) != 1:
            continue
        archive = bundles[0]
        expected = {str(p.relative_to(folder)): p for p in paths_under(folder)}
        seen, different = set(), []
        with tarfile.open(archive, 'r:gz') as bundle:
            for member in bundle:
                assert member.isfile() and not member.name.startswith('/') and '..' not in Path(member.name).parts
                relative = str(Path(*Path(member.name).parts[1:]))
                assert relative in expected and relative not in seen
                path = expected[relative]
                assert not path.is_symlink()
                with bundle.extractfile(member) as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest() != digest(path):
                        different.append(relative)
                seen.add(relative)
        if seen != set(expected) or different:
            skipped.append({'folder': str(folder.relative_to(ROOT)),
                            'reason': 'Extra build/manual files or changed payload; preserve this entire tree.',
                            'extra_files': sorted(set(expected)-seen), 'different_files': different})
            save(HERE / 'package-copies-retained.json', skipped)
            print('Preserved expanded package with extra build/manual content:', folder.relative_to(ROOT), flush=True)
            continue
        rows = {str(p.relative_to(ROOT)): before['files'][str(p.relative_to(ROOT))] for p in expected.values()}
        record = {'folder': str(folder.relative_to(ROOT)), 'archive': str(archive.relative_to(ROOT)),
                  'archive_sha256': digest(archive), 'files': rows}
        records.append(record)
        save(HERE / 'package-copies.json', records)
        for name, row in rows.items():
            path = ROOT / name
            assert digest(path) == row['sha256']
            path.unlink()
        for current, directories, names in os.walk(folder, topdown=False):
            if Path(current) != folder:
                Path(current).rmdir()
        (folder / 'ARCHIVED_PACKAGE.md').write_text('This exact expanded copy was verified against '+
            str(archive.relative_to(ROOT))+'.\nRestore using the cleanup recovery tool before rebuilding from this copy.\n')
        print('Removed verified expanded copy:', folder.relative_to(ROOT), flush=True)
    save(HERE / 'package-copies.json', records)
    save(HERE / 'package-copies-retained.json', skipped)


def verify():
    before = load_before()
    shared = json.loads((HERE / 'shared-executables.json').read_text())
    archived = json.loads((HERE / 'field-archives.json').read_text())
    packaged = json.loads((HERE / 'package-copies.json').read_text())
    removed = {name for row in [*archived, *packaged] for name in row['files']}
    shared_names = {name for row in shared for name in row['paths']}
    cached, count = {}, 0
    for name, row in before['files'].items():
        path = ROOT / name
        if name in removed:
            assert not path.exists(), name
        elif 'link' in row:
            assert path.is_symlink() and os.readlink(path) == row['link'], name
        else:
            info = path.stat()
            key = (info.st_dev, info.st_ino)
            if key not in cached:
                cached[key] = digest(path)
            assert cached[key] == row['sha256'], name
            if name not in shared_names:
                assert stat.S_IMODE(info.st_mode) == row['mode'] and info.st_mtime_ns == row['mtime_ns'], name
            count += 1
    for row in shared:
        anchor = ROOT / row['store']
        assert digest(anchor) == row['sha256'] and stat.S_IMODE(anchor.stat().st_mode) == row['mode']
        assert all(os.path.samefile(anchor, ROOT / name) for name in row['paths'])
    result = {'retained_regular_files_hash_verified': count, 'retained_symlinks_verified': sum('link' in r for r in before['files'].values()),
              'shared_executable_groups': len(shared), 'shared_executable_paths': len(shared_names),
              'shared_executable_allocated_bytes_reclaimed': sum(r['allocated_bytes_reclaimed'] for r in shared),
              'historical_csv_files_archived': sum(len(r['files']) for r in archived),
              'archived_csv_original_bytes': sum(r['original_bytes'] for r in archived),
              'compressed_csv_archive_bytes': sum(r['bytes'] for r in archived),
              'verified_expanded_packages_removed': len(packaged),
              'expanded_package_allocated_bytes_reclaimed': sum(r['allocated_bytes'] for p in packaged for r in p['files'].values()),
              'free_bytes_before': before['free_bytes'], 'free_bytes_after': free_bytes(),
              'no_unique_scientific_data_deleted': True, 'live_build_and_current_bp3_runs_unchanged': True,
              'checkpoint_files_unchanged': True}
    save(HERE / 'verification.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['inventory', 'deduplicate', 'archive-fields', 'packages', 'verify'])
    arguments = parser.parse_args()
    {'inventory': inventory, 'deduplicate': deduplicate, 'archive-fields': archive_fields,
     'packages': packages, 'verify': verify}[arguments.action]()
