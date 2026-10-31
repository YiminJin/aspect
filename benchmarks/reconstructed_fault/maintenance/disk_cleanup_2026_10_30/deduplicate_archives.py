#!/usr/bin/env python3
"""Retain unique field records; remove only independently verified archive copies."""
from pathlib import Path
import collections, hashlib, json, subprocess, tarfile

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent

def digest(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()

def main():
    candidates = json.loads((HERE/'field-archive-dedup-candidates.json').read_text())
    expected = {}
    archives = {a['path']: a for a in json.loads((HERE/'external-archive-members.json').read_text())}
    for a in candidates:
        for name, (archive, member) in a['duplicate_files'].items():
            expected.setdefault(archive, set()).add(member)
    verified = {}
    for archive, names in expected.items():
        matched = 0
        with tarfile.open(ROOT/archive, 'r|xz') as stream:
            for member in stream:
                if member.name not in names or not member.isfile():
                    continue
                h = hashlib.sha256(stream.extractfile(member).read()).hexdigest()
                assert h == archives[archive]['members'][member.name], (archive, member.name)
                verified[h] = (archive, member.name)
                matched += 1
        print('Verified', archive, matched, 'actual members', flush=True)
    (HERE/'verified-external-members.json').write_text(json.dumps(verified, indent=2)+'\n')
    deletions = []
    replacements = []
    for item in candidates:
        original = ROOT/item['path']
        old = next(a for a in json.loads((ROOT/item['manifest']).read_text()) if a['path'] == item['path'])
        assert digest(original) == old['sha256'], original
        duplicates = {n: verified[r['sha256']] for n, r in old['files'].items() if r['sha256'] in verified}
        if not duplicates:
            continue
        kept = {n: r for n, r in old['files'].items() if n not in duplicates}
        replacement = original.with_name(original.name.replace('.tar.zst', '.unique.tar.zst'))
        assert not replacement.exists()
        if kept:
            reader = subprocess.Popen(['zstd', '-q', '-dc', str(original)], stdout=subprocess.PIPE)
            with replacement.open('wb') as output:
                writer = subprocess.Popen(['zstd', '-q', '-T1', '-3', '-c'], stdin=subprocess.PIPE, stdout=output)
                with tarfile.open(fileobj=reader.stdout, mode='r|') as source, tarfile.open(fileobj=writer.stdin, mode='w|') as target:
                    for member in source:
                        if member.name not in kept:
                            continue
                        assert member.isfile(), member.name
                        data = source.extractfile(member).read()
                        assert hashlib.sha256(data).hexdigest() == kept[member.name]['sha256']
                        import io
                        target.addfile(member, io.BytesIO(data))
                writer.stdin.close()
                assert writer.wait() == 0
            reader.stdout.close()
            assert reader.wait() == 0
            # Re-read the newly compressed archive, independently of write buffers.
            reader = subprocess.Popen(['zstd', '-q', '-dc', str(replacement)], stdout=subprocess.PIPE)
            found = set()
            with tarfile.open(fileobj=reader.stdout, mode='r|') as source:
                for member in source:
                    assert hashlib.sha256(source.extractfile(member).read()).hexdigest() == kept[member.name]['sha256']
                    found.add(member.name)
            reader.stdout.close()
            assert reader.wait() == 0 and found == kept.keys()
        record = dict(original=item['path'], replacement=str(replacement.relative_to(ROOT)) if kept else None,
                      retained_members=kept, external_duplicate_members=duplicates,
                      replacement_sha256=digest(replacement) if kept else None)
        replacements.append(record)
        s = original.stat()
        deletions.append(dict(path=item['path'], inode=s.st_ino, size=s.st_size, allocated=s.st_blocks*512,
                              reason='Every member retained exactly once: independently read/hash-verified external reference archive or verified unique-member replacement',
                              retained='archive-replacements.json: '+str(replacement.relative_to(ROOT)) if kept else 'archive-replacements.json: verified external reference archives'))
        (HERE/'archive-replacements.json').write_text(json.dumps(replacements, indent=2)+'\n')
        (HERE/'batch08-duplicate-field-archives.json').write_text(json.dumps(deletions, indent=2)+'\n')
        print('Prepared', original.name, len(duplicates), 'duplicates,', len(kept), 'unique records', flush=True)

if __name__ == '__main__':
    main()
