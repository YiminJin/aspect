#!/usr/bin/env python3
"""Execute one explicitly reviewed, recorded build-particle-free file batch."""
from pathlib import Path
import argparse, datetime, hashlib, json, os, stat

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    rows = json.loads(args.manifest.read_text())
    protected = json.loads((HERE/'protected-text-sha256.json').read_text())
    seen = set()
    archive_hashes = {}
    for row in rows:
        name = row['path']; path = ROOT/name
        assert name not in seen, name
        if name in protected:
            if 'retained_text' in row:
                assert path.name in ['bp3_accepted_state.txt', 'last_good_checkpoint.txt'], name
                assert '/restart/' in name, name
                assert hashlib.sha256(row['retained_text'].encode()).hexdigest() == protected[name], name
            else:
                # Only an exact, independently read archive copy can replace source.
                archive = row['verified_archive']
                if archive not in archive_hashes:
                    archive_hashes[archive] = hashlib.file_digest((ROOT/archive).open('rb'), 'sha256').hexdigest()
                assert archive_hashes[archive] == row['archive_sha256'], archive
                assert row['member_sha256'] == protected[name], name
        seen.add(name)
        assert path.is_relative_to(ROOT/'build-particle-free'), name
        assert path.resolve() == path and '..' not in Path(name).parts, name
        s = path.lstat()
        assert stat.S_ISREG(s.st_mode), name
        assert s.st_size == row['size'] and s.st_ino == row['inode'], name
        assert row['reason'] and row['retained'], name
    # Flush the complete manifest, including pre-deletion hashes, before mutation.
    for row in rows:
        row['sha256'] = hashlib.file_digest((ROOT/row['path']).open('rb'), 'sha256').hexdigest()
    args.manifest.write_text(json.dumps(rows, indent=2)+'\n')
    before = os.statvfs(ROOT); free_before = before.f_bavail*before.f_frsize
    for row in rows:
        (ROOT/row['path']).unlink()
    os.sync()
    after = os.statvfs(ROOT)
    result = dict(files=len(rows), logical_allocated_bytes=sum(r['allocated'] for r in rows),
                  filesystem_free_change_bytes=after.f_bavail*after.f_frsize-free_before,
                  completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    args.manifest.with_suffix('.result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))

if __name__ == '__main__':
    main()
