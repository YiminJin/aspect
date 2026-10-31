#!/usr/bin/env python3
"""Locate retained cleanup evidence, optionally recover one file without overwrite."""
from pathlib import Path
import argparse, gzip, hashlib, json, subprocess, tarfile

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent

def locate(name):
    p = ROOT/name
    if p.is_file():
        return dict(file=name, sha256=hashlib.file_digest(p.open('rb'), 'sha256').hexdigest())
    with gzip.open(HERE/'recovery-map.json.gz', 'rt') as stream:
        mapping = json.load(stream)
    if name in mapping:
        return mapping[name]
    raise SystemExit('No retained original at this name. Consult batch manifests for intentional pruning/rebuild instructions.')

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('original_path')
    parser.add_argument('--destination', type=Path, help='Recover to this new file; never overwrite different contents.')
    args = parser.parse_args()
    item = locate(args.original_path)
    print(json.dumps(item, indent=2))
    if args.destination is None:
        return
    if 'text' in item:
        data = item['text'].encode()
    elif 'file' in item:
        data = (ROOT/item['file']).read_bytes()
        item.setdefault('mode', (ROOT/item['file']).stat().st_mode & 0o777)
    elif 'compressed' in item:
        data = subprocess.check_output(['zstd', '-q', '-dc', str(ROOT/item['compressed'])])
    else:
        archive = ROOT/item['archive']
        process = subprocess.Popen(['zstd', '-q', '-dc', str(archive)], stdout=subprocess.PIPE) if archive.name.endswith('.zst') else None
        stream = tarfile.open(fileobj=process.stdout, mode='r|') if process else tarfile.open(archive, 'r|*')
        data = None
        with stream:
            for member in stream:
                if member.name == item['member']:
                    assert member.isfile()
                    data = stream.extractfile(member).read()
                    item.setdefault('mode', member.mode)
                    break
        if process:
            process.stdout.close()
            process.wait()  # Early member completion can give the reader SIGPIPE.
        assert data is not None, item
    assert hashlib.sha256(data).hexdigest() == item['sha256']
    if args.destination.exists():
        assert args.destination.read_bytes() == data, 'Refusing to overwrite different contents'
        return
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with args.destination.open('xb') as output:
        output.write(data)
    args.destination.chmod(item.get('mode', 0o644))

if __name__ == '__main__':
    main()
