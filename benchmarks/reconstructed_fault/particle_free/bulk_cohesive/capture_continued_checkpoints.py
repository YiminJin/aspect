#!/usr/bin/env python3
"""Preserve completed rolling native checkpoints for this one bounded run.

Read only the last-good pointer. Never alter a checkpoint or request a solve.
The small metadata read is specific to this native v2 archive/host ABI; it is
not checkpoint conversion. Standard ASPECT restart still owns all decoding.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import struct
import tarfile
import time
import zlib

RUN = Path('build-particle-free/bulk-continued/runs/continued')


def capture(run, dest):
    pointer = run/'restart/last_good_checkpoint.txt'
    if not pointer.exists(): return None
    number = int(pointer.read_text()); directory = pointer.parent/f'{number:02d}'
    b = zlib.decompress((directory/'resume.z').read_bytes()[16:])
    tag = b'bulk cohesive BE native remainder v2'; start = b.index(tag)+len(tag)
    quadrature, step, publications, physical_time = struct.unpack_from('<IIQd', b, start)
    assert quadrature == 8 and step == publications
    if step < 95: return None
    archive = dest/f'accepted-{step:04d}.tar.gz'
    if archive.exists(): return None
    records = {}
    with tarfile.open(archive, 'x:gz', compresslevel=1) as tar:
        for p in sorted(directory.iterdir()):
            if not p.is_file(): continue
            before = hashlib.sha256(p.read_bytes()).hexdigest()
            tar.add(p, arcname=f'restart/{number:02d}/{p.name}')
            after = hashlib.sha256(p.read_bytes()).hexdigest(); assert before == after
            records[p.name] = dict(bytes=p.stat().st_size, sha256=before)
        data = str(number).encode(); info = tarfile.TarInfo('restart/last_good_checkpoint.txt')
        info.size = len(data); tar.addfile(info, io.BytesIO(data))
    record = dict(step=step, time=physical_time, publications=publications,
                  checkpoint_id=number, files=records,
                  archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    archive.with_suffix('.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(dict(captured=step, time=physical_time, bytes=archive.stat().st_size)), flush=True)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--seconds', type=float, default=14400)
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--output', type=Path, default=Path('build-particle-free/bulk-continued/saved-checkpoints'))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic()+args.seconds
    result = Path('benchmarks/reconstructed_fault/refactoring_boundary/evidence/bulk-continued-native.json')
    while time.monotonic() < deadline:
        capture(args.run, args.output)
        if result.exists(): break
        time.sleep(10)
