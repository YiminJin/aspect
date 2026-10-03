#!/usr/bin/env python3
"""Negative control: change only saved manager RNG words in a cloned snapshot.

Keep binary string lengths, generator streams and every physical/particle value
unchanged. This is test fixture corruption, never a production restart utility.
"""
from pathlib import Path
import hashlib, json, struct, zlib, sys
r = Path(__file__).resolve().parent
source = r/(sys.argv[1] if len(sys.argv)>1 else 'output-create-fixed')
target = r/(sys.argv[2] if len(sys.argv)>2 else 'output-rng-negative')
assert source.parent==r and target.parent==r and 'rng-negative' in target.name
label = (target/'restart/last_good_checkpoint.txt').read_text().strip().zfill(2)
p = target/'restart'/label/'resume.z'
packed = p.read_bytes()
blocks, size, last, compressed = struct.unpack('=4I', packed[:16])
assert blocks == 1 and size == last and compressed == len(packed)-16
raw = zlib.decompress(packed[16:]); assert len(raw) == size
before = raw
changes = []
for rank in range(2):
    stream = (source/f'rng-checkpoint-2-rank{rank}.txt').read_bytes()
    manager, generator = stream.split(b'\n')
    words = manager.split(b' ')
    assert len(words) == 625
    changed = b' '.join(list(reversed(words[:-1]))+[words[-1]])+b'\n'+generator
    assert len(changed) == len(stream) and raw.count(stream) == 1
    offset = raw.index(stream)
    raw = raw[:offset]+changed+raw[offset+len(stream):]
    changes.append(dict(rank=rank, offset=offset, bytes=len(stream),
                        before=hashlib.sha256(stream).hexdigest(),
                        after=hashlib.sha256(changed).hexdigest()))
assert len(raw) == len(before)
encoded = zlib.compress(raw,9)
p.write_bytes(struct.pack('=4I',1,size,size,len(encoded))+encoded)
(r/'evidence'/(target.name+'-fixture.json')).write_text(json.dumps(changes,indent=2)+'\n')
