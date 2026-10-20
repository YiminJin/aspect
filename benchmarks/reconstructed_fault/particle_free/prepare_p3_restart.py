#!/usr/bin/env python3
"""Copy a native P3 checkpoint to a fresh branch; never edit its source."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('source',type=Path)
p.add_argument('destination',type=Path)
p.add_argument('--side',choices=['before','after'],required=True)
a=p.parse_args()
assert not a.destination.exists(), 'Refusing to overwrite a restart branch'
checkpoint='1' if a.side=='before' else '2'
assert (a.source/'restart'/checkpoint.zfill(2)).is_dir()
a.destination.mkdir(parents=True)
shutil.copytree(a.source/'restart',a.destination/'restart')
(a.destination/'restart/last_good_checkpoint.txt').write_text(checkpoint+'\n')
root=Path(__file__).resolve().parents[3]
(a.destination/'input.prm').write_text(
    'include '+str(root/'benchmarks/reconstructed_fault/particle_free/p3_native.prm')+'\n'
    'set Resume computation = true\nset Output directory = '+str(a.destination.resolve())+'\n')
files={str(f.relative_to(a.source)):hashlib.sha256(f.read_bytes()).hexdigest()
       for f in sorted((a.source/'restart'/checkpoint.zfill(2)).rglob('*')) if f.is_file()}
(a.destination/'origin.json').write_text(json.dumps(dict(source=str(a.source.resolve()),checkpoint=checkpoint,files=files),indent=2)+'\n')
print(a.destination/'input.prm')
