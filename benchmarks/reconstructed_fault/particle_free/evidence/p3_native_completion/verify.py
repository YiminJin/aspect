#!/usr/bin/env python3
"""Verify archived native P3 files and repeat the independent transfer checks."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import tempfile

here=Path(__file__).resolve().parent
sys.path.insert(0,str(here.parents[1]))
from assess_p3_extension import assess

manifest=json.loads((here/'archive-manifest.json').read_text())
archive=here/manifest['archive']
assert hashlib.sha256(archive.read_bytes()).hexdigest()==manifest['sha256']
with tempfile.TemporaryDirectory(prefix='p3-native-replay-') as directory:
    target=Path(directory)
    with tarfile.open(archive) as source:
        members=source.getmembers()
        assert len(members)==len(manifest['files'])
        for member in members:
            assert member.isfile() and member.name in manifest['files']
            data=source.extractfile(member).read()
            assert hashlib.sha256(data).hexdigest()==manifest['files'][member.name]['sha256']
            path=target/member.name
            assert path.resolve().is_relative_to(target.resolve())
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    for case in ['original','sensitive']:
        measured=assess(target/case,target/(case+'-after'),target/(case+'-before'))
        # assess applies the predeclared numerical screens and checks persistent
        # native binary identity, both restart sides and native stresses/loads.
        assert measured['native_restart_before_append_and_after_commit_exact']
        print(case,'relative L2',measured['reconstructed_full_tensor_relative_L2'],
              'unchanged-T',measured['unchanged_T_negative_control_relative_L2'])
print('Verified',len(manifest['files']),'native exports/checkpoint files; both independent assessments pass.')
