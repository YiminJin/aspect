#!/usr/bin/env python3
"""Compare the accepted evolving-history checkpoint from the supplemental case."""
from pathlib import Path
import hashlib,json,struct,zlib
root=Path(__file__).resolve().parent
paths=[root/f'output-{v}-stage-j-open-top-2/restart/01' for v in ('reference','candidate')]
checks={}
for name in ('mesh','mesh.info','mesh_fixed.data','mesh_variable.data'):
 checks[name]= (paths[0]/name).read_bytes()==(paths[1]/name).read_bytes()
def fingerprint(path):
 data=zlib.decompress((path/'resume.z').read_bytes()[16:])
 key=b'StageJRestart';assert data.count(key)==1
 start=data.index(key)+len(key)
 size=struct.unpack_from('<Q',data,start)[0];start+=8
 blob=data[start:start+size]
 assert len(blob)==size and b'serialization::archive' in blob[:40]
 return blob
left,right=map(fingerprint,paths)
checks['StageJRestart-history-V-geometry-bulk-fingerprint']=left==right
(root/'evidence/cohesive-checkpoint-comparison.json').write_text(json.dumps(checks,indent=2)+'\n')
print('Evolving cohesive checkpoint:',checks)
assert all(checks.values())
