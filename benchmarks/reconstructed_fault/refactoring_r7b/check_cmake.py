#!/usr/bin/env python3
"""Exercise the committed package-check block; this does not test a Tpetra backend."""
from pathlib import Path
import subprocess,json,hashlib
r=Path(__file__).resolve().parent;repo=r.parents[2]
s=(repo/'CMakeLists.txt').read_text();block=s[s.index('# Determine if Trilinos provides all necessary packages'):s.index('if(NOT DEAL_II_WITH_SUNDIALS)')]
checks=[]
for name,version,epetra,tpetra,muelu,requested,ok,selected in [
 ('old-epetra','9.6.2',1,0,0,0,1,0),('old-tpetra-rejected','9.6.2',1,1,1,1,0,1),
 ('new-epetra','9.8.0',1,0,0,0,1,0),('new-tpetra-missing','9.8.0',1,0,0,1,0,1),
 ('new-both-epetra','9.8.0',1,1,1,0,1,0),('new-both-tpetra','9.8.0',1,1,1,1,1,1),
 ('new-muelu-missing','9.8.0',1,1,0,1,0,1),('new-forced-tpetra','9.8.0',0,1,1,0,1,1),
 ('new-neither','9.8.0',0,0,0,0,0,1)]:
 p=r/'build'/f'{name}.cmake'
 text=f'set(DEAL_II_PACKAGE_VERSION {version})\nset(DEAL_II_TRILINOS_WITH_EPETRA {epetra})\nset(DEAL_II_TRILINOS_WITH_TPETRA {tpetra})\nset(DEAL_II_TRILINOS_WITH_TPETRA_MUELU {muelu})\nset(ASPECT_USE_TPETRA {"ON" if requested else "OFF"} CACHE BOOL "test")\n'+block
 p.write_text(text);x=subprocess.run(['cmake','-P',str(p)],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 (r/'evidence'/f'cmake-{name}.log').write_text(x.stdout)
 checks.append(dict(case=name,passed=(x.returncode==0)==bool(ok) and f"Using ASPECT_USE_TPETRA = '{'ON' if selected else 'OFF'}'" in x.stdout,exit_code=x.returncode))
(r/'evidence/cmake-branches.json').write_text(json.dumps({'block_sha256':hashlib.sha256(block.encode()).hexdigest(),'checks':checks},indent=2)+'\n')
print(sum(x['passed'] for x in checks),'/',len(checks));assert all(x['passed'] for x in checks)
