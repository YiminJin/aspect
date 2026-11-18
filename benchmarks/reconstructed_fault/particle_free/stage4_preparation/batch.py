#!/usr/bin/env python3
"""Durable sequential execution of an explicitly selected JSON case list."""
from pathlib import Path
import argparse,json,subprocess,sys,time,gzip,hashlib,shutil
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];BUILD=ROOT/'build-particle-free/stage4-preparation'
p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--worker',action='store_true');a=p.parse_args()
assert a.name.isidentifier()
cases=json.loads((BUILD/f'{a.name}-cases.json').read_text())
assert 1<=len(cases)<=3
if not a.worker:
 assert subprocess.run(['pgrep','-x','aspect-release'],capture_output=True).returncode==1,'Existing solver ranks are active.'
 with (BUILD/f'{a.name}.log').open('x') as log:
  job=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),a.name,'--worker'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 (BUILD/f'{a.name}-pid.json').write_text(json.dumps(dict(pid=job.pid,started_unix=time.time()),indent=2)+'\n')
 print('Preparation supervisor PID',job.pid);raise SystemExit(0)
started=time.monotonic();records=[]
for args in cases:
 if (BUILD/'stop-batch').exists():break
 now=time.monotonic();cmd=[sys.executable,str(HERE/'run.py'),*args]
 print('START',args,time.strftime('%Y-%m-%d %H:%M:%S %Z'),flush=True)
 code=subprocess.call(cmd,cwd=ROOT)
 row=dict(command=cmd,exit_code=code,seconds=time.monotonic()-now)
 # Compress only completed new rank-local observations, with no data loss.
 # No numerical ranks run while compression executes; checkpoints stay in place.
 out=ROOT/'build-particle-free/bulk-tip/runs'/('stage4-preparation-'+args[0])/'output'
 compressed={}
 for source in sorted([*out.glob('preload-scan-*.csv'),*out.glob('release-phase-*.csv'),*out.glob('prepared-history-*.bin'),*out.glob('tip-balance-*.csv'),*out.glob('tip-seed-fracture-*.csv'),*out.glob('restore-*.csv')]):
  data=source.read_bytes();destination=source.with_suffix(source.suffix+'.gz')
  with destination.open('xb') as raw:
   with gzip.GzipFile(filename='',fileobj=raw,mode='wb',compresslevel=4,mtime=0) as stream:stream.write(data)
  assert hashlib.sha256(gzip.decompress(destination.read_bytes())).digest()==hashlib.sha256(data).digest()
  compressed[source.name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),gzip_bytes=destination.stat().st_size)
  source.unlink()
 (out/'lossless-compression.json').write_text(json.dumps(compressed,indent=2)+'\n')
 records.append(row)
 (BUILD/f'{a.name}-progress.json').write_text(json.dumps(records,indent=2)+'\n');print('FINISHED',json.dumps(row),flush=True)
 if code:break
(BUILD/f'{a.name}-finished.json').write_text(json.dumps(dict(cases=records,seconds=time.monotonic()-started,complete=len(records)==len(cases) and all(r['exit_code']==0 for r in records)),indent=2)+'\n')
