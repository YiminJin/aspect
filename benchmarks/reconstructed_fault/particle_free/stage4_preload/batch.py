#!/usr/bin/env python3
"""Durable sequential execution of an explicitly selected JSON case list."""
from pathlib import Path
import argparse,json,subprocess,sys,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];BUILD=ROOT/'build-particle-free/stage4-preload'
p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--worker',action='store_true');a=p.parse_args()
assert a.name.isidentifier()
cases=json.loads((BUILD/f'{a.name}-cases.json').read_text())
assert 1<=len(cases)<=3
if not a.worker:
 assert subprocess.run(['pgrep','-x','aspect-release'],capture_output=True).returncode==1,'Existing solver ranks are active.'
 with (BUILD/f'{a.name}.log').open('x') as log:
  job=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),a.name,'--worker'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 (BUILD/f'{a.name}-pid.json').write_text(json.dumps(dict(pid=job.pid,started_unix=time.time()),indent=2)+'\n')
 print('Preload supervisor PID',job.pid);raise SystemExit(0)
started=time.monotonic();records=[]
for args in cases:
 if (BUILD/'stop-batch').exists():break
 now=time.monotonic();cmd=[sys.executable,str(HERE/'run.py'),*args]
 print('START',args,time.strftime('%Y-%m-%d %H:%M:%S %Z'),flush=True)
 code=subprocess.call(cmd,cwd=ROOT)
 row=dict(command=cmd,exit_code=code,seconds=time.monotonic()-now);records.append(row)
 (BUILD/f'{a.name}-progress.json').write_text(json.dumps(records,indent=2)+'\n');print('FINISHED',json.dumps(row),flush=True)
 if code:break
(BUILD/f'{a.name}-finished.json').write_text(json.dumps(dict(cases=records,seconds=time.monotonic()-started,complete=len(records)==len(cases) and all(r['exit_code']==0 for r in records)),indent=2)+'\n')
