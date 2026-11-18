#!/usr/bin/env python3
"""Durable sequential supervisor for precisely the two selected release cases."""
from pathlib import Path
import argparse,json,subprocess,sys,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];BUILD=ROOT/'build-particle-free/stage4-release'
p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');a=p.parse_args()
if not a.worker:
 assert subprocess.run(['pgrep','-x','aspect-release'],capture_output=True).returncode==1,'Existing solver ranks are active.'
 with (BUILD/'batch.log').open('x') as log:
  job=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 (BUILD/'batch-pid.json').write_text(json.dumps(dict(pid=job.pid,started_unix=time.time()),indent=2)+'\n')
 print('Release pair supervisor PID',job.pid);raise SystemExit(0)
started=time.monotonic();records=[]
for law in ('weakening','neutral'):
 if (BUILD/'stop-batch').exists():break
 now=time.monotonic();cmd=[sys.executable,str(HERE/'run.py'),law]
 print('START',law,time.strftime('%Y-%m-%d %H:%M:%S %Z'),flush=True)
 code=subprocess.call(cmd,cwd=ROOT)
 row=dict(law=law,command=cmd,exit_code=code,seconds=time.monotonic()-now);records.append(row)
 (BUILD/'batch-progress.json').write_text(json.dumps(records,indent=2)+'\n');print('FINISHED',json.dumps(row),flush=True)
 # Diagnostic failures need inspection before repeating the same observer.
 if code:break
(BUILD/'batch-finished.json').write_text(json.dumps(dict(cases=records,seconds=time.monotonic()-started,complete=len(records)==2 and all(r['exit_code']==0 for r in records)),indent=2)+'\n')
