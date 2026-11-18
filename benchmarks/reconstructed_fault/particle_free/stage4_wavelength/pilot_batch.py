#!/usr/bin/env python3
"""Seven explicitly selected short histories, sequentially; durable supervision.

Start only after the initial 256 weakening/background history has finished.
The detached supervisor survives a terminal interruption and retains each
existing runner's timeout and exit status. A stop-batch file prevents the next
case from starting; it never interrupts a checkpoint write in the active case.
"""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];BUILD=ROOT/'build-particle-free/stage4-pilot'
CASES=[(512,'weakening','background'),(256,'weakening','perturbed'),(512,'weakening','perturbed'),
       (256,'neutral','background'),(512,'neutral','background'),(256,'neutral','perturbed'),(512,'neutral','perturbed')]
p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');a=p.parse_args()
if not a.worker:
    running=subprocess.run(['pgrep','-x','aspect-release'],capture_output=True,text=True)
    assert running.returncode==1, 'Wait for all existing solver ranks to exit before starting this batch.'
    with (BUILD/'batch.log').open('x') as log:
        job=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (BUILD/'batch-pid.json').write_text(json.dumps(dict(pid=job.pid,started_unix=time.time(),cases=CASES),indent=2)+'\n')
    print('Native pilot supervisor PID',job.pid);raise SystemExit(0)
first=ROOT/'build-particle-free/bulk-tip/runs/stage4-native-256-weakening-background'
assert 'Termination requested by criterion: end step' in (first/'run.log').read_text()
assert ',0.000640000000000000' in (first/'output/pilot-summary.csv').read_text() or (first/'output/pilot-summary.csv').read_text().splitlines()[-1].startswith('8,')
start=time.monotonic();records=[]
for nx,law,case in CASES:
    remaining=7200-(time.monotonic()-start)
    if (BUILD/'stop-batch').exists() or remaining<900:break
    command=[sys.executable,str(HERE/'run_pilot.py'),str(nx),law,case,'--steps','4','--seconds',str(min(1500,int(remaining)))]
    now=time.monotonic();print('START',nx,law,case,time.strftime('%Y-%m-%d %H:%M:%S %Z'),flush=True)
    code=subprocess.call(command,cwd=ROOT)
    record=dict(nx=nx,law=law,case=case,command=command,exit_code=code,seconds=time.monotonic()-now);records.append(record)
    (BUILD/'batch-progress.json').write_text(json.dumps(records,indent=2)+'\n');print('FINISHED',json.dumps(record),flush=True)
    if code:break
(BUILD/'batch-finished.json').write_text(json.dumps(dict(cases=records,seconds=time.monotonic()-start,complete=len(records)==len(CASES) and all(r['exit_code']==0 for r in records)),indent=2)+'\n')
