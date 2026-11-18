"""Finish selected B/C stages after A, reserving reduction time within eight hours."""
import datetime
import json
import subprocess
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/resolution60_overnight'
RUNS=ROOT/'build-particle-free/bulk-tip/runs'
start=datetime.datetime.fromisoformat(json.loads((OUT/'entry.json').read_text())['start']).timestamp()
deadline=start+8*3600
status={'deadline_UTC':datetime.datetime.fromtimestamp(deadline,datetime.timezone.utc).isoformat(),'stages':[]}

def save():
    (OUT/'campaign.json').write_text(json.dumps(status,indent=2)+'\n')

def remaining():return deadline-time.time()

# A is already running in its own bounded wrapper. Never start two simulations.
a=RUNS/'resolution60-quarter-window/command.json'
while not a.exists() or 'exit_code' not in json.loads(a.read_text()):time.sleep(10)
# The outer runner may still be verifying/compressing the final closed frame.
while True:
    jobs=subprocess.check_output(['ps','-eo','comm,args'],text=True).splitlines()
    if not any(s.split()[0]=='python3' and 'run_resolution_matched.py coarse --growth --label resolution60-quarter-window' in s for s in jobs):break
    time.sleep(5)
status['stages'].append(dict(stage='A',command=json.loads(a.read_text())));save()

def run(stage,label,mesh,dt,seconds,extra=()):
    command=['python3',str(Path(__file__).with_name('run_resolution_matched.py')),mesh,'--growth','--label',label,
             '--evidence-directory',str(OUT),'--dt',str(dt),'--end','.08','--end-step',str(16 if stage=='B' else 20),
             '--ranks','2','--checkpoint-steps','1','--seconds',str(seconds),'--wall-hours',str((seconds-120)/3600),*extra]
    item=dict(stage=stage,label=label,command=command,started_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds_cap=seconds)
    status['active']=item;save()
    code=subprocess.call(command,cwd=ROOT)
    item.update(exit_code=code,finished_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
    status['stages'].append(item);status.pop('active');save()

b_seconds=int(min(18000,remaining()-4800))
if b_seconds>=1800:run('B','resolution60-axial-growth','axial',.005,b_seconds)
else:status['B_missing']='Insufficient budget after A; preserving accepted evidence.';save()
c_seconds=int(min(7200,remaining()-1200))
if c_seconds>=1500:
    run('C','resolution60-halfstep-late-extended','coarse',.0025,c_seconds,
        ['--resume-from',str(RUNS/'resolution60-halfstep-window/output'),'--snapshot-id','2',
         '--checker',str(ROOT/'build-particle-free/resolution60-overnight/checker/libbulk_tip_checks.release.so'),
         '--extend-terminal-restart'])
else:status['C_missing']='Remaining budget reserved for reduction; no late replay launched.';save()
status['finished_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
