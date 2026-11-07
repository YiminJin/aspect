#!/usr/bin/env python3
"""Sequential native SIPG/obstacle checks; preserve failures and command identity."""
from pathlib import Path
import json,subprocess,os,time,sys,signal
root=Path(__file__).resolve().parents[4]
out=root/'benchmarks/reconstructed_fault/particle_free/evidence/dgq2_phase_field'/sys.argv[1]
out.mkdir(exist_ok=False)
env=os.environ.copy()
for k in ('OMP_NUM_THREADS','DEAL_II_NUM_THREADS','OPENBLAS_NUM_THREADS'):env[k]='1'
exe=root/'build-particle-free/dgq2-phase/operators/dgq2_phase_checks'
records=[]
cases=[(m,l,1) for m in ['elliptic','obstacle','no-growth','moving-obstacle'] for l in [2,3,4]]
cases += [(m,3,n) for n in [2,4] for m in ['elliptic','obstacle','moving-obstacle']]
for mode,level,ranks in cases:
 label=f'{mode}-{level}-{ranks}'
 command=['/opt/openmpi/5.0.6/bin/mpirun','-np',str(ranks),str(exe),str(level),mode]
 start=time.monotonic()
 with (out/(label+'.log')).open('w') as stream:
  process=subprocess.Popen(command,env=env,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=process.wait(timeout=120)
  except subprocess.TimeoutExpired:
   os.killpg(process.pid,signal.SIGTERM)
   try:process.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
   code=124
 r=dict(label=label,command=command,exit_code=code,seconds=time.monotonic()-start)
 records.append(r);(out/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
 print(label,code,round(r['seconds'],2),flush=True)
 if code:raise SystemExit(code)
