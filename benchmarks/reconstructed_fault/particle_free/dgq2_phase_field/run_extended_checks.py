#!/usr/bin/env python3
"""Sequential many-crossing, time, refined-face and coercivity checks."""
from pathlib import Path
import json,subprocess,os,time,sys,signal
root=Path(__file__).resolve().parents[4]
out=root/'benchmarks/reconstructed_fault/particle_free/evidence/dgq2_phase_field'/sys.argv[1]
out.mkdir(exist_ok=False)
exe=root/'build-particle-free/dgq2-phase/operators/dgq2_phase_checks'
records=[]
cases=[]
# Fixed travel .75 and width 1.4: 3, 6, 12 cell crossings, respectively.
for mode in ('translate','reverse','rotation','divergence','inflow'):
 for level in (4,5,6):
  for limiting in (0,1):
   cases.append((mode,level,limiting,.05,.75,0,1))
# Temporal control for the smooth bounded field; no limiter activation.
for dt in (.01,.005,.0025,.00125):cases.append(('inflow',4,1,dt,.75,0,1))
for ranks in (1,4):
 for mode in ('elliptic','obstacle','moving-obstacle','constant','inflow','divergence'):
  cases.append((mode,2,1,.025,.4,1,ranks))
cases=[case for case in cases if case[5]]+[('quadratic-shear',level,limiting,.01,.4,0,1) for level in (3,4,5) for limiting in (0,1)]+[case for case in cases if not case[5]]
if len(sys.argv)>2:
 if sys.argv[2]=='refined':cases=[case for case in cases if case[5]]
 elif sys.argv[2]=='transport':cases=[case for case in cases if not case[5]]
 else:raise ValueError(sys.argv[2])
for mode,level,limiting,dt,end,refined,ranks in cases:
 label=f'{mode}-{level}-{limiting}-{dt}-{end}-{refined}-{ranks}'
 command=['/opt/openmpi/5.0.6/bin/mpirun','-np',str(ranks),str(exe),str(level),mode,str(limiting),str(dt),str(end),str(refined)]
 start=time.monotonic()
 with (out/(label+'.log')).open('w') as stream:
  process=subprocess.Popen(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=process.wait(timeout=180)
  except subprocess.TimeoutExpired:
   os.killpg(process.pid,signal.SIGTERM)
   try:process.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
   code=124
 record=dict(label=label,command=command,exit_code=code,seconds=time.monotonic()-start)
 records.append(record);(out/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
 print(label,code,round(record['seconds'],2),flush=True)
 if code:raise SystemExit(code)
