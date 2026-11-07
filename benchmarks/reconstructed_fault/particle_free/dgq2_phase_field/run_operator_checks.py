#!/usr/bin/env python3
"""Sequential bounded checks of the compiled native phase operator."""
from pathlib import Path
import json,subprocess,os,time,sys
root=Path(__file__).resolve().parents[4]
out=root/'benchmarks/reconstructed_fault/particle_free/evidence/dgq2_phase_field'/(sys.argv[1] if len(sys.argv)>1 else 'transport-native')
out.mkdir(exist_ok=True)
env=os.environ.copy()
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','DEAL_II_NUM_THREADS'):env[name]='1'
exe=root/'build-particle-free/dgq2-phase/operators/dgq2_phase_checks'
records=[]
for mode in ('translate','reverse','rotation','divergence','inflow'):
 for level in (3,4,5):
  for limited in (0,1):
   label=f'{mode}-{level}-{limited}'
   log=out/(label+'.log')
   if log.exists():raise RuntimeError(f'Preserve {log}')
   command=['/opt/openmpi/5.0.6/bin/mpirun','-np','1',str(exe),str(level),mode,str(limited)]
   start=time.monotonic()
   result=subprocess.run(command,env=env,cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
   log.write_text(result.stdout)
   record=dict(label=label,command=command,exit_code=result.returncode,seconds=time.monotonic()-start)
   records.append(record);(out/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
   print(label,result.returncode,round(record['seconds'],2),flush=True)
   if result.returncode:raise SystemExit(result.returncode)
