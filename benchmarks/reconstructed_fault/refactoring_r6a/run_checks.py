#!/usr/bin/env python3
"""Isolated off/on/missing/open-failure and rollback checks with matched settings."""
from pathlib import Path
import os,subprocess,sys
r=Path(__file__).resolve().parent;repo=r.parents[2];v=sys.argv[1];assert v in ('reference','candidate')
binary=repo/('build-refactor-r5b2/aspect-r5b2-qualified' if v=='reference' else 'build-refactor-r6a/aspect-release')
cases=[(f'bp3-{mode}-{rank}',mode,rank) for mode,rank in [('off',1),('on',1),('off',2),('on',2),('missing',1),('blocked',2)]]
cases += [(f'rollback-{mode}-{rank}',mode,rank) for mode in ('off','on') for rank in (1,2)]
for label,mode,rank in cases:
 if len(sys.argv)>2 and label not in sys.argv[2:]:continue
 env={k:x for k,x in os.environ.items() if not k.startswith('ASPECT_')};env['ASPECT_SOURCE_DIR']=str(repo)
 if label.startswith('bp3'):
  env.update(ASPECT_FAULT_PERFORMANCE='1',ASPECT_FAULT_EXPLICIT_B='1',ASPECT_FAULT_EXPLICIT_G='1')
 if mode!='off':
  # Exercise actual presence semantics, including both "0" and empty values.
  value=('0' if rank==1 else '') if mode=='on' else '1'
  env.update(ASPECT_STRESS_CYCLE_TRACE=value,ASPECT_FAULT_SOURCE_HISTORY_DIAGNOSTIC=value)
 subprocess.run([sys.executable,str(r/'run_logged.py'),v+'-'+label,'300','mpirun','-np',str(rank),str(binary),str(r/f'inputs/{v}-{label}.prm')],cwd=repo,env=env,check=True)
