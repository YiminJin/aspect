#!/usr/bin/env python3
"""Run the bounded matched R5b2 matrix without changing physical settings."""
from pathlib import Path
import os,subprocess,sys,json
r=Path(__file__).resolve().parent;repo=r.parents[2];v=sys.argv[1];assert v in ('reference','candidate')
binary=repo/('build-refactor-r5b1/aspect-r5b1-qualified' if v=='reference' else 'build-refactor-r5b2/aspect-release')
base={k:x for k,x in os.environ.items() if not k.startswith('ASPECT_')};base['ASPECT_SOURCE_DIR']=str(repo)
cases=[]
if v=='reference' and (len(sys.argv)<3 or any(n!='frozen' for n in sys.argv[2:])):
 raise SystemExit('Reuse qualified R5b1 surface outputs; only frozen reference run is fresh.')
for rank in (1,2):
 cases.append((f'unit-{rank}',rank,['--test','[fault_surface_direct],[fault_normal_filter]'],0,{}))
 for name in ('dynamic','adiabatic','rate','explicit','filter','singular','singular-current','pressure','rollback','bp3'):
  if v=='reference' and name in ('pressure','rollback'):continue
  env={}
  if name not in ('pressure','rollback'):env['ASPECT_FAULT_PERFORMANCE']='1'
  if name=='explicit':env.update(ASPECT_FAULT_EXPLICIT_G='1',ASPECT_FAULT_EXPLICIT_B='1',ASPECT_FAULT_COMPARE_COUPLING='1')
  if name=='filter':env['ASPECT_TEST_NORMAL_FILTER']='1'
  if name=='bp3':env.update(ASPECT_FAULT_EXPLICIT_G='1',ASPECT_FAULT_EXPLICIT_B='1')
  cases.append((f'{name}-{rank}',rank,[str(r/f'inputs/{v}-{name}-{rank}.prm')],1 if name.startswith('singular') else 0,env))
cases.append(('frozen',4,[str(r/f'inputs/{v}-frozen.prm')],1,
 json.loads((r.with_name('frozen_gmg_repair')/'evidence/environment.json').read_text())))
failures=[]
for name,rank,args,expected,extra in cases:
 if len(sys.argv)>2 and name not in sys.argv[2:]:continue
 env=base|extra
 code=subprocess.run([sys.executable,str(r/'run_logged.py'),f'{v}-{name}', '720' if name=='frozen' else '300','mpirun','-np',str(rank),str(binary),*args],cwd=repo,env=env).returncode
 if code!=expected:failures.append((name,code,expected))
assert not failures,failures
