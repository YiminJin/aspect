#!/usr/bin/env python3
"""Bounded matched projection, material-caller and restart verification."""
from pathlib import Path
import os,subprocess,sys
r=Path(__file__).resolve().parent;repo=r.parents[2];v=sys.argv[1]
assert v in ('reference','candidate')
binary=repo/('build-refactor-r5a1/aspect-r5a1-qualified' if v=='reference' else 'build-refactor-r5a2/aspect-release')
env={k:x for k,x in os.environ.items() if not k.startswith('ASPECT_')};env['ASPECT_SOURCE_DIR']=str(repo)
env['ASPECT_FAULT_PERFORMANCE']='1'
filters='[fault_domain_quadrature],Fault normal-profile*,Fault projection*,ReconstructedFaultManager owns*,ReconstructedFaultManager checkpoint*,[fault_slip_restart]'
cases=[]
for rank in (1,2):
 cases.append((f'unit-{rank}',rank,['--test',filters],0))
 for name in ('ih','no-composition','surface','cache-remote','cache-cell','pressure','rollback-original'):
  # The accepted R5a1 coupled outputs remain usable directly as reference.
  if v=='reference' and name in ('pressure','rollback-original'):continue
  cases.append((f'{name}-{rank}',rank,[str(r/f'inputs/{v}-{name}-{rank}.prm')],0))
cases += [('cold-warm-2',2,[str(r/f'inputs/{v}-cold-warm-2.prm')],0)]
# Reuse the preserved accepted reference restart trace; run a private candidate branch.
if v=='candidate':cases += [('restart',1,[str(r/'inputs/candidate-restart.prm')],1)]
failures=[]
for name,rank,args,expected in cases:
 if len(sys.argv)>2 and name not in sys.argv[2:]:continue
 case_env=env.copy()
 if name.startswith(('pressure-','rollback-original-')) or name=='restart':case_env.pop('ASPECT_FAULT_PERFORMANCE',None)
 code=subprocess.run([sys.executable,str(r/'run_logged.py'),f'{v}-{name}','240','mpirun','-np',str(rank),str(binary),*args],cwd=repo,env=case_env).returncode
 if code != expected:failures.append((name,code,expected))
assert not failures,failures
