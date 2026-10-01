#!/usr/bin/env python3
"""Run the selected lifecycle checks with bounded logs; reuse unrelated R4 evidence."""
from pathlib import Path
import os,subprocess,sys
r=Path(__file__).resolve().parent;repo=r.parents[2];v=sys.argv[1]
assert v in ('reference','candidate')
binary=repo/('build-refactor-r4c/aspect-r4c-verified' if v=='reference' else 'build-refactor-r5a1/aspect-release')
env={k:x for k,x in os.environ.items() if not k.startswith('ASPECT_')};env['ASPECT_SOURCE_DIR']=str(repo)
filters='[fault_slip_restart],[fault_prescribed_v],ReconstructedFaultManager slip*,ReconstructedFaultManager validates slip*,ReconstructedFaultManager checkpoint*,Stage-I captured BP3*,[fault_slip_interpolation]'
failures=[]
for ranks in (1,2):
 cases=[('unit',['--test',filters])]
 if v=='candidate':
  cases += [(name,[str(r/f'inputs/candidate-{name}-{ranks}.prm')]) for name in ('pressure','rollback-original')]
 for name,args in cases:
  code=subprocess.run([sys.executable,str(r/'run_logged.py'),f'{v}-{name}-{ranks}','180','mpirun','-np',str(ranks),str(binary),*args],cwd=repo,env=env).returncode
  if code:failures.append((name,ranks,code))
assert not failures,failures
