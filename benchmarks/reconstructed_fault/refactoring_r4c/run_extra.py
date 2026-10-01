#!/usr/bin/env python3
from pathlib import Path
import json,subprocess,sys,os
r=Path(__file__).resolve().parent;repo=r.parents[2];v=sys.argv[1]
binary=repo/('build-refactor-r4b-residual/aspect-r4b-residual-qualified' if v=='reference' else 'build-refactor-r4c/aspect-release')
for name in ('ordinary-amg','bfbt','melt','fail-s','fail-budget'):
 env=os.environ.copy()
 if name=='frozen':env.update(json.loads((r/'evidence/frozen-environment.json').read_text()))
 command=['python3',str(r/'run_logged.py'),f'{v}-{name}','600' if name=='frozen' else '180','mpirun','-np','4' if name=='frozen' else '1',str(binary),str(r/f'inputs/{v}-{name}.prm')]
 subprocess.run(command,env=env,check=False)
