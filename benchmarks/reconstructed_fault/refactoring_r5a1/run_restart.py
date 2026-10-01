#!/usr/bin/env python3
"""Resume separate copies of the preserved pre-R3-fix checkpoint, without tuning."""
from pathlib import Path
import os,subprocess,sys
r=Path(__file__).resolve().parent;repo=r.parents[2];v=sys.argv[1]
assert v in ('reference','candidate')
binary=repo/('build-refactor-r4c/aspect-r4c-verified' if v=='reference' else 'build-refactor-r5a1/aspect-release')
env={k:x for k,x in os.environ.items() if not k.startswith('ASPECT_')};env['ASPECT_SOURCE_DIR']=str(repo)
code=subprocess.run([sys.executable,str(r/'run_logged.py'),f'{v}-restart','180','mpirun','-np','1',str(binary),str(r/f'inputs/{v}-restart.prm')],cwd=repo,env=env).returncode
# Known cohesive step-two nonconvergence must also be checked in the comparator.
assert code==1,code
