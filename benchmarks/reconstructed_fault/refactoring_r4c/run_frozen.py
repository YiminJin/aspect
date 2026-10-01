#!/usr/bin/env python3
from pathlib import Path
import json,subprocess,sys,os
r=Path(__file__).resolve().parent;repo=r.parents[2];v=sys.argv[1]
binary=repo/('build-refactor-r4b-residual/aspect-r4b-residual-qualified' if v=='reference' else 'build-refactor-r4c/aspect-release')
env=os.environ.copy();env.update(json.loads((r/'evidence/frozen-environment.json').read_text()))
raise SystemExit(subprocess.run(['python3',str(r/'run_logged.py'),f'{v}-frozen-replay','600','mpirun','-np','4',str(binary),str(r/f'inputs/{v}-frozen-replay.prm')],env=env).returncode)
