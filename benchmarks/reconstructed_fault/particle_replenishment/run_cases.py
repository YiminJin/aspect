#!/usr/bin/env python3
"""Bounded isolated runs; no server or production input writes."""
from pathlib import Path
import os,subprocess,sys,json
r=Path(__file__).resolve().parent;repo=r.parents[2]
case=sys.argv[1];ranks=sys.argv[2] if len(sys.argv)>2 else '1'
env={k:v for k,v in os.environ.items() if not k.startswith('ASPECT_')};env['ASPECT_SOURCE_DIR']=str(repo)
label=case+(('-'+sys.argv[3]) if len(sys.argv)>3 else '')+'-np'+ranks
used=sum(json.loads(p.read_text())['seconds'] for p in (r/'evidence').glob('*-np*.json'))
assert used<900,'Local simulation budget exhausted'
subprocess.run([sys.executable,str(r/'run_logged.py'),label,str(min(180,900-used)),'mpirun','-np',ranks,str(repo/'build-refactor-r6b/aspect-r6b-qualified'),*(['--validate'] if case=='candidate-parse' else []),str(r/'inputs'/f'{case}.prm')],cwd=repo,env=env,check=True)
