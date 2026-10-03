#!/usr/bin/env python3
"""Bounded Section-3 cases with immutable logs and a 20-minute total budget."""
from pathlib import Path
import json,os,subprocess,sys
root=Path(__file__).resolve().parent;repo=root.parents[2]
if sys.argv[1]=='--batch':
 for pair in sys.argv[2:]:
  case,ranks=pair.split(':')
  result=subprocess.run([sys.executable,__file__,case,ranks])
  if result.returncode:raise SystemExit(result.returncode)
 raise SystemExit(0)
case,ranks=sys.argv[1:3]
label=case+('-'+sys.argv[3] if len(sys.argv)>3 else '')+'-np'+ranks
used=sum(json.loads(p.read_text())['seconds'] for p in (root/'evidence').glob('*-np*.json'))
assert used<1200,'Local simulation budget exhausted'
env={k:v for k,v in os.environ.items() if not k.startswith('ASPECT_')};env['ASPECT_SOURCE_DIR']=str(repo)
result=subprocess.run([sys.executable,str(root/'run_logged.py'),label,str(min(180,1200-used)),
 'mpirun','-np',ranks,str(repo/'build-refactor-r6b/aspect-filter-derivative-qualified'),str(root/'inputs'/(case+'.prm'))],cwd=repo,env=env)
raise SystemExit(result.returncode)
