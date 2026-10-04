#!/usr/bin/env python3
"""Bounded birth/completion cases with immutable logs and a 20-minute total budget."""
from pathlib import Path
import json,os,subprocess,sys,shutil
root=Path(__file__).resolve().parent;repo=root.parents[2]
if sys.argv[1]=='--batch':
 for pair in sys.argv[2:]:
  case,ranks=pair.split(':')
  result=subprocess.run([sys.executable,__file__,case,ranks])
  if result.returncode:raise SystemExit(result.returncode)
 raise SystemExit(0)
case,ranks=sys.argv[1:3]
label=case+('-'+sys.argv[3] if len(sys.argv)>3 else '')+'-np'+ranks
used=sum(json.loads(p.read_text())['seconds'] for p in (root/'evidence').glob('*-np*.json') if not p.name.startswith('production'))
assert used<1200,'Local simulation budget exhausted'
env={k:v for k,v in os.environ.items() if not k.startswith('ASPECT_')};env['ASPECT_SOURCE_DIR']=str(repo)
assert not (root/"evidence"/(label+".log")).exists(), "Use a new label; evidence is immutable"
shutil.copy2(root/"inputs"/(case+".prm"),root/"evidence"/(label+".prm"))
result=subprocess.run([sys.executable,str(root/'run_logged.py'),label,str(600 if case.startswith('production') else min(360,1200-used)),
 'mpirun','-np',ranks,str(repo/'build-refactor-r6b/aspect-birth-identity-qualified'),*(['--validate'] if case=='production-parse' else []),str(root/'inputs'/(case+'.prm'))],cwd=repo,env=env)
raise SystemExit(result.returncode)
