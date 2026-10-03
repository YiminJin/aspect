#!/usr/bin/env python3
"""Bounded isolated runs; no server or production input writes."""
from pathlib import Path
import os,subprocess,sys,json
r=Path(__file__).resolve().parent;repo=r.parents[2]
if sys.argv[1]=='--batch':
 for item in sys.argv[2:]:
  case,ranks=item.split(':')
  expected_failure=case.startswith('corrupt-') or case in ('legacy-active','changed-active')
  result=subprocess.run([sys.executable,__file__,case,ranks])
  assert (result.returncode!=0)==expected_failure, (case,result.returncode)
 raise SystemExit(0)
case=sys.argv[1];ranks=sys.argv[2] if len(sys.argv)>2 else '1'
env={k:v for k,v in os.environ.items() if not k.startswith('ASPECT_')};env['ASPECT_SOURCE_DIR']=str(repo)
label=case+(('-'+sys.argv[3]) if len(sys.argv)>3 else '')+'-np'+ranks
used=sum(json.loads(p.read_text())['seconds'] for p in (r/'evidence').glob('*-np*.json'))
assert used<1200,'Local simulation budget exhausted'
subprocess.run([sys.executable,str(r/'run_logged.py'),label,str(min(180,1200-used)),'mpirun','-np',ranks,str(repo/'build-refactor-r6b/aspect-particle-lifecycle-qualified'),*(['--validate'] if case.endswith('-parse') else []),str(r/'inputs'/f'{case}.prm')],cwd=repo,env=env,check=True)
