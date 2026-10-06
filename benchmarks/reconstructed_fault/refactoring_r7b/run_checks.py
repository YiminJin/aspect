#!/usr/bin/env python3
from pathlib import Path
import os,subprocess,sys,shutil
r=Path(__file__).resolve().parent;repo=r.parents[2];v=sys.argv[1]
bin=r/'build/reference-aspect' if v=='reference' else repo/'build-refactor-post-r6-gcc12-unity/aspect'
env={k:x for k,x in os.environ.items() if not k.startswith('ASPECT_')};env['ASPECT_SOURCE_DIR']=str(repo)
cases=[('amg',1,0),('bfbt',1,0),('melt',1,0),('gmg',1,0),('particles',2,0),('particles-resume',2,0),('cache',2,0),('empty-cache',2,1),('empty-fixed-cache',2,1),('empty-owner',2,1),('empty-owner-qualified',2,0),('singular',1,1),('cohesive',1,1),('cohesive-resume',1,1),('particles-cross-resume',2,0),('cohesive-cross-resume',1,1)]
failures=[]
for name,ranks,expected in cases:
 if v=="reference" and "-cross-" in name:continue
 if len(sys.argv)>2 and name not in sys.argv[2:]:continue
 out=r/f'output-{v}-{name}';assert not out.exists(),out
 if name.endswith('-resume'):
  base=name.removesuffix('-resume').removesuffix('-cross')
  parent_variant='reference' if '-cross-' in name else v
  parent=r/f'output-{parent_variant}-{base}'
  shutil.copytree(parent,out)
  text=(r/f'inputs/{parent_variant}-{base}.prm').read_text().replace(str(parent),str(out))+'\nset Resume computation = true\n'
  (r/f'inputs/{v}-{name}.prm').write_text(text)
 args=[sys.executable,str(r/'run_logged.py'),v+'-'+name,'240','mpirun','-np',str(ranks),str(bin),str(r/f'inputs/{v}-{name}.prm')]
 result=subprocess.run(args,cwd=repo,env=env)
 # Preserve unexpected outcomes and continue independent cases; markers/comparison determine qualification.
 print(name,'exit',result.returncode,'expected',expected,flush=True)
 if result.returncode!=expected:failures.append(name)

if failures:raise SystemExit("Unexpected exits: "+", ".join(failures))
