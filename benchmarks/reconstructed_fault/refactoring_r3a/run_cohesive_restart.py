#!/usr/bin/env python3
"""Read the same reference Stage-J checkpoint; retain its known step-two failure."""
from pathlib import Path
import shutil,subprocess,sys,os
root=Path(__file__).resolve().parent; repo=root.parents[2]
variant=sys.argv[1]
source=root/'output-reference-stage-j-open-top-2'
dest=root/f'output-{variant}-cohesive-resume-direct-2'
assert (source/'restart/01/resume.z').exists()
shutil.copytree(source,dest)
text=(root/'inputs'/f'{variant}-stage-j-open-top-2.prm').read_text().replace(f'output-{variant}-stage-j-open-top-2',f'output-{variant}-cohesive-resume-direct-2')
# The create plugin already contains the shared restart observer. Loading it
# avoids the resume wrapper's hard-coded cwd checkpoint-copy side effect.
text+='set Resume computation = true\n'
prm=root/'inputs'/f'{variant}-cohesive-resume-direct-2.prm';prm.write_text(text)
env=os.environ.copy();env['ASPECT_FAULT_EXPLICIT_B']='1';env['ASPECT_FAULT_EXPLICIT_G']='1'
binary='build-refactor-r3a/aspect-release' if variant=='candidate' else 'build-refactor-r2b-cache/aspect-cache-qualified'
result=subprocess.run(['python3',str(root/'run_logged.py'),f'{variant}-cohesive-resume-direct-2','300','mpirun','-np','2',str(repo/binary),str(prm)],env=env)
raise SystemExit(result.returncode)
