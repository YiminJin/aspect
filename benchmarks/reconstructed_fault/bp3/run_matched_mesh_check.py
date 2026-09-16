"""One mesh-only export after a failed paired mesh guard; no mechanics."""
import json
import os
from pathlib import Path
import subprocess
import time

here=Path(__file__).resolve().parent;repo=here.parents[2]
pair=here/'junction-matched-local4';out=pair/'mesh-check';out.mkdir()
(out/'run.prm').write_text(f'include {pair}/refined/run.prm\nset Output directory = {out}\n')
env={k:v for k,v in os.environ.items() if not k.startswith('ASPECT_')}
env.update(ASPECT_SOURCE_DIR=str(repo),ASPECT_BP3_SHARED_CLOCK=str(pair/'clock'),
  ASPECT_BP3_PAIR_LABEL='refined',ASPECT_BP3_TARGET_MESH=str(pair/'refined/target_cells.txt'),
  ASPECT_BP3_MESH_ONLY='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',DEAL_II_NUM_THREADS='1')
cmd=['mpirun','-np','4',str(repo/'build-pf-cpdi/aspect-release'),str(out/'run.prm')]
start=time.monotonic()
with (out/'run.log').open('x') as f:result=subprocess.run(cmd,cwd=here,env=env,stdout=f,stderr=subprocess.STDOUT)
(out/'execution.json').write_text(json.dumps(dict(command=cmd,status=result.returncode,seconds=time.monotonic()-start),indent=2)+'\n')
print((out/'execution.json').read_text())
