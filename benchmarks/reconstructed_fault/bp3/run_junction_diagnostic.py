"""One 35-km noncommitting mechanical probe from the local step-11 checkpoint."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

here=Path(__file__).resolve().parent
repo=here.parents[2]
checkpoint=here/'normal-stress-complete-local4/restart/03'
out=here/'normal-stress-junction35-local4'
out.mkdir()
digest=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
source_hashes={p.name:digest(p) for p in checkpoint.iterdir() if p.is_file()}
shutil.copytree(checkpoint,out/'restart/03')
(out/'restart/last_good_checkpoint.txt').write_text('3\n')
assert source_hashes=={p.name:digest(p) for p in (out/'restart/03').iterdir() if p.is_file()}
prm=out/'diagnostic.prm'
prm.write_text(f'include {here}/normal_stress_fresh_local4.prm\n'
               f'set Output directory = {out}\nset Resume computation = true\n')
env={k:v for k,v in os.environ.items() if not k.startswith('ASPECT_')}
env.update(ASPECT_SOURCE_DIR=str(repo),ASPECT_FAULT_EXPLICIT_B='1',ASPECT_FAULT_EXPLICIT_G='1',
           ASPECT_FAULT_SURFACE_SOLVER='tridiagonal',ASPECT_FAULT_STRESS_SAMPLE_DIAGNOSTIC='1',
           ASPECT_FAULT_NONCOMMITTING_DIAGNOSTIC='1',ASPECT_BP3_JUNCTION_DIAGNOSTIC='1',
           OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',DEAL_II_NUM_THREADS='1')
binary=repo/'build-pf-cpdi/aspect-release'
command=['mpirun','-np','4',str(binary),str(prm)]
record=dict(command=command,checkpoint=str(checkpoint),checkpoint_sha256=source_hashes,
            environment={k:v for k,v in env.items() if k.startswith('ASPECT_')},
            files={str(p):digest(p) for p in [binary,here/'build/libbp3.release.so',here/'bp3.cc',
                    here/'junction_diagnostic.h',repo/'source/simulator/solver.cc',prm]})
(out/'provenance.json').write_text(json.dumps(record,indent=2)+'\n')
start=time.monotonic()
with (out/'run.log').open('x') as stream:
    result=subprocess.run(command,cwd=here,env=env,stdout=stream,stderr=subprocess.STDOUT)
log=(out/'run.log').read_text()
record.update(status=result.returncode,wall_seconds=time.monotonic()-start,
    converged='Noncommitting fault diagnostic converged:' in log,
    rollback_verified='BP3 noncommitting rollback verified:' in log,
    intentional_stop='Noncommitting fault diagnostic completed; intentional rollback stop.' in log,
    source_checkpoint_unchanged=source_hashes=={p.name:digest(p) for p in checkpoint.iterdir() if p.is_file()},
    diagnostic_checkpoint_unchanged=source_hashes=={p.name:digest(p) for p in (out/'restart/03').iterdir() if p.is_file()},
    accepted_output_written=(out/'accepted_steps.csv').exists())
(out/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k not in ['files','checkpoint_sha256']},indent=2))
