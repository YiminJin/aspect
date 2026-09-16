"""One fresh fully prescribed mature-Vp experiment: initialization + two steps."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
BASE=HERE/'mature-fault-50-local4'
OUT=HERE/'uniform-sliding-50-local4'

def main():
    OUT.mkdir() # Preserve earlier evidence; never overwrite/retry a run.
    with (BASE/'clock.csv').open() as f:clock=list(csv.DictReader(f))[:3]
    with (OUT/'clock.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(clock[0]));w.writeheader();w.writerows(clock)
    (OUT/'run.prm').write_text(f'''include {BASE}/run.prm
set Output directory = {OUT}
set End time = {clock[-1]['time']}
subsection Termination criteria
  set End step = 2
end
''')
    env={k:v for k,v in os.environ.items() if not k.startswith('ASPECT_')}
    env.update(ASPECT_SOURCE_DIR=str(REPO),ASPECT_BP3_UNIFORM_SLIDING='1',
        ASPECT_FAULT_EXPLICIT_B='1',ASPECT_FAULT_EXPLICIT_G='1',
        ASPECT_FAULT_SURFACE_SOLVER='tridiagonal',ASPECT_FAULT_STRESS_SAMPLE_DIAGNOSTIC='1',
        ASPECT_FAULT_HISTORY_AUDIT='1',ASPECT_BP3_TIMESTEP_SEQUENCE=str(OUT/'clock.csv'),
        ASPECT_BP3_TARGET_MESH=str(HERE/'junction-matched-qualified-local4/refined/target_cells.txt'),
        ASPECT_BP3_EXACT_TARGET='1',ASPECT_BP3_EXPECTED_FAULT=str(HERE/'fault-grid-50-local4/fault.txt'),
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',DEAL_II_NUM_THREADS='1')
    command=['timeout','--signal=TERM','--kill-after=15','900','mpirun','-np','4',
             '--bind-to','core','--map-by','core',str(REPO/'build-pf-cpdi/aspect-release'),str(OUT/'run.prm')]
    paths=[HERE/'bp3.cc',HERE/'uniform_sliding.h',Path(__file__),HERE/'build/libbp3.release.so',
           REPO/'build-pf-cpdi/aspect-release',OUT/'run.prm',OUT/'clock.csv',BASE/'prestress.txt']
    (OUT/'provenance.json').write_text(json.dumps(dict(command=command,
        environment={k:v for k,v in env.items() if k.startswith('ASPECT_')},
        sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}),indent=2)+'\n')
    start=time.monotonic()
    with (OUT/'run.log').open('x') as f:
        result=subprocess.run(command,cwd=HERE,env=env,stdout=f,stderr=subprocess.STDOUT)
    record=dict(status=result.returncode,seconds=time.monotonic()-start)
    (OUT/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
    assert result.returncode==0,'Preserve failed evidence; no automatic retry.'

if __name__=='__main__':main()
