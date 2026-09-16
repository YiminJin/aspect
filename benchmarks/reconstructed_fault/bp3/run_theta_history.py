"""One fresh functional-state replay on the saved 50-m mesh and clock."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
BASE=HERE/'fault-grid-50-local4'
OUT=HERE/'theta-history-50-local4'
OUT.mkdir()
prm=OUT/'run.prm'
prm.write_text(f'include {BASE}/run.prm\nset Output directory = {OUT}\nset Resume computation = false\n')
env={k:v for k,v in os.environ.items() if not k.startswith('ASPECT_')}
env.update(ASPECT_SOURCE_DIR=str(REPO),ASPECT_FAULT_EXPLICIT_B='1',ASPECT_FAULT_EXPLICIT_G='1',
    ASPECT_FAULT_SURFACE_SOLVER='tridiagonal',ASPECT_FAULT_STRESS_SAMPLE_DIAGNOSTIC='1',
    ASPECT_FAULT_HISTORY_AUDIT='1',ASPECT_BP3_TIMESTEP_SEQUENCE=str(BASE/'accepted_steps.csv'),
    ASPECT_BP3_TARGET_MESH=str(HERE/'junction-matched-qualified-local4/refined/target_cells.txt'),
    ASPECT_BP3_EXACT_TARGET='1',ASPECT_BP3_EXPECTED_FAULT=str(BASE/'fault.txt'),
    ASPECT_FAULT_THETA_HISTORY_DIAGNOSTIC=str(OUT/'theta_function_history.txt'),
    OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',DEAL_II_NUM_THREADS='1')
cmd=['mpirun','-np','4','--bind-to','core','--map-by','core',str(REPO/'build-pf-cpdi/aspect-release'),str(prm)]
paths=[Path(__file__),REPO/'build-pf-cpdi/aspect-release',HERE/'build/libbp3.release.so',HERE/'bp3.cc',
       HERE/'theta_history_diagnostic.h',REPO/'source/material_model/phase_field_fault.cc',
       REPO/'source/material_model/fault_theta_history_diagnostic.h',prm,BASE/'run.prm',BASE/'fault.txt',BASE/'accepted_steps.csv']
record=dict(command=cmd,environment={k:v for k,v in env.items() if k.startswith('ASPECT_')},
            sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
(OUT/'provenance.json').write_text(json.dumps(record,indent=2)+'\n')
start=time.monotonic()
with (OUT/'run.log').open('x') as log:
    result=subprocess.run(cmd,cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT)
record=dict(status=result.returncode,seconds=time.monotonic()-start)
(OUT/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
assert result.returncode==0,'Preserve failed evidence; do not retry or change the prescribed clock automatically.'
