"""One 50-m replay, freezing only the post-initialization surface resistance."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import argparse

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--adaptive',action='store_true',help='Allow production-controlled substeps between comparison times.')
args=parser.parse_args()
BASE=HERE/'fault-grid-50-local4';OUT=HERE/('frozen-cohesion-adaptive-50-local4' if args.adaptive else 'frozen-cohesion-50-local4')
OUT.mkdir()
prm=OUT/'run.prm'
prm.write_text(f'include {BASE}/run.prm\nset Output directory = {OUT}\nset Resume computation = false\n'
    + ('subsection Termination criteria\n  set Termination criteria = end time\nend\n' if args.adaptive else ''))
env={k:v for k,v in os.environ.items() if not k.startswith('ASPECT_')}
env.update(ASPECT_SOURCE_DIR=str(REPO),ASPECT_FAULT_EXPLICIT_B='1',ASPECT_FAULT_EXPLICIT_G='1',
    ASPECT_FAULT_SURFACE_SOLVER='tridiagonal',ASPECT_FAULT_STRESS_SAMPLE_DIAGNOSTIC='1',
    ASPECT_FAULT_HISTORY_AUDIT='1',ASPECT_BP3_TIMESTEP_SEQUENCE=str(BASE/'accepted_steps.csv'),
    ASPECT_BP3_TARGET_MESH=str(HERE/'junction-matched-qualified-local4/refined/target_cells.txt'),
    ASPECT_BP3_EXACT_TARGET='1',ASPECT_BP3_EXPECTED_FAULT=str(BASE/'fault.txt'),
    ASPECT_FAULT_FROZEN_COHESION_DIAGNOSTIC=str(OUT/'initial_cohesion.txt'),
    OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',DEAL_II_NUM_THREADS='1')
if args.adaptive:env['ASPECT_BP3_ADAPTIVE_REPLAY']='1'
cmd=['mpirun','-np','4','--bind-to','core','--map-by','core',str(REPO/'build-pf-cpdi/aspect-release'),str(prm)]
paths=[Path(__file__),REPO/'build-pf-cpdi/aspect-release',HERE/'build/libbp3.release.so',HERE/'bp3.cc',
    HERE/'cohesion_diagnostic.h',HERE/'replay_time_step.h',REPO/'source/material_model/phase_field_fault.cc',
    REPO/'source/material_model/fault_cohesion_diagnostic.h',prm,BASE/'run.prm',BASE/'fault.txt',BASE/'accepted_steps.csv']
(OUT/'provenance.json').write_text(json.dumps(dict(command=cmd,
    environment={k:v for k,v in env.items() if k.startswith('ASPECT_')},
    sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}),indent=2)+'\n')
start=time.monotonic()
with (OUT/'run.log').open('x') as log:r=subprocess.run(cmd,cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT)
record=dict(status=r.returncode,seconds=time.monotonic()-start)
(OUT/'execution.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
assert r.returncode==0,'Preserve failure and timestep guards; no automatic retuning/retry.'
