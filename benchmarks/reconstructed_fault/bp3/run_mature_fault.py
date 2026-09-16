"""One fresh mature-fault comparison; no retry, continuation or timestep retuning."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
BASE=HERE/'fault-grid-50-local4'
REFERENCE=HERE/'frozen-cohesion-adaptive-50-local4'
OUT=HERE/'mature-fault-50-local4'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('action',choices=['prepare','run'])
args=parser.parse_args()

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

if args.action=='prepare':
    OUT.mkdir()
    with (REFERENCE/'fault_0.csv').open() as f:initial=list(csv.DictReader(f))
    dt0=4e6;G=32038120320.;eta=1e26
    beta=math.exp(-dt0*G/eta);kappa=-eta*math.expm1(-dt0*G/eta)
    # Store the captured evaluated C_star as a rational field. The old shear
    # remains Q1, while the three correction coefficients are not a nodal C_star.
    with (OUT/'prestress.txt').open('x') as f:
        f.write(str(len(initial))+'\n')
        for r in initial:
            values=[float(r[k]) for k in ('x','y','tau_bg','sigma_n_bg')]
            values += [beta*float(r['C']),kappa*float(r['V']),float(r['Ih'])]
            f.write(' '.join(format(v,'.17g') for v in values)+'\n')
    with (BASE/'accepted_steps.csv').open() as f:clock=list(csv.DictReader(f))[:11]
    with (OUT/'clock.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(clock[0]));writer.writeheader();writer.writerows(clock)
    assert float(clock[-1]['time'])==922804465.5975173
    (OUT/'run.prm').write_text(f'''include {BASE}/run.prm
set Output directory = {OUT}
set Resume computation = false
set End time = 922804465.5975173
subsection Material model
  subsection Phase field fault
    set Fault constitutive mode = mature frictional
  end
end
subsection Postprocess
  subsection BP3
    set Mature prestress file = {OUT}/prestress.txt
  end
end
subsection Termination criteria
  set End step = 10
end
''')
    (OUT/'preparation.json').write_text(json.dumps(dict(
        source_initial=str(REFERENCE/'fault_0.csv'),source_sha256=digest(REFERENCE/'fault_0.csv'),
        beta0=beta,kappa0=kappa,prestress_sha256=digest(OUT/'prestress.txt'),
        cap_seconds=2400,end_time_seconds=float(clock[-1]['time'])),indent=2)+'\n')
    print('Prepared one 4-rank case; expected 15–22 min, hard cap 40 min. Not run.')
else:
    assert (OUT/'preparation.json').exists() and not (OUT/'run.log').exists()
    env={k:v for k,v in os.environ.items() if not k.startswith('ASPECT_')}
    env.update(ASPECT_SOURCE_DIR=str(REPO),ASPECT_FAULT_EXPLICIT_B='1',ASPECT_FAULT_EXPLICIT_G='1',
        ASPECT_FAULT_SURFACE_SOLVER='tridiagonal',ASPECT_FAULT_STRESS_SAMPLE_DIAGNOSTIC='1',
        ASPECT_FAULT_HISTORY_AUDIT='1',ASPECT_BP3_TIMESTEP_SEQUENCE=str(OUT/'clock.csv'),
        ASPECT_BP3_TARGET_MESH=str(HERE/'junction-matched-qualified-local4/refined/target_cells.txt'),
        ASPECT_BP3_EXACT_TARGET='1',ASPECT_BP3_EXPECTED_FAULT=str(BASE/'fault.txt'),
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',DEAL_II_NUM_THREADS='1')
    cmd=['timeout','--signal=TERM','--kill-after=15','2400','mpirun','-np','4','--bind-to','core','--map-by','core',
         str(REPO/'build-pf-cpdi/aspect-release'),str(OUT/'run.prm')]
    paths=[Path(__file__),OUT/'run.prm',OUT/'prestress.txt',OUT/'clock.csv',
           REPO/'build-pf-cpdi/aspect-release',HERE/'build/libbp3.release.so',HERE/'bp3.cc',HERE/'mature_fault.h',
           REPO/'source/material_model/phase_field_fault.cc',REPO/'source/simulator/phase_field.cc']
    (OUT/'provenance.json').write_text(json.dumps(dict(command=cmd,
        environment={k:v for k,v in env.items() if k.startswith('ASPECT_')},
        sha256={str(p):digest(p) for p in paths}),indent=2)+'\n')
    start=time.monotonic()
    with (OUT/'run.log').open('x') as f:result=subprocess.run(cmd,cwd=HERE,env=env,stdout=f,stderr=subprocess.STDOUT)
    record=dict(status=result.returncode,seconds=time.monotonic()-start)
    (OUT/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))
    assert result.returncode==0,'Preserve the partial result; no retry or controller retuning.'
