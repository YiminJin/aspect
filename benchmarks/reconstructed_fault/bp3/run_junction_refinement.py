"""One fresh local-mesh refinement, with no controller bypass or automatic retry."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

here = Path(__file__).resolve().parent
repo = here.parents[2]
out = here/'normal-stress-junction-refined-local4'
out.mkdir()
prm = here/'normal_stress_junction_refined.prm'
sequence = here/'normal-stress-complete-local4/accepted_steps.csv'
binary = repo/'build-pf-cpdi/aspect-release'
plugin = here/'build/libbp3.release.so'
env = {k: v for k, v in os.environ.items() if not k.startswith('ASPECT_')}
env.update(ASPECT_SOURCE_DIR=str(repo), ASPECT_FAULT_EXPLICIT_B='1',
           ASPECT_FAULT_EXPLICIT_G='1', ASPECT_FAULT_SURFACE_SOLVER='tridiagonal',
           ASPECT_FAULT_STRESS_SAMPLE_DIAGNOSTIC='1',
           ASPECT_FAULT_HISTORY_AUDIT='1', ASPECT_BP3_TIMESTEP_SEQUENCE=str(sequence),
           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', DEAL_II_NUM_THREADS='1')
command = ['mpirun', '-np', '4', str(binary), str(prm)]
sources = [binary, plugin, prm, sequence, here/'bp3.cc', here/'replay_time_step.h',
           here/'first_cycle_coarse/original.prm', here/'normal_stress_fresh_local4.prm']
record = dict(command=command, fresh_start=True, no_retry=True, end_step=12,
              environment={k: v for k, v in env.items() if k.startswith('ASPECT_')},
              sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
(out/'provenance.json').write_text(json.dumps(record, indent=2)+'\n')
start = time.monotonic()
with (out/'run.log').open('x') as log:
    result = subprocess.run(command, cwd=here, env=env, stdout=log, stderr=subprocess.STDOUT)
record.update(status=result.returncode, seconds=time.monotonic()-start)
(out/'execution.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k!='sha256'}, indent=2))
