"""Fresh four-rank diagnostic through step 12, without a wall-time cap.

Refuses to overwrite previous evidence. No checkpoint input or automatic retry.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

here = Path(__file__).resolve().parent
repo = here.parents[2]
out = here / 'normal-stress-complete-local4'
out.mkdir()
prm = here / 'normal_stress_complete_local4.prm'
binary = repo / 'build-pf-cpdi/aspect-release'
plugin = here / 'build/libbp3.release.so'
env = {k: v for k, v in os.environ.items() if not k.startswith('ASPECT_')}
env.update(ASPECT_SOURCE_DIR=str(repo), ASPECT_FAULT_EXPLICIT_B='1',
           ASPECT_FAULT_EXPLICIT_G='1', ASPECT_FAULT_SURFACE_SOLVER='tridiagonal',
           ASPECT_FAULT_STRESS_SAMPLE_DIAGNOSTIC='1', OMP_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1', DEAL_II_NUM_THREADS='1')
command = ['mpirun', '-np', '4', str(binary), str(prm)]
sources = [prm, binary, plugin, here/'bp3.cc', here/'bp3_model.h', here/'fault.txt',
           here/'normal_stress_fresh_local4.prm', here/'first_cycle_coarse/original.prm',
           repo/'source/reconstructed_fault/surface_system.cc',
           repo/'source/simulator/assemblers/reconstructed_fault_stokes.cc',
           repo/'source/simulator/solver.cc']
record = dict(command=command, cwd=str(here), mpi_ranks=4, fresh_start=True,
              configured_end_step=12, wall_cap_seconds=None,
              sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              environment={k: v for k, v in env.items() if k.startswith('ASPECT_')})
(out/'provenance.json').write_text(json.dumps(record, indent=2)+'\n')
start = time.monotonic()
with (out/'run.log').open('x') as log:
    result = subprocess.run(command, cwd=here, env=env, stdout=log, stderr=subprocess.STDOUT)
record.update(status=result.returncode, wall_seconds=time.monotonic()-start)
(out/'execution.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps({k: v for k, v in record.items() if k != 'sha256'}, indent=2))
