"""Copy local step 11; perform only step-12 transfer/frozen-load extraction."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

here = Path(__file__).resolve().parent
repo = here.parents[2]
checkpoint = here / 'normal-stress-complete-local4/restart/03'
out = here / 'normal-stress-history-load-local4'
out.mkdir()  # Preserve evidence: never overwrite a previous attempt.
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
hashes = {p.name: digest(p) for p in checkpoint.iterdir() if p.is_file()}
shutil.copytree(checkpoint, out / 'restart/03')
(out / 'restart/last_good_checkpoint.txt').write_text('3\n')
prm = out / 'diagnostic.prm'
prm.write_text(f'include {here}/normal_stress_fresh_local4.prm\n'
               f'set Output directory = {out}\nset Resume computation = true\n')
env = {k: v for k, v in os.environ.items() if not k.startswith('ASPECT_')}
env.update(ASPECT_SOURCE_DIR=str(repo), ASPECT_FAULT_EXPLICIT_B='1',
           ASPECT_FAULT_EXPLICIT_G='1', ASPECT_FAULT_SURFACE_SOLVER='tridiagonal',
           ASPECT_BP3_HISTORY_LOAD_DIAGNOSTIC='1', OMP_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1', DEAL_II_NUM_THREADS='1')
binary = repo / 'build-pf-cpdi/aspect-release'
command = ['mpirun', '-np', '4', str(binary), str(prm)]
record = dict(command=command, checkpoint=str(checkpoint), checkpoint_sha256=hashes,
              files={str(p): digest(p) for p in [binary, here/'build/libbp3.release.so',
                    here/'bp3.cc', here/'history_load_diagnostic.h', prm]},
              environment={k: v for k, v in env.items() if k.startswith('ASPECT_')})
(out/'provenance.json').write_text(json.dumps(record, indent=2)+'\n')
start = time.monotonic()
with (out/'run.log').open('x') as stream:
    result = subprocess.run(command, cwd=here, env=env, stdout=stream, stderr=subprocess.STDOUT)
log = (out/'run.log').read_text()
record.update(status=result.returncode, wall_seconds=time.monotonic()-start,
    extraction_complete='Frozen history extraction completed before Newton;' in log,
    intentional_stop='BP3 frozen history audit completed; intentional pre-Newton stop.' in log,
    source_checkpoint_unchanged=hashes == {p.name:digest(p) for p in checkpoint.iterdir() if p.is_file()},
    copied_checkpoint_unchanged=hashes == {p.name:digest(p) for p in (out/'restart/03').iterdir() if p.is_file()},
    accepted_output_written=(out/'accepted_steps.csv').exists())
(out/'execution.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k not in ['files','checkpoint_sha256']}, indent=2))
if not (record['extraction_complete'] and record['intentional_stop']
        and record['source_checkpoint_unchanged'] and record['copied_checkpoint_unchanged']
        and not record['accepted_output_written']):
    raise SystemExit('Frozen extraction did not meet its completion/preservation checks.')
