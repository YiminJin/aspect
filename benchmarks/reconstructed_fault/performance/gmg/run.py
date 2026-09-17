"""One frozen wide-BP3 linearization, four ranks, no accepted step-2 update."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
BP3=REPO/'benchmarks/reconstructed_fault/bp3'
OUT=HERE/'frozen-wide-verified-local4'
subprocess.run(['python3',str(BP3/'run_research.py'),'--configuration','wide',
                '--velocity-preconditioner','amg',
                '--prepare-only','--output',str(OUT)],check=True,cwd=REPO)
record=json.loads((OUT/'provenance.json').read_text())
env=__import__('os').environ.copy()
env={k:v for k,v in env.items() if not k.startswith('ASPECT_')}
env.update(record['environment'])
env.update(ASPECT_FAULT_LINEAR_PERFORMANCE='1',ASPECT_FAULT_GMG_HIERARCHY='1',ASPECT_FROZEN_GMG_STEP='2',
           ASPECT_FROZEN_GMG_NEWTON='4',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
           DEAL_II_NUM_THREADS='1')
plugin=HERE.parent/'build-gmg/libfault_frozen_gmg.release.so'
prm=OUT/'probe.prm'
prm.write_text(f'''include {OUT/'run.prm'}
set Additional shared libraries = {BP3/'build/libbp3.release.so'}, {plugin}
''')
command=['timeout','--signal=TERM','--kill-after=15','600','mpirun','-np','4',
         '--bind-to','core','--map-by','core',str(REPO/'build-pf-cpdi/aspect-release'),str(prm)]
record.update(command=command,environment={k:v for k,v in env.items() if k.startswith('ASPECT_')})
for p in (Path(__file__),prm,plugin,REPO/'tests/reconstructed_fault_frozen_gmg.cc'):
    record['sha256'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
(OUT/'probe_provenance.json').write_text(json.dumps(record,indent=2)+'\n')
start=time.monotonic()
with (OUT/'probe.log').open('x') as log:
    result=subprocess.run(command,cwd=BP3,env=env,stdout=log,stderr=subprocess.STDOUT)
record=dict(exit_status=result.returncode,wall_seconds=time.monotonic()-start)
log=(OUT/'probe.log').read_text()
record['intentional_stop']='FROZEN AMG/GMG COMPARISON PASSED' in log
(OUT/'probe_execution.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2),flush=True)
assert record['intentional_stop'] and result.returncode not in (0,124,137)
with (OUT/'frozen_gmg.csv').open() as stream: rows=list(csv.DictReader(stream))
assert [r['backend'] for r in rows]==['AMG','GMG']
assert all(float(r['fresh'])<=float(r['tolerance']) for r in rows)
assert rows[0]['rhs_norm']==rows[1]['rhs_norm'] and rows[0]['tolerance']==rows[1]['tolerance']
for r in rows: print(r,flush=True)
