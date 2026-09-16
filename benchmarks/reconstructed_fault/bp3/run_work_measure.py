"""One fresh four-rank free-top work-measure qualification; no continuation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
BASE=HERE/'top-source-paired-50-local4'
OUT=HERE/'work-measure-free-top-local4'

def main():
    OUT.mkdir() # Never overwrite failed evidence.
    (OUT/'run.prm').write_text(f'include {BASE}/run.prm\nset Output directory = {OUT}\n'
        'set Resume computation = false\nsubsection Termination criteria\n  set End step = 0\nend\n')
    env={k:v for k,v in os.environ.items() if not k.startswith('ASPECT_')}
    env.update(json.loads((BASE/'provenance.json').read_text())['environment'])
    for key in ['ASPECT_BP3_UNIFORM_SLIDING','ASPECT_BP3_ALL_SOURCE_QPS','ASPECT_FAULT_SOURCE_HISTORY_DIAGNOSTIC',
                'ASPECT_FAULT_HISTORY_AUDIT','ASPECT_FAULT_STRESS_SAMPLE_DIAGNOSTIC','ASPECT_BP3_TIMESTEP_SEQUENCE']:
        env.pop(key,None)
    env.update(ASPECT_BP3_WORK_MEASURE='1',ASPECT_FAULT_NONCOMMITTING_DIAGNOSTIC='1',
               OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',DEAL_II_NUM_THREADS='1')
    command=['timeout','--signal=TERM','--kill-after=15','600','mpirun','-np','4',
             '--bind-to','core','--map-by','core',str(REPO/'build-pf-cpdi/aspect-release'),str(OUT/'run.prm')]
    paths=[Path(__file__),OUT/'run.prm',BASE/'completion.txt',HERE/'mature-fault-50-local4/prestress.txt',
           HERE/'bp3.cc',HERE/'work_measure_diagnostic.h',HERE/'junction_diagnostic.h',
           HERE/'build/libbp3.release.so',REPO/'build-pf-cpdi/aspect-release',
           REPO/'source/reconstructed_fault/surface_system.cc',REPO/'include/aspect/reconstructed_fault/surface_system.h']
    (OUT/'provenance.json').write_text(json.dumps(dict(command=command,
        environment={k:v for k,v in env.items() if k.startswith('ASPECT_')},
        sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}),indent=2)+'\n')
    start=time.monotonic()
    with (OUT/'run.log').open('x') as log:
        result=subprocess.run(command,cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT)
    text=(OUT/'run.log').read_text()
    verified=('Noncommitting fault diagnostic converged:' in text
              and 'BP3 noncommitting rollback verified:' in text
              and 'Work-measure nonuniform free-top production checks passed' in text)
    record=dict(status=result.returncode,seconds=time.monotonic()-start,qualified=verified,
                expected_exit='nonzero: intentional rollback stop after verified convergence')
    (OUT/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
    assert verified,'Preserve the blocker; no automatic simulation retry.'

if __name__=='__main__':main()
