"""Run one selected frozen periodic-domain regression, preserving earlier output."""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

p=argparse.ArgumentParser()
p.add_argument('case', choices=['one','two','one-remote','two-remote'])
args=p.parse_args()
here=Path(__file__).resolve().parent
root=here.parents[4]
assert not (here/args.case).exists(), 'Preserve failed/completed evidence before another invocation.'
command=[str(root/'build-pf-cpdi/aspect-release'),str(here/f'{args.case}.prm')]
if args.case.startswith('two'): command=['mpirun','-np','2']+command
start=time.monotonic()
with (here/f'{args.case}.log').open('w') as log:
    process=subprocess.Popen(command,cwd=here/'build',stdout=log,stderr=subprocess.STDOUT,
                             start_new_session=True,
                             env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
                                  **({'K3_REQUIRE_REMOTE_PERIODIC_IMAGES':'1'} if args.case=='two-remote' else {})})
    try: status=process.wait(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL)
        process.wait()
        status=124
usage=resource.getrusage(resource.RUSAGE_CHILDREN)
report=dict(command=command,status=status,wall_seconds=time.monotonic()-start,peak_rss_KiB=usage.ru_maxrss)
(here/f'{args.case}.resources.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
raise SystemExit(status)
