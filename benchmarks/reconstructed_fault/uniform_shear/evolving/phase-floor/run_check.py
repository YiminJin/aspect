"""Run one small phase precision regression with a hard wall cap."""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

parser=argparse.ArgumentParser()
parser.add_argument('case',choices=['one','two','refined'])
args=parser.parse_args()
here=Path(__file__).resolve().parent
root=here.parents[4]
assert not (here/args.case).exists()
command=[str(root/'build-pf-cpdi/aspect-release'),str(here/f'{args.case}.prm')]
if args.case=='two': command=['mpirun','-np','2']+command
start=time.monotonic()
with (here/f'{args.case}.log').open('w') as stream:
    process=subprocess.Popen(command,cwd=here/'tests/build',stdout=stream,stderr=subprocess.STDOUT,
        start_new_session=True,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
    try: status=process.wait(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL);process.wait();status=124
report=dict(status=status,seconds=time.monotonic()-start,
    peak_rss_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
(here/f'{args.case}.resources.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
raise SystemExit(status)
