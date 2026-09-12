"""Exactly one one-rank Release K3 invocation; process-group cap, no retry."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

directory=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--case',choices=('smoke','normal256','spatial0375_n128',
                                    'spatial0375_n256','spatial0375_n512',
                                    'spatial0375_n128_f32','spatial0375_n256_f32'),default='smoke')
parser.add_argument('--wall-cap',type=float,default=180.)
args=parser.parse_args()
root=directory.parents[3]
binary=root/'build-pf-cpdi/aspect-release'
parameter=directory/f'{args.case}.prm'
assert not (directory/args.case).exists(), 'Preserve existing case evidence; no automatic retry.'
command=[str(binary),str(parameter)]
start=time.monotonic()
with (directory/f'{args.case}.log').open('w') as log:
    process=subprocess.Popen(command,cwd=directory/'build',stdout=log,stderr=subprocess.STDOUT,
                             start_new_session=True,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
    try:
        status=process.wait(timeout=args.wall_cap)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL)
        process.wait()
        status=124
usage=resource.getrusage(resource.RUSAGE_CHILDREN)
digest=lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
report=dict(command=command,cwd=str(directory/'build'),ranks=1,wall_cap_seconds=args.wall_cap,
            exit_status=status,wall_seconds=time.monotonic()-start,peak_rss_KiB=usage.ru_maxrss,
            user_seconds=usage.ru_utime,system_seconds=usage.ru_stime,
            binary_sha256=digest(binary),plugin_sha256=digest(directory/'build/libuniform_shear.release.so'),
            parameter_sha256=digest(parameter),git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip())
(directory/f'{args.case}.resources.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
raise SystemExit(status)
