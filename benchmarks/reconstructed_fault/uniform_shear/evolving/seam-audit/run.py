"""One disposable frozen-data run; never restart an existing output directory."""
import json
import argparse
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

here = Path(__file__).resolve().parent
root = here.parents[4]
parser=argparse.ArgumentParser()
parser.add_argument('--corrected',action='store_true')
args=parser.parse_args()
suffix='-corrected' if args.corrected else ''
assert not (here / ('output'+suffix)).exists(), 'Preserve prior diagnostic evidence.'
start = time.monotonic()
with (here / ('run'+suffix+'.log')).open('w') as log:
    process = subprocess.Popen(
        [str(root / 'build-pf-cpdi/aspect-release'), str(here / ('audit'+suffix+'.prm'))],
        cwd=here / 'build', stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
        env={**os.environ, 'ASPECT_SOURCE_DIR': str(root),
             'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
             **({'K3_CORRECTED_PERIODIC_AUDIT':'1'} if args.corrected else {})})
    try:
        status = process.wait(timeout=180)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        status = 124
usage = resource.getrusage(resource.RUSAGE_CHILDREN)
report = dict(status=status, wall_seconds=time.monotonic()-start,
              peak_rss_KiB=usage.ru_maxrss,
              complete='K3_SEAM_AUDIT_COMPLETE: production residual' in (here / ('run'+suffix+'.log')).read_text())
(here / ('resources'+suffix+'.json')).write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
# Completion deliberately throws before mechanics, not a nonlinear success.
raise SystemExit(0 if report['complete'] and status != 124 else 1)
