"""Bounded frozen phase-linearization audit; preserve prior evidence."""
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

here=Path(__file__).resolve().parent
root=here.parents[4]
assert not (here/'output').exists()
start=time.monotonic()
command=[str(root/'build-pf-cpdi/aspect-release'),str(here/'audit.prm')]
with (here/'audit.log').open('w') as log:
    p=subprocess.Popen(command,cwd=here/'build',stdout=log,stderr=subprocess.STDOUT,
        start_new_session=True,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
    try: status=p.wait(timeout=300)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid,signal.SIGKILL); p.wait(); status=124
report=dict(status=status,seconds=time.monotonic()-start,
    peak_rss_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    complete='K3_PHASE_FLOOR_AUDIT_COMPLETE' in (here/'audit.log').read_text())
(here/'resources.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
raise SystemExit(0 if status!=124 and report['complete'] else 1)
