"""Bounded fresh small run; preserve outputs and never retry automatically."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import re
import subprocess
import time

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--build',type=Path,default=Path(__file__).resolve().parent/'build')
    p.add_argument('--prm',type=Path,default=Path(__file__).resolve().with_name('clean_stress_cycle.prm'))
    p.add_argument('--label',default='clean-cycle')
    a=p.parse_args(); build=a.build.resolve()
    log=build/(a.label+'-run.log')
    output=re.findall(r'^set Output directory = (.+)$',a.prm.read_text(),re.M)[-1]
    if log.exists() or (build/output).exists():
        raise SystemExit('Existing evidence: select a fresh build/staging directory; no overwrite.')
    env=os.environ.copy()
    env.update(ASPECT_STRESS_CYCLE_TRACE='1',ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION='1',
               ASPECT_FAULT_EXPLICIT_B='1',ASPECT_FAULT_EXPLICIT_G='1',
               ASPECT_FAULT_SURFACE_SOLVER='tridiagonal',OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1',DEAL_II_NUM_THREADS='1')
    cmd=['timeout','--kill-after=10','120',str(a.binary.resolve()),str(a.prm.resolve())]
    start=time.monotonic()
    with log.open('x') as stream:
        run=subprocess.run(cmd,cwd=build,env=env,stdout=stream,stderr=subprocess.STDOUT)
    result=dict(command=cmd,returncode=run.returncode,wall_s=time.monotonic()-start,
                peak_child_rss_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest())
    (build/(a.label+'-execution.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    raise SystemExit(run.returncode)
