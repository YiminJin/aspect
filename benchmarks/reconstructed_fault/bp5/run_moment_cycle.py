"""Run exactly one selected bounded moment branch, never overwrite or retry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['production','native_history_reference','horizontal_moment_update'])
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--ranks',type=int,default=1,choices=[1,2])
    p.add_argument('--compatibility-audit',action='store_true')
    p.add_argument('--inclined',action='store_true',help='60-degree A/B fixture; never the horizontal moment correction')
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent/'moment-consistency')
    a=p.parse_args();base=Path(__file__).resolve().parent
    case=a.root.resolve()/(a.mode+('-mpi2' if a.ranks==2 else ''))
    case.mkdir(parents=True,exist_ok=False)
    if a.inclined and a.mode=='horizontal_moment_update':
        p.error('The inclined comparison supports A/B only.')
    shutil.copy2(base/(('inclined_' if a.inclined else 'moment_')+a.mode+'.prm'),case/'run.prm')
    shutil.copy2(base/'build/libbp5_moment_cycle.release.so',case/'libbp5_moment_cycle.release.so')
    for f in ['moment_cycle.cc','clean_stress_cycle.cc']:shutil.copy2(base/f,case/f)
    env=os.environ.copy();env.update(ASPECT_STRESS_CYCLE_TRACE='1',ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION='1',
       ASPECT_FAULT_EXPLICIT_B='1',ASPECT_FAULT_EXPLICIT_G='1',ASPECT_FAULT_SURFACE_SOLVER='tridiagonal',
       OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',DEAL_II_NUM_THREADS='1')
    if a.compatibility_audit:
        env['ASPECT_FAULT_COMPATIBILITY_DIAGNOSTIC']='1'
    cmd=['timeout','--kill-after=10','120','mpirun','-np',str(a.ranks),str(a.binary.resolve()),'run.prm']
    start=time.monotonic()
    with (case/'run.log').open('x') as stream:r=subprocess.run(cmd,cwd=case,env=env,stdout=stream,stderr=subprocess.STDOUT)
    record=dict(command=cmd,returncode=r.returncode,wall_s=time.monotonic()-start,
        compatibility_audit=a.compatibility_audit,
        inclined=a.inclined,
        peak_child_rss_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        binary_sha256=sha(a.binary),plugin_sha256=sha(case/'libbp5_moment_cycle.release.so'),
        sources={f:sha(case/f) for f in ['moment_cycle.cc','clean_stress_cycle.cc','run.prm']})
    (case/'execution.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
    raise SystemExit(r.returncode)
