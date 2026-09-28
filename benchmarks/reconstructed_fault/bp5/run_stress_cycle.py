"""Run three independent single-step trials; abort on the first failed branch."""
import argparse
import csv
import json
import os
import resource
import shlex
import subprocess
import time
from pathlib import Path
from stage_normal_stress_diagnostic import sha


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--ranks',type=int,default=4)
    p.add_argument('--cap',type=int,default=3000)
    p.add_argument('--launcher',help="MPI command, for example 'ibrun'; default: mpirun -np RANKS")
    args=p.parse_args()
    root=args.directory.resolve();binary=args.binary.resolve()
    env=os.environ.copy()
    env.update(ASPECT_STRESS_CYCLE_TRACE='1',ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION='1',ASPECT_FAULT_EXPLICIT_B='1',ASPECT_FAULT_EXPLICIT_G='1',
               ASPECT_FAULT_SURFACE_SOLVER='tridiagonal',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',DEAL_II_NUM_THREADS='1')
    start=time.monotonic()
    for branch in ('dt','half','quarter'):
        path=root/branch
        manifest=json.loads((path/'cycle.json').read_text())
        if (path/'run.log').exists(): raise RuntimeError('No automatic retry: existing log')
        if sha(path/'normal_stress_diagnostic_restart.prm')!=manifest['prm_sha256']: raise RuntimeError('Input changed')
        if sha(path/'libbp5_stress_cycle.release.so')!=manifest['cycle_library_sha256']: raise RuntimeError('Cycle plugin changed')
        staging=json.loads((path/'staging.json').read_text())
        for name,digest in {**staging['staged_input_sha256'],**staging['libraries']}.items():
            if sha(path/name)!=digest: raise RuntimeError(f'Dependency changed: {name}')
        for name,digest in staging['checkpoint_sha256'].items():
            if sha(path/'output-normal-diagnostic/restart/01'/name)!=digest:
                raise RuntimeError(f'Incoming checkpoint changed: {name}')
        cap=min(900,int(args.cap-(time.monotonic()-start)))
        if cap<30: raise RuntimeError('Aggregate budget exhausted')
        launcher=shlex.split(args.launcher) if args.launcher else ['mpirun','-np',str(args.ranks)]
        command=['timeout','--kill-after=15',str(cap)]+launcher+[
                 str(binary),'normal_stress_diagnostic_restart.prm']
        now=time.monotonic()
        with (path/'run.log').open('x') as log:
            result=subprocess.run(command,cwd=path,env=env,stdout=log,stderr=subprocess.STDOUT)
        summary=path/'output-normal-diagnostic/normal_summary.csv'
        rows=list(csv.DictReader(summary.open())) if summary.exists() else []
        # The existing observer emits a row only after genuine solver/history checks.
        ok=result.returncode==0 and len(rows)==1 and int(rows[0]['step'])==5613
        record=dict(command=command,binary_sha256=sha(binary),seconds=time.monotonic()-now,
                    returncode=result.returncode,passed=ok,accepted=rows,
                    peak_child_rss_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                    environment={k:v for k,v in env.items() if k.startswith(('ASPECT_','OMP_','DEAL_II_','OPENBLAS_'))})
        (path/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
        print(branch,record['seconds'],ok,flush=True)
        if not ok: raise SystemExit('First unsuccessful branch preserved; no retry or later trial.')
