#!/usr/bin/env python3
"""Sequential bounded controls for the material cleanup; no growth campaign."""
import hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/material_overnight'
BUILD=ROOT/'build-particle-free/material-overnight'
RUN=ROOT/'benchmarks/reconstructed_fault/particle_free/bulk_tip/run.py'
NEW=BUILD/'native/aspect-release'
CHECK=BUILD/'checks/libbulk_tip_checks.release.so'
OLD=ROOT/'build-particle-free/bulk-tip/native/aspect-release'
OLD_CHECK=ROOT/'build-particle-free/bulk-tip/synthetic-checks/libbulk_tip_checks.release.so'
env=os.environ.copy()
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','DEAL_II_NUM_THREADS'):env[key]='1'
records=[]
def execute(label,command,seconds):
    started=time.monotonic()
    with (OUT/f'{label}.log').open('x') as f:
        try:code=subprocess.run([str(x) for x in command],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=seconds).returncode
        except subprocess.TimeoutExpired:code=124
    r=dict(label=label,command=[str(x) for x in command],exit_code=code,seconds=time.monotonic()-started)
    records.append(r);(OUT/'cleanup-controls.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(r),flush=True)
    if code:raise SystemExit(code)
execute('units-final',[NEW,'--test'],240)
execute('point-controls',[BUILD/'checks/bulk_tip_points'],120)
common=['--fixture','benchmarks/reconstructed_fault/particle_free/bulk_tip/synthetic_propagation.prm','--dt','.0025','--end','.06','--ranks','2','--resume-from','build-particle-free/bulk-tip/runs/synthetic-growth-be-dt0025-local/output','--snapshot-id','3','--phase-balance','--seconds','300']
execute('be-old-restart',['python3',RUN,'material-cleanup-be-old01','--executable',OLD,'--checker',OLD_CHECK]+common,330)
execute('be-new-restart',['python3',RUN,'material-cleanup-be-new01','--executable',NEW,'--checker',CHECK,'--rejection','--rejection-step','24']+common,330)
execute('midpoint-control',['python3',RUN,'material-cleanup-midpoint01','--executable',NEW,'--checker',CHECK,'--fixture','benchmarks/reconstructed_fault/particle_free/bulk_tip/localized.prm','--time-update','midpoint','--dt','.005','--end','.09','--ranks','1','--tangent','--tangent-step','8','--seconds','180'],210)
execute('midpoint-restart',['python3',RUN,'material-cleanup-midpoint-restart01','--executable',NEW,'--checker',CHECK,'--fixture','benchmarks/reconstructed_fault/particle_free/bulk_tip/localized.prm','--time-update','midpoint','--dt','.005','--end','.08','--ranks','2','--resume-from','build-particle-free/bulk-tip/runs/tip-time-local01/output','--snapshot-id','1','--rejection','--rejection-step','16','--seconds','180'],210)
# Retained mature control, with fresh diagnostics compiled for the new core.
run=BUILD/'mature-control';run.mkdir(exist_ok=False)
text=(ROOT/'build-particle-free/bulk-tip/consolidation-mature/input.prm').read_text()
text=text.replace(str(ROOT/'build-particle-free/bulk-tip/consolidation-mature-checks'),str(BUILD/'mature-checks')).replace(str(ROOT/'build-particle-free/bulk-tip/consolidation-layered-provider'),str(BUILD/'layered-provider')).replace(str(ROOT/'build-particle-free/bulk-tip/consolidation-checks'),str(BUILD/'checks'))
text+='\nset Output directory = '+str(run/'output')+'\n'
(run/'input.prm').write_text(text)
execute('mature-control',['/opt/openmpi/5.0.6/bin/mpirun','-np','1',NEW,run/'input.prm'],180)
