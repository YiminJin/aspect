"""Disposable checkpoint export, frozen comparison, or explicitly selected FE-history solve."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

p=argparse.ArgumentParser(); p.add_argument('mode',choices=['export','frozen','solve'])
p.add_argument('--tag',default='local4'); p.add_argument('--bulk-dir',default='history-mechanics-export-local4'); args=p.parse_args()
here=Path(__file__).resolve().parent; repo=here.parents[2]
out=here/('history-mechanics-'+args.mode+'-'+args.tag); out.mkdir()
slot='01' if args.mode=='export' else '03'
checkpoint=here/'normal-stress-complete-local4/restart'/slot
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
hashes={p.name:digest(p) for p in checkpoint.iterdir() if p.is_file()}
shutil.copytree(checkpoint,out/'restart'/slot)
(out/'restart/last_good_checkpoint.txt').write_text(str(int(slot))+'\n')
prm=out/'diagnostic.prm'
prm.write_text(f'include {here}/normal_stress_fresh_local4.prm\nset Output directory = {out}\nset Resume computation = true\n')
env={k:v for k,v in os.environ.items() if not k.startswith('ASPECT_')}
env.update(ASPECT_SOURCE_DIR=str(repo),ASPECT_FAULT_EXPLICIT_B='1',ASPECT_FAULT_EXPLICIT_G='1',
           ASPECT_FAULT_SURFACE_SOLVER='tridiagonal',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',DEAL_II_NUM_THREADS='1')
if args.mode=='export': env['ASPECT_BP3_EXPORT_CHECKPOINT_BULK']='1'
else:
    env.update(ASPECT_FAULT_HISTORY_AUDIT='1',ASPECT_FAULT_NONCOMMITTING_DIAGNOSTIC='1',ASPECT_FAULT_STRESS_SAMPLE_DIAGNOSTIC='1')
    if args.mode=='frozen':
        env['ASPECT_BP3_FROZEN_SURFACE']=str(here/args.bulk_dir)
        shutil.copyfile(here/'normal-stress-complete-local4/fault_12.csv',here/args.bulk_dir/'fault_12.csv')
    else: env.update(ASPECT_FAULT_HISTORY_FE='1',ASPECT_FAULT_NONLINEAR_DIAGNOSTIC='1')
command=['mpirun','-np','4',str(repo/'build-pf-cpdi/aspect-release'),str(prm)]
record=dict(mode=args.mode,command=command,checkpoint=str(checkpoint),checkpoint_sha256=hashes,
            environment={k:v for k,v in env.items() if k.startswith('ASPECT_')},
            files={str(p):digest(p) for p in [repo/'build-pf-cpdi/aspect-release',here/'build/libbp3.release.so',
                  repo/'source/reconstructed_fault/surface_system.cc',here/'history_mechanics_diagnostic.h',prm]})
(out/'provenance.json').write_text(json.dumps(record,indent=2)+'\n')
start=time.monotonic()
with (out/'run.log').open('x') as log:
    result=subprocess.run(command,cwd=here,env=env,stdout=log,stderr=subprocess.STDOUT)
text=(out/'run.log').read_text()
marker={'export':'Checkpoint bulk export complete;', 'frozen':'Frozen surface comparison complete',
        'solve':'Noncommitting fault diagnostic converged:'}[args.mode]
record.update(status=result.returncode,seconds=time.monotonic()-start,completed=marker in text,
    source_checkpoint_unchanged=hashes=={p.name:digest(p) for p in checkpoint.iterdir() if p.is_file()},
    copied_checkpoint_unchanged=hashes=={p.name:digest(p) for p in (out/'restart'/slot).iterdir() if p.is_file()},
    accepted_output_written=(out/'accepted_steps.csv').exists())
if args.mode=='export' and record['completed']:
    cells={}
    valid=True
    for f in out.glob('bulk_owned_rank*.csv'):
        with f.open() as stream:
            for r in csv.DictReader(stream):
                try:
                    value=float(r['value']); local=int(r['local'])
                    entries=cells.setdefault(r['cell'],set())
                    valid=valid and local not in entries;entries.add(local)
                except (ValueError,TypeError,KeyError):valid=False
    record['complete_cell_export']=valid and len(cells)==36106 and all(len(v)==22 for v in cells.values())
    record['completed']=record['complete_cell_export']
(out/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k not in ['files','checkpoint_sha256']},indent=2))
assert record['completed'] and record['source_checkpoint_unchanged'] and record['copied_checkpoint_unchanged'] and not record['accepted_output_written']
