"""Read one accepted bulk checkpoint; then one exact comparison/conditional solve."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
SAVED=HERE/'fault-grid-50-local4'
ROOT=HERE/'theta-exact-local4'
parser=argparse.ArgumentParser()
parser.add_argument('mode',choices=['export','verify-export','conditional'])
parser.add_argument('--source-dir',type=Path,help='Alternate 50-m case for read-only export only.')
parser.add_argument('--accepted-step',type=int,default=13)
parser.add_argument('--checkpoint-slot',default='02')
args=parser.parse_args()
if args.source_dir is not None:
    assert args.mode in ('export','verify-export'),'Alternate cases are export-only, never mechanical replays.'
    SAVED=args.source_dir.resolve();ROOT=SAVED/f'pressure-extraction-{args.accepted_step}'
def rows(path):
    with path.open() as f:return list(csv.DictReader(f))

def verify_export(out):
    metadata=rows(out/'checkpoint_bulk_metadata.csv')[0]
    accepted=rows(SAVED/f'fault_{args.accepted_step}.csv')[0]
    # ASPECT advances the clock before checkpointing; the stored solution is
    # still accepted step 13, while metadata already describes upcoming step 14.
    assert int(metadata['step'])==args.accepted_step+1 and int(metadata['cells'])==42880
    assert abs(float(metadata['time'])-float(metadata['dt'])-float(accepted['time']))<1e-6
    cells={}
    for rank in range(4):
        with (out/f'bulk_owned_rank{rank}.csv').open() as f:
            for row in csv.DictReader(f):
                key=(row['cell'],int(row['local']))
                assert key not in cells
                cells[key]=float(row['value'])
    assert len({key[0] for key in cells})==42880
    assert len(cells)==42880*22
    assert 'Checkpoint bulk export complete;' in (out/'run.log').read_text()
    return dict(verified=True,accepted_step=args.accepted_step,checkpoint_clock_step=args.accepted_step+1,
                accepted_time=float(accepted['time']),cell_count=42880,coefficients=len(cells))

if args.mode=='verify-export':
    out=ROOT/'export'
    original=json.loads((out/'execution.json').read_text())
    assert original['original_checkpoint_unchanged'] and original['copied_checkpoint_unchanged']
    assert not original['accepted_output_written']
    verification=verify_export(out)
    (out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps(verification,indent=2))
    raise SystemExit(0)
out=ROOT/args.mode
out.mkdir(parents=True)
slot=args.checkpoint_slot if args.mode=='export' else '01'
checkpoint=SAVED/'restart'/slot
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
hashes={p.name:digest(p) for p in checkpoint.iterdir() if p.is_file()}
shutil.copytree(checkpoint,out/'restart'/slot)
(out/'restart/last_good_checkpoint.txt').write_text(str(int(slot))+'\n')
prm=out/'diagnostic.prm'
prm.write_text(f'include {SAVED}/run.prm\nset Output directory = {out}\nset Resume computation = true\n')
env={k:v for k,v in os.environ.items() if not k.startswith('ASPECT_')}
env.update(ASPECT_SOURCE_DIR=str(REPO),ASPECT_FAULT_EXPLICIT_B='1',ASPECT_FAULT_EXPLICIT_G='1',
           ASPECT_FAULT_SURFACE_SOLVER='tridiagonal',ASPECT_BP3_TIMESTEP_SEQUENCE=str(SAVED/'accepted_steps.csv'),
           ASPECT_BP3_TARGET_MESH=str(HERE/'junction-matched-qualified-local4/refined/target_cells.txt'),
           OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',DEAL_II_NUM_THREADS='1')
if args.mode=='export':
    env['ASPECT_BP3_EXPORT_CHECKPOINT_BULK']='1'
else:
    exported=ROOT/'export'
    assert json.loads((exported/'verification.json').read_text())['verified']
    f11,f12,f13=[rows(SAVED/f'fault_{k}.csv') for k in [11,12,13]]
    assert len(f12)==1236 and all(int(f13[i]['prescribed'])==p for i,p in [(795,1),(796,0),(797,0)])
    for i in range(len(f12)):
        v=float(f12[i]['V']);dt=float(f12[i]['dt']);x=v*dt/.008
        theta=float(f11[i]['Theta'])*np.exp(-x)-(.008/v)*np.expm1(-x)
        assert abs(theta/float(f12[i]['Theta'])-1)<1e-12
    for k in [11,12,13]:shutil.copyfile(SAVED/f'fault_{k}.csv',exported/f'fault_{k}.csv')
    (exported/'frozen_update.txt').write_text(''.join(
        ' '.join(['13','0',str(j),f12[j]['dt'],f11[j]['Theta'],f11[j+1]['Theta'],f12[j]['V'],f12[j+1]['V']])+'\n'
        for j in [795,796]))
    parts=[rows(SAVED/f'history_surface_step13_rank{r}.csv') for r in range(4)]
    with (exported/'expected_weak.csv').open('w',newline='') as f:
        writer=csv.writer(f);keys=['weight','particle_q','particle_C','particle_friction','particle_damping','particle_R','particle_sigma']
        writer.writerow(['node']+keys)
        for i in range(1236):writer.writerow([i]+[sum(float(p[i][key]) for p in parts) for key in keys])
    env.update(ASPECT_BP3_THETA_EXACT_DIAGNOSTIC=str(exported),ASPECT_FAULT_NONCOMMITTING_DIAGNOSTIC='1',
               ASPECT_FAULT_THETA_AUDIT_SEGMENTS='795 796',ASPECT_FAULT_STRESS_SAMPLE_DIAGNOSTIC='1',
               ASPECT_FAULT_HISTORY_AUDIT='1')

command=['mpirun','-np','4','--bind-to','core','--map-by','core',str(REPO/'build-pf-cpdi/aspect-release'),str(prm)]
record=dict(mode=args.mode,command=command,checkpoint=str(checkpoint),checkpoint_sha256=hashes,
            environment={k:v for k,v in env.items() if k.startswith('ASPECT_')},
            files={str(p):digest(p) for p in [REPO/'build-pf-cpdi/aspect-release',HERE/'build/libbp3.release.so',
              HERE/'theta_exact_diagnostic.h',HERE/'junction_diagnostic.h',HERE/'bp3.cc',prm,
              REPO/'source/material_model/phase_field_fault.cc',REPO/'source/reconstructed_fault/surface_system.cc']})
(out/'provenance.json').write_text(json.dumps(record,indent=2)+'\n')
start=time.monotonic()
with (out/'run.log').open('x') as log:
    result=subprocess.run(command,cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT)
log=(out/'run.log').read_text()
unchanged=hashes=={p.name:digest(p) for p in checkpoint.iterdir() if p.is_file()}
copy_unchanged=hashes=={p.name:digest(p) for p in (out/'restart'/slot).iterdir() if p.is_file()}
if args.mode=='export':
    verification=verify_export(out)
    verified=verification['verified']
    (out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
else:
    verified=('Theta exact gate and derivatives passed;' in log
              and 'Noncommitting fault diagnostic converged:' in log
              and 'BP3 noncommitting rollback verified:' in log)
    if 'Theta exact comparison complete: no sign reversal, no solve.' in log:
        verified=True
record.update(status=result.returncode,seconds=time.monotonic()-start,verified=verified,
              original_checkpoint_unchanged=unchanged,copied_checkpoint_unchanged=copy_unchanged,
              accepted_output_written=(out/'accepted_steps.csv').exists())
(out/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k not in ['files','checkpoint_sha256','environment']},indent=2))
assert verified and unchanged and copy_unchanged and not record['accepted_output_written'], 'Preserve evidence; no automatic retry.'
