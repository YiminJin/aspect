"""One copied revised-work checkpoint; bounded noncommitting A/B mechanics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
SAVED=HERE/'work-replay-50-local4'
ROOT=HERE/'within-step-50-local4'
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('case',choices=['A','B']);args=parser.parse_args()
    assert json.loads((SAVED/'pressure-extraction-9/export/verification.json').read_text())['accepted_step']==9
    if args.case=='B':assert json.loads((ROOT/'A/analysis.json').read_text())['baseline_reproduced']
    out=ROOT/args.case;out.mkdir(parents=True)
    checkpoint=SAVED/'restart/01'
    hashes={p.name:digest(p) for p in checkpoint.iterdir() if p.is_file()}
    shutil.copytree(checkpoint,out/'restart/01')
    (out/'restart/last_good_checkpoint.txt').write_text('1\n')
    prm=out/'run.prm'
    prm.write_text(f'''include {SAVED}/run.prm
set Output directory = {out}
set Resume computation = true
subsection Postprocess
  subsection BP3
    set Committing work-measure replay = false
  end
end
''')
    env={k:v for k,v in os.environ.items() if not k.startswith('ASPECT_')}
    env.update(ASPECT_SOURCE_DIR=str(REPO),ASPECT_FAULT_EXPLICIT_B='1',ASPECT_FAULT_EXPLICIT_G='1',
        ASPECT_FAULT_SURFACE_SOLVER='tridiagonal',ASPECT_FAULT_COMPARE_SURFACE_INVERSE='1',
        ASPECT_FAULT_NONCOMMITTING_DIAGNOSTIC='1',ASPECT_BP3_WITHIN_STEP_DIAGNOSTIC=str(SAVED),
        ASPECT_FAULT_STRESS_SAMPLE_DIAGNOSTIC='1',ASPECT_BP3_TIMESTEP_SEQUENCE=str(SAVED/'accepted_steps.csv'),
        ASPECT_BP3_TARGET_MESH=str(HERE/'junction-matched-qualified-local4/refined/target_cells.txt'),
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',DEAL_II_NUM_THREADS='1')
    if args.case=='B':env['ASPECT_FAULT_WITHIN_STEP_STATE']='1'
    command=['timeout','--signal=TERM','--kill-after=15','300' if args.case=='A' else '600',
        'mpirun','-np','4','--bind-to','core','--map-by','core',str(REPO/'build-pf-cpdi/aspect-release'),str(prm)]
    record=dict(command=command,checkpoint=str(checkpoint),checkpoint_sha256=hashes,
        environment={k:v for k,v in env.items() if k.startswith('ASPECT_')},
        executable_sha256=digest(REPO/'build-pf-cpdi/aspect-release'),plugin_sha256=digest(HERE/'build/libbp3.release.so'))
    (out/'provenance.json').write_text(json.dumps(record,indent=2)+'\n')
    start=time.monotonic()
    with (out/'run.log').open('x') as stream:
        result=subprocess.run(command,cwd=HERE,env=env,stdout=stream,stderr=subprocess.STDOUT)
    log=(out/'run.log').read_text()
    linear=re.findall(r'Fault linear solve: iterations=(\d+), estimated=[^,]+, fresh=([^,]+), target=([^,]+)',log)
    record.update(status=result.returncode,seconds=time.monotonic()-start,
        verified=('Within-step input/derivative checks passed;' in log and
                  'Noncommitting fault diagnostic converged:' in log and 'BP3 noncommitting rollback verified:' in log),
        fresh_linear_passed=bool(linear) and all(float(a)<=float(b) for _,a,b in linear),
        fresh_linear_checks=len(linear),krylov=sum(int(i) for i,_,_ in linear),
        original_unchanged=hashes=={p.name:digest(p) for p in checkpoint.iterdir() if p.is_file()},
        copy_unchanged=hashes=={p.name:digest(p) for p in (out/'restart/01').iterdir() if p.is_file()},
        accepted_output_written=(out/'accepted_steps.csv').exists())
    (out/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ('checkpoint_sha256','environment')},indent=2))
    assert all(record[k] for k in ('verified','fresh_linear_passed','original_unchanged','copy_unchanged'))
    assert not record['accepted_output_written'],'No continuation or accepted-history publication is authorized.'

if __name__=='__main__':main()
