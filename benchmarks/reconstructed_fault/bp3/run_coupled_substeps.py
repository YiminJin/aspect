"""Only two/four coupled-state substeps from the identical revised step-9 state."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import struct
import subprocess
import time
import zlib

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
SAVED=HERE/'work-replay-50-local4'
ROOT=HERE/'coupled-substeps-50-local4'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def retime(data, old_clock, new_time, new_dt):
    # Same narrowly checked archive operation as run_contact_subdivision.py.
    # No state/history/plugin byte, preceding dt, or step index is changed.
    header=struct.unpack('<4I',data[:16]);raw=zlib.decompress(data[16:])
    assert header==(1,len(raw),len(raw),len(data)-16)
    key=struct.pack('<dddI',*old_clock)
    assert raw.count(key)==1,'Unknown checkpoint clock layout.'
    offset=raw.index(key)
    assert offset==237,'Previously qualified archive layout changed.'
    changed=raw[:offset]+struct.pack('<dd',new_time,new_dt)+raw[offset+16:]
    assert changed[:offset]+struct.pack('<dd',*old_clock[:2])+changed[offset+16:]==raw
    compressed=zlib.compress(changed,9)
    return struct.pack('<4I',1,len(changed),len(changed),len(compressed))+compressed,offset


def prepare(count,out):
    out.mkdir(parents=True) # Preserve all previous evidence; refuse overwrite.
    with (SAVED/'accepted_steps.csv').open() as f:
        accepted=list(csv.DictReader(f))
    rows=[{k:r[k] for k in ('step','time','dt')} for r in accepted[:10]]
    start=float(accepted[9]['time']);end=float(accepted[10]['time'])
    dt=float(accepted[10]['dt'])/count
    for i in range(count):
        rows.append(dict(step=10+i,time=start+(i+1)*dt,dt=dt))
    assert abs(rows[-1]['time']-end)<1e-6
    with (out/'clock.csv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=['step','time','dt']);writer.writeheader();writer.writerows(rows)
    checkpoint=SAVED/'restart/01'
    hashes={p.name:digest(p) for p in checkpoint.iterdir() if p.is_file()}
    shutil.copytree(checkpoint,out/'restart/01')
    (out/'restart/last_good_checkpoint.txt').write_text('1\n')
    clock=(end,float(accepted[10]['dt']),float(accepted[9]['dt']),10)
    changed,offset=retime((checkpoint/'resume.z').read_bytes(),clock,rows[10]['time'],dt)
    (out/'restart/01/resume.z').write_bytes(changed)
    assert all(digest(out/'restart/01'/p)==h for p,h in hashes.items() if p!='resume.z')
    (out/'checkpoint_retime.json').write_text(json.dumps(dict(source=str(checkpoint),source_sha256=hashes,
        old_clock=clock,new_clock=[rows[10]['time'],dt],offset=offset,
        all_bytes_outside_pending_time_dt_unchanged=True),indent=2)+'\n')
    (out/'run.prm').write_text(f'''include {SAVED}/run.prm
set Output directory = {out}
set Resume computation = true
subsection Termination criteria
  set Termination criteria = end time, BP3 replay complete
end
''')
    print(json.dumps(dict(substeps=count,start=start,end=end,dt=dt),indent=2))


def run(count,out):
    env={k:v for k,v in os.environ.items() if not k.startswith('ASPECT_')}
    env.update(ASPECT_SOURCE_DIR=str(REPO),ASPECT_FAULT_EXPLICIT_B='1',ASPECT_FAULT_EXPLICIT_G='1',
        ASPECT_FAULT_SURFACE_SOLVER='tridiagonal',ASPECT_FAULT_COMPARE_SURFACE_INVERSE='1',
        ASPECT_FAULT_WITHIN_STEP_STATE='1',ASPECT_BP3_COUPLED_STATE_REPLAY=str(count),
        ASPECT_BP3_COUPLED_STATE_INPUT=str(SAVED),ASPECT_FAULT_STRESS_SAMPLE_DIAGNOSTIC='1',
        ASPECT_FAULT_SOURCE_HISTORY_DIAGNOSTIC='1',ASPECT_BP3_TIMESTEP_SEQUENCE=str(out/'clock.csv'),
        ASPECT_BP3_TARGET_MESH=str(HERE/'junction-matched-qualified-local4/refined/target_cells.txt'),
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',DEAL_II_NUM_THREADS='1')
    command=['timeout','--signal=TERM','--kill-after=15','1200','mpirun','-np','4','--bind-to','core',
        '--map-by','core',str(REPO/'build-pf-cpdi/aspect-release'),str(out/'run.prm')]
    record=dict(command=command,environment={k:v for k,v in env.items() if k.startswith('ASPECT_')},
        sha256={str(p):digest(p) for p in [Path(__file__),out/'run.prm',out/'clock.csv',
            HERE/'bp3.cc',HERE/'within_step_diagnostic.h',HERE/'work_replay.h',
            HERE/'build/libbp3.release.so',REPO/'build-pf-cpdi/aspect-release']})
    (out/'provenance.json').write_text(json.dumps(record,indent=2)+'\n')
    start=time.monotonic()
    with (out/'run.log').open('x') as log:
        result=subprocess.run(command,cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT)
    text=(out/'run.log').read_text()
    linear=re.findall(r'Fault linear solve: iterations=(\d+), estimated=[^,]+, fresh=([^,]+), target=([^,]+)',text)
    old=json.loads((out/'checkpoint_retime.json').read_text())
    checkpoint=Path(old['source'])
    record=dict(status=result.returncode,seconds=time.monotonic()-start,
        child_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        source_unchanged=old['source_sha256']=={p.name:digest(p) for p in checkpoint.iterdir() if p.is_file()},
        candidate_commits=text.count('Coupled-state commit verified:'),
        completed='BP3 REPLAY COMPLETE:' in text,fresh_linear_checks=len(linear),
        fresh_linear_passed=bool(linear) and all(float(a)<=float(b) for _,a,b in linear),
        krylov=sum(int(i) for i,_,_ in linear))
    (out/'execution.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
    assert result.returncode==0 and record['source_unchanged'] and record['completed']
    assert record['candidate_commits']==count and record['fresh_linear_passed']
    with (out/'accepted_steps.csv').open() as f:
        accepted=list(csv.DictReader(f))
    assert [int(r['step']) for r in accepted]==list(range(10,10+count))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run']);parser.add_argument('count',type=int,choices=[2,4])
    args=parser.parse_args();out=ROOT/f'substeps{args.count}'
    (prepare if args.action=='prepare' else run)(args.count,out)
