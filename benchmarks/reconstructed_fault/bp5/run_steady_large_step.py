"""Large-clock qualification; immutable runs and disposable checkpoint clocks.

Only the copied pending time/dt may change for a subdivision. Constitutive
history, old dt, accepted step, mesh and frozen prestress bytes are preserved.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time

import run_steady_startup as steady
from startup_30km import BP3, ROOT, STUDY, parameters, render, digest, restore_prefix
from run_coupled_substeps import retime

OUT = STUDY / 'steady-large-step'


def prepare(case, ceiling):
    if case == 'startup':
        steady.OUT = OUT
        steady.prepare(case)
        path = OUT / case
        values = parameters((path/'run.prm').read_text())
        values['Maximum time step',] = '4e6'
        values['End time',] = '1.6e7'
        values['Postprocess','BP3','Last accepted step'] = '4'
        values['Checkpointing','Steps between checkpoint'] = '2'
        record = json.loads((path/'launch.json').read_text())
    else:
        path = OUT / case
        path.mkdir(parents=True)
        source = OUT/'startup'
        assert json.loads((source/'execution.json').read_text())['passed']
        clock = list(csv.DictReader((source/'accepted_steps.csv').open()))
        t0, original_dt = float(clock[1]['time']), float(clock[2]['dt'])
        interval = min(original_dt, ceiling)
        count = 2 if case.startswith('half') else 1
        dt = interval/count
        checkpoint = next(p for p in (source/'restart').glob('[0-9][0-9]')
                          if int((p/'bp3_accepted_state.txt').read_text().split()[0]) == 1)
        shutil.copytree(checkpoint,path/'restart/01')
        (path/'restart/last_good_checkpoint.txt').write_text('1\n')
        for p in (checkpoint/'bp3_output_metadata').iterdir():
            shutil.copy2(p,path/p.name)
        shutil.copy2(source/'cumulative_slip.csv',path/'cumulative_slip.csv')
        restore_prefix(path/'cumulative_slip.csv',1)
        old_clock = (t0+original_dt,original_dt,float(clock[1]['dt']),2)
        changed, offset = retime((checkpoint/'resume.z').read_bytes(),old_clock,t0+dt,dt)
        (path/'restart/01/resume.z').write_bytes(changed)
        hashes={str(p.relative_to(checkpoint)):digest(p) for p in checkpoint.rglob('*') if p.is_file()}
        assert all(digest(path/'restart/01'/p)==h for p,h in hashes.items() if p!='resume.z')
        (path/'checkpoint_source.json').write_text(json.dumps(dict(source=str(checkpoint),hashes=hashes,
            old_clock=old_clock,new_pending_clock=[t0+dt,dt],offset=offset,
            all_other_uncompressed_bytes_identical=True),indent=2)+'\n')
        values=parameters((source/'run.prm').read_text())
        values['Resume computation',]='true'
        values['Output directory',]=str(path)
        values['Maximum time step',]=str(dt)
        values['End time',]=str(t0+interval)
        values['Postprocess','BP3','Last accepted step']=str(1+count)
        values['Checkpointing','Steps between checkpoint']='0'
        record=json.loads((source/'launch.json').read_text())
        record['hashes']={p:h for p,h in record['hashes'].items() if not p.endswith('/run.prm')}
        record['command'][-1]=str(path/'run.prm')
    values['Postprocess','BP3','Audit full state every step']='false'
    (path/'run.prm').write_text(render(values))
    record['hashes'][str(path/'run.prm')]=digest(path/'run.prm')
    record['hashes'][str(Path(__file__).resolve())]=digest(Path(__file__))
    record['environment']['ASPECT_BP5_TIMESTEP_AUDIT']='1'
    (path/'launch.json').write_text(json.dumps(record,indent=2)+'\n')


def run(case):
    path=OUT/case
    record=json.loads((path/'launch.json').read_text())
    for p,h in record['hashes'].items():
        assert digest(Path(p))==h,p
    assert shutil.disk_usage(OUT).free>1024**3
    env={k:v for k,v in os.environ.items() if not k.startswith('ASPECT_')}
    env.update(record['environment'])
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',DEAL_II_NUM_THREADS='1',LD_BIND_NOW='1')
    start=time.monotonic()
    with (path/'run.log').open('x') as log:
        result=subprocess.run(['timeout','--kill-after=15','1200']+record['command'],cwd=BP3,
                              env=env,stdout=log,stderr=subprocess.STDOUT)
    rows=list(csv.DictReader((path/'accepted_steps.csv').open())) if (path/'accepted_steps.csv').exists() else []
    values=parameters((path/'run.prm').read_text())
    passed=result.returncode==0 and bool(rows) and (
        int(rows[-1]['step'])==int(values['Postprocess','BP3','Last accepted step'])
        or float(rows[-1]['time'])==float(values['End time',]))
    info=dict(seconds=time.monotonic()-start,status=result.returncode,passed=passed,
        peak_child_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        accepted_steps=[int(r['step']) for r in rows])
    (path/'execution.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2))
    assert passed,'Unexpected result preserved; no automatic retry.'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run'])
    parser.add_argument('case')
    parser.add_argument('--ceiling',type=float,default=4e6)
    args=parser.parse_args()
    prepare(args.case,args.ceiling) if args.action=='prepare' else run(args.case)
