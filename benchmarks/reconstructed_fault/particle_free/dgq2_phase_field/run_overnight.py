#!/usr/bin/env python3
"""Prepare and run only the selected A/B/D/C restored-composition matrix.

Reuses prepare_integration_fixture and the existing layered reference checker.
No core build, architecture change, old-output cleanup or implicit restart.
"""
import argparse
import csv
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
SUPPORT = HERE.parent / 'transported_objective'
BUILD = ROOT / 'build-particle-free/native-phase-integration'
RUNS = ROOT / 'build-particle-free/transported-objective/stage-c/material-runs'
OUT = ROOT / 'benchmarks/reconstructed_fault/particle_free/evidence/dgq2_overnight'
CASES = dict(A=(2,5,1,.0125,7200), B=(2,5,1,.00625,7200),
             D=(2,4,3,.00625,14400), C=(-2,5,1,.0125,7200))

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def dump(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n'); temporary.replace(path)

def parameters(text):
    """Resolve repeated deal.II sections/assignments without changing their values."""
    data = {}; stack = [data]
    for raw in text.splitlines():
        line = raw.split('#',1)[0].strip()
        if line.startswith('subsection '): stack.append(stack[-1].setdefault(line[11:], {}))
        elif line == 'end': stack.pop()
        elif line.startswith('set '):
            key,value = line[4:].split('=',1); stack[-1][key.strip()] = value.strip()
        elif line: raise ValueError(line)
    assert len(stack)==1
    return data

def render(data, indent=''):
    lines=[]
    for key,value in data.items():
        if isinstance(value,dict):
            lines += [indent+'subsection '+key, render(value,indent+'  '),indent+'end']
        else: lines.append(indent+'set '+key+' = '+value)
    return '\n'.join(lines)

def prepare():
    for case,(U,level,depth,dt,seconds) in CASES.items():
        label='overnight-'+case.lower(); name='layered-phase-integration-'+label; path=RUNS/name
        subprocess.run([sys.executable,str(HERE/'prepare_integration_fixture.py'),label,
                        '--end','.8','--level',str(level)],check=True,cwd=ROOT)
        data=parameters((path/'input.prm').read_text())
        data['Particles']={'Load balancing strategy':'none'}; data['Postprocess'].pop('Particles',None)
        data['Checkpointing']['Steps between checkpoint']='16'
        data['Maximum time step']=data['Maximum first time step']=str(dt)
        data['Time stepping']['Function']['Function expression']=str(dt)
        clock=0.
        for _ in range(round(.8/dt)): clock+=dt
        data['End time']=format(min(.8,clock),'.17g')
        data['Geometry model']['Box'].update({'X repetitions':str(3*depth),'Y repetitions':str(depth)})
        data['Postprocess']['Layered background']['Translation']=str(U)
        mat=data['Material model']['Phase field fault']
        mat['Material exterior stress identity']=mat['Material exterior stress identity'].rsplit('U=',1)[0]+'U='+str(U)
        assert mat['Constitutive spin']=='bulk slip subtracted' and mat['Evolve phase field']=='false'
        assert data['Compositional fields']['Names of fields']=='phase_field'
        assert data['Time stepping']['List of model names']=='function'
        assert data['Discretization']['Composition polynomial degree']=='2'
        assert data['Boundary composition model']['Function']['Function expression']=='0'
        (path/'input.prm').write_text('# Frozen restored native-composition overnight case '+case+'\n'+render(data)+'\n')
        args=dict(label=name,translation=U,level=level,depth_repetitions=depth,dt=dt,end=float(data['End time']),
                  halfwidth=6.,ranks=4,spin='bulk slip subtracted',background_shear=3.7,
                  phase_discretization='DGQ2',adaptive=False,schedule=False,checkpoint_steps=16)
        record=dict(case=case,arguments=args,command=['/opt/openmpi/5.0.6/bin/mpirun','-n','4',str(BUILD/'native/aspect-release'),str(path/'input.prm')],
                    core_sha256=sha(BUILD/'native/aspect-release'),provider_sha256=sha(BUILD/'provider/liblayered_background.release.so'),
                    checker_sha256=sha(BUILD/'checks/libtransported_material_energy.release.so'),input_sha256=sha(path/'input.prm'),
                    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),time_limit_s=seconds)
        dump(path/'command.json',record)
        (OUT/'inputs'/case).mkdir(parents=True);shutil.copy2(path/'input.prm',OUT/'inputs'/case/'input.prm');shutil.copy2(path/'fault.txt',OUT/'inputs'/case/'fault.txt')
    dump(OUT/'queue.json',{'order':list(CASES),'budget_s':28800,'analysis_reserve_s':3600,'state':'prepared'})

def rows(path):
    p=path/'output/material-summary.csv'
    if not p.exists():return []
    with p.open() as f:return list(csv.DictReader(f))

def progress(path,start,case):
    rr=rows(path); last=rr[-1] if rr else {}; elapsed=time.monotonic()-start
    step=int(last.get('step',0)); physical=float(last.get('time',0)); remaining=(.8-physical)*elapsed/physical if physical else None
    record=dict(case=case,step=step,time=physical,elapsed_s=elapsed,remaining_s=remaining,retries=int(last.get('retries',0)))
    dump(OUT/'progress.json',record)
    with (OUT/'progress.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    print(json.dumps(record),flush=True)
    return record

def health(path,case,completed):
    rr=rows(path); U,level,depth,dt,_=CASES[case]; expected=round(.8/dt)
    for i,r in enumerate(rr):
        assert int(r['step'])==i
        assert int(r['history_publications'])==int(r['geometry_publications'])==i
        assert int(r['retries'])==0, 'Unexpected retry: schedule must be reviewed'
        assert abs(float(r['time'])-i*dt)<2e-12
        if i:assert abs(float(r['dt'])-dt)<2e-12
    if completed:assert len(rr)==expected+1
    log=(path/'run.log').read_text(errors='replace'); cells=3*depth*depth*4**level
    startup=re.search(r'Number of active cells:\s*([\d,]+)',log)
    if rr:
        assert startup and int(startup[1].replace(',',''))==cells,(startup,cells)
        with (path/'output/material-0-surface.csv').open() as f:assert len(list(csv.DictReader(f)))==20
    balance=[]
    for line in log.splitlines():
        if 'DG phase transport:' in line:
            balance.append({k:float(v) for k,v in (p.split('=') for p in line.split('DG phase transport:',1)[1].strip().split(', '))})
    for r in balance:assert all(math.isfinite(v) for v in r.values())
    dump(OUT/('health-'+case+'.json'),dict(case=case,completed=completed,cells=cells,accepted_steps=len(rr)-1,
         last_time=float(rr[-1]['time']) if rr else None,stage_balance=balance,
         signed_balance=sum(r['balance error'] for r in balance),absolute_balance=sum(abs(r['balance error']) for r in balance)))

def compress(path):
    """Lossless compact retention only after this new run has exited."""
    ledger=[]
    for p in sorted((path/'output').glob('material-*-rank*.csv')):
        target=p.with_suffix('.csv.gz'); assert not target.exists()
        digest=sha(p)
        with p.open('rb') as src,gzip.open(target,'wb',compresslevel=1) as dst:shutil.copyfileobj(src,dst)
        with gzip.open(target,'rb') as src:assert hashlib.file_digest(src,'sha256').hexdigest()==digest
        ledger.append(dict(path=str(target.relative_to(ROOT)),uncompressed_sha256=digest,bytes=p.stat().st_size,gzip_bytes=target.stat().st_size))
        p.unlink() # Only this task's new file; verified identical compressed copy retained.
    dump(path/'compressed-fields.json',ledger)

def run():
    lock=(RUNS.parent/'simulation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start_all=time.monotonic();deadline=start_all+28800; result=[]; a_seconds=None
    env=os.environ.copy();env.update(LD_LIBRARY_PATH='/opt/openmpi/5.0.6/lib',OMP_NUM_THREADS='1',DEAL_II_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMP_PROC_BIND='false',OMPI_MCA_mpi_yield_when_idle='1',ASPECT_TRUE_STRESS_ENERGY='1')
    for k in list(env):
        if k.startswith('ASPECT_') and k!='ASPECT_TRUE_STRESS_ENERGY':env.pop(k)
    env['LAYERED_EVIDENCE']=str(OUT)
    dump(OUT/'environment.json',{k:v for k,v in env.items() if k.startswith(('ASPECT_','OMP','DEAL_II','OPENBLAS','LD_LIBRARY_PATH'))})
    for case in CASES:
        path=RUNS/('layered-phase-integration-overnight-'+case.lower());record=json.loads((path/'command.json').read_text());assert not (path/'run.log').exists()
        estimate=(a_seconds or 2700)*dict(A=1,B=1.9,D=4.4,C=1.15)[case]
        available=deadline-time.monotonic()-3600
        if estimate>available:
            result.append(dict(case=case,state='deferred-budget',estimate_s=estimate,available_s=available));continue
        limit=min(CASES[case][-1],available);start=time.monotonic();reason=None
        with (path/'run.log').open('x') as log:
            proc=subprocess.Popen(record['command'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            dump(OUT/'active.json',dict(case=case,pid=proc.pid,unix_start=time.time(),deadline_s=limit))
            while proc.poll() is None:
                try: proc.wait(timeout=30)
                except subprocess.TimeoutExpired: pass
                try:
                    status=progress(path,start,case);health(path,case,False)
                    if time.monotonic()-start>limit:raise TimeoutError('Selected wall limit')
                    if shutil.disk_usage(ROOT).free<400*1024**2:raise RuntimeError('Disk reserve reached')
                except Exception as exc:
                    reason=str(exc);os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL)
                    break
            code=proc.wait()
        record.update(exit_code=code,wall_s=time.monotonic()-start,stop_reason=reason);dump(path/'command.json',record)
        if case=='A' and code==0:a_seconds=record['wall_s']
        try:
            health(path,case,code==0)
            if code==0:subprocess.run([sys.executable,str(SUPPORT/'check_layered_reference.py'),path.name],env=env,cwd=ROOT,check=True,timeout=120)
        except Exception as exc:reason='health: '+str(exc)
        compress(path)
        result.append(dict(case=case,exit_code=code,wall_s=record['wall_s'],reason=reason,completed=code==0 and not reason))
        dump(OUT/'queue.json',dict(order=list(CASES),results=result,elapsed_s=time.monotonic()-start_all,budget_s=28800,state='running'))
        if code!=0 and not rows(path):
            print('Common startup failure; queue stopped before another case.',flush=True);break
    dump(OUT/'queue.json',dict(order=list(CASES),results=result,elapsed_s=time.monotonic()-start_all,budget_s=28800,state='simulations-finished'))
    print('Simulation queue finished',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run']);a=p.parse_args()
    prepare() if a.action=='prepare' else run()
