"""Bounded qualification of the 0--30/30--33 km modified BP5 research fixture."""
import argparse
import csv
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time
import numpy as np
from numpy.polynomial.legendre import leggauss
from run_short import HERE, BP3, ROOT, BIN, LIB, GENERATOR, parameters, render, digest
from run_mechanical_width import VirtualProfile, NORMAL, SN
from slip_history import restore_prefix

STUDY=HERE/'weakening30-dc010-ell100'
FIXTURE=STUDY/'fixture'
PLUGIN=HERE/'build/libbp5_initialization.release.so'


def setup():
    FIXTURE.mkdir(parents=True)
    with (FIXTURE/'mesh.log').open('x') as log:
        subprocess.run([str(GENERATOR),'100','24.4140625','0',str(FIXTURE/'target_cells.txt')],
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    length=100000/SN
    s=np.r_[np.arange(0,33001,100),np.linspace(33000,length,int(np.ceil((length-33000)/100))+1)[1:]][::-1]
    fault=np.column_stack([50000*(1+.5/SN)-.5*s,100000-SN*s,np.full(len(s),.6)])
    fault[0,1]=0.;fault[-1,1]=100000.
    np.savetxt(FIXTURE/'fault.txt',fault,fmt='%.17g')
    prestress=np.column_stack([fault[:,:2],np.tile([26546122.365139291,50e6,0,0,1],(len(s),1))])
    np.savetxt(FIXTURE/'prestress.txt',prestress,fmt='%.17g',header=str(len(s)),comments='')
    profile=VirtualProfile(100,24.4140625)
    rows=[];error=0.
    for j,(a,b) in enumerate(zip(fault[:-1,:2],fault[1:,:2])):
        for q,z in enumerate((leggauss(3)[0]+1)/2):
            p=(1-z)*a+z*b;e=profile.extent
            intervals=[(-e,min(-p[1]/NORMAL[1],e)),(max((100000-p[1])/NORMAL[1],-e),e)]
            value=sum(profile.integrate(p,lo,hi) for lo,hi in intervals)
            check=sum(profile.integrate(p,lo,hi,16) for lo,hi in intervals) if value else 0.
            error=max(error,abs(value-check));rows.append([3*j+q,*p,value])
    assert error<1e-6
    np.savetxt(FIXTURE/'completion.txt',rows,fmt=['%d','%.17g','%.17g','%.17g'],header=str(len(rows)),comments='')
    values=parameters((HERE/'dc010-ell100/weak-state-init/run.prm').read_text())
    values['Additional shared libraries',]=str(PLUGIN)+', '+str(LIB)
    values['Fault reconstruction','Prescribed faults file']=str(FIXTURE/'fault.txt')
    values['Mesh refinement','BP3 saved mesh','Target cells file']=str(FIXTURE/'target_cells.txt')
    values['Postprocess','BP3','Mature prestress file']=str(FIXTURE/'prestress.txt')
    values['Postprocess','BP3','Bottom normalization completion file']=str(FIXTURE/'completion.txt')
    values['Postprocess','BP3','Weakening region length']='30000'
    values['Postprocess','BP3','Last accepted step']='4'
    values['Postprocess','BP3','Audit full state every step']='true'
    values['Postprocess','BP3','Graceful wall seconds']='1750'
    values['Checkpointing','Steps between checkpoint']='3'
    values['Termination criteria','Checkpoint on termination']='true'
    values['End time',]='16000000'
    (FIXTURE/'startup.prm').write_text(render(values))
    old=HERE/'dc010-ell100/fixtures/candidate/target_cells.txt'
    info=dict(weakening=[0,30000],transition=[30000,33000],fault_vertices=len(s),
              spacing=[float(min(-np.diff(s))),float(max(-np.diff(s)))],
              completion_absolute_check=error,mesh_identical_after_regeneration=digest(old)==digest(FIXTURE/'target_cells.txt'),
              hashes={p.name:digest(p) for p in FIXTURE.iterdir() if p.is_file()})
    (FIXTURE/'manifest.json').write_text(json.dumps(info,indent=2)+'\n')


def prepare(kind):
    out=STUDY/kind;out.mkdir()
    values=parameters((FIXTURE/'startup.prm').read_text())
    values['Output directory',]=str(out)
    if kind=='half':
        assert json.loads((STUDY/'startup/execution.json').read_text())['passed']
        times=[float(r['time']) for r in csv.DictReader((STUDY/'startup/accepted_steps.csv').open())]
        targets=[]
        for a,b in zip(times[:-1],times[1:]):targets.extend([(a+b)/2,b])
        expression='2e6'
        for target in reversed(targets):expression=f'if(time<{target:.17g},{target:.17g}-time,{expression})'
        values['Maximum time step',]='2e6'
        values['Time stepping','List of model names']+=', function'
        values['Time stepping','Function','Function expression']=expression
        values['Postprocess','BP3','Last accepted step']='2147483647'
        values['Postprocess','BP3','Audit full state every step']='false'
        values['End time',]=str(times[-1])
        values['Checkpointing','Steps between checkpoint']='0'
        values['Termination criteria','Checkpoint on termination']='false'
    elif kind=='restart':
        source=STUDY/'startup'
        assert json.loads((source/'execution.json').read_text())['passed']
        candidates=[p for p in (source/'restart').glob('[0-9][0-9]')
                    if int((p/'bp3_accepted_state.txt').read_text().split()[0])==2]
        assert len(candidates)==1
        checkpoint=candidates[0]
        shutil.copytree(checkpoint,out/'restart/01')
        (out/'restart/last_good_checkpoint.txt').write_text('1\n')
        for p in (checkpoint/'bp3_output_metadata').iterdir():shutil.copy2(p,out/p.name)
        shutil.copy2(source/'cumulative_slip.csv',out/'cumulative_slip.csv')
        restore_prefix(out/'cumulative_slip.csv',2)
        values['Resume computation',]='true'
        values['Checkpointing','Steps between checkpoint']='0'
        values['Termination criteria','Checkpoint on termination']='false'
        (out/'checkpoint_source.json').write_text(json.dumps(dict(source=str(checkpoint),
            hashes={str(p.relative_to(checkpoint)):digest(p) for p in checkpoint.rglob('*') if p.is_file()}),indent=2)+'\n')
    (out/'run.prm').write_text(render(values))
    env=dict(ASPECT_SOURCE_DIR=str(ROOT),ASPECT_FAULT_EXPLICIT_B='1',ASPECT_FAULT_EXPLICIT_G='1',
             ASPECT_FAULT_SURFACE_SOLVER='tridiagonal',ASPECT_BP5_SHORT_TEST='1',
             ASPECT_BP3_LENGTH_COUPLED_DIAGNOSTIC='1',ASPECT_MECHANICAL_WIDTH_PROBE='1',
             ASPECT_MECHANICAL_PROBE_STEP='0',ASPECT_MECHANICAL_PROBE_NEWTON='0',ASPECT_BP3_LENGTH_QUALIFICATION='1')
    inputs=[BIN,PLUGIN,LIB,out/'run.prm',Path(__file__).resolve(),HERE/'weak_initialization.h',
            BP3/'bp3.cc',BP3/'bp3_model.h',BP3/'work_replay.h']+list(FIXTURE.glob('*.txt'))
    record=dict(command=['mpirun','-np','4','--bind-to','core','--map-by','core',str(BIN),str(out/'run.prm')],
                environment=env,cap_seconds=1800,hashes={str(p):digest(p) for p in inputs})
    (out/'launch.json').write_text(json.dumps(record,indent=2)+'\n')


def run(kind):
    out=STUDY/kind;record=json.loads((out/'launch.json').read_text())
    used=sum(json.loads(p.read_text())['seconds'] for p in STUDY.glob('*/execution.json'))
    cap=min(record['cap_seconds'],int(5400-used)-20);assert cap>0
    assert shutil.disk_usage(STUDY).free>3*1024**3
    for p,sha in record['hashes'].items():assert digest(Path(p))==sha,p
    env={k:v for k,v in os.environ.items() if not k.startswith('ASPECT_')}
    env.update(record['environment']);env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',DEAL_II_NUM_THREADS='1',LD_BIND_NOW='1')
    start=time.monotonic()
    with (out/'run.log').open('x') as log:
        result=subprocess.run(['timeout','--kill-after=15',str(cap)]+record['command'],cwd=BP3,env=env,stdout=log,stderr=subprocess.STDOUT)
    rows=list(csv.DictReader((out/'accepted_steps.csv').open())) if (out/'accepted_steps.csv').exists() else []
    passed=result.returncode==0 and bool(rows) and float(rows[-1]['time'])>=16e6
    info=dict(seconds=time.monotonic()-start,status=result.returncode,passed=passed,
              accepted_steps=[int(r['step']) for r in rows],peak_child_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    (out/'execution.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info,indent=2))
    assert passed,'Preserved failure; do not retry or retune'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['setup','prepare','run'])
    parser.add_argument('kind',nargs='?',choices=['startup','half','restart'])
    a=parser.parse_args()
    if a.action=='setup':setup()
    elif a.action=='prepare':prepare(a.kind)
    else:run(a.kind)
