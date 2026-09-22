"""Explicit diagnostic exception to the failed profile-width qualification gate.

This launcher never changes comparison.json or the production guard. It starts
fresh and compares coupled mechanics; neither mesh is promoted to production.
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

import numpy as np
from numpy.polynomial.legendre import leggauss
from length_scale_study import HERE, ROOT, BIN, LIB, digest, parameters, render
from run_mechanical_width import VirtualProfile, NORMAL

STUDY=HERE/'length-scale-study'
OUT=STUDY/'coupled-diagnostic'


def completion(mesh):
    path=OUT/'completion_reference.txt'
    assert not path.exists()
    profile=VirtualProfile(50,6.103515625)
    fault=np.loadtxt(HERE/'fixtures/modified_bp3_dc024_ell50/fault.txt')[:,:2]
    rows=[];error=0.
    for j,(a,b) in enumerate(zip(fault[:-1],fault[1:])):
        for q,z in enumerate((leggauss(3)[0]+1)/2):
            p=(1-z)*a+z*b;e=profile.extent
            intervals=[(-e,min(-p[1]/NORMAL[1],e)),(max((100000-p[1])/NORMAL[1],-e),e)]
            missing=sum(profile.integrate(p,lo,hi) for lo,hi in intervals)
            check=sum(profile.integrate(p,lo,hi,16) for lo,hi in intervals)
            error=max(error,abs(check-missing));rows.append([3*j+q,*p,missing])
    assert error<1e-6
    with path.open('x') as stream:
        stream.write(str(len(rows))+'\n');np.savetxt(stream,rows,fmt=['%d','%.17g','%.17g','%.17g'])
    (OUT/'reference_inputs.json').write_text(json.dumps(dict(mesh_hash=digest(mesh),completion_hash=digest(path),
        completion_order_error=error,endpoint_h=profile.h,ell=50,reference_mode=2),indent=2)+'\n')


def prepare(label,reference=False):
    assert json.loads((STUDY/'comparison.json').read_text())['profile_gate'] is False
    out=OUT/label;out.mkdir(parents=True)
    values=parameters((HERE/'bp3_dc024_ell50.prm').read_text())
    values['Output directory',]=str(out)
    values['Additional shared libraries',]=str(HERE/'build/libbp3.release.so')+', '+str(LIB)
    values['Resume computation',]='false'
    values['Time stepping','List of model names']+=', mechanical probe clock'
    values['Postprocess','BP3','Profile time interval']='1'
    values['Postprocess','BP3','Last accepted step']='2'
    values['Postprocess','BP3','Graceful wall seconds']='2350'
    values['Checkpointing','Time between checkpoint']='0'
    values['Checkpointing','Steps between checkpoint']='2'
    if reference:
        values['Mesh refinement','BP3 saved mesh','Target cells file']=str(OUT/'target_cells_reference.txt')
        values['Mesh refinement','Initial adaptive refinement']='12'
        values['Postprocess','BP3','Bottom normalization completion file']=str(OUT/'completion_reference.txt')
    (out/'run.prm').write_text('# Diagnostic only; original 1% width criterion remains failed.\n'+render(values))
    (out/'clock.csv').write_text('step,time,dt\n0,0,0\n1,4000000,4000000\n2,8000000,4000000\n')
    env=dict(ASPECT_SOURCE_DIR=str(ROOT),ASPECT_FAULT_EXPLICIT_B='1',ASPECT_FAULT_EXPLICIT_G='1',
        ASPECT_FAULT_SURFACE_SOLVER='tridiagonal',ASPECT_MECHANICAL_WIDTH_PROBE='1',
        ASPECT_MECHANICAL_PROBE_STEP='0',ASPECT_MECHANICAL_PROBE_NEWTON='0',
        ASPECT_BP3_LENGTH_STUDY='1',ASPECT_BP3_LENGTH_QUALIFICATION='1',
        ASPECT_BP3_LENGTH_COUPLED_DIAGNOSTIC='1',ASPECT_BP3_TIMESTEP_SEQUENCE=str(out/'clock.csv'))
    paths=[out/'run.prm',out/'clock.csv',BIN,LIB,HERE/'build/libbp3.release.so',Path(__file__)]
    for section,key in [(('Fault reconstruction',),'Prescribed faults file'),
                        (('Mesh refinement','BP3 saved mesh'),'Target cells file'),
                        (('Postprocess','BP3'),'Mature prestress file'),
                        (('Postprocess','BP3'),'Bottom normalization completion file')]:
        paths.append(Path(values[section+(key,)]))
    record=dict(diagnostic_exception=True,production_profile_gate=False,first_step=0,last_step=2,
        cap_seconds=2400,environment=env,hashes={str(p):digest(p) for p in paths},
        command=['mpirun','-np','4','--bind-to','core','--map-by','core',str(BIN),str(out/'run.prm')])
    (out/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
    print('Prepared explicit diagnostic:',out)


def run(label):
    out=OUT/label;record=json.loads((out/'launch.json').read_text())
    used=sum(json.loads(p.read_text())['seconds'] for p in STUDY.rglob('execution.json'))
    cap=min(record['cap_seconds'],int(7200-used));assert cap>0
    for path,sha in record['hashes'].items(): assert digest(Path(path))==sha,path
    env={k:v for k,v in os.environ.items() if not k.startswith('ASPECT_')}
    env.update(record['environment']);env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',DEAL_II_NUM_THREADS='1',LD_BIND_NOW='1')
    start=time.monotonic()
    with (out/'run.log').open('x') as log:
        result=subprocess.run(['timeout','--kill-after=20',str(cap)]+record['command'],cwd=HERE,env=env,
            stdout=log,stderr=subprocess.STDOUT)
    path=out/'accepted_steps.csv';rows=list(csv.DictReader(path.open())) if path.exists() else []
    expected=list(range(record['first_step'],record['last_step']+1))
    actual=[int(r['step']) for r in rows if int(r['step'])>=record['first_step']]
    passed=result.returncode==0 and actual==expected and all(int(r['fresh_linear_checks_passed'])==1 for r in rows)
    info=dict(seconds=time.monotonic()-start,status=result.returncode,expected_completion=passed,
        accepted_steps=actual,previous_simulation_seconds=used,cap_seconds=cap,
        peak_child_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    (out/'execution.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info,indent=2))
    assert passed,'Preserved failure: no automatic retry or solver adjustment'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['completion','prepare','run'])
    p.add_argument('--label',default='candidate-two');p.add_argument('--reference',action='store_true')
    args=p.parse_args()
    if args.action=='completion': completion(OUT/'target_cells_reference.txt')
    elif args.action=='prepare': prepare(args.label,args.reference)
    else: run(args.label)
