#!/usr/bin/env python3
"""Bounded serial native finite-interface cases; no production driver changes."""
import os,json,sys
from pathlib import Path
import run_cohesive_interface_native as native
ROOT=Path(__file__).resolve().parents[3]
WORK=ROOT/'build-particle-free/cohesive-interface-insertion'
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_insertion'

def setup(length=1.,height=.125):
    WORK.mkdir(exist_ok=True,parents=True);OUT.mkdir(exist_ok=True,parents=True)
    text=(ROOT/'benchmarks/reconstructed_fault/particle_free/cohesive_interface_native.prm').read_text().replace('/cohesive-interface-native/plugin/','/cohesive-interface-insertion/plugin-final/').replace('set Steps between checkpoint = 1','set Steps between checkpoint = 2')
    text=text.replace('set X extent = 1',f'set X extent = {length}').replace('set Y extent = 0.125',f'set Y extent = {height}')
    template=WORK/'template.prm';template.write_text(text)
    native.SOURCE=template;native.WORK=WORK;native.OUT=OUT
    os.environ.update(CI_EXACT_JUMP='1',CI_FINITE='1',CI_LENGTH=str(length),CI_HEIGHT=str(height),CI_FIRST='.25',CI_LAST='.75',CI_APPEND_FIRST='.125',CI_APPEND_LAST='.875',CI_LOAD_END='.3')

def run(name,first=.25,last=.75,insert=-1.,**kw):
    os.environ.update(CI_FIRST=str(first),CI_LAST=str(last),CI_INSERT_TIME=str(insert))
    return native.run(name,mode='front',**kw)

def main():
    setup();stage=sys.argv[1] if len(sys.argv)>1 else 'qualification'
    if stage=='qualification':
        records=[]
        base=dict(nx=32,ny=40,dt=.002,end=.32)
        records.append(run('installed',first=.125,last=.875,**base))
        records.append(run('insert-early',insert=.10,**base))
        records.append(run('insert-late',insert=.18,**base))
        records.append(run('insert-short',first=.3125,last=.6875,insert=.10,**base))
        split=0.
        for _ in range(51):split+=.002
        records.append(run('split',insert=.10,**dict(base,end=split)))
        records.append(run('resume',insert=.10,resume=True,folder='split',**base))
    elif stage=='identity':
        os.environ['CI_FINITE']='0'
        records=[native.run('front-identity',nx=32,ny=40,dt=.002,end=.02,mode='front')]
    elif stage=='legacy-check':
        os.environ['CI_FINITE']='0';os.environ['CI_LOAD_END']='.84'
        final=native.SOURCE.read_text()
        native.SOURCE.write_text(final.replace('/cohesive-interface-insertion/plugin-final/','/cohesive-interface-front/plugin/'))
        records=[native.run('prior-front',nx=32,ny=40,dt=.002,end=.04,mode='front')]
        native.SOURCE.write_text(final)
        records.append(native.run('current-front',nx=32,ny=40,dt=.002,end=.04,mode='front'))
    elif stage=='fields':
        import numpy as np
        folder=OUT/'growth-128-half';h=np.genfromtxt(folder/'history.csv',delimiter=',',names=True);nodal=np.genfromtxt(folder/'nodal_history.csv',delimiter=',',names=True)
        sample=WORK/'field-input.txt'
        with sample.open('w') as f:
            for t in [2.,2.42]:
                i=np.argmin(abs(h['time_s']-t));assert abs(h['time_s'][i]-t)<1e-10
                values=nodal[i*128:(i+1)*128]['relaxed_jump_m'];assert len(values)==128
                f.write(' '.join(format(v,'.17g') for v in [h['time_s'][i],h['wall_memory_m'][i],*values])+'\n')
        setup(8.,1.);os.environ.update(CI_LOAD_END='2.7',CI_SEED_FIRST='.3125',CI_SEED_LAST='.6875',CI_FIELD_INPUT=str(sample),CI_ALL_FIELDS='1')
        records=[run_field_export()]
    elif stage=='final-checks':
        base=dict(nx=32,ny=40,dt=.002,end=.32)
        records=[run('cache-control',first=.125,last=.875,**base),run('insert-loaded',insert=.28,**base)]
        split=0.
        for _ in range(141):split+=.002
        records.append(run('final-split',insert=.28,**dict(base,end=split)))
        records.append(run('final-resume',insert=.28,resume=True,folder='final-split',**base))
        os.environ.update(CI_APPEND_FIRST='.25',CI_APPEND_LAST='.75')
        records.append(run('no-extension',insert=.1,**base))
    elif stage=='buffers':
        records=[run('buffer-short',first=.3125,last=.6875,nx=32,ny=40,dt=.002,end=.38),run('buffer-long',first=.125,last=.875,nx=32,ny=40,dt=.002,end=.38)]
    elif stage=='focused':
        assert json.loads((OUT/'qualification_summary.json').read_text())['passed']
        setup(8.,1.);os.environ.update(CI_LOAD_END='2.7',CI_SEED_FIRST='.3125',CI_SEED_LAST='.6875')
        records=[run_long(name,128,80,dt,2.42) for name,dt in [('growth-128',.005),('growth-128-half',.0025)]]
    elif stage=='growth':
        assert json.loads((OUT/'qualification_summary.json').read_text())['passed']
        # Use a local explicit runner for the longer but still bounded component
        # test; no change to the existing reference runner or its timeout.
        setup(8.,1.)
        os.environ.update(CI_LOAD_END='2.7',CI_SEED_FIRST='.3125',CI_SEED_LAST='.6875')
        records=[run_long('growth-64',64,40,.005,expected_buffer_stop=True)]
    else:raise ValueError(stage)
    (OUT/f'runs-{stage}.json').write_text(json.dumps(records,indent=2)+'\n')
def run_field_export():
    import time,subprocess,shutil
    folder=WORK/'runs/physical-fields/output';folder.parent.mkdir(parents=True,exist_ok=True)
    if folder.exists():folder.rename(folder.parent/('prior-'+str(time.time_ns())))
    text=(WORK/'growth-128-half.prm').read_text().replace('set End time = 2.42','set End time = 0').replace(str(WORK/'runs/growth-128-half/output'),str(folder))
    prm=WORK/'physical-fields.prm';prm.write_text(text)
    env=dict(os.environ,ASPECT_SOURCE_DIR=str(ROOT),CI_NX='128',CI_WIDTH='.2',CI_MODE='front',CI_FIRST='.125',CI_LAST='.875',CI_INSERT_TIME='-1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    start=time.perf_counter()
    with (WORK/'physical-fields.log').open('w') as f:r=subprocess.run([str(native.EXE),str(prm)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=240)
    result=dict(name='physical-fields',seconds=time.perf_counter()-start,exit_code=r.returncode);print(json.dumps(result),flush=True)
    assert r.returncode==0
    destination=OUT/'physical-fields';destination.mkdir(exist_ok=True)
    for name in ['field_checks.csv','operator_checks.csv','trace_checks.csv']:shutil.copy2(folder/name,destination/name)
    return result

def run_long(name,nx,ny,dt,end=2.75,expected_buffer_stop=False):
    import subprocess,time,shutil
    folder=WORK/'runs'/name;output=folder/'output';folder.mkdir(parents=True,exist_ok=True)
    if output.exists():output.rename(folder/('prior-'+str(time.time_ns())))
    text=native.SOURCE.read_text().replace('set X repetitions = 4',f'set X repetitions = {nx}').replace('set Y repetitions = 40',f'set Y repetitions = {ny}').replace('set End time = 0.004',f'set End time = {end}').replace('set Maximum time step = 0.002',f'set Maximum time step = {dt}').replace('set Maximum first time step = 0.002',f'set Maximum first time step = {dt}').replace('$ASPECT_SOURCE_DIR/build-particle-free/cohesive-interface-native/smoke/output',str(output))
    text=text.replace('set Steps between checkpoint = 2','set Steps between checkpoint = 20')
    prm=WORK/(name+'.prm');prm.write_text(text)
    env=dict(os.environ,ASPECT_SOURCE_DIR=str(ROOT),CI_NX=str(nx),CI_WIDTH='.2',CI_QUADRATURE='3',CI_MODE='front',CI_FIRST='.125',CI_LAST='.875',CI_INSERT_TIME='-1',CI_BUFFER_M='.25',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    start=time.perf_counter()
    with (WORK/(name+'.log')).open('w') as log:
        r=subprocess.run([str(native.EXE),str(prm)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
    result=dict(name=name,nx=nx,ny=ny,dt_s=dt,seconds=time.perf_counter()-start,exit_code=r.returncode)
    destination=OUT/name;destination.mkdir(exist_ok=True,parents=True)
    for p in output.glob('*.csv'):shutil.copy2(p,destination/p.name)
    print(json.dumps(result),flush=True)
    (OUT/(name+'-run.json')).write_text(json.dumps(result,indent=2))
    if r.returncode and not (expected_buffer_stop and 'Cohesive zone reached declared finite candidate buffer' in (WORK/(name+'.log')).read_text()):raise RuntimeError(f'{name} stopped; preserve trial and diagnose before progression.')
    if expected_buffer_stop:assert r.returncode!=0,'Expected finite-buffer control did not stop'
    return result

if __name__=='__main__':main()
