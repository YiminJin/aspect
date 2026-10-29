#!/usr/bin/env python3
"""Bounded sequential native kernel/lifecycle runs; each has a hard timeout."""
import os,json,time,subprocess,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
WORK=ROOT/'build-particle-free/cohesive-interface-native'
SOURCE=ROOT/'benchmarks/reconstructed_fault/particle_free/cohesive_interface_native.prm'
EXE=ROOT/'build-particle-free/coulomb-native/aspect-release'
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_native'

def run(name,nx=4,ny=40,dt=.002,end=0.,mode='formation',width=.025,quad=3,resume=False,folder=None):
    folder=folder or name;output=WORK/'runs'/folder/'output';output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists() and not resume:
        backup=output.parent/('prior-'+str(time.time_ns()));output.rename(backup)
    text=SOURCE.read_text().replace('set X repetitions = 4',f'set X repetitions = {nx}').replace('set Y repetitions = 40',f'set Y repetitions = {ny}')
    text=text.replace('set End time = 0.004',f'set End time = {end}').replace('set Maximum time step = 0.002',f'set Maximum time step = {dt}').replace('set Maximum first time step = 0.002',f'set Maximum first time step = {dt}')
    text=text.replace('$ASPECT_SOURCE_DIR/build-particle-free/cohesive-interface-native/smoke/output',str(output))
    text=text.replace('set Steps between checkpoint = 1','set Steps between checkpoint = 5' if mode=='nonuniform' else 'set Steps between checkpoint = 100000')
    if resume:text+='\nset Resume computation = true\n'
    prm=WORK/(name+'.prm');prm.write_text(text)
    env=dict(os.environ,ASPECT_SOURCE_DIR=str(ROOT),CI_NX=str(nx),CI_WIDTH=str(width),CI_QUADRATURE=str(quad),CI_MODE=mode,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    start=time.perf_counter()
    with (WORK/(name+'.log')).open('w') as log:
        result=subprocess.run([str(EXE),str(prm)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
    record=dict(name=name,nx=nx,ny=ny,dt_s=dt,end_s=end,mode=mode,width_m=width,quad=quad,resume=resume,output=str(output.relative_to(ROOT)),seconds=time.perf_counter()-start,exit_code=result.returncode)
    with (WORK/'runs.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    print(json.dumps(record),flush=True)
    if result.returncode:raise RuntimeError(f'{name} failed; inspect {WORK/(name+".log")}')
    destination=OUT/name;destination.mkdir(exist_ok=True,parents=True)
    for p in output.glob('*.csv'):shutil.copy2(p,destination/p.name)
    return record

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    cases=[dict(name='formation',end=.42),dict(name='formation-half',dt=.001,end=.42),dict(name='mature',mode='mature',dt=.01,end=6),
           dict(name='nonuniform',mode='nonuniform',nx=8,ny=80,dt=.001,end=.15),
           dict(name='split',mode='nonuniform',nx=8,ny=80,dt=.001,end=.07500000000000005),
           dict(name='resume',mode='nonuniform',nx=8,ny=80,dt=.001,end=.15,resume=True,folder='split'),
           dict(name='operator-16',nx=16,ny=160),dict(name='quadrature6',quad=6),dict(name='width-half',nx=16,ny=160,width=.0125),dict(name='operator-32',nx=32,ny=160),dict(name='width-half-32',nx=32,ny=160,width=.0125)]
    records=[run(**case) for case in cases]
    (OUT/'runs.json').write_text(json.dumps(records,indent=2)+'\n')

if __name__=='__main__':main()
