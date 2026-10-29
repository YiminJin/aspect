#!/usr/bin/env python3
"""Branch an immutable native accepted checkpoint with a shorter pending step."""
from pathlib import Path
import os
import argparse,hashlib,json,shutil,struct,zlib
ROOT=Path(os.environ.get('ASPECT_P4_ROOT',Path(__file__).resolve().parents[4])).resolve()
BASE=ROOT/'build-particle-free/bulk-continuation'
def prepare(name,source,dt,end,include="benchmarks/reconstructed_fault/particle_free/bulk_cohesive/fracture.prm"):
    source=source.resolve()
    target=BASE/'runs'/name
    assert not target.exists(),target
    data=zlib.decompress((source/'resume.z').read_bytes()[16:]);tag=b'bulk cohesive cell history v3';off=data.index(tag)+len(tag)
    q,step,pub,time=struct.unpack_from('<IIQd',data,off)
    assert pub==step and dt>0 and end>time
    # Discover the native clock tuple from its known pending step and accepted
    # time, rather than assuming a Boost archive offset or rewriting its bytes.
    clocks=[];needle=struct.pack('<I',step+1);pos=0
    while True:
        pos=data.find(needle,pos,4096)
        if pos<0:break
        if pos>=24:
            t,pending,old=struct.unpack_from('<ddd',data,pos-24)
            if t>time and 0<pending<1 and 0<old<1 and abs((t-pending)-time)<1e-12:
                clocks.append(dict(offset=pos-24,pending_time=t,pending_dt=pending,old_dt=old,pending_step=step+1))
        pos+=1
    assert len(clocks)==1,clocks
    clock=clocks[0]
    assert dt<=clock['pending_dt']*(1+1e-12),'Cannot increase the saved pending interval: '+str(clock)
    (target/'restart').mkdir(parents=True);shutil.copytree(source,target/'restart'/source.name)
    (target/'restart/last_good_checkpoint.txt').write_text(str(int(source.name))+'\n')
    finalstep=step+round((end-time)/dt)
    assert abs(time+(finalstep-step)*dt-end)<1e-12
    prm=BASE/(name+'.prm')
    prm.write_text(f'include $ASPECT_SOURCE_DIR/{include}\n'
      'set Additional shared libraries = $ASPECT_SOURCE_DIR/build-particle-free/bulk-continuation/plugin/libbulk_propagation_checks.release.so\n'
      f'set Output directory = $ASPECT_SOURCE_DIR/build-particle-free/bulk-continuation/runs/{name}\n'
      f'set Resume computation = true\nset Maximum time step = {dt:.17g}\nset Maximum first time step = {dt:.17g}\nset End time = 0.185\n'
      f'subsection Termination criteria\n  set Termination criteria = end step\n  set End step = {finalstep}\nend\n')
    manifest=dict(source=str(source.relative_to(ROOT)),accepted_step=step,accepted_time=time,publications=pub,quadrature=q,dt=dt,end=end,clock=clock,
      files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()})
    (BASE/(name+'-input.json')).write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('source',type=Path);p.add_argument('dt',type=float);p.add_argument('end',type=float);p.add_argument('--include',default='benchmarks/reconstructed_fault/particle_free/bulk_cohesive/fracture.prm');a=p.parse_args();prepare(a.name,a.source,a.dt,a.end,a.include)
