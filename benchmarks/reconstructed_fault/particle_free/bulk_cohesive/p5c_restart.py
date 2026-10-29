#!/usr/bin/env python3
"""Fork a verified same-mode P5c checkpoint, without changing any saved clock/history."""
from pathlib import Path
import argparse,json,zlib,struct,shutil,hashlib

def prepare(work,mode):
    run=work/'runs'/f'seed-{mode}';found=[]
    for p in (run/'restart').glob('[0-9][0-9]'):
        raw=zlib.decompress((p/'resume.z').read_bytes()[16:]);tag=f'bulk cohesive {mode} tensor cell history v1'.encode();off=raw.index(tag)+len(tag)
        q,step,pub,t=struct.unpack_from('<IIQd',raw,off)
        if step==95:found.append((p,q,pub,t))
    assert len(found)==1,found
    source,q,pub,t=found[0];assert q==8 and pub==95 and abs(t-.11875)<1e-12
    target=work/'runs'/f'seed-{mode}-half';assert not target.exists();(target/'restart').mkdir(parents=True);shutil.copytree(source,target/'restart'/source.name)
    (target/'restart/last_good_checkpoint.txt').write_text(str(int(source.name))+'\n')
    prm=f'include $ASPECT_SOURCE_DIR/build-particle-free/p5c-native/seed-{mode}.prm\nset Output directory = $ASPECT_SOURCE_DIR/build-particle-free/p5c-native/runs/seed-{mode}-half\nset Resume computation = true\nset Maximum time step = 0.000625\nset Maximum first time step = 0.000625\nsubsection Termination criteria\n set End step = 137\nend\n'
    (work/f'seed-{mode}-half.prm').write_text(prm)
    record={'same_mode':mode,'accepted_parent_step':95,'accepted_parent_publications':pub,'accepted_parent_time_s':t,'dt_s':.000625,'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()},'clock_changes':'none; existing post_resume_time_step hook reduces pending dt'}
    (work/f'seed-{mode}-half-input.json').write_text(json.dumps(record,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('work',type=Path);p.add_argument('mode',choices=['fixed','radial']);a=p.parse_args();prepare(a.work,a.mode)
