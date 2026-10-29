#!/usr/bin/env python3
"""Branch a valid native checkpoint; never edit its archive or accepted owners."""
from pathlib import Path
import argparse,shutil,zlib,struct,json,hashlib


def main(rank):
    base=Path('build-particle-free/bulk-parallel');run=base/'runs'/f'r{rank}';target=base/'runs'/f'resume-r{rank}'
    assert not target.exists();target.mkdir()
    selected=None
    for p in (run/'restart').glob('*/resume.z'):
        data=zlib.decompress(p.read_bytes()[16:]);tag=b'bulk cohesive cell history v3';offset=data.find(tag);assert offset>=0
        q,step,pub,t=struct.unpack_from('<IIQd',data,offset+len(tag))
        print(p,step,pub,t)
        if step==47:selected=p.parent
    assert selected is not None,'The pre-growth accepted checkpoint must still be retained'
    (target/'restart').mkdir();shutil.copytree(selected,target/'restart'/selected.name)
    (target/'restart/last_good_checkpoint.txt').write_text(selected.name+'\n')
    prm=base/f'resume-r{rank}.prm';prm.write_text('include $ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/particle_free/bulk_cohesive/parallel.prm\nset Output directory = $ASPECT_SOURCE_DIR/build-particle-free/bulk-parallel/runs/resume-r'+str(rank)+'\nset Resume computation = true\n')
    manifest={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in selected.iterdir() if p.is_file()}
    (base/f'restart-r{rank}-input.json').write_text(json.dumps(dict(source=str(selected),accepted_step=47,files=manifest,parameter_file=str(prm)),indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('rank',type=int);main(p.parse_args().rank)
