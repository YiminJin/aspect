#!/usr/bin/env python3
"""Lossless compact copies of this task's generated evidence; deletes nothing."""
from pathlib import Path
import csv,json,hashlib,tarfile,shutil
import numpy as np
ROOT=Path('build-particle-free/bulk-parallel');OUT=Path('benchmarks/reconstructed_fault/particle_free/evidence/bulk_parallel')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(8*1024*1024):h.update(block)
    return h.hexdigest()

def main():
    dataset=OUT/'quadrature';dataset.mkdir(exist_ok=True)
    selected=[0,96,120,121,124,128,139,140,148];manifest={}
    fields=['phi','Txx','Tyy','Txy','tauxx','tauyy','tauxy','a','q','k','L','driving','pressure']
    for n in selected:
        source=ROOT/f'runs/mixed/bulk-{n:04d}.csv';target=dataset/f'accepted-{n:04d}.npz'
        with source.open() as f:header=f.readline().strip().split(',')
        if not target.exists():
            raw=np.loadtxt(source,delimiter=',',skiprows=1,usecols=range(1,len(header)))
            data={name:raw[:,header.index(name)-1].reshape(-1,64) for name in fields}
            np.savez_compressed(target,**data)
            loaded=np.load(target);assert all(np.array_equal(loaded[k],v) for k,v in data.items())
            if n==0:
                cell_ids=[r['cell'] for r in csv.DictReader((ROOT/'runs/mixed/means-0000.csv').open())]
                np.savez_compressed(dataset/'geometry.npz',points=raw[:,:2].reshape(-1,64,2),JxW=raw[:,2].reshape(-1,64),cell_ids=cell_ids)
        manifest[str(n)]=dict(source=str(source),source_sha256=sha(source),compact=str(target),sha256=sha(target),float64_roundtrip_verified=True)
        print('quadrature',n,target.stat().st_size,flush=True)
    (dataset/'manifest.json').write_text(json.dumps(dict(fields=fields,note='13 native float64 fields, original CellId/Gauss8 order; coordinates/JxW stored once; no history interpolation',states=manifest),indent=2)+'\n')
    # Preserve all nine DG modes, native FE coefficients and all QP histories in
    # canonical small-control records. Dense text is redundant with these and
    # remains untouched in the raw staging directory.
    archive=OUT/'qualification.tar.xz'
    if not archive.exists():
        with tarfile.open(archive,'x:xz',preset=1) as tar:
            for name in ['r1','r2','r4','resume-r1','resume-r2','resume-r4']:
                path=ROOT/'runs'/name
                for p in sorted(path.iterdir()):
                    if p.is_file() and not (p.name.startswith('bulk-') and p.name[5:9].isdigit()):tar.add(p,arcname=name+'/'+p.name)
                if name.startswith('r') and not name.startswith('resume'):
                    for p in sorted((path/'restart/03').iterdir()):tar.add(p,arcname=name+'/restart/03/'+p.name)
                    data=str(3).encode();import io
                    info=tarfile.TarInfo(name+'/restart/last_good_checkpoint.txt');info.size=len(data);tar.addfile(info,io.BytesIO(data))
    archive=OUT/'mixed-diagnostics.tar.xz'
    if not archive.exists():
        with tarfile.open(archive,'x:xz',preset=1) as tar:
            for p in sorted((ROOT/'runs/mixed').iterdir()):
                if p.is_file() and not (p.name.startswith('bulk-') and p.name[5:9].isdigit()) and not p.name.startswith('record-'):tar.add(p,arcname=p.name)
    raw={}
    for name in ['baseline','r1','r2','r4','resume-r1','resume-r2','resume-r4','mixed','layout']:
        for p in sorted((ROOT/'runs'/name).rglob('*')):
            if p.is_file():raw[str(p)]=dict(bytes=p.stat().st_size,sha256=sha(p))
    (OUT/'raw-manifest.json').write_text(json.dumps(raw,indent=2)+'\n')
    for name in ['entry.json','mixed-launch.json','restart-r1-input.json','restart-r2-input.json','restart-r4-input.json']:
        shutil.copy2(ROOT/name,OUT/name)
    logdir=OUT/'logged';logdir.mkdir(exist_ok=True)
    for p in Path('benchmarks/reconstructed_fault/refactoring_boundary/evidence').glob('bulk-parallel-*'):
        if p.suffix in ['.log','.json']:shutil.copy2(p,logdir/p.name)
    print('archives',[(p.name,p.stat().st_size) for p in OUT.glob('*.tar.xz')],flush=True)
if __name__=='__main__':main()
