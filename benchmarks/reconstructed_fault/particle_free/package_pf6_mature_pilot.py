#!/usr/bin/env python3
"""Archive bounded pilot evidence and exact source/artifacts; preserve raw runs."""
import argparse,hashlib,json,subprocess,tarfile
from pathlib import Path
from reduce_pf6_mature_pilot import FUNCTIONAL
ROOT=Path(__file__).resolve().parents[3]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('evidence',type=Path);a=p.parse_args()
    a.directory=a.directory.resolve();a.evidence.mkdir(parents=True,exist_ok=True)
    archive=a.evidence/'reference.tar.xz';assert not archive.exists(),archive
    members={};artifacts={}
    def add(path,name=None):
        assert path.is_file(),path
        members[name or 'snapshot/'+str(path.resolve().relative_to(ROOT))]=path
    def artifact(path,expected):
        assert digest(path)==expected,path
        name='artifacts/'+expected+'/'+path.name
        if name not in artifacts:add(path,name);artifacts[name]=dict(sha256=expected,bytes=path.stat().st_size)
    steps={name:{0,3} for name in FUNCTIONAL}
    steps.update({'dg-r1-restart-reference':{0,5},'dg-r1-restart-split':{0,4},
      'dg-r1-restart-second-leg':{0,5},'dg-r1-to-r4-restart':{0,5},
      'dg-r4-to-r2-restart':{0,6},'dg-r4-restart-reference':{0,6},
      'dg-outer-cutback-r2-final':{0,4},'dg-nonlinear-cutback-r4':{0,4},
      'dg-cutback-matching-time':{0,4},'dg-fixed-ff-h8-final':{0,1,2,3},
      'dg-fixed-ff-h8-dt100':{0,2,4,6},'dg-ff-dt100-final':{0,2,4,6},
      'dg-ff-dt50-final':{0,4,8,12}})
    heavy=('authoritative-stress-','physical-fe-','dg-coefficients-')
    for directory in (a.directory/'runs',a.directory/'current/runs'):
        for record in sorted(directory.glob('*/record.json')):
            case=record.parent;data=json.loads(record.read_text())
            for name in ('record.json','input.prm','source.patch','run.log'):
                path=case/name
                if path.exists():add(path)
            for name,sha in data['artifacts'].items():artifact(case/Path(name).name,sha)
            for path in (case/'run').glob('*.csv'):
                if case.name in steps:
                    if path.name.startswith(heavy):
                        if int(path.stem.rsplit('-',1)[1]) not in steps[case.name]:continue
                    elif path.name.startswith('audit_particles_'):
                        if int(path.name.split('_')[2]) not in steps[case.name]:continue
                    add(path)
                elif path.stat().st_size<500000:add(path)
            # Retain actual serial/split/cross-rank native restart checkpoints.
            if case.name in ('dg-life-r1-final','dg-r1-restart-split','dg-r1-to-r4-restart'):
                for path in (case/'run/restart').rglob('*'):
                    if path.is_file():add(path)
    baseline=ROOT/'build-particle-free/mature-resolution/runs/dg-ff'
    for path in (baseline/'run').glob('*.csv'):
        if path.name.startswith(heavy+('coupled-profile-','surface-weak-')) or path.name in ('initial-mesh.csv','dg-support.csv'):add(path)
    add(baseline/'record.json')
    add(ROOT/'build-particle-free/mature-resolution/inputs-lf/physical-profile.csv')
    for directory in ('inputs','surface-full','surface-full-final','surface-full-intended','server-pilot-final'):
        folder=a.directory/directory
        for path in folder.rglob('*'):
            if not path.is_file():continue
            if path.suffix in ('.so',) or path.name=='aspect-release':continue
            if any(s in path.parts for s in ('solution','particles','profiles','reconstructed_faults')):continue
            if path.stat().st_size>5000000 and path.suffix not in ('.csv','.xz'):continue
            add(path)
        manifest=folder/'manifest.json'
        if manifest.exists():
            for name,sha in json.loads(manifest.read_text()).get('artifacts',{}).items():artifact(folder/Path(name).name,sha)
    for pattern in ('*.json','*.log','*.patch'):
        for path in a.directory.glob(pattern):add(path)
    for path in (ROOT/'benchmarks/reconstructed_fault/refactoring_boundary/evidence').glob('pf6-mature-*.json'):add(path)
    for path in (ROOT/'benchmarks/reconstructed_fault/refactoring_boundary/evidence').glob('pf6-mature-*.log'):add(path)
    paths=subprocess.check_output(['git','ls-files','-c','-o','--exclude-standard','source','include','cmake','CMakeLists.txt','unit_tests','tests',
      'benchmarks/reconstructed_fault/bp3_native','benchmarks/reconstructed_fault/particle_free','doc/reconstructed_fault','AGENTS.md'],cwd=ROOT,text=True).splitlines()
    for relative in sorted(set(paths)):
        path=ROOT/relative
        if '/evidence/' in relative:continue
        if path.is_file() and (path.suffix in ('.cc','.h','.prm','.cmake','.py','.md','.tex') or path.name=='CMakeLists.txt'):add(path)
    manifest={name:dict(bytes=path.stat().st_size,sha256=digest(path)) for name,path in sorted(members.items())}
    (a.evidence/'archive-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(packaging_members=len(manifest),uncompressed_bytes=sum(e['bytes'] for e in manifest.values()))),flush=True)
    with tarfile.open(archive,'x:xz',preset=3) as tar:
        for name,path in sorted(members.items()):
            info=tarfile.TarInfo(name);info.size=path.stat().st_size;info.mode=0o644;info.mtime=0
            with path.open('rb') as stream:tar.addfile(info,stream)
    provenance=dict(archive=archive.name,bytes=archive.stat().st_size,sha256=digest(archive),members=len(manifest),
      source_and_data_bytes=sum(e['bytes'] for e in manifest.values()),artifacts=artifacts,
      head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    (a.evidence/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n');print(json.dumps(provenance,indent=2))
if __name__=='__main__':main()
