#!/usr/bin/env python3
"""Package selected compact fields and exact sources without deleting raw outputs."""
import argparse,hashlib,io,json,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('evidence',type=Path);a=p.parse_args()
    a.evidence.mkdir(parents=True,exist_ok=True)
    archive=a.evidence/'reference.tar.xz'
    if archive.exists():raise SystemExit('Refusing to overwrite '+str(archive))
    members={}
    def add(path,name):
        if not path.is_file():raise ValueError(path)
        members[name]=path
    for folder in sorted((a.directory/'runs').iterdir()):
        if not folder.is_dir():continue
        label=folder.name
        for name in ['record.json','input.prm','source.patch','run.log']:
            if (folder/name).exists():add(folder/name,'study/runs/'+label+'/'+name)
        if label in ['dg-cc','dg-fc-final','dg-ff','particle-cc','particle-fc','particle-ff']:
            for step in range(4):
                for prefix in ['common-stress','coupled-profile','surface-weak']:
                    name=f'{prefix}-{step}.csv';add(folder/'run'/name,'study/runs/'+label+'/run/'+name)
            for name in ['physical-fe-0.csv','initial-mesh.csv','coupled-checks.csv','accepted_steps.csv','initial-particle-identity.csv','mechanical-quadrature.csv','bulk-normal-cuts-3.csv']:
                add(folder/'run'/name,'study/runs/'+label+'/run/'+name)
            if label.startswith('dg'):
                for name in ['dg-support.csv','dg-coefficients-3.csv']:
                    add(folder/'run'/name,'study/runs/'+label+'/run/'+name)
    default=a.directory/'runs/particle-default-regression/run'
    for name in ['coupled-profile-0.csv','coupled-profile-1.csv','coupled-checks.csv']:
        add(default/name,'study/runs/particle-default-regression/run/'+name)
    for label,baseline in [('dg-cc','dg-q6'),('dg-ff','dg-fine'),('particle-default-regression','gate-particle-v3')]:
        add(ROOT/'build-particle-free/mature-dg/runs'/baseline/'record.json','retained/'+label+'-record.json')
    for f in (a.directory/'derived').iterdir():add(f,'study/derived/'+f.name)
    for folder in ['inputs','inputs-lf']:
        for f in (a.directory/folder).iterdir():add(f,'study/'+folder+'/'+f.name)
    for f in a.directory.glob('*.json'):add(f,'study/'+f.name)
    for f in a.directory.glob('*.log'):add(f,'study/'+f.name)
    for f in a.directory.glob('*.patch'):add(f,'study/'+f.name)
    files=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard',
        'source','include','CMakeLists.txt','cmake','benchmarks/reconstructed_fault/bp3_native',
        'tests/phase_field_fault_test_access.h'],cwd=ROOT,text=True).splitlines()
    for relative in sorted(set(files)):
        f=ROOT/relative
        if f.is_file():add(f,'source_snapshot/'+relative)
    for f in Path(__file__).parent.glob('*pf6_mature_resolution*.py'):
        add(f,'source_snapshot/'+str(f.resolve().relative_to(ROOT)))
    for name in ['analyze_pf6_mature.py','run_pf6_mature.py','export_pf6_mature_tractions.py']:
        add(Path(__file__).with_name(name),'source_snapshot/benchmarks/reconstructed_fault/particle_free/'+name)
    for name in ['PF6_MATURE_RESOLUTION_FORMULATION.md','PF6_MATURE_RESOLUTION_REPORT.md','SOURCE_AUDIT.md','STATUS.md']:
        add(ROOT/'doc/reconstructed_fault/particle_free'/name,'source_snapshot/doc/reconstructed_fault/particle_free/'+name)
    for relative in ['doc/reconstructed_fault/current_design.md','doc/reconstructed_fault/specification.tex','AGENTS.md']:
        add(ROOT/relative,'source_snapshot/'+relative)
    for relative in ['build-particle-free/native/aspect-release',
        'build-particle-free/mature-resolution/bp3-plugin/libbp3_restore_150x50.release.so',
        'build-particle-free/mature-resolution/observers/libbp3_coupled_checks.release.so']:
        add(ROOT/relative,'artifacts/'+Path(relative).name)
    manifest={name:dict(bytes=f.stat().st_size,sha256=sha(f.read_bytes())) for name,f in sorted(members.items())}
    (a.evidence/'archive-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with tarfile.open(archive,'x:xz',preset=4) as t:
        for name,f in sorted(members.items()):
            data=f.read_bytes();info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644;info.mtime=0
            t.addfile(info,io.BytesIO(data))
    provenance=dict(archive=archive.name,bytes=archive.stat().st_size,sha256=sha(archive.read_bytes()),members=len(manifest),
                    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    (a.evidence/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n');print(json.dumps(provenance))
if __name__=='__main__':main()
