#!/usr/bin/env python3
"""Preserve completed native solver output separately from compact figures/CSV."""
import hashlib,json,os,shutil,subprocess,tarfile
from pathlib import Path
from fracture_analysis import ROOT,BUILD,OUT

def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def archive(path,files):
    assert not path.exists(),path
    names={str(p.relative_to(ROOT)):p for p in files}
    rows={n:dict(bytes=p.stat().st_size,sha256=digest(p)) for n,p in names.items()}
    with path.with_suffix(path.suffix+'.partial').open('xb') as output:
        compressor=subprocess.Popen(['zstd','-q','-T2','-6','-c'],stdin=subprocess.PIPE,stdout=output)
        with tarfile.open(fileobj=compressor.stdin,mode='w|',format=tarfile.PAX_FORMAT,dereference=True) as bundle:
            for name,p in names.items():bundle.add(p,arcname=name,recursive=False)
        compressor.stdin.close();assert compressor.wait()==0
    path.with_suffix(path.suffix+'.partial').replace(path)
    reader=subprocess.Popen(['zstd','-q','-dc',str(path)],stdout=subprocess.PIPE);seen=set()
    with tarfile.open(fileobj=reader.stdout,mode='r|') as bundle:
        for member in bundle:
            assert member.isfile() and member.name in rows and member.name not in seen
            with bundle.extractfile(member) as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==rows[member.name]['sha256']
            seen.add(member.name)
    while reader.stdout.read(2**20):pass
    reader.stdout.close();assert reader.wait()==0 and seen==set(rows)
    result=dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=digest(path),files=rows)
    path.with_name(path.name+'.json').write_text(json.dumps(result,indent=2)+'\n')
    return {k:v for k,v in result.items() if k!='files'}|dict(file_count=len(rows),original_bytes=sum(r['bytes'] for r in rows.values()))

def main():
    status=ROOT/'benchmarks/reconstructed_fault/refactoring_boundary/evidence/bulk-fracture-m4.json'
    assert status.exists(),'Native bounded run has not completed.'
    run=json.loads(status.read_text())
    logdir=OUT/'logged';logdir.mkdir(exist_ok=True)
    for p in status.parent.glob('bulk-fracture-*'):
        if p.suffix in {'.json','.log'}:shutil.copy2(p,logdir/p.name)
    # Full numeric output and native restart stay in BUILD, not the deliverable.
    folder=BUILD/'archives';folder.mkdir(exist_ok=True)
    numeric=[]
    for relative in ['runs/candidate','runs/startup-m3-rejected','startup-m3']:
        numeric.extend(p for p in (BUILD/relative).rglob('*') if p.is_file() and not p.is_symlink())
    numeric+=list(BUILD.glob('*.log'))+list(BUILD.glob('strip-softening-state.npz'))
    native=archive(folder/'native-solver.tar.zst',sorted(set(numeric)))
    fixture=ROOT/'benchmarks/reconstructed_fault/particle_free/bulk_cohesive'
    sources=[*fixture.glob('fracture*'),fixture/'propagation.cc',fixture/'parallel.cc',fixture/'strip_oracle.cc',fixture/'CMakeLists.txt',
        fixture/'native.prm',fixture/'parallel.prm',fixture/'mixed_boundary.prm',
        ROOT/'benchmarks/reconstructed_fault/particle_free/CMakeLists.txt',
        ROOT/'source/material_model/phase_field_fault/bulk_cohesive.cc',
        ROOT/'source/material_model/phase_field_fault/bulk_cohesive_history.cc',
        ROOT/'source/simulator/solver/bulk_cohesive_phase.cc',
        *list((ROOT/'include/aspect/material_model/phase_field_fault').glob('bulk_cohesive*.h')),
        ROOT/'doc/reconstructed_fault/particle_free/BULK_FRACTURE_REPORT.md',
        ROOT/'doc/reconstructed_fault/particle_free/BULK_FRACTURE_SELECTION.md',
        ROOT/'doc/reconstructed_fault/particle_free/SOURCE_AUDIT.md',ROOT/'doc/reconstructed_fault/particle_free/STATUS.md']
    source=archive(folder/'source-overlay.tar.zst',sorted(set(p for p in sources if p.is_file())))
    result=dict(native=native,source=source,run=run,root_free_bytes=os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize,
                excluded_from_deliverable='Full native output/checkpoints, frozen failed startup and source overlay remain under build-particle-free/bulk-fracture/archives; no raw outputs deleted.')
    (OUT/'archives.json').write_text(json.dumps(result,indent=2)+'\n')
    manifest={str(p.relative_to(OUT)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in OUT.rglob('*') if p.is_file() and p.name not in {'manifest.json','SHA256SUMS'}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (OUT/'SHA256SUMS').write_text(''.join(f'{r["sha256"]}  {p}\n' for p,r in sorted(manifest.items())))
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
