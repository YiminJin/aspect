#!/usr/bin/env python3
"""One compact evidence archive; full native records remain separately archived."""
from pathlib import Path
import os
import argparse,hashlib,json,re,shutil,subprocess,tarfile,zipfile
ROOT=Path(os.environ.get('ASPECT_P4_ROOT',Path(__file__).resolve().parents[4])).resolve();BASE=ROOT/'build-particle-free/bulk-continuation';OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/bulk_continuation'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def fixtures():
    pending=list(BASE.glob('*.prm'));seen=set()
    while pending:
        p=pending.pop()
        if p in seen:continue
        seen.add(p)
        for line in p.read_text().splitlines():
            match=re.match(r'\s*include \$ASPECT_SOURCE_DIR/(.*)',line)
            if match:pending.append(ROOT/match[1].strip())
    return sorted(seen)

def full():
    destination=BASE/'archives';destination.mkdir(exist_ok=True);archive=destination/'native-controls.tar.zst';assert not archive.exists()
    files=[]
    for p in (BASE/'runs').rglob('*'):
        if p.is_file() and not any('unrun' in part or 'preparation-only' in part for part in p.parts):files.append(p)
    for pattern in ['*.prm','*-input.json','native*/identity.json','native*/build-records.json','native*/bulk_cohesive_phase.cc','plugin*/identity.json','plugin*/*.cc','plugin*-build.json','plugin*.log']:
        files.extend(BASE.glob(pattern))
    files.extend([BASE/'native-v6/aspect-release',BASE/'plugin/libbulk_propagation_checks.release.so',BASE/'plugin-boundary/libbulk_propagation_checks.release.so'])
    files.extend(BASE.glob('build_native*.py'))
    files.extend(fixtures())
    files.append(ROOT/'build-particle-free/bulk-fracture/strip/libbulk_strip_oracle.so')
    original=ROOT/'build-particle-free/bulk-fracture/runs/candidate'
    files.extend(p for p in original.iterdir() if p.is_file() and p.suffix in ['.csv','.prm','.json'])
    files.extend((BASE/'stability').glob('*.npz'))
    source=ROOT/'build-particle-free/bulk-fracture/runs/candidate/restart/02'
    files.extend(p for p in source.iterdir() if p.is_file())
    files=sorted(set(files));listing=destination/'native-controls-files.txt';listing.write_text(''.join(str(p.relative_to(ROOT))+'\n' for p in files))
    with archive.open('wb') as stream:
        result=subprocess.run(['tar','--zstd','-cf','-','-T',str(listing)],cwd=ROOT,stdout=stream,stderr=subprocess.PIPE)
    assert result.returncode==0,result.stderr.decode()
    manifest=dict(path=str(archive.relative_to(ROOT)),bytes=archive.stat().st_size,sha256=sha(archive),files={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files})
    (destination/'native-controls.json').write_text(json.dumps(manifest,indent=2)+'\n');(OUT/'full-archive.json').write_text(json.dumps({k:v for k,v in manifest.items() if k!='files'},indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='files'},indent=2))

def compact():
    for file in ['BULK_CONTINUATION_REPORT.md','BULK_CONTINUATION_SELECTION.md','P5_DIRECTION_SPECIFICATION.md']:
        text=(ROOT/'doc/reconstructed_fault/particle_free'/file).read_text();text=text.replace('../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_continuation/','').replace('](SOURCE_AUDIT.md)','](SOURCE_AUDIT_P4.md)');(OUT/file).write_text(text)
    audit=(ROOT/'doc/reconstructed_fault/particle_free/SOURCE_AUDIT.md').read_text()
    marker='## P4 continuation/stability diagnostic'
    (OUT/'SOURCE_AUDIT_P4.md').write_text(audit[audit.index(marker):])
    inputs=OUT/'inputs';inputs.mkdir(exist_ok=True)
    for p in list(BASE.glob('*.prm'))+list(BASE.glob('*-input.json')):
        if 'unrun' not in p.name and 'preparation-only' not in p.name:shutil.copy2(p,inputs/p.name)
    for p in fixtures():
        if p.parent==BASE:continue
        target=inputs/'repository'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    for name in ['seed-prescription.csv','fracture-parameters.csv']:
        shutil.copy2(ROOT/'build-particle-free/bulk-fracture/runs/candidate'/name,inputs/name)
    logs=OUT/'logged';logs.mkdir(exist_ok=True)
    for p in (ROOT/'benchmarks/reconstructed_fault/refactoring_boundary/evidence').glob('bulk-continuation-*'):
        if p.suffix in ['.json','.log']:shutil.copy2(p,logs/p.name)
    for run in (BASE/'runs').iterdir():
        if not run.is_dir():continue
        diagnostic=list(run.glob('bulk-*tangent*.csv'))+list(run.glob('continuation-state.json'))
        if diagnostic:
            target=OUT/'diagnostics'/run.name;target.mkdir(parents=True,exist_ok=True)
            for p in diagnostic:shutil.copy2(p,target/p.name)
    sources=OUT/'reproduction';sources.mkdir(exist_ok=True)
    for p in (ROOT/'benchmarks/reconstructed_fault/particle_free/bulk_cohesive').glob('continuation_*.py'):shutil.copy2(p,sources/p.name)
    for name in ['fracture_analysis.py','fracture_strip.py']:
        shutil.copy2(ROOT/'benchmarks/reconstructed_fault/particle_free/bulk_cohesive'/name,sources/name)
    overlay=OUT/'source-overlay'
    for name in ['source/simulator/solver/bulk_cohesive_phase.cc','benchmarks/reconstructed_fault/particle_free/bulk_cohesive/parallel.cc','benchmarks/reconstructed_fault/particle_free/bulk_cohesive/propagation.cc']:
        target=overlay/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    pinned=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/bulk_fracture/entry.json';protected=json.load(open(pinned))['protected_bp3']
    assert all(sha(ROOT/p)==v['sha256'] for p,v in protected.items())
    (OUT/'bp3-preservation.json').write_text(json.dumps(dict(files=len(protected),all_unchanged=True,reference=str(pinned.relative_to(ROOT))),indent=2)+'\n')
    payload=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name not in ['manifest.json','SHA256SUMS'])
    manifest={str(p.relative_to(OUT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in payload}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    checks=sorted(payload+[OUT/'manifest.json']);(OUT/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(OUT))+'\n' for p in checks))
    archive=OUT.parent/'bulk_continuation.zip';assert not archive.exists()
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(OUT))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,entry in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==entry['sha256']
    (archive.with_suffix('.zip.sha256')).write_text(sha(archive)+'  '+archive.name+'\n')
    print(json.dumps(dict(archive=str(archive.relative_to(ROOT)),files=len(manifest),bytes=archive.stat().st_size,sha256=sha(archive)),indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['full','compact']);a=p.parse_args();full() if a.mode=='full' else compact()
