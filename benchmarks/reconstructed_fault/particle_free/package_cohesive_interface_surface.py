#!/usr/bin/env python3
"""Verify/preserve the bounded surface study and create its compact archive."""
import ast
import csv
import difflib
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import py_compile
import shutil
import subprocess
import sys
import zipfile
from PIL import Image

ROOT=Path(__file__).resolve().parents[3]
BENCH=ROOT/'benchmarks/reconstructed_fault/particle_free'
OUT=BENCH/'evidence/cohesive_interface_surface'
WORK=ROOT/'build-particle-free/cohesive-interface-surface'
DOC=ROOT/'doc/reconstructed_fault/particle_free'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def main():
    assert Path.cwd()==ROOT
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip()
    assert branch=='pf-rsf-particle-free'
    protected=json.loads((OUT/'entry-protection.json').read_text())
    archives=json.loads((OUT/'entry-archives.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in protected.items())
    assert all(sha(OUT.parent/p)==h for p,h in archives.items())
    write('preservation.json',dict(protected_files=len(protected),old_archives=len(archives),
          changed_protected_files=[],changed_archives=[],production_BP3_unchanged=True))
    modified=[];patch=[]
    with zipfile.ZipFile(WORK/'entry-source.zip') as z:
        for name in z.namelist():
            old=z.read(name).decode();new=(ROOT/name).read_text()
            if name.endswith(('STATUS.md','SOURCE_AUDIT.md')):assert new.endswith(old)
            if old!=new:
                modified.append(name)
                patch.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='task-entry/'+name,tofile='current/'+name))
    (OUT/'source-entry.patch').write_text(''.join(patch))
    added=list(BENCH.glob('*cohesive_interface_surface.py'))+list(DOC.glob('COHESIVE_INTERFACE_SURFACE_*.md'))
    write('changed-files.json',dict(revision=revision,modified_since_task_entry=modified,
          added_for_task=[str(p.relative_to(ROOT)) for p in sorted(added)],
          native_or_production_source_changes=False,evidence_files_additional=True))
    sources={ROOT/p for p in modified}|set(added)
    sources.update([ROOT/'source/reconstructed_fault/cohesive_interface.cc',
                    ROOT/'include/aspect/reconstructed_fault/cohesive_interface.h'])
    queue=[p for p in sources if p.suffix=='.py']
    while queue:
        p=queue.pop();py_compile.compile(str(p),doraise=True)
        for node in ast.walk(ast.parse(p.read_text())):
            names=[a.name for a in node.names] if isinstance(node,ast.Import) else ([node.module] if isinstance(node,ast.ImportFrom) and node.module else [])
            for name in names:
                for directory in [BENCH,BENCH/'bulk_cohesive']:
                    dep=directory/(name.replace('.','/')+'.py')
                    if dep.is_file() and dep not in sources:sources.add(dep);queue.append(dep)
    diagnostics=OUT/'diagnostics';diagnostics.mkdir(exist_ok=True)
    for p in sorted(WORK.glob('*.log')):shutil.copy2(p,diagnostics/p.name)
    for name in ['surface-512/failure.json','surface-512/resume-info.json']:
        shutil.copy2(WORK/name,diagnostics/('surface-512-'+Path(name).name))
    # Useful, small reference checkpoints; no native archive is duplicated.
    for case in ['surface-256','surface-512','surface-512-half']:
        for clock in [2423750,2550000,2580000]:
            p=WORK/case/f'state-{clock}.npz'
            if p.exists():
                dest=OUT/'checkpoints'/case;dest.mkdir(parents=True,exist_ok=True)
                shutil.copy2(p,dest/p.name)
    results=json.loads((OUT/'verification.json').read_text())
    assert all(results['gates'].values())
    assert json.loads((OUT/'qualification.json').read_text())['all_passed']
    lifecycle=json.loads((OUT/'lifecycle-verification.json').read_text())
    assert all(r['all_state_arrays_bitwise'] and r['counter_increment']==1 for r in lifecycle['serialization'])
    counts={}
    for p in OUT.glob('*.csv'):
        with p.open() as f:data=list(csv.DictReader(f))
        for r in data:
            assert None not in r
            for value in r.values():
                assert value is not None
                try:v=float(value)
                except ValueError:continue
                assert math.isfinite(v),(p,value)
        counts[p.name]=len(data)
    images={}
    for p in sorted(OUT.glob('*')):
        if p.suffix not in ['.png','.gif']:continue
        with Image.open(p) as im:
            images[p.name]=dict(width=im.width,height=im.height,frames=getattr(im,'n_frames',1));im.verify()
    assert images['surface-evolution.gif']['frames']>=20
    write('artifact-verification.json',dict(csv_rows=counts,images=images,checks_passed=True,
          python=sys.version,packages={p:importlib.metadata.version(p) for p in ['numpy','scipy','mpmath','matplotlib','Pillow']}))
    runs=[json.loads(p.read_text()) for p in sorted(OUT.glob('*-run.json'))]
    write('cost-summary.json',dict(evolution_seconds=sum(r['seconds'] for r in runs),
          evolution_peak_RSS_MiB=max(r['peak_RSS_MiB'] for r in runs),
          sequential=True,BLAS_threads=1,native_runs_launched=0,
          note='Includes preserved fine predictor stop/completion. Excludes native archive reading/qualification, plotting and packaging.'))
    write('operator-identities.json',{p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in WORK.glob('operators-*.npz')})
    files=sources|{p for p in OUT.rglob('*') if p.is_file() and not p.is_symlink()}
    files-={OUT/'manifest.json',OUT/'checksums.sha256'}
    write('manifest.json',dict(revision=revision,branch=branch,
          purpose='Compact independent surface-reference evidence/source snapshot; not native ASPECT restart data.',
          files={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)}))
    files.add(OUT/'manifest.json')
    (OUT/'checksums.sha256').write_text(''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in sorted(files)))
    files.add(OUT/'checksums.sha256')
    archive=OUT.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(files):z.write(p,str(p.relative_to(ROOT)))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    archive.with_suffix('.zip.sha256').write_text(sha(archive)+'  '+archive.name+'\n')
    print(json.dumps(dict(archive=str(archive.relative_to(ROOT)),bytes=archive.stat().st_size,
          files=len(files),gates=results['gates'],protected_files=len(protected),old_archives=len(archives)),indent=2))


if __name__=='__main__':main()
