#!/usr/bin/env python3
"""Preserve task boundaries and package compact curved-interface evidence."""
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
import numpy as np
from PIL import Image

from cohesive_interface_turn import ROOT, WORK, OUT, Kernel, save

BENCH=ROOT/'benchmarks/reconstructed_fault/particle_free'
DOC=ROOT/'doc/reconstructed_fault/particle_free'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    assert Path.cwd()==ROOT
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip()
    assert branch=='pf-rsf-particle-free'
    protected=json.loads((OUT/'entry-protection.json').read_text())
    archives=json.loads((OUT/'entry-archives.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in protected.items())
    assert all(sha(OUT.parent/p)==h for p,h in archives.items())
    save(OUT/'preservation.json',dict(protected_files=len(protected),old_archives=len(archives),
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
    added=list(BENCH.glob('*cohesive_interface_turn*.py'))+[BENCH/'cohesive_interface_turn_solver.cc']
    added+=list(DOC.glob('COHESIVE_INTERFACE_TURN_*.md'))
    save(OUT/'changed-files.json',dict(revision=revision,modified_since_task_entry=modified,
          added_for_task=[str(p.relative_to(ROOT)) for p in sorted(added)],
          production_source_changes=False,evidence_files_additional=True))
    sources={ROOT/p for p in modified}|set(added)
    sources.update([ROOT/'source/reconstructed_fault/cohesive_interface.cc',
                    ROOT/'include/aspect/reconstructed_fault/cohesive_interface.h',
                    BENCH/'cohesive_interface_evolution.h'])
    queue=[p for p in sources if p.suffix=='.py']
    while queue:
        p=queue.pop();py_compile.compile(str(p),doraise=True)
        for node in ast.walk(ast.parse(p.read_text())):
            names=[a.name for a in node.names] if isinstance(node,ast.Import) else ([node.module] if isinstance(node,ast.ImportFrom) and node.module else [])
            for name in names:
                dep=BENCH/(name.replace('.','/')+'.py')
                if dep.is_file() and dep not in sources:sources.add(dep);queue.append(dep)
    diagnostics=OUT/'diagnostics';diagnostics.mkdir(exist_ok=True)
    for p in sorted(WORK.glob('*.log')):shutil.copy2(p,diagnostics/p.name)
    for p in sorted((WORK/'runs').glob('*/failure.json')):
        dest=diagnostics/p.parent.name;dest.mkdir(exist_ok=True)
        for name in ['failure.json','last-accepted.json','retry-from.json','initial.json']:
            q=p.parent/name
            if q.exists():shutil.copy2(q,dest/name)
    for name in ['predictor-failed.npz','interruption.json','entry-source.zip','entry-status.txt']:
        if (WORK/name).exists():shutil.copy2(WORK/name,diagnostics/name)
    checkpoints=OUT/'checkpoints';checkpoints.mkdir(exist_ok=True)
    for label in ['accepted-selection','half-p30','maturehalf128']:
        dest=checkpoints/label;dest.mkdir(exist_ok=True)
        for name in ['initial.json','final.json','restart-check.json']:
            shutil.copy2(WORK/'runs'/label/name,dest/name)
    shutil.copy2(WORK/'runs/selected-context-checkpoint.json',checkpoints/'selected-context-checkpoint.json')
    operators=OUT/'operators';operators.mkdir(exist_ok=True)
    for label in ['half-p30','maturehalf128','wide-fine-p60']:
        info=json.loads((OUT/(label+'-summary.json')).read_text());k=Kernel(WORK/info['kernel'].removesuffix('-q1'),True,12.,.0021)
        np.savez_compressed(operators/(label+'.npz'),K=k.K,Q=k.Q,Nm=k.Nm,Np=k.Np,E8=k.E,
            x_m=k.s['x_m'],y_m=k.s['y_m'],weights_m=k.w,tx=k.s['tx'],ty=k.s['ty'],
            Jx=k.Jx,Jy=k.Jy,full_Jx=k.full_Jx,full_Jy=k.full_Jy,full_Q=k.full_Q,
            prolongation=k.prolongation,field_transform=k.field_transform)
        k.close()
    raw={};kernel_seconds=0.;raw_bytes=0
    for folder in sorted(WORK.glob('kernel-*')):
        if not folder.is_dir():continue
        files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in folder.iterdir() if p.is_file()}
        raw_bytes+=sum(v['bytes'] for v in files.values());metadata=None
        if (folder/'native.json').exists():
            try:metadata=json.loads((folder/'native.json').read_text())
            except json.JSONDecodeError:pass
        if metadata:kernel_seconds+=metadata['seconds']
        raw[str(folder.relative_to(ROOT))]=dict(metadata=metadata,files=files)
    save(OUT/'raw-kernel-index.json',raw)
    summaries=[json.loads(p.read_text()) for p in OUT.glob('*-summary.json')]
    runs=[r for r in summaries if all(key in r for key in ['label','steps','seconds'])]
    save(OUT/'cost-summary.json',dict(completed_evolution_runs=len(runs),
        completed_evolution_seconds=sum(r['seconds'] for r in runs),
        complete_native_kernel_seconds=kernel_seconds,raw_native_bytes=raw_bytes,
        selected_owner_peak_RSS_MiB=json.loads((OUT/'selected-lifecycle.json').read_text())['peak_process_RSS_MiB'],
        native_peak_RSS_measured=False,sequential_timing_runs=True,BLAS_threads=1,
        note='Includes retained pilots and controls; excludes failed partial evolution, compilation, plotting and packaging. No server/full-event run.'))
    verification=json.loads((OUT/'verification.json').read_text());selection=verification['primary_selection']
    lifecycle=json.loads((OUT/'selected-lifecycle.json').read_text())
    assert selection['half_step_score_gap']>selection['twice_observed_combined_uncertainty']
    assert selection['width_changes_ranking'] and not selection['unique_geometry_independent_direction']
    assert lifecycle['selected_native_evolution_bitwise_to_preview'] and lifecycle['actual_JSON_restart_retry_bitwise']
    regression=json.loads((OUT/'default-regression.json').read_text())
    assert all(regression['rebuilt-smooth']['bitwise_files'].values())
    assert regression['legacy-default']['full_tensor_and_pressure_error_Pa']<1e-7
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
    for p in sorted(OUT.iterdir()):
        if p.suffix not in ['.png','.gif']:continue
        with Image.open(p) as im:
            images[p.name]=dict(width=im.width,height=im.height,frames=getattr(im,'n_frames',1));im.verify()
    assert images['selected-deflection.gif']['frames']==25
    save(OUT/'artifact-verification.json',dict(csv_rows=counts,images=images,checks_passed=True,
        python=sys.version,packages={p:importlib.metadata.version(p) for p in ['numpy','scipy','matplotlib','Pillow']}))
    subprocess.run(['git','diff','--check','--',str(BENCH/'CMakeLists.txt'),str(DOC/'STATUS.md'),str(DOC/'SOURCE_AUDIT.md')],check=True)
    files=sources|{p for p in OUT.rglob('*') if p.is_file() and not p.is_symlink()}
    files-={OUT/'manifest.json',OUT/'checksums.sha256'}
    save(OUT/'manifest.json',dict(revision=revision,branch=branch,
        purpose='Compact isolated native reference source/evidence snapshot. Requires the matching ASPECT worktree/build; not a standalone ASPECT distribution or production checkpoint.',
        files={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)}))
    files.add(OUT/'manifest.json')
    (OUT/'checksums.sha256').write_text(''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in sorted(files)))
    files.add(OUT/'checksums.sha256');archive=OUT.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(files):z.write(p,str(p.relative_to(ROOT)))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    archive.with_suffix('.zip.sha256').write_text(sha(archive)+'  '+archive.name+'\n')
    print(json.dumps(dict(archive=str(archive.relative_to(ROOT)),bytes=archive.stat().st_size,
        files=len(files),production_protected=len(protected),old_archives=len(archives),
        primary_angle=selection['angle_deg'],geometry_independent_angle=False),indent=2))


if __name__=='__main__':main()
