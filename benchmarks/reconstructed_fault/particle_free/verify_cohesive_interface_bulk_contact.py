#!/usr/bin/env python3
"""Verify preservation, measured gates and compact artifacts without another ZIP."""
import csv
import difflib
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import py_compile
import subprocess
import zipfile
from PIL import Image
from investigate_cohesive_interface_bulk_contact import ROOT, WORK, OUT, PATH_WORK, save

BENCH=ROOT/'benchmarks/reconstructed_fault/particle_free'
DOC=ROOT/'doc/reconstructed_fault/particle_free'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def main():
    branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip();assert branch=='pf-rsf-particle-free'
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    protected=json.loads((OUT/'entry-protection.json').read_text())
    allowed='benchmarks/reconstructed_fault/particle_free/cohesive_interface_path.cc'
    bad=[name for name,h in protected.items() if name!=allowed and digest(ROOT/name)!=h];assert not bad,bad
    old=OUT.parent/'cohesive_interface_path';manifest=json.loads((old/'manifest.json').read_text())['files'];count=0;states=0
    for name,entry in manifest.items():
        if not name.startswith(str(old.relative_to(ROOT))+'/'):continue
        assert digest(ROOT/name)==entry['sha256'],name;count+=1
        prefix=str((old/'checkpoints').relative_to(ROOT))+'/'
        if name.startswith(prefix):
            equivalent=PATH_WORK/name.removeprefix(prefix)
            if equivalent.is_file():assert digest(equivalent)==entry['sha256'];states+=1
    save(OUT/'preservation.json',dict(branch=branch,revision=revision,unchanged_protected_files=len(protected)-1,
        changed_protected_files=[],explicitly_revised_benchmark=allowed,old_compact_artifacts_verified=count,
        original_raw_states_verified=states,original_source_checkpoint_sha256=digest(PATH_WORK/'work-temporal-incoming.json'),
        production_BP3_unchanged=True,cleanup_storage_mapping='benchmarks/reconstructed_fault/maintenance/disk_cleanup_2026_10_27/actions.json'))
    modified=[];patch=[]
    with zipfile.ZipFile(WORK/'entry-source.zip') as archive:
        for name in archive.namelist():
            previous=archive.read(name).decode();current=(ROOT/name).read_text()
            if name.endswith(('STATUS.md','SOURCE_AUDIT.md')):assert current.endswith(previous)
            if previous!=current:
                modified.append(name);patch.extend(difflib.unified_diff(previous.splitlines(True),current.splitlines(True),fromfile='task-entry/'+name,tofile='current/'+name))
    (OUT/'source-entry.patch').write_text(''.join(patch))
    scripts=['investigate_cohesive_interface_bulk_contact.py','assess_cohesive_interface_bulk_contact.py',
        'evolve_cohesive_interface_bulk_contact.py','repeat_cohesive_interface_bulk_contact.py','fresh_cohesive_interface_bulk_contact.py',
        'analyze_cohesive_interface_bulk_evolution.py','plot_cohesive_interface_bulk_contact.py','check_cohesive_interface_lifting.py',
        'verify_cohesive_interface_bulk_contact.py']
    for name in scripts:py_compile.compile(str(BENCH/name),doraise=True)
    added=[str((BENCH/name).relative_to(ROOT)) for name in scripts]+[str(p.relative_to(ROOT)) for p in DOC.glob('COHESIVE_INTERFACE_BULK_CONTACT_*.md')]
    save(OUT/'changed-files.json',dict(modified_since_task_entry=modified,added_scientific_files=added,
        maintenance='benchmarks/reconstructed_fault/maintenance/disk_cleanup_2026_10_27',production_changes=False))
    raw={};seconds=0.;raw_bytes=0
    for folder in sorted(WORK.iterdir()):
        if not folder.is_dir() or not (folder/'native.json').is_file():continue
        meta=json.loads((folder/'native.json').read_text());seconds+=meta['seconds']
        files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in folder.iterdir() if p.is_file()}
        raw_bytes+=sum(v['bytes'] for v in files.values());raw[str(folder.relative_to(ROOT))]=dict(metadata=meta,files=files)
    save(OUT/'raw-index.json',raw)
    states_index={}
    for folder in sorted(WORK.iterdir()):
        if not folder.is_dir() or not (folder/'final.json').is_file():continue
        for p in folder.glob('*.json'):
            states_index[str(p.relative_to(ROOT))]=dict(bytes=p.stat().st_size,sha256=digest(p))
    save(OUT/'state-index.json',states_index)
    short=[json.loads(p.read_text()) for p in OUT.glob('short-*-evolution.json')]
    own=[json.loads(p.read_text()) for p in OUT.glob('own-time-*-evolution.json')]
    fresh=[json.loads((OUT/(label+'-summary.json')).read_text()) for label in ['fresh-wide64','fresh-wide96','fresh-fitted64','fresh-fitted96']]
    causal=[json.loads((OUT/(label+'-summary.json')).read_text()) for label in ['causal-wide64','causal-wide96']]
    save(OUT/'cost-summary.json',dict(unique_native_kernels=len(raw),unique_native_seconds=seconds,raw_kernel_bytes=raw_bytes,
        short_control_seconds=sum(d['seconds'] for d in short),fresh_fixed_seconds=sum(d['seconds'] for d in fresh),
        own_accepted_time_control_seconds=sum(d['seconds'] for d in own),
        causal_controller_seconds=[d['seconds'] for d in causal],causal_owner_peak_RSS_MiB=max(d['peak_RSS_MiB'] for d in causal),
        native_peak_RSS_measured=False,sequential_timing_runs=True,BLAS_threads=1,
        timing_note='Causal controller clocks include on-demand kernels but exclude initial kernel. Unique native and controller times overlap; do not add. Existing kernels are reused, so these are not cold-start speedup measurements.'))
    assessment=json.loads((OUT/'evolution-assessment.json').read_text())
    assert assessment['all_native_rejection_retry_restart'] and assessment['all_once_only'] and assessment['maximum_FD_relative']<1e-6
    assert all(d['relative_fracture']<.01 and d['relative_sliding']<.01 for d in short)
    assert len(own)==2 and all(d['initial']['same_basis_accepted_restart'] and d['publication_increment']==d['steps'] and d['checks']['native_rejection_retry_restart_bitwise'] and d['checks']['FD_relative']<1e-6 and d['relative_fracture']<.01 for d in own)
    assert all(d['physical_publications']==1780 and d['geometry_publications']==4 for d in causal)
    assert all(all(e['transfer']['old_stress_pressure_columns_bitwise'] and e['transfer']['old_histories_bitwise'] and e['transfer']['physical_tensor_pressure_error_Pa']==0 and e['transfer']['energy_change_J_m']==0 for e in d['events']) for d in causal)
    with (OUT/'fresh-errors.csv').open() as f:
        assert all(float(r['damage_relative'])<.05 and float(r['full_front_difference_m'])<.09375 and float(r['process_front_difference_m'])<.09375 for r in csv.DictReader(f))
    frozen=json.loads((OUT/'frozen-assessment.json').read_text())['errors']
    assert any(e['label']=='wide96' and not e['normal_target_pass'] for e in frozen), 'Keep the ahead-region failure visible'
    counts={};images={}
    for p in OUT.glob('*.csv'):
        with p.open() as f:rows=list(csv.DictReader(f))
        for r in rows:
            assert None not in r
            for v in r.values():
                assert v is not None
                try:value=float(v)
                except ValueError:continue
                assert math.isfinite(value),(p,v)
        counts[p.name]=len(rows)
    for p in OUT.glob('*.png'):
        with Image.open(p) as im:images[p.name]=dict(width=im.width,height=im.height);im.verify()
    assert len(images)==6
    subprocess.run(['git','diff','--check','--',str(DOC/'STATUS.md'),str(DOC/'SOURCE_AUDIT.md')],check=True)
    save(OUT/'verification.json',dict(checks_passed=True,failed_scientific_targets_retained=True,csv_rows=counts,images=images,
        native_executable_sha256=digest(WORK/'build/cohesive_interface_path'),native_C_API_sha256=digest(PATH_WORK/'build/libcohesive_interface_turn_solver.so'),
        packages={p:importlib.metadata.version(p) for p in ['numpy','scipy','matplotlib','Pillow']}))
    files={p for p in OUT.rglob('*') if p.is_file()}|{ROOT/p for p in modified+added}
    maintenance=ROOT/'benchmarks/reconstructed_fault/maintenance/disk_cleanup_2026_10_27'
    files.update(p for p in maintenance.iterdir() if p.is_file())
    files.add(WORK/'entry-source.zip')
    files-={OUT/'manifest.json',OUT/'checksums.sha256'}
    save(OUT/'manifest.json',dict(revision=revision,branch=branch,files={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(files)}))
    files.add(OUT/'manifest.json')
    (OUT/'checksums.sha256').write_text(''.join(digest(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in sorted(files)))
    print(json.dumps(dict(compact_files=len(files),csv_tables=len(counts),PNGs=len(images),protected=len(protected)-1,old_states=states)))


if __name__=='__main__':main()
