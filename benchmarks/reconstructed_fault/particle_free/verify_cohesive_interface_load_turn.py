#!/usr/bin/env python3
"""Preservation, publication accounting, retained failures and evidence index."""
import csv
import difflib
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
from PIL import Image
import numpy as np
from cohesive_interface_load_turn import ROOT, WORK, OUT, save, table, sha


def main():
    revised={str(Path('benchmarks/reconstructed_fault/particle_free')/n) for n in
        ['cohesive_interface_path.cc','cohesive_interface_evolution.h','cohesive_interface_turn_solver.cc']}
    protected=json.loads((OUT/'entry-protection.json').read_text());unchanged=[]
    for name,digest in protected.items():
        if name in revised:continue
        assert sha(ROOT/name)==digest,name
        unchanged.append(name)
    entry=zipfile.ZipFile(WORK/'entry-source.zip');patch=[]
    for name in sorted(revised):
        before=entry.read(name).decode().splitlines(keepends=True);after=(ROOT/name).read_text().splitlines(keepends=True)
        patch.extend(difflib.unified_diff(before,after,fromfile='task-entry/'+name,tofile=name))
    (OUT/'native-changes.diff').write_text(''.join(patch))
    for name in ['STATUS.md','SOURCE_AUDIT.md']:
        path='doc/reconstructed_fault/particle_free/'+name
        assert (ROOT/path).read_bytes().endswith(entry.read(path)),name+' historical content changed'
    previous=json.loads((OUT.parent/'cohesive_interface_bulk_contact/manifest.json').read_text());old_count=0
    for name,meta in previous['files'].items():
        if '/evidence/cohesive_interface_bulk_contact/' in name:
            assert sha(ROOT/name)==meta['sha256'],name
            old_count+=1
    labels=['turn-small64-reproduced','turn-large64','constant-small64','constant-large64','turn-small96','turn-small-half64','turn-small-quantum64']
    counts=[]
    for label in labels:
        folder=WORK/label;initial=json.loads((folder/'initial.json').read_text());final=json.loads((folder/'last-accepted.json').read_text())
        path=OUT/(label+'-history.csv')
        if not path.exists():path=OUT/(label+'-partial.csv')
        rows=list(csv.DictReader(path.open()));events=json.loads((folder/'events.json').read_text())
        assert len(rows)==final['physical_publications']-initial['physical_publications']
        assert len(events)==final['publications']-initial['publications']
        assert max(float(r['weak_error_Pa']) for r in rows)<=1e-8
        assert max(abs(float(r['ledger_error_J_m']))/max(1,abs(float(r['endpoint_work_J_m']))) for r in rows)<1e-10
        assert min(float(r['minimum_normal_Pa']) for r in rows)>0
        for e in events:
            t=e['transfer'];assert t['tensor_error_Pa']<1e-7 and abs(t['energy_change_J_m'])<1e-10
            assert t['new_fracture_J_m']==0 and t['new_bond_J_m']==0 and t['old_histories_bitwise']
        counts.append(dict(label=label,physical_steps=len(rows),physical_publication_increment=final['physical_publications']-initial['physical_publications'],
            new_geometry_commits=len(events),final_s=final['clock_s'],failed=(folder/'failure.json').exists()))
    table(OUT/'publication-accounting.csv',counts)
    assert json.loads((WORK/'turn-small64/last-accepted.json').read_text())==json.loads((WORK/'turn-small64-reproduced/last-accepted.json').read_text())
    rejected=list(WORK.glob('kernel-*/rejected-birth.json'));assert len(rejected)>=4
    for p in rejected:
        r=json.loads(p.read_text());assert r['required_multiplier_Pa']>3000
    native=[]
    for p in WORK.glob('kernel-*/native.json'):
        m=json.loads(p.read_text());assert m['energy_relative']<1e-4 and m['gradient_FD_relative']<1e-6
        native.append(dict(kernel=p.parent.name,**m))
    table(OUT/'native-cost.csv',native)
    images={}
    for p in OUT.glob('*.png'):
        with Image.open(p) as im:im.verify()
        images[p.name]=p.stat().st_size
    animation=json.loads((OUT/'animation-verification.json').read_text())
    with Image.open(OUT/'evolution.gif') as im:frames=im.n_frames;assert frames==len(animation['times_s']) and frames>1
    assert len(images)>=5
    size=sum(p.stat().st_size for p in WORK.rglob('*') if p.is_file());assert size<2*2**30
    save(OUT/'verification.json',dict(engineering_checks_passed=True,scientific_comparison_failures_retained=True,
        protected_files_unchanged=len(unchanged),prior_bulk_contact_artifacts_unchanged=old_count,
        reproduced_failed_last_state_bitwise=True,rejected_tied_birth_archives=len(rejected),
        experimental_native_source_changes=sorted(revised),png_bytes=images,gif_frames=frames,
        work_bytes=size,kernel_count=len(native),scope='serial experimental native two-affine causal evolution; no production qualification'))
    files=[p for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json']
    files += [ROOT/name for name in revised]
    files += list((ROOT/'benchmarks/reconstructed_fault/particle_free').glob('*cohesive_interface_load_turn.py'))
    files += [ROOT/'doc/reconstructed_fault/particle_free'/name for name in
        ['COHESIVE_INTERFACE_LOAD_TURN_SELECTION.md','COHESIVE_INTERFACE_LOAD_TURN_REPORT.md','STATUS.md','SOURCE_AUDIT.md']]
    save(OUT/'manifest.json',dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        files={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(set(files))}))
    print(json.dumps(json.loads((OUT/'verification.json').read_text()),indent=2))


if __name__=='__main__':main()
