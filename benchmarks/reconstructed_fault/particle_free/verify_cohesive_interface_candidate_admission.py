#!/usr/bin/env python3
"""Preservation and publication checks, with scientific failures kept explicit."""
import csv
import difflib
import json
import subprocess
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from cohesive_interface_candidate_admission import ROOT,WORK,OUT,open_state,counts,zero_tail,formed_count
from cohesive_interface_path import sha,save,table


def main():
    revised={str(Path('benchmarks/reconstructed_fault/particle_free')/name) for name in
        ['cohesive_interface_path.cc','cohesive_interface_evolution.h','cohesive_interface_turn_solver.cc','cohesive_interface_load_turn.py']}
    docs={'doc/reconstructed_fault/particle_free/STATUS.md','doc/reconstructed_fault/particle_free/SOURCE_AUDIT.md'}
    protected=json.loads((OUT/'entry-protection.json').read_text());unchanged=0
    for name,digest in protected.items():
        if name in revised|docs:continue
        assert sha(ROOT/name)==digest,name;unchanged+=1
    archive=zipfile.ZipFile(WORK/'entry-source.zip');diff=[]
    for name in sorted(revised):diff.extend(difflib.unified_diff(archive.read(name).decode().splitlines(keepends=True),
        (ROOT/name).read_text().splitlines(keepends=True),fromfile='task-entry/'+name,tofile=name))
    (OUT/'native-changes.diff').write_text(''.join(diff))
    for name in docs:assert (ROOT/name).read_bytes().endswith(archive.read(name)),name+' prior evidence changed'
    labels=['candidates2','candidates8','half-step','cadence-half','cadence-onset'];account=[];scientific=[]
    for label in labels:
        folder=WORK/label;initial=json.loads((folder/'initial.json').read_text());events=json.loads((folder/'events.json').read_text())
        k,s=open_state(folder/'final.json');rows=list(csv.DictReader((OUT/(label+'-history.csv')).open()))
        zero_tail(k,s)
        assert len(rows)==s['physical_publications']-initial['physical_publications']
        assert len(events)==s['publications']-initial['publications']
        assert sum(int(r['newly_formed_cells']) for r in rows)==s['formed_arcs']-initial['formed_arcs']
        assert max(float(r['weak_error_Pa']) for r in rows)<=1e-8
        assert min(float(r['minimum_normal_Pa']) for r in rows)>0
        assert max(abs(float(r['ledger_error_J_m']))/max(1,abs(float(r['endpoint_work_J_m']))) for r in rows)<1e-10
        for r in rows:
            assert float(r['metadata_commit_work_J_m'])==0 and float(r['metadata_commit_fracture_J_m'])==0
        for i,e in enumerate(events,1):
            before=json.loads((folder/f'before-formation-{i}.json').read_text());after=json.loads((folder/f'after-formation-{i}.json').read_text())
            n=before['formed_arcs'];q0,_=counts(before)
            assert after['geometry'][:n]==before['geometry'][:n]
            assert after['record_ids'][:q0]==before['record_ids']
            assert after['physical_publications']==before['physical_publications']+1
            assert after['formed_generation']==before['formed_generation']+1
            assert after['formed_arcs']==formed_count(k,after,n)
            for meta in after['arc_metadata'][n:after['formed_arcs']]:
                assert meta['selected_from_time_s']==before['clock_s'] and meta['committed_time_s']==after['clock_s']
            assert e['transfer']['stress_defect_Pa']<1e-7 and abs(e['transfer']['storage_defect_J_m'])<=1e-10*max(1,float(rows[0]['bulk_J_m']))
        result=json.loads((OUT/(label+'-summary.json')).read_text())
        account.append(dict(label=label,physical_steps=len(rows),physical_publication_increment=s['physical_publications']-initial['physical_publications'],
            registry_commits=len(events),cells_formed=s['formed_arcs']-initial['formed_arcs'],persistent_records=len(s['record_ids']),
            temporary_cells=len(s['geometry'])-s['formed_arcs'],retained_initial_registry_unchanged=s['geometry'][:initial['formed_arcs']]==initial['geometry'][:initial['formed_arcs']]))
        scientific.append(dict(label=label,absolute_physical_work_remainder_J_m=result['physical_error_J_m'],
            relative_to_sliding=result['physical_vs_sliding'],relative_to_fracture=result['physical_vs_fracture'],threshold=.01,
            sliding_passed=result['physical_vs_sliding']<.01,fracture_passed=result['physical_vs_fracture']<.01))
        k.close()
    table(OUT/'publication-accounting.csv',account);save(OUT/'physical-energy-gates.json',scientific)
    assert not scientific[0]['fracture_passed'] and scientific[2]['fracture_passed']
    identity=json.loads((OUT/'default-identity.json').read_text());assert all(identity.values())
    q=json.loads((OUT/'initial-qualification.json').read_text());assert q['coupled_FD_relative']<1e-6 and q['zero_material_commit_rejected']
    life=json.loads((OUT/'active-lifecycle.json').read_text());assert life['successful_retry_bitwise'] and life['formed_history_JSON_restart_bitwise']
    assert life['active_coupled_FD_relative']<1e-6
    onset=json.loads((OUT/'onset-cadence.json').read_text());assert onset['old_admission_clocks_s']==onset['new_admission_clocks_s']
    assert max(onset['physical_history_errors'].values())<1e-12
    complete=json.loads((OUT/'candidate-completion.json').read_text());assert complete['completion_retries']>0 and complete['remaining_wholly_zero_cells']>0
    overloaded=json.loads((OUT/'overloaded-joint-admission.json').read_text());assert overloaded['original_pure_tied_birth_still_rejected']
    assert overloaded['transfer']['intact_virtual_peak_Pa']>3000 and overloaded['local_admission']['fracture_J_m']>0
    independent=json.loads((OUT/'reserve-independence.json').read_text());assert independent['persistent_geometry_bitwise']
    assert independent['maximum_full_plane_strain_deviator_difference_Pa']<1e-7
    targets=list(csv.DictReader((OUT/'uncertainty.csv').open()));assert all(abs(float(r['growth_relative']))<.05 for r in targets)
    kernels=[]
    for p in WORK.glob('kernel-*/native.json'):
        m=json.loads(p.read_text());assert m['energy_relative']<1e-4 and m['gradient_FD_relative']<1e-6
        kernels.append(dict(kernel=p.parent.name,**m))
    table(OUT/'native-cost.csv',kernels)
    animation=json.loads((OUT/'animation.json').read_text())
    with Image.open(OUT/'formation.gif') as im:frames=im.n_frames;assert frames==len(animation['times_s'])
    for p in OUT.glob('*.png'):
        with Image.open(p) as im:im.verify()
    size=sum(p.stat().st_size for p in WORK.rglob('*') if p.is_file());assert size<1.2*2**30
    save(OUT/'verification.json',dict(engineering_qualification_passed=True,coarse_physical_energy_failure_preserved=True,
        earlier_point_traction_and_normal_failures_preserved=True,protected_files_unchanged=unchanged,
        publication_accounting=account,native_kernels=len(kernels),native_seconds=sum(m['seconds'] for m in kernels),
        work_bytes=size,gif_frames=frames,legacy_checkpoint_conversion=False,production_changed=False,
        scope='Serial fixed bulk mesh, homogeneous law, experimental native/JSON ownership; no quantitative mature propagation qualification'))
    files=[p for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json']
    files += [ROOT/name for name in revised]
    files += list((ROOT/'benchmarks/reconstructed_fault/particle_free').glob('*cohesive_interface_candidate_admission.py'))
    files += [ROOT/'doc/reconstructed_fault/particle_free'/name for name in
        ['COHESIVE_INTERFACE_CANDIDATE_ADMISSION_SELECTION.md','COHESIVE_INTERFACE_CANDIDATE_ADMISSION_FORMULATION.md','COHESIVE_INTERFACE_CANDIDATE_ADMISSION_REPORT.md','STATUS.md','SOURCE_AUDIT.md']]
    save(OUT/'manifest.json',dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        binaries={str(p.relative_to(ROOT)):sha(p) for p in [WORK/'build/cohesive_interface_path',WORK/'build/libcohesive_interface_turn_solver.so']},
        files={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(set(files))}))
    print(json.dumps(json.loads((OUT/'verification.json').read_text()),indent=2))


if __name__=='__main__':main()
