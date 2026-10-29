#!/usr/bin/env python3
"""Preservation, scientific screens and once-only publication accounting."""
import csv
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image

from cohesive_interface_formed_turn import ROOT,WORK,OUT,SOURCE,SPACING,configure
import cohesive_interface_candidate_admission as owner
from cohesive_interface_path import save,table,sha


def main():
    configure('coarse');protected=json.loads((OUT/'entry-protection.json').read_text())
    docs=['doc/reconstructed_fault/particle_free/STATUS.md','doc/reconstructed_fault/particle_free/SOURCE_AUDIT.md']
    for path,digest in protected.items():
        if path not in docs:assert sha(ROOT/path)==digest,path
    with zipfile.ZipFile(WORK/'entry-docs.zip') as z:
        for path in docs:assert (ROOT/path).read_bytes().endswith(z.read(path)),path
    initial=json.loads(SOURCE.read_text());qbase,_=owner.counts(initial);accounts=[];gates=[]
    for label in ['coarse','fine','temporal-half']:
        run_initial=initial if label!='temporal-half' else json.loads((WORK/'temporal-prefix.json').read_text())
        expected_steps=120 if label!='temporal-half' else 40;dt=.000125 if label!='temporal-half' else .0000625
        qbase,_=owner.counts(run_initial)
        folder=WORK/label;ik,s=owner.open_state(folder/'initial.json');assert s==run_initial
        k,t=owner.open_state(folder/'final.json');owner.zero_tail(k,t)
        h=np.genfromtxt(OUT/(label+'-history.csv'),delimiter=',',names=True)
        assert len(h)==expected_steps and h['time_s'][-1]==.440 and max(abs(h['dt_s']-dt))<1e-12
        assert min(h['minimum_normal_Pa'])>0 and min(h['raw_minimum_normal_Pa'])>0
        assert max(h['weak_error_Pa'])<=1e-8
        assert max(abs(h['ledger_error_J_m'])/np.maximum(1,abs(h['endpoint_work_J_m'])))<1e-10
        assert np.all(h['metadata_commit_work_J_m']==0) and np.all(h['metadata_commit_fracture_J_m']==0)
        assert max(h['transfer_stress_Pa'])<1e-7
        assert t['physical_publications']-run_initial['physical_publications']==len(h)
        assert t['geometry'][:run_initial['formed_arcs']]==run_initial['geometry'][:run_initial['formed_arcs']]
        assert t['record_ids'][:qbase]==run_initial['record_ids']
        events=json.loads((folder/'events.json').read_text())
        for number,event in enumerate(events,1):
            before=json.loads((folder/f'before-formation-{number}.json').read_text());after=json.loads((folder/f'after-formation-{number}.json').read_text())
            n=before['formed_arcs'];q0,_=owner.counts(before)
            assert after['geometry'][:n]==before['geometry'][:n] and after['record_ids'][:q0]==before['record_ids']
            assert after['physical_publications']==before['physical_publications']+1
            assert after['publications']==before['publications']+1
            for a in after['arc_metadata'][n:after['formed_arcs']]:
                assert a['selected_from_time_s']==before['clock_s'] and a['committed_time_s']==after['clock_s']
            assert np.all(np.array(after['histories']['damage'][:q0])>=np.array(before['histories']['damage'][:q0]))
        result=json.loads((OUT/(label+'-summary.json')).read_text())
        accounts.append(dict(label=label,physical_steps=len(h),physical_publications=result['physical_publications'],
            formed_transactions=len(events),new_cells=t['formed_arcs']-run_initial['formed_arcs'],
            old_records_preserved=True,minimum_contact_Pa=float(min(h['minimum_normal_Pa'])),max_weak_Pa=float(max(h['weak_error_Pa']))))
        gates.append(dict(label=label,energy_limit=.01,relative_to_sliding=result['physical_vs_sliding'],relative_to_fracture=result['physical_vs_fracture'],
            sliding_pass=result['physical_vs_sliding']<.01,fracture_pass=result['physical_vs_fracture']<.01))
        if label!='temporal-half':
            life=json.loads((OUT/(label+'-lifecycle.json')).read_text())
            assert life['coupled_FD_relative']<1e-6 and life['rebuilt_context_JSON_restart_bitwise'] and life['native_contact_loss_rejected'] and life['duplicate_rejected']
        ik.close();k.close()
    table(OUT/'publication-accounting.csv',accounts)
    difference=json.loads((OUT/'spatial-difference.json').read_text())
    save(OUT/'accuracy-gates.json',dict(energy=gates,spatial=dict(
        growth_pass=abs(difference['incremental_damage_relative'])<.05,
        new_material_growth_pass=abs(difference['new_material_damage_relative'])<.05,
        new_material_slip_pass=abs(difference['new_material_slip_relative'])<.05,
        slip_pass=abs(difference['signed_slip_increment_relative'])<.05,
        process_front_pass=abs(difference['process_front_difference_m'])<=.09375,
        developed_front_pass=abs(difference['developed_front_difference_m'])<=.09375,
        preferred_direction_pass=abs(difference['sampled_preference_difference_deg'])<1.),
        earlier_point_traction_and_normal_failures_open=True,two_levels_not_order=True))
    record=json.loads((OUT/'accuracy-gates.json').read_text())
    record['spatial_at_all_common_times']=dict(growth_pass=difference['max_common_time_growth_relative']<.05,
        slip_pass=difference['max_common_time_signed_slip_increment_relative']<.05,
        process_front_pass=difference['max_common_time_process_front_difference_m']<=.09375,
        developed_front_pass=difference['max_common_time_full_front_difference_m']<=.09375,
        preferred_direction_pass=difference['max_common_time_preferred_difference_deg']<1.)
    save(OUT/'accuracy-gates.json',record)
    temporal=json.loads((OUT/'temporal-difference.json').read_text())
    # A failed scientific screen stays a reported result, not a relaxed gate.
    gates_record=json.loads((OUT/'accuracy-gates.json').read_text())
    gates_record['late_window_temporal']=dict(growth_pass=abs(temporal['growth_relative'])<.05,
        slip_pass=abs(temporal['slip_increment_relative'])<.05,
        process_front_pass=abs(temporal['process_front_difference_m'])<=.09375,
        developed_front_pass=abs(temporal['developed_front_difference_m'])<=.09375)
    gates_record['frozen_weak_loads']=json.loads((OUT/'frozen-assessment.json').read_text())
    save(OUT/'accuracy-gates.json',gates_record)
    kernels=[]
    for p in WORK.glob('*/native.json'):
        d=json.loads(p.read_text());assert d['energy_relative']<1e-4 and d['gradient_FD_relative']<1e-6
        kernels.append(dict(kernel=p.parent.name,**d))
    table(OUT/'native-cost.csv',kernels)
    animation=json.loads((OUT/'animation.json').read_text())
    with Image.open(OUT/'turn.gif') as im:assert im.n_frames==len(animation['times_s'])
    for p in OUT.glob('*.png'):
        with Image.open(p) as im:im.verify()
    size=sum(p.stat().st_size for p in WORK.rglob('*') if p.is_file() and not p.is_symlink());assert size<1.1*2**30
    save(OUT/'verification.json',dict(engineering_checks_passed=True,protected_files=len(protected)-2,
        production_BP3_defaults_unchanged=True,old_evidence_unchanged=True,raw_bytes=size,native_kernels=len(kernels),
        native_setup_seconds=sum(k['seconds'] for k in kernels),publication_accounting=accounts,
        scientific_accuracy_separate='accuracy-gates.json',scope='Serial experimental fixed-grid native/JSON owner, not production/MPI qualification'))
    sources=list((ROOT/'benchmarks/reconstructed_fault/particle_free').glob('*cohesive_interface_formed_turn.py'))
    sources+=list((ROOT/'doc/reconstructed_fault/particle_free').glob('COHESIVE_INTERFACE_FORMED_TURN_*.md'))
    sources += [ROOT/path for path in docs]
    files=sources+[p for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json']
    save(OUT/'manifest.json',dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        binaries={str(p.relative_to(ROOT)):sha(p) for p in [WORK/'build/cohesive_interface_path',WORK/'build/libcohesive_interface_turn_solver.so']},
        files={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))}))
    print((OUT/'verification.json').read_text())


if __name__=='__main__':main()
