#!/usr/bin/env python3
"""Final native gates, immutable parents, source identities and media checks."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
from continuation_analysis import ROOT,BASE,OUT,read,save_csv

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def verify(names):
    cases={};residual_work=[]
    for name in names:
        run=BASE/'runs'/name
        with (run/'resources.csv').open() as stream:resource=list(csv.DictReader(stream))
        # Keep uint64 fingerprints as exact strings, never floating point.
        assert {r['freeze'] for r in resource}=={'1'}
        assert len({r['particle_fingerprint'] for r in resource})==1
        assert len({r['particles'] for r in resource})==1
        a=read(run/'bulk-accepted.csv');assert np.all(a['step']==a['publications'])
        assert np.all(np.diff(a['step'])==1) and np.all(a['dt']>0)
        bounds={'mechanical_residual':1e-9,'phase_KKT':1e-7,'projection_error':1e-12,'ledger_error':1e-10}
        maxima={key:float(np.max(abs(a[key]))) for key in bounds}
        assert all(maxima[key]<=bound for key,bound in bounds.items())
        g=read(run/'gates.csv');assert np.all(g['passed']==1)
        p=read(run/'pressure.csv');positive=p[p['step']>0]
        # The retained native observer excludes step zero, before the first
        # bulk mechanical solve has replaced ASPECT's initial pressure field.
        assert max(positive['free_side_scaled_residual'])<=1e-9
        cases[name]=dict(accepted_step_range=[int(a['step'][0]),int(a['step'][-1])],
                         accepted_time_range_s=[float(a['time'][0]),float(a['time'][-1])],
                         maxima=maxima,bounds=bounds,all_native_gates_passed=True,
                         frozen_particle_count=int(resource[0]['particles']),
                         unchanged_ID_coordinate_fingerprint=resource[0]['particle_fingerprint'],
                         maximum_positive_step_free_side_scaled_residual=float(max(positive['free_side_scaled_residual'])),
                         peak_rank_KiB=max(int(r['max_rank_KiB']) for r in resource),
                         sum_of_rank_peaks_KiB=max(int(r['sum_rank_peaks_KiB']) for r in resource))
        for row in a:
            residual_work.append([name,row['time'],row['step'],row['dt'],row['phase_work'],row['mechanical_work'],row['pressure_work'],row['ledger_error']])
    pairs=[]
    for a,b in [('reference-hold-critical','reference-relax'),('half-hold','half-relax'),('quarter-hold','quarter-relax')]:
        ia=json.load(open(BASE/(a+'-input.json')));ib=json.load(open(BASE/(b+'-input.json')))
        assert ia['source']==ib['source'] and ia['files']==ib['files']
        pairs.append(dict(candidates=[a,b],identical_parent=True,source=ia['source']))
    parent_count=0
    for p in BASE.glob('*-input.json'):
        a=json.load(open(p))
        for name,digest in a['files'].items():
            assert sha(ROOT/a['source']/name)==digest,(p,name)
        parent_count+=1
    law=json.load(open(OUT/'independent-reference-checks.json'))['constitutive_and_history_sources_unchanged']
    assert all(sha(ROOT/name)==digest for name,digest in law.items())
    identity=json.load(open(BASE/'native-v6/identity.json'))
    assert sha(ROOT/'source/simulator/solver/bulk_cohesive_phase.cc')==identity['source_sha256']
    assert sha(BASE/'native-v6/aspect-release')==identity['binary_sha256']
    observers={}
    for folder in ['plugin','plugin-boundary']:
        value=json.load(open(BASE/folder/'identity.json'))
        key='library_sha256' if folder=='plugin' else 'libbulk_propagation_checks.release.so'
        assert sha(BASE/folder/'libbulk_propagation_checks.release.so')==value[key]
        observers[folder]=value
    assert sha(ROOT/'benchmarks/reconstructed_fault/particle_free/bulk_cohesive/parallel.cc')==observers['plugin-boundary']['parallel.cc']
    assert sha(ROOT/'benchmarks/reconstructed_fault/particle_free/bulk_cohesive/propagation.cc')==observers['plugin-boundary']['propagation.cc']
    entry=json.load(open(OUT/'entry.json'))
    preserved_baselines={}
    for name,digest in entry['files'].items():
        if 'aspect-release' in name or name.endswith('.so'):
            assert sha(ROOT/name)==digest;preserved_baselines[name]=digest
    old=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/bulk_fracture'
    protected=json.load(open(old/'entry.json'))['protected_bp3'];assert all(sha(ROOT/name)==v['sha256'] for name,v in protected.items())
    manifest=json.load(open(old/'manifest.json'));assert all(sha(old/name)==v['sha256'] for name,v in manifest.items())
    media={}
    for p in OUT.glob('*.png'):
        with Image.open(p) as im:im.verify()
        with Image.open(p) as im:media[p.name]=dict(size=list(im.size),frames=1)
    gifs=list(OUT.glob('*.gif'));assert len(gifs)==1
    with Image.open(gifs[0]) as im:
        assert im.n_frames==65
        for i in range(im.n_frames):im.seek(i);im.load()
        media[gifs[0].name]=dict(size=list(im.size),frames=im.n_frames)
    result=dict(cases=cases,hold_parents=pairs,immutable_input_manifests=parent_count,
                constitutive_history_sources_unchanged=law,native_v6=identity,
                observers=observers,
                preserved_baseline_binaries=preserved_baselines,protected_BP3_files=len(protected),
                preserved_prior_evidence_files=len(manifest),media=media,
                strict_reference_match_passed=False)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    save_csv(OUT/'native-residual-work.csv',['case','time_s','step','dt_s','phase_residual_work_J_m','mechanical_residual_work_J_m','pressure_work_J_m','native_ledger_error_J_m'],residual_work)
    print(json.dumps(dict(cases=len(cases),immutable_parents=parent_count,protected_BP3=len(protected),prior_evidence=len(manifest),media=len(media)),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('names',nargs='+');a=p.parse_args();verify(a.names)
