#!/usr/bin/env python3
"""Verify immutable artifacts, matching controls, defaults and retained regressions."""
import argparse,hashlib,json,re,sys
from pathlib import Path
import numpy as np
from analyze_pf6_mature import read
ROOT=Path(__file__).resolve().parents[3]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    runs=a.directory/'runs';old=ROOT/'build-particle-free/mature-dg/runs'
    labels=['dg-cc','dg-fc-final','dg-ff','particle-cc','particle-fc','particle-ff']
    records={label:json.loads((runs/label/'record.json').read_text()) for label in labels}
    for label,r in records.items():
        assert r['exit_code']==0
        folder=runs/label
        assert r['input_sha256']==sha(folder/'input.prm')
        for relative,h in r['artifacts'].items():assert sha(folder/Path(relative).name)==h,(label,relative)
        for relative,h in r['source_hashes'].items():assert sha(ROOT/relative)==h,(label,relative)
        for name,h in r['csv_hashes'].items():assert sha(folder/'run'/name)==h,(label,name)
        for path,h in r.get('external_inputs',{}).items():assert sha(Path(path))==h,(label,path)
    assert len({tuple(sorted(r['artifacts'].items())) for r in records.values()})==1
    assert len({tuple(sorted(r['source_hashes'].items())) for r in records.values()})==1
    matched={}
    for dg,particle in [('dg-cc','particle-cc'),('dg-fc-final','particle-fc'),('dg-ff','particle-ff')]:
        matched[dg+'__'+particle]={}
        for name in ['initial-mesh.csv','initial-phase.csv','mechanical-quadrature.csv','initial-particle-identity.csv']:
            assert (runs/dg/'run'/name).read_bytes()==(runs/particle/'run'/name).read_bytes(),(dg,particle,name)
            matched[dg+'__'+particle][name]='byte identical'
    regressions={}
    for label,baseline in [('dg-cc','dg-q6'),('dg-ff','dg-fine')]:
        counts=0
        for prefix in ['coupled-profile','authoritative-stress','dg-coefficients','physical-fe']:
            for step in range(4):
                name=f'{prefix}-{step}.csv'
                assert (runs/label/'run'/name).read_bytes()==(old/baseline/'run'/name).read_bytes(),(label,name)
                counts+=1
        regressions[label]=dict(baseline=baseline,byte_identical_field_files=counts)
    if (runs/'particle-default-regression/record.json').exists():
        r=json.loads((runs/'particle-default-regression/record.json').read_text());assert r['exit_code']==0
        diffs={}
        for step in [0,1]:
            name=f'coupled-profile-{step}.csv'
            left,right=read(old/'gate-particle-v3/run'/name),read(runs/'particle-default-regression/run'/name)
            diffs[str(step)]={k:float(max(abs(left[k]-right[k]))) for k in left}
            assert max(diffs[str(step)].values())==0.
        regressions['particle_default_Gauss3']=diffs
    incoming=json.loads((a.directory/'incoming.json').read_text());changed=[];missing=[]
    for relative,h in incoming.items():
        path=ROOT/relative
        if not path.is_file():missing.append(relative)
        elif sha(path)!=h:changed.append(relative)
    assert not missing,missing
    for name in ['dg_history','dg_transport','dg_limiter','dg_accepted','dg_transfer','fe_normalization','fe_phase_energy','normalization','history','constitutive']:
        f='source/material_model/phase_field_fault/'+name+'.cc'
        assert sha(ROOT/f)==incoming[f],f
    for f in ['source/particle/manager.cc','source/particle/particle_domain.cc','source/reconstructed_fault/manager.cc','source/simulator/phase_field.cc']:
        assert sha(ROOT/f)==incoming[f],f
    entry=a.directory/'entry/source/material_model/phase_field_fault.cc';current=ROOT/'source/material_model/phase_field_fault.cc'
    defaults=lambda p:re.findall(r'prm.declare_entry\s*\(\s*"([^"]+)"\s*,\s*"([^"]*)"',p.read_text())
    assert defaults(entry)==defaults(current)
    all_records=[json.loads(f.read_text()) for f in runs.glob('*/record.json')]
    result=dict(source_files_per_case=len(records['dg-cc']['source_hashes']),
                same_core_and_plugins=True,same_native_sources=True,matched_controls=matched,
                regressions=regressions,unchanged_material_defaults=len(defaults(current)),
                missing_entry_files=missing,changed_entry_files=changed,
                total_native_attempts=len(all_records),total_native_seconds=sum(r['seconds'] for r in all_records),
                prior_mature_archive_sha256=sha(ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/pf6_mature_dg/reference.tar.xz'),
                prior_domain_free_archive_sha256=sha(ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/pf6_domain_free/reference.tar.xz'),
                strict_reference_match_passed=False)
    assert result['total_native_seconds']<3600
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
