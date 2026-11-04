#!/usr/bin/env python3
"""Preservation, source recovery and exact binary/run identities for this phase."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[4]
E=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence'
OUT=E/'transported_objective/bounded_profile'
BUILD=ROOT/'build-particle-free/transported-objective/stage-c'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(name,data):(OUT/name).write_text(json.dumps(data,indent=2)+'\n')

def main():
    entry=read(OUT/'entry.json');old=read(E/'transported_objective/stage_c/entry.json')
    assert subprocess.check_output(['git','rev-parse','--show-toplevel'],text=True).strip()==entry['root']
    assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()==entry['branch']
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==entry['head']
    sets=dict(entry_controls=entry['saved_controls'],protected=old['protected_files'],
              server=read(E/'objective_stress/bp3-input-manifest.json'),
              cleanup=read(ROOT/'benchmarks/reconstructed_fault/maintenance/disk_cleanup_2026_10_30/protected-critical-sha256.json'),
              pre_C_controls={r['copy']:r['sha256'] for r in old['controls']},
              coupled_controls=read(E/'transported_objective/stage_c_coupled/entry.json')['saved_controls'])
    for name in ['pre-support','support-admission']:
        saved=read(OUT/(name+'.json'));sets[name]={saved['core_copy']:saved['core_sha256']}
    sets['pre_support_checkers']=read(OUT/'pre-support-checkers.json')
    preservation={}
    for name,records in sets.items():
        bad=[p for p,v in records.items() if not (ROOT/p).exists() or sha(ROOT/p)!=(v['sha256'] if isinstance(v,dict) else v)]
        preservation[name]=dict(checked=len(records),mismatches=bad);assert not bad,(name,bad)
    write('preservation.json',preservation)
    native={str(p.relative_to(ROOT)):sha(p) for folder in ['source','include','unit_tests'] for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ['.h','.cc','.hpp','.cpp']}
    # Entry also records native build/config templates, beyond C++ suffixes.
    for name in entry['native_source']:
        assert (ROOT/name).is_file(),('An incoming source file disappeared.',name)
        native[name]=sha(ROOT/name)
    changed=[p for p,h in entry['native_source'].items() if native[p]!=h]
    added=sorted(set(native)-set(entry['native_source']))
    allowed={'source/material_model/phase_field_fault.cc','include/aspect/material_model/phase_field_fault.h','source/material_model/phase_field_fault/dg_material.cc','source/material_model/phase_field_fault/dg_accepted.cc','source/reconstructed_fault/manager.cc','include/aspect/reconstructed_fault/manager.h'}
    assert set(changed)<=allowed,changed
    assert added==['source/material_model/phase_field_fault/dg_material_profile.cc'],added
    write('native-source-sha256.json',native)
    paths={ROOT/p for p in native}
    paths.update(p for p in (ROOT/'benchmarks/reconstructed_fault/particle_free/transported_objective').rglob('*') if p.is_file() and p.suffix in ['.py','.cc','.h','.md','.txt'])
    paths.update((ROOT/'doc/reconstructed_fault/particle_free').glob('*.md'))
    paths.update(ROOT/p for p in ['AGENTS.md','CMakeLists.txt','doc/reconstructed_fault/current_design.md','doc/reconstructed_fault/specification.tex'])
    recovery={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}
    write('source-recovery-sha256.json',recovery)
    with tarfile.open(OUT/'source-inputs.tar.xz','w:xz') as archive:
        for p in sorted(paths):archive.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
    with tarfile.open(OUT/'source-inputs.tar.xz') as archive:
        for member in archive.getmembers():assert hashlib.sha256(archive.extractfile(member).read()).hexdigest()==recovery[member.name]
    runs={p.parent.name:read(p) for p in (BUILD/'material-runs').glob('bp*/command.json')}
    binaries=[ROOT/'build-particle-free/native/aspect-release',BUILD/'checks/libtransported_material_native.release.so',
              BUILD/'checks/libtransported_material_energy.release.so',BUILD/'checks/libtransported_objective_native.release.so',
              BUILD/'bp3-plugin/libbp3_restore_150x50.release.so',BUILD/'cohesive-checks/libcohesive_rsf_native_checks.release.so']
    logs={str(p.relative_to(ROOT)):sha(p) for p in BUILD.glob('bounded-profile-*.log')}
    unit=ROOT/'benchmarks/reconstructed_fault/refactoring_boundary/evidence/bounded-profile-final-unit.log'
    assert 'All tests passed (183170 assertions in 152 test cases)' in unit.read_text()
    records={str(p.relative_to(ROOT)):sha(p) for p in OUT.glob('*') if p.is_file() and p.name!='final-provenance.json'}
    write('final-provenance.json',dict(utc=datetime.now(timezone.utc).isoformat(),root=entry['root'],branch=entry['branch'],head=entry['head'],
          changed_incoming_native=changed,added_native=added,preservation=preservation,
          binaries={str(p.relative_to(ROOT)):sha(p) for p in binaries},runs=runs,build_logs=logs,
          unit_suite=dict(assertions=183170,cases=152,sha256=sha(unit)),evidence=records,
          data_losses=['39 of 45 historical animation frames lost in inherited cleanup.','Earlier temporary /tmp build logs were lost; this phase retains durable logs.'],
          no_server_or_production_restart=True))
    print('Preserved all protected data; verified source recovery and recorded exact local provenance.')

if __name__=='__main__':main()
