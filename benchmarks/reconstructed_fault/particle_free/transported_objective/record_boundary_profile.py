#!/usr/bin/env python3
"""Verify preservation and archive this bounded diagnostic experiment."""
import hashlib
import json
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
E=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence'
OUT=E/'transported_objective/boundary_profile'
BUILD=ROOT/'build-particle-free/transported-objective/stage-c'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(name,data):(OUT/name).write_text(json.dumps(data,indent=2)+'\n')


def main():
    entry=read(OUT/'entry.json')
    assert subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=ROOT,text=True).strip()==entry['root']
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()==entry['branch']
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==entry['head']
    for name,expected in entry['native_source'].items():assert sha(ROOT/name)==expected,name
    for name,meta in entry['artifacts'].items():assert sha(ROOT/name)==meta['sha256'],name
    previous=read(E/'transported_objective/stage_c/entry.json')
    preservation={}
    sets={'protected':previous['protected_files'],
          'server':read(E/'objective_stress/bp3-input-manifest.json'),
          'cleanup':read(ROOT/'benchmarks/reconstructed_fault/maintenance/disk_cleanup_2026_10_30/protected-critical-sha256.json'),
          'coupled_entry_controls':read(E/'transported_objective/stage_c_coupled/entry.json')['saved_controls'],
          'pre_C_controls':{r['copy']:r['sha256'] for r in previous['controls']}}
    for label,values in sets.items():
        for name,meta in values.items():assert sha(ROOT/name)==(meta['sha256'] if isinstance(meta,dict) else meta),name
        preservation[label]=dict(checked=len(values),unchanged=True)
    write('preservation.json',preservation)
    names=['boundary-profile-n8','boundary-profile-n16','boundary-profile-n32','compact-profile-full','compact-profile-half']
    cases={name:read(BUILD/'material-runs'/name/'command.json') for name in names}
    assert all(c['exit_code']==0 and c['arguments']['ranks']==2 for c in cases.values())
    transport=read(OUT/'boundary-transport.json');profile=read(OUT/'profile-projection.json');tensor=read(OUT/'tensor-defect.json')
    assert len(transport['rows'])==30
    assert max(r['relative_L2_consistent'] for r in transport['rows'] if r['kind']==0)<1e-10
    assert max(r['max_semidiscrete_identity_defect'] for r in transport['rows'])<1e-10
    assert max(r['nodal_replay_Linf'] for c in profile['native_replays'] for r in c['steps'])<1e-11
    assert max(abs(r['defect']) for r in profile['variance_controls'] if r['domain_halfwidth']==8)<1e-12
    native=BUILD/'checks/libtransported_isolation_native.release.so'
    assert 'not found' not in subprocess.check_output(['ldd',str(native)],text=True)
    sources=set()
    for p in (ROOT/'benchmarks/reconstructed_fault/particle_free/transported_objective').rglob('*'):
        if p.is_file() and p.suffix in ['.py','.h','.cc','.txt','.md']:sources.add(p)
    for name in ['BOUNDARY_PROFILE_ISOLATION_REPORT.md','STATUS.md','RESEARCH_HANDOFF.md','SOURCE_AUDIT.md',
                 'REVIEW_PLAN.md','TRANSPORTED_OBJECTIVE_TASK_STATE.md','IMPLEMENTATION_MAPS.md','TRANSPORTED_OBJECTIVE_SPEC.md',
                 'TRANSPORTED_OBJECTIVE_STAGE_C_COUPLED_REPORT.md']:
        sources.add(ROOT/'doc/reconstructed_fault/particle_free'/name)
    sources.update(ROOT/p for p in ['AGENTS.md','doc/reconstructed_fault/current_design.md','doc/reconstructed_fault/specification.tex'])
    for name in names:
        sources.update(BUILD/'material-runs'/name/f for f in ['command.json','input.prm','fault.txt','runner.py'])
    manifest={str(p.relative_to(ROOT)):sha(p) for p in sorted(sources)}
    write('source-input-manifest.json',manifest)
    archive=OUT/'source-inputs.tar.xz'
    with tarfile.open(archive,'w:xz') as stream:
        for p in sorted(sources):stream.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
    with tarfile.open(archive) as stream:
        for member in stream.getmembers():assert hashlib.sha256(stream.extractfile(member).read()).hexdigest()==manifest[member.name]
    fields={}
    read_names=set(names)|{c['case'] for c in tensor['cases']}|{c['case'] for c in profile['native_replays']}
    for name in read_names:
        for p in (BUILD/'material-runs'/name/'output').glob('*.csv'):
            if p.name.startswith(('material-','boundary-profile-')):fields[str(p.relative_to(ROOT))]=sha(p)
    write('input-fields-sha256.json',fields)
    logs=list(BUILD.glob('boundary-profile-*.log'))+[BUILD/'material-runs'/name/'run.log' for name in names]
    record=dict(captured_utc=datetime.now(timezone.utc).isoformat(),root=str(ROOT),branch=entry['branch'],revision=entry['head'],
      selected_experiments_complete=True,production_ready=False,full_coupled_energy_qualified=False,
      unchanged_native_source_files=len(entry['native_source']),unchanged_existing_artifacts=entry['artifacts'],
      new_checker=dict(path=str(native.relative_to(ROOT)),sha256=sha(native),runtime_resolves=True),
      cases=cases,nonpublishing_transport_rows=len(transport['rows']),source_input_files=len(manifest),
      source_archive_sha256=sha(archive),input_field_files=len(fields),
      earlier_complete_native_source_archive_sha256=sha(E/'transported_objective/stage_c_coupled/current-source.tar.xz'),
      preservation=preservation,logs={str(p.relative_to(ROOT)):sha(p) for p in logs},
      results_sha256={p.name:sha(p) for p in sorted(OUT.glob('*.json')) if p.name!='final-provenance.json'},
      retained_diagnostic_failures=['Build before CMake regeneration: new target absent; initial log retained.',
        'Strict mathematical Q1 replay initially differs by9.11e-8 near a cell face; native lookup tolerance/averaging is separately modeled, not a relaxed screen.'],
      verification='Native identity/reference/snapshot checks, independent Q1 variance/replay, syntax and diff checks; no unchanged-core full suite rerun.',
      limits=['No production/event continuation.','No full coupled work/energy qualification.','No change of zero-inflow/Q1 scheme.',
              'Prior cohesive energy and tip failures remain.','39/45 historical frame loss and old temporary-log loss remain.'])
    write('final-provenance.json',record)
    print(json.dumps(dict(native_source_unchanged=record['unchanged_native_source_files'],cases=len(cases),scratch_rows=len(transport['rows']),preservation=preservation)))


if __name__=='__main__':main()
