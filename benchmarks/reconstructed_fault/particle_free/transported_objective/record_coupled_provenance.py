#!/usr/bin/env python3
"""Record exact local Stage C continuation source, controls and evidence."""
import hashlib
import json
import re
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
E = ROOT/'benchmarks/reconstructed_fault/particle_free/evidence'
OUT = E/'transported_objective/stage_c_coupled'
BUILD = ROOT/'build-particle-free/transported-objective/stage-c'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(name, data):
    (OUT/name).write_text(json.dumps(data, indent=2)+'\n')


def main():
    entry = read(OUT/'entry.json')
    assert subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=ROOT,text=True).strip() == str(ROOT)
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip() == entry['branch']
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip() == entry['head']
    prior = read(E/'transported_objective/stage_c/entry.json')
    sets = {'protected':prior['protected_files'],
      'server':read(E/'objective_stress/bp3-input-manifest.json'),
      'cleanup':read(ROOT/'benchmarks/reconstructed_fault/maintenance/disk_cleanup_2026_10_30/protected-critical-sha256.json'),
      'entry_controls':entry['saved_controls'],
      'pre_C_controls':{r['copy']:r['sha256'] for r in prior['controls']}}
    preservation = {}
    for label, records in sets.items():
        bad = []
        for name, value in records.items():
            expected = value['sha256'] if isinstance(value,dict) else value
            p = ROOT/name
            if not p.exists() or sha(p) != expected:
                bad.append(name)
        preservation[label] = dict(checked=len(records), mismatches=bad)
        assert not bad, (label,bad)
    write('preservation.json', preservation)
    native = {str(p.relative_to(ROOT)):sha(p) for folder in ['source','include','unit_tests']
              for p in sorted((ROOT/folder).rglob('*')) if p.is_file() and p.suffix in ['.h','.cc','.hpp','.cpp']}
    original = read(OUT/'entry-source-sha256.json')
    for name in original:
        assert (ROOT/name).is_file(), name
        native[name] = sha(ROOT/name)
    assert set(original) <= set(native), 'An incoming native source file disappeared.'
    write('current-source-sha256.json', native)
    paths = {ROOT/p for p in native}
    paths.update(p for p in (ROOT/'benchmarks/reconstructed_fault/particle_free/transported_objective').rglob('*')
                 if p.is_file() and p.suffix in ['.py','.cc','.h','.md','.txt'])
    paths.update(p for p in (ROOT/'doc/reconstructed_fault/particle_free').glob('*.md'))
    paths.update(ROOT/p for p in ['AGENTS.md','CMakeLists.txt','doc/reconstructed_fault/current_design.md',
                                 'doc/reconstructed_fault/specification.tex','doc/reconstructed_fault/CURRENT_STATUS.md'])
    recovery = {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}
    write('source-recovery-sha256.json', recovery)
    archive = OUT/'current-source.tar.xz'
    with tarfile.open(archive,'w:xz') as stream:
        for p in sorted(paths):
            stream.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
    with tarfile.open(archive) as stream:
        for member in stream.getmembers():
            assert hashlib.sha256(stream.extractfile(member).read()).hexdigest() == recovery[member.name]
    artifacts = [ROOT/'build-particle-free/native/aspect-release',
                 BUILD/'checks/libtransported_material_native.release.so',
                 BUILD/'checks/libtransported_objective_native.release.so',
                 BUILD/'bp3-plugin/libbp3_restore_150x50.release.so',
                 BUILD/'cohesive-checks/libcohesive_rsf_native_checks.release.so']
    libraries = {}
    for p in artifacts:
        text = subprocess.check_output(['ldd',str(p)],text=True)
        assert 'not found' not in text, p
        libraries[str(p.relative_to(ROOT))] = dict(sha256=sha(p),bytes=p.stat().st_size,runtime_resolves=True)
    cases = {p.parent.name:read(p) for p in sorted((BUILD/'material-runs').glob('*/command.json'))}
    cases.update({name:read(BUILD/'runs'/name/'command.json') for name in
                  ['tip-trace-cut','tip-trace-legacy','tip-trace-half','coupled-legacy-final']})
    unit = (BUILD/'coupled-unit-final.log').read_text()
    assert 'All tests passed (183170 assertions in 152 test cases)' in unit
    a=BUILD/'runs/legacy-pre-c-control/output/accepted_steps.csv'
    b=BUILD/'runs/coupled-legacy-final/output/accepted_steps.csv'
    assert a.read_bytes()==b.read_bytes()
    logs = list(BUILD.glob('coupled-build-*.log')) + list(BUILD.glob('material-checks-build-*.log'))
    logs += [BUILD/'coupled-bp3-build.log',BUILD/'coupled-unit-final.log',BUILD/'tip-checks-build.log']
    summary = dict(captured_utc=datetime.now(timezone.utc).isoformat(),root=str(ROOT),branch=entry['branch'],revision=entry['head'],
      scope='Narrowed Stage C mature material transaction and heterogeneous coupling; cohesive energy/tip diagnosis.',
      stage_c_complete=False,production_ready=False,source_identity='HEAD plus retained dirty source and exact recovery archive; not HEAD alone.',
      native_source_count=len(native),incoming_source_count=len(original),
      changed_incoming_source=[p for p in original if original[p]!=native[p]],new_native_source=sorted(set(native)-set(original)),
      source_manifest_sha256=sha(OUT/'current-source-sha256.json'),source_recovery_files=len(recovery),source_recovery_sha256=sha(archive),
      artifacts=libraries,cases=cases,preservation=preservation,
      unit_suite=dict(assertions=183170,cases=152,exit_code=0,log_sha256=sha(BUILD/'coupled-unit-final.log')),
      legacy_accepted_steps_byte_identical=True,legacy_accepted_steps_sha256=sha(b),
      build_and_unit_logs={str(p.relative_to(ROOT)):sha(p) for p in logs if p.exists()},
      data_losses=['Cleanup:39 of45 historical animation frames lost.','Earlier temporary /tmp build logs lost; new logs are durable.'],
      limits=['No production readiness or full temporal convergence claim.','No material resampling/birth/boundary completion/moving AMR.',
              'No transported cohesive shape/pressure or mature moving energy-flux qualification.',
              'No server jobs, protected checkpoint conversion, long replay or physical campaign.'])
    write('final-provenance.json',summary)
    print(json.dumps(dict(native_files=len(native),recovery_files=len(recovery),preservation=preservation)))


if __name__=='__main__':
    main()
