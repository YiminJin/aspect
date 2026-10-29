#!/usr/bin/env python3
"""Package useful review evidence; retain full native archives separately."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile
from p5c_campaign import ROOT, WORK, OUT, sha, save


def main():
    entry = json.loads((WORK/'entry.json').read_text())
    for section in ['protected', 'artifacts', 'native_source_sha256']:
        for name, digest in entry[section].items():
            assert sha(ROOT/name) == digest, name
    assert sha(WORK/'plugin/libbulk_p5c_campaign_checks.release.so') == entry['new_plugin_sha256']
    archive = OUT.with_suffix('.zip')
    assert not archive.exists(), 'Preserve completed evidence'
    bench = ROOT/'benchmarks/reconstructed_fault/particle_free/bulk_cohesive'
    changed = [ROOT/'benchmarks/reconstructed_fault/particle_free/CMakeLists.txt', *sorted(bench.glob('p5c_campaign*'))]
    changed = [p for p in changed if p.is_file() and p.suffix in ['.py', '.cc', '.txt']]
    changed += [ROOT/'doc/reconstructed_fault/particle_free'/name for name in
                ['P5C_CONTINUATION_REPORT.md', 'P5C_CONTINUATION_SELECTION.md', 'STATUS.md', 'SOURCE_AUDIT.md']]
    changed.append(ROOT/'doc/reconstructed_fault/CURRENT_STATUS.md')
    context = [bench/name for name in ['p5c_native.cc', 'parallel.cc', 'propagation.cc', 'p5c_analysis.py',
               'p5c_specimen_analysis.py', 'p5c_axial_analysis.py', 'p5c_seed_fixed.prm', 'p5c_seed_radial.prm',
               'native.prm', 'parallel.prm', 'mixed_boundary.prm', 'fracture.prm']]
    context += [ROOT/'source/material_model/phase_field_fault/bulk_cohesive_tensor.cc',
                ROOT/'source/material_model/phase_field_fault/bulk_cohesive_history.cc',
                ROOT/'source/simulator/solver/bulk_cohesive_phase.cc']
    for path in changed+context:
        target = OUT/'context'/path.relative_to(ROOT); target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
    for path in WORK.glob('*.prm'):
        target = OUT/'inputs'/path.name; target.parent.mkdir(exist_ok=True); shutil.copy2(path, target)
    for path in (WORK/'runs').glob('*/parameters.prm'):
        target = OUT/'inputs'/path.parent.name/path.name; target.parent.mkdir(exist_ok=True); shutil.copy2(path, target)
    for name in ['entry.json', 'build-provenance.json', 'replay-plan.json', 'state.json', 'shutdown-checkpoint.json']:
        shutil.copy2(WORK/name, OUT/name)
    report = ROOT/'doc/reconstructed_fault/particle_free/P5C_CONTINUATION_REPORT.md'
    report_text = report.read_text().replace('../../../benchmarks/reconstructed_fault/particle_free/evidence/p5c_continuation/', '')
    report_text = report_text.replace('[review archive](../../../benchmarks/reconstructed_fault/particle_free/evidence/p5c_continuation.zip)', 'review archive (this package)')
    report_text = report_text.replace('(P5C_CONTINUATION_SELECTION.md)', '(SELECTION.md)')
    (OUT/'REPORT.md').write_text(report_text)
    shutil.copy2(ROOT/'doc/reconstructed_fault/particle_free/P5C_CONTINUATION_SELECTION.md', OUT/'SELECTION.md')
    shutil.copy2(WORK/'RESUME.md', OUT/'COMPLETION.md')
    commands = {}
    for path in sorted((ROOT/'benchmarks/reconstructed_fault/refactoring_boundary/evidence').glob('p5c-campaign-*.json')):
        commands[path.stem] = json.loads(path.read_text())
    save(OUT/'run-commands.json', commands)
    checkpoints = {path.parent.name: json.loads(path.read_text()) for path in sorted((WORK/'checkpoints').glob('*/metadata.json'))}
    for name, record in checkpoints.items():
        directory = WORK/'checkpoints'/name/'restart'
        checkpoint_id = int((directory/'last_good_checkpoint.txt').read_text())
        for filename, digest in record['files'].items():
            assert sha(directory/f'{checkpoint_id:02}'/filename) == digest, (name, filename)
    save(OUT/'retained-checkpoints.json', checkpoints)
    save(OUT/'provenance.json', {'revision': entry['revision'], 'protected_hashes_verified': len(entry['protected']),
         'context_sha256': {str(p.relative_to(ROOT)): sha(p) for p in context},
         'full_native_outputs': str((WORK/'runs').relative_to(ROOT)),
         'protected_checkpoints': str((WORK/'checkpoints').relative_to(ROOT)),
         'strict_reference_match_passed': False, 'production_defaults_changed': False,
         'scope': 'selected fine fixed-mesh finite-body continuation and local timestep replays; no server job or additional mesh'})
    (OUT/'changed-files.txt').write_text('\n'.join(str(p.relative_to(ROOT)) for p in changed)+'\n'+str(OUT.relative_to(ROOT))+'/ (compact evidence)\n')
    for path in OUT.rglob('*.json'):
        json.loads(path.read_text())
    manifest = {str(p.relative_to(OUT)): {'bytes': p.stat().st_size, 'sha256': sha(p)}
                for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'manifest.json'}
    save(OUT/'manifest.json', manifest)
    temporary = archive.with_suffix('.zip.partial')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as target:
        for path in sorted(OUT.rglob('*')):
            if path.is_file(): target.write(path, str(path.relative_to(OUT)))
    with zipfile.ZipFile(temporary) as target:
        assert target.testzip() is None
        for name, record in manifest.items():
            assert hashlib.sha256(target.read(name)).hexdigest() == record['sha256']
    temporary.replace(archive)
    archive.with_suffix('.zip.sha256').write_text(sha(archive)+'  '+archive.name+'\n')
    print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size, 'sha256': sha(archive)}, indent=2))


if __name__ == '__main__':
    main()
