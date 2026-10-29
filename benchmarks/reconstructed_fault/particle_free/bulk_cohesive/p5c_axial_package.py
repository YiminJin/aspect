#!/usr/bin/env python3
"""Compact review archive; leave complete native runs/checkpoints separately."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT/'build-particle-free/p5c-axial'
OUT = ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/p5c_axial'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    archive = OUT.with_suffix('.zip')
    assert not archive.exists(), 'Preserve a completed evidence package'
    files = [ROOT/'doc/reconstructed_fault/particle_free'/name for name in
             ['P5C_AXIAL_REPORT.md', 'P5C_AXIAL_SELECTION.md', 'STATUS.md', 'SOURCE_AUDIT.md']]
    bench = ROOT/'benchmarks/reconstructed_fault/particle_free/bulk_cohesive'
    files += sorted(bench.glob('p5c_axial_*.py'))
    context = [bench/name for name in ['p5c_native.cc', 'parallel.cc', 'propagation.cc',
               'p5c_specimen_analysis.py', 'p5c_analysis.py', 'p5c_seed_fixed.prm', 'p5c_seed_radial.prm',
               'native.prm', 'parallel.prm', 'mixed_boundary.prm', 'fracture.prm']]
    context += [ROOT/name for name in ['source/material_model/phase_field_fault/bulk_cohesive_tensor.cc',
                'source/material_model/phase_field_fault/bulk_cohesive_history.cc',
                'source/simulator/solver/bulk_cohesive_phase.cc']]
    for path in files+context:
        target = OUT/'context'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for path in WORK.glob('*.prm'):
        target = OUT/'inputs'/path.name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(path, target)
    for name in ['fixed-prefix', 'fixed-active', 'radial-prefix', 'radial-active']:
        for filename in ['parameters.prm', 'original.prm']:
            target = OUT/'inputs'/name/filename
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(WORK/'runs'/name/filename, target)
    for name in ['entry.json', 'fixed-active-input.json', 'radial-active-input.json', 'excluded-launch.json']:
        shutil.copy2(WORK/name, OUT/name)
    report = ROOT/'doc/reconstructed_fault/particle_free/P5C_AXIAL_REPORT.md'
    (OUT/'REPORT.md').write_text(report.read_text().replace('../../../benchmarks/reconstructed_fault/particle_free/evidence/p5c_axial/', ''))
    (OUT/'changed-files.txt').write_text('\n'.join(str(p.relative_to(ROOT)) for p in files)+'\n'+str(OUT.relative_to(ROOT))+'/ (compact measured evidence)\n')
    (OUT/'provenance.json').write_text(json.dumps({
        'revision': json.loads((WORK/'entry.json').read_text())['revision'],
        'native_executable': 'build-particle-free/p5c-native/aspect-release',
        'native_source_or_plugin_changed': False,
        'context_hashes': {str(p.relative_to(ROOT)): sha(p) for p in context},
        'full_solver_outputs': str((WORK/'runs').relative_to(ROOT)),
        'analysis_views': 'build-particle-free/p5c-axial/analysis; hardlinks plus derived combined ledgers, never restart inputs',
        'previous_comparison': 'p5c_native.zip; SHA256 recorded in entry.json and verified unchanged',
        'strict_reference_match_passed': False,
        'scope': 'two fresh fixed-geometry axial-refinement trajectories, four ranks, same physical inputs and complete matched schedule',
        'no_additional_loading_or_server_job': True}, indent=2)+'\n')
    for path in OUT.rglob('*.json'):
        json.loads(path.read_text())
    manifest = {str(p.relative_to(OUT)): {'bytes': p.stat().st_size, 'sha256': sha(p)}
                for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'manifest.json'}
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    temporary = archive.with_suffix('.zip.partial')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as target:
        for path in sorted(OUT.rglob('*')):
            if path.is_file():
                target.write(path, str(path.relative_to(OUT)))
    with zipfile.ZipFile(temporary) as target:
        assert target.testzip() is None
        for name, record in manifest.items():
            assert hashlib.sha256(target.read(name)).hexdigest() == record['sha256']
    temporary.replace(archive)
    archive.with_suffix('.zip.sha256').write_text(sha(archive)+'  '+archive.name+'\n')
    print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size, 'files': len(manifest)+1, 'sha256': sha(archive)}, indent=2))


if __name__ == '__main__':
    main()
