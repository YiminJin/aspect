#!/usr/bin/env python3
"""Freeze bounded quadrature/local-transfer evidence, preserving failed attempts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('output', type=Path)
p.add_argument('--prefix', default='pf6-local-final')
a = p.parse_args()
root = Path(__file__).resolve().parents[3]
runs = root/'build-particle-free/runs'
diag = root/'build-particle-free/quad-local'
suffixes = ['final-frozen-q', 'final-q12', 'trajectory-q24', 'trajectory-q32']
suffixes += ['uniform-half', 'uniform-continued']
suffixes += [f'r{r}' for r in [1, 2, 4]]+[f'restart-r{r}' for r in [1, 2, 4]]+['cross-1-to-4']
labels = [a.prefix+'-'+s for s in suffixes]
a.output.mkdir(parents=True, exist_ok=True)
archive = a.output/'reference.tar.xz'
assert not archive.exists()
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
files = {}


def add(path, name):
    if path.is_file():
        files[name] = path


records = {}
for label in labels:
    folder = runs/label
    record = json.loads((folder/'record.json').read_text())
    assert record['exit_code'] == 0
    records[label] = record
    for name in ['record.json', 'command.json', 'fixture.prm', 'input.prm', 'run.log', 'source.patch', 'observer-source.cc']:
        add(folder/name, label+'/'+name)
    for path in (folder/'run').rglob('*'):
        if path.is_file():
            add(path, label+'/run/'+str(path.relative_to(folder/'run')))
assert len({r['executable_sha256'] for r in records.values()}) == 1
assert len({r['plugin_sha256'] for r in records.values()}) == 1
for name in ['aspect-release', 'libpf6_transfer_checks.release.so']:
    add(runs/labels[0]/name, 'build/'+name)
for label in ['pf6-local-regression-fe', 'pf6-transfer-regression-fe']:
    folder = runs/label
    for name in ['record.json', 'input.prm', 'run.log', 'source.patch']:
        add(folder/name, 'controls/'+label+'/'+name)
    for path in (folder/'run').glob('*.csv'):
        add(path, 'controls/'+label+'/run/'+path.name)
for folder in sorted(runs.glob('pf6-local-*')):
    if folder.name in labels:
        continue
    for name in ['record.json', 'command.json', 'fixture.prm', 'input.prm', 'run.log', 'source.patch', 'observer-source.cc']:
        add(folder/name, 'other-attempts/'+folder.name+'/'+name)
    for path in (folder/'run').glob('*'):
        if path.is_file() and (path.name.startswith('first-local-') or path.name in ['transfer-gates.csv', 'transfer-errors.csv', 'transfer-cost.csv']):
            add(path, 'other-attempts/'+folder.name+'/run/'+path.name)
for path in diag.iterdir():
    if path.suffix in ['.json', '.log', '.patch']:
        add(path, 'diagnosis/'+path.name)
for name in ['assess_pf6_local.py', 'assess_pf6_transfer.py', 'run_pf6_local.py',
             'run_pf6_local_cases.py', 'run_pf6_transfer.py', 'run_fixed_profile.py',
             'package_pf6_local.py', 'replay_pf6_local.py', 'pf6_transfer_checks.cc',
             'pf6a-common.prm', 'pf6a-initial-phase.txt', 'voro_closed_patch.cc', 'CMakeLists.txt']:
    add(Path(__file__).with_name(name), 'source/benchmarks/reconstructed_fault/particle_free/'+name)
add(root/'tests/phase_field_test_access.h', 'source/tests/phase_field_test_access.h')
for name in subprocess.check_output(['git', 'ls-files', '-m', '-o', '--exclude-standard', 'source', 'include'], cwd=root, text=True).splitlines():
    add(root/name, 'source/'+name)
for name in ['PF6_LOCAL_FORMULATION.md', 'PF6_LOCAL_REPORT.md', 'PF6_PARTICLE_READ_AUDIT.md', 'STATUS.md', 'SOURCE_AUDIT.md']:
    add(root/'doc/reconstructed_fault/particle_free'/name, 'source/doc/reconstructed_fault/particle_free/'+name)
logroot = root/'benchmarks/reconstructed_fault/refactoring_boundary/evidence'
for path in logroot.glob('pf6-local-*'):
    if path.suffix in ['.json', '.log']:
        add(path, 'launch-logs/'+path.name)
manifest = {name: sha(path) for name, path in sorted(files.items())}
with tarfile.open(archive, 'w:xz', preset=2) as f:
    for name, path in sorted(files.items()):
        f.add(path, arcname=name, recursive=False)
(a.output/'archive-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
(a.output/'provenance.json').write_text(json.dumps(dict(prefix=a.prefix, records=records,
    archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
    original_transfer_evidence_sha256=sha(root/'benchmarks/reconstructed_fault/particle_free/evidence/pf6_transfer/reference.tar.xz')), indent=2)+'\n')
print(json.dumps(dict(archive=str(archive), files=len(files), bytes=archive.stat().st_size)))
