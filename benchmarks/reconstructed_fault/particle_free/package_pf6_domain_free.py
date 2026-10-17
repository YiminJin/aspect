#!/usr/bin/env python3
"""Freeze homogeneous mesh-data checks, native checkpoints and failed CPDI control."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('output', type=Path)
p.add_argument('--prefix', default='pf6-domain-free')
a = p.parse_args()
root = Path(__file__).resolve().parents[3]
runs, diag = root/'build-particle-free/runs', root/'build-particle-free/domain-free'
if a.prefix != 'pf6-domain-free':
    diag = diag/a.prefix
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
a.output.mkdir(parents=True, exist_ok=True)
archive = a.output/'reference.tar.xz'
assert not archive.exists()
files, records = {}, {}


def add(path, name):
    if path.is_file():
        files[name] = path


for folder in sorted(runs.glob(a.prefix+'-*')):
    record = json.loads((folder/'record.json').read_text())
    if 'domain_free_case' not in record:
        continue
    records[folder.name] = record
    for name in ['record.json', 'command.json', 'input.prm', 'fixture.prm', 'run.log', 'source.patch', 'observer-source.cc']:
        add(folder/name, folder.name+'/'+name)
    for f in (folder/'run').rglob('*'):
        add(f, folder.name+'/run/'+str(f.relative_to(folder/'run')))
assert len(records) == 17
assert len({r['executable_sha256'] for r in records.values()}) == 1
assert len({r['plugin_sha256'] for r in records.values()}) == 1
for name in ['aspect-release', 'libpf6_transfer_checks.release.so']:
    add(runs/(a.prefix+'-r1')/name, 'build/'+name)
for label in [a.prefix+'-regression-fe', 'pf6-local-regression-fe']:
    folder = runs/label
    for name in ['record.json', 'input.prm', 'run.log', 'source.patch']:
        add(folder/name, 'controls/'+label+'/'+name)
    for f in (folder/'run').glob('*.csv'):
        add(f, 'controls/'+label+'/run/'+f.name)
for f in diag.iterdir():
    if f.suffix in ['.json', '.patch', '.log']:
        add(f, 'diagnosis/'+f.name)
for name in ['assess_pf6_domain_free.py', 'assess_pf6_transfer.py', 'assess_pf6_local.py',
             'run_pf6_domain_free.py', 'run_pf6_domain_free_cases.py', 'run_pf6_local.py',
             'run_pf6_transfer.py', 'run_fixed_profile.py', 'package_pf6_domain_free.py',
             'replay_pf6_domain_free.py', 'pf6_transfer_checks.cc', 'pf6a-initial-phase.txt', 'CMakeLists.txt']:
    add(Path(__file__).with_name(name), 'source/benchmarks/reconstructed_fault/particle_free/'+name)
add(root/'tests/phase_field_test_access.h', 'source/tests/phase_field_test_access.h')
for name in subprocess.check_output(['git', 'ls-files', '-m', '-o', '--exclude-standard', 'source', 'include'], cwd=root, text=True).splitlines():
    add(root/name, 'source/'+name)
for name in ['PF6_DOMAIN_FREE_FORMULATION.md', 'PF6_DOMAIN_FREE_REPORT.md', 'PF6_PARTICLE_READ_AUDIT.md', 'STATUS.md', 'SOURCE_AUDIT.md']:
    add(root/'doc/reconstructed_fault/particle_free'/name, 'source/doc/reconstructed_fault/particle_free/'+name)
for f in (root/'benchmarks/reconstructed_fault/refactoring_boundary/evidence').glob(a.prefix+'-*'):
    if f.suffix in ['.json', '.log']:
        add(f, 'launch-logs/'+f.name)
manifest = {name: sha(f) for name, f in sorted(files.items())}
with tarfile.open(archive, 'w:xz', preset=2) as f:
    for name, path in sorted(files.items()):
        f.add(path, arcname=name, recursive=False)
(a.output/'archive-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
previous = root/'benchmarks/reconstructed_fault/particle_free/evidence/pf6_local/reference.tar.xz'
(a.output/'provenance.json').write_text(json.dumps(dict(prefix=a.prefix, records=records,
    archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
    previous_local_evidence_sha256=sha(previous)), indent=2)+'\n')
print(json.dumps(dict(files=len(files), bytes=archive.stat().st_size)))
