#!/usr/bin/env python3
"""Archive the completed transfer fixture without changing native outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[3]
    runs = root/'build-particle-free/runs'
    diagnosis = root/'build-particle-free/transfer-qualification'
    labels = [f'pf6-transfer-checked-{part}r{rank}'
              for part in ['', 'half-', 'restart-'] for rank in [1, 2, 4]]
    labels += ['pf6-transfer-checked-cross-1-to-4']
    a.output.mkdir(parents=True, exist_ok=True)
    archive = a.output/'reference.tar.xz'
    assert not archive.exists(), 'Do not overwrite frozen evidence'
    files = {}

    def add(path, name):
        if path.is_file():
            files[name] = path

    records = {}
    for label in labels:
        folder = runs/label
        r = json.loads((folder/'record.json').read_text())
        assert r['exit_code'] == 0
        records[label] = r
        for name in ['input.prm', 'fixture.prm', 'record.json', 'command.json',
                     'run.log', 'source.patch', 'observer-source.cc']:
            add(folder/name, label+'/'+name)
        for path in (folder/'run').rglob('*'):
            # Save the actual native checkpoints as well as every compact export.
            if path.is_file():
                add(path, label+'/run/'+str(path.relative_to(folder/'run')))
    first = runs/labels[0]
    for name in ['aspect-release', 'libpf6_transfer_checks.release.so']:
        add(first/name, 'build/'+name)
    assert len({r['executable_sha256'] for r in records.values()}) == 1
    assert len({r['plugin_sha256'] for r in records.values()}) == 1
    for name in ['assess_pf6_transfer.py', 'run_pf6_transfer.py',
                 'run_pf6_transfer_restarts.py', 'package_pf6_transfer.py',
                 'replay_pf6_transfer.py', 'pf6_transfer_checks.cc',
                 'pf6a-common.prm', 'pf6a-initial-phase.txt', 'run_fixed_profile.py',
                 'reference_checks.py', 'CMakeLists.txt']:
        add(Path(__file__).with_name(name), 'source/benchmarks/reconstructed_fault/particle_free/'+name)
    # A diff alone omits incoming untracked source, including dg_transfer.cc.
    # Save complete current contents of every modified/untracked production file.
    paths = subprocess.check_output(['git', 'ls-files', '-m', '-o', '--exclude-standard',
                                     'source', 'include'], cwd=root, text=True).splitlines()
    for name in sorted(set(paths)):
        add(root/name, 'source/'+name)
    for name in ['PF6_TRANSFER_FORMULATION.md', 'PF6_TRANSFER_REPORT.md',
                 'PF6_PARTICLE_READ_AUDIT.md', 'STATUS.md', 'SOURCE_AUDIT.md']:
        add(root/'doc/reconstructed_fault/particle_free'/name, 'source/doc/reconstructed_fault/particle_free/'+name)
    for path in diagnosis.iterdir():
        if path.suffix in ['.json', '.log', '.patch', '.prm', '.c']:
            add(path, 'diagnosis/'+path.name)
    # Retain failed/capped attempts as failures, with their commands and logs.
    # Their original per-run binaries/full outputs remain untouched locally.
    failures = []
    for folder in sorted(runs.glob('pf6-transfer-*')):
        if folder.name in labels:
            continue
        record = folder/'record.json'
        if record.exists():
            failures.append(dict(label=folder.name, record=json.loads(record.read_text())))
        for name in ['record.json', 'command.json', 'input.prm', 'fixture.prm',
                     'run.log', 'observer-source.cc', 'source.patch']:
            add(folder/name, 'other-attempts/'+folder.name+'/'+name)
        for path in (folder/'run').glob('*'):
            if path.is_file() and path.name in ['transfer-errors.csv', 'transfer-gates.csv', 'matching-cell-trace.csv']:
                add(path, 'other-attempts/'+folder.name+'/run/'+path.name)
    logroot = root/'benchmarks/reconstructed_fault/refactoring_boundary/evidence'
    for path in logroot.glob('pf6-transfer-*'):
        if path.suffix in ['.json', '.log']:
            add(path, 'launch-logs/'+path.name)
    manifest = {name: sha(path) for name, path in sorted(files.items())}
    with tarfile.open(archive, 'w:xz', preset=2) as out:
        for name, path in sorted(files.items()):
            out.add(path, arcname=name, recursive=False)
    (a.output/'archive-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    provenance = dict(base_revision=records[labels[0]]['head'],
                      archive_sha256=sha(archive), records=records,
                      other_attempts=failures,
                      scope='Uniform Cartesian fixed-surface transfers; 1/2/4 ranks; no transport or moving geometry')
    (a.output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(dict(files=len(files), bytes=archive.stat().st_size, archive=str(archive))))


if __name__ == '__main__':
    main()
