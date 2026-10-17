#!/usr/bin/env python3
"""Verify the frozen archive, then independently reassess transfers and restarts.

Requires NumPy. Launches no ASPECT process; creates a fresh output directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('evidence', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    evidence = a.evidence.resolve()
    archive = evidence/'reference.tar.xz'
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    provenance = json.loads((evidence/'provenance.json').read_text())
    assert sha(archive) == provenance['archive_sha256']
    manifest = json.loads((evidence/'archive-manifest.json').read_text())
    with tarfile.open(archive) as f:
        assert {m.name for m in f.getmembers()} == set(manifest)
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
                   for m in f.getmembers())
        f.extractall(a.output, filter='data')
    for name, expected in manifest.items():
        assert sha(a.output/name) == expected, name
    script = a.output/'source/benchmarks/reconstructed_fault/particle_free/assess_pf6_transfer.py'
    reports = a.output/'replayed'
    reports.mkdir()
    for rank in [1, 2, 4]:
        subprocess.run([sys.executable, str(script), str(a.output/f'pf6-transfer-checked-r{rank}'),
                        '--output', str(reports/f'r{rank}.json')], check=True)
    pairs = [('rank-1-2', 'r1', 'r2'), ('rank-1-4', 'r1', 'r4')]
    pairs += [(f'restart-r{r}', f'r{r}', f'restart-r{r}') for r in [1, 2, 4]]
    pairs += [('cross-1-to-4', 'r4', 'cross-1-to-4')]
    for label, left, right in pairs:
        subprocess.run([sys.executable, str(script), str(a.output/('pf6-transfer-checked-'+left)),
                        '--compare', str(a.output/('pf6-transfer-checked-'+right)),
                        '--output', str(reports/(label+'.json'))], check=True)
    quadrature_flags = [json.loads((reports/f'r{r}.json').read_text())['quadrature']
                        ['native_rule_vs_high_rule_screen_passed'] for r in [1, 2, 4]]
    assert quadrature_flags == [False]*3, 'Retain the documented quadrature limitation'
    result = dict(archive_members_verified=len(manifest), independent_transfer_checks_passed=True,
                  ownership_restart_comparisons_passed=True,
                  native_rule_vs_high_rule_screen_passed=False,
                  strict_reference_match_passed=False,
                  strict_reference_note='Unchanged prior publication result, not rerun or repaired by this transfer study')
    (reports/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
