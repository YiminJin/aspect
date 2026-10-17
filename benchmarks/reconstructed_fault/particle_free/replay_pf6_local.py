#!/usr/bin/env python3
"""Checksum and independently reassess the bounded evidence; launch no ASPECT job."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('evidence', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
provenance = json.loads((a.evidence/'provenance.json').read_text())
archive = a.evidence/'reference.tar.xz'
assert sha(archive) == provenance['archive_sha256']
manifest = json.loads((a.evidence/'archive-manifest.json').read_text())
with tarfile.open(archive) as f:
    members = f.getmembers()
    assert {m.name for m in members} == set(manifest)
    assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
    f.extractall(a.output, filter='data')
for name, expected in manifest.items():
    assert sha(a.output/name) == expected, name
prefix = provenance['prefix']
script = a.output/'source/benchmarks/reconstructed_fault/particle_free/assess_pf6_local.py'
out = a.output/'replayed'
out.mkdir()


def check(label, args):
    subprocess.run([sys.executable, str(script)]+args+['--output', str(out/(label+'.json'))], check=True)


case = lambda s: str(a.output/(prefix+'-'+s))
check('frozen', [case('final-frozen-q'), '--frozen'])
check('fresh', [case('final-q12')])
for q in [24, 32]:
    check(f'trajectory-q12-q{q}', [case('final-q12'), '--compare', case(f'trajectory-q{q}')])
subprocess.run([sys.executable, str(a.output/'source/benchmarks/reconstructed_fault/particle_free/assess_pf6_transfer.py'),
                case('final-q12'), '--compare', case('uniform-continued'), '--output', str(out/'uniform-continued.json')], check=True)
for r in [1, 2, 4]:
    check(f'local-r{r}', [case(f'r{r}'), '--local'])
    check(f'restart-r{r}', [case(f'r{r}'), '--local', '--compare', case(f'restart-r{r}'), '--restart-comparison'])
for r in [2, 4]:
    check(f'rank-1-{r}', [case('r1'), '--local', '--compare', case(f'r{r}')])
check('cross-1-to-4', [case('r1'), '--local', '--compare', case('cross-1-to-4'), '--restart-comparison'])
check('cross-vs-fresh4', [case('r4'), '--local', '--compare', case('cross-1-to-4'), '--restart-comparison'])
assert 'Particle RNG replay unavailable' in (a.output/(prefix+'-cross-1-to-4')/'run.log').read_text()
left = a.output/'controls/pf6-local-regression-fe/run'
right = a.output/'controls/pf6-transfer-regression-fe/run'
controls = list(left.glob('*.csv'))
assert len(controls) == 21 and all(f.read_bytes() == (right/f.name).read_bytes() for f in controls)
frozen = json.loads((out/'frozen.json').read_text())
assert frozen['G12_gate_passed'] and frozen['original_G6_failure_retained']
result = dict(archive_members_verified=len(manifest), independent_replay_passed=True,
              retained_G6_native_quadrature_passed=False, strict_reference_match_passed=False,
              legacy_FE_identical_CSVs=21, changed_rank_RNG_replay_available=False,
              native_jobs_launched=0)
(out/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
