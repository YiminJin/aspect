#!/usr/bin/env python3
"""Verify archive hashes and independently replay checks without ASPECT or MPI."""
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
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
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
source = a.output/'source/benchmarks/reconstructed_fault/particle_free'
sys.path.insert(0, str(source))
from assess_pf6_local import compare_exports
from assess_pf6_domain_free import domain_absence
out = a.output/'replayed'
out.mkdir()
prefix = provenance['prefix']
case = lambda s: a.output/(prefix+'-'+s)


def check(label, folder, args=()):
    subprocess.run([sys.executable, str(source/'assess_pf6_domain_free.py'), str(folder)]+list(args)+['--output', str(out/(label+'.json'))], check=True)


for r in [1, 2, 4]:
    check(f'r{r}', case(f'r{r}'))
    c = compare_exports(case(f'r{r}'), case(f'restart-r{r}'), 'mesh-checkpoint', 'mesh-restored')
    c['domain_absence'] = domain_absence(case(f'restart-r{r}'))
    (out/f'restart-r{r}.json').write_text(json.dumps(c, indent=2)+'\n')
    if r > 1:
        check(f'rank-1-{r}', case('r1'), ['--compare', str(case(f'r{r}'))])
check('legacy-comparison', case('r1'), ['--compare', str(case('legacy-control')), '--legacy-comparison'])
check('continued', case('r1'), ['--compare', str(case('continued'))])
c = compare_exports(case('r1'), case('cross-1-to-4'), 'mesh-checkpoint', 'mesh-restored')
c['domain_absence'] = domain_absence(case('cross-1-to-4'))
(out/'cross-1-to-4.json').write_text(json.dumps(c, indent=2)+'\n')
assert 'Particle RNG replay unavailable' in (case('cross-1-to-4')/'run.log').read_text()
for label, record in provenance['records'].items():
    expected_failure = '-negative-' in label or label.endswith('-legacy-deep-failure')
    assert record['exit_code'] == (1 if expected_failure else 0), label
    if label.endswith('-legacy-deep-failure'):
        assert 'crosses the boundary' in (a.output/label/'run.log').read_text()
left = a.output/'controls'/(prefix+'-regression-fe')/'run'
right = a.output/'controls/pf6-local-regression-fe/run'
controls = list(left.glob('*.csv'))
assert len(controls) == 21 and all(f.read_bytes() == (right/f.name).read_bytes() for f in controls)
result = dict(archive_members_verified=len(manifest), independent_replay_passed=True,
              legacy_FE_identical_CSVs=21, domain_free_deep_remap_passed=True,
              original_CPDI_deep_failure_retained=True, native_Gauss6_failure_retained=True,
              strict_reference_match_passed=False, changed_rank_RNG_replay_available=False,
              native_jobs_launched=0)
(out/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
