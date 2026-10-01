#!/usr/bin/env python3
"""Require all selected checks and freeze the qualified residual candidate."""
from pathlib import Path
import hashlib, json, shutil, subprocess

root = Path(__file__).resolve().parent
repo = root.parents[2]
e = root / 'evidence'
h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
labels = ['configure', 'independent-final', 'build', 'plugin-configure', 'plugin-build']
labels += [f'candidate-{name}-{rank}' for name in
           ('residual', 'exhaustion', 'pressure', 'unit', 'rollback-original') for rank in (1, 2)]
labels += ['candidate-gmg-1']
labels += ['candidate-bp3-' + mode for mode in
           ('legacy-one', 'legacy-two', 'automatic-one', 'automatic-two')]
for label in labels:
    record = json.loads((e / (label + '.json')).read_text())
    assert record['exit_code'] == (
        1 if '-exhaustion-' in label else 0), label
    if label.startswith('candidate-'):
        baseline = json.loads((root.with_name('refactoring_r4b_linear') /
                               'evidence' / (label + '.json')).read_text())
        assert record['environment'] == baseline['environment'], label
for name, count in [('source-verification', 8), ('focused-comparison', 31), ('lifecycle-comparison', 28)]:
    checks = json.loads((e / (name + '.json')).read_text())
    assert len(checks) == count and all(checks.values()), name
fields = json.loads((e / 'state-comparison.json').read_text())
assert len(fields) == 372
assert all(f['max_abs'] == 0 for row in fields.values() for f in row['fields'].values())
for name in ('candidate-source-hashes', 'executed-artifacts', 'protected-hashes'):
    manifest = json.loads((e / (name + '.json')).read_text())
    assert all(h(repo / p) == v for p, v in manifest.items()), name
reference = json.loads((e / 'reference.json').read_text())
assert h(repo / reference['binary']) == reference['sha256']
for name in ('residual-1', 'residual-2', 'exhaustion-1', 'exhaustion-2',
             'pressure-1', 'pressure-2', 'gmg-1', 'rollback-original-1', 'rollback-original-2',
             'bp3-legacy-one', 'bp3-legacy-two', 'bp3-automatic-one', 'bp3-automatic-two'):
    prm = (root / f'output-candidate-{name}/parameters.prm').read_text()
    libraries = next(line for line in prm.splitlines() if 'set Additional shared libraries' in line)
    assert all('refactoring_r4b_residual/plugin-build/' in p
               for p in libraries.split('=', 1)[1].split(',')), libraries
build = repo / 'build-refactor-r4b-residual'
obj = build / 'CMakeFiles/aspect.exe.release.dir/source/simulator/solver/reconstructed_fault_stokes.cc.o'
assert obj.stat().st_mtime >= max((repo / p).stat().st_mtime for p in (
    'include/aspect/simulator.h', 'source/simulator/solver/reconstructed_fault_stokes.cc'))
symbols = subprocess.check_output(['nm', '-C', '--defined-only', str(obj)], text=True)
names = ('solve_reconstructed_fault_stokes', 'solve_reconstructed_fault_condensed_system',
         'evaluate_reconstructed_fault_coupled_residual')
lines = [line for line in symbols.splitlines() if '{' not in line and ' W ' in line
         and any('::' + name + '(' in line for name in names)]
for dim in (2, 3):
    for name in names:
        assert sum(f' W aspect::Simulator<{dim}>::{name}(' in line for line in lines) == 1, (dim, name)
(e / 'linked-symbols.txt').write_text('\n'.join(lines) + '\n')
binary = build / 'aspect-release'
qualified = build / 'aspect-r4b-residual-qualified'
if not qualified.exists():
    shutil.copy2(binary, qualified)
assert h(binary) == h(qualified)
record = dict(reference_head=reference['head'], reference_patch=reference['patch'],
              reference_sha256=reference['sha256'],
              candidate_revision='uncommitted second R4b subpass over accepted first subpass',
              candidate_sha256=h(qualified), build_and_runtime_checks=len(labels),
              expected_exhaustion_failures=2, exact_field_groups=372,
              bp3_counter_and_decision_checks=28, focused_checks=31,
              source_and_protection_checks=8, instantiations=[2, 3])
(e / 'qualification.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
