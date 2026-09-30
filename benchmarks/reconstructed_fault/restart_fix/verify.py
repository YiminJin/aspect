#!/usr/bin/env python3
"""Verify the bounded correction and capture the corrected R3a reference."""
from pathlib import Path
import hashlib
import json
import shutil

root = Path(__file__).resolve().parent
repo = root.parents[2]
evidence = root/'evidence'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

entry = json.loads((evidence/'entry-source-hashes.json').read_text())
current = {name: sha(repo/name) for name in entry}
changed = sorted(name for name in entry if current[name] != entry[name])
assert changed == ['source/reconstructed_fault/manager.cc', 'unit_tests/reconstructed_fault.cc'], changed
insertion = ('    // Rebuild the transient per-fault layout; callers reapply prescribed rows.\n'
             '    prescribed_slip_rates.assign(reconstructed_faults.size(), {});\n')
manager = (repo/changed[0]).read_text()
assert manager.count(insertion) == 1
assert manager.replace(insertion, '') == (evidence/'manager.cc.before').read_text()
expected = {'regression-before-fix': 139, 'unit-1': 0, 'unit-2': 0,
            'cohesive-create-one': 1, 'cohesive-resume-one': 1}
expected.update({f'candidate-bp3-{mode}': 0 for mode in
                 ('legacy-one', 'legacy-two', 'automatic-one', 'automatic-two', 'automatic-split')})
for label, code in expected.items():
    assert json.loads((evidence/f'{label}.json').read_text())['exit_code'] == code, label
for name in ('cohesive-comparison', 'lifecycle-comparison'):
    checks = json.loads((evidence/f'{name}.json').read_text())
    assert checks and all(checks.values()), name
fields = json.loads((evidence/'state-comparison.json').read_text())
assert len(fields) == 405
assert all(f['max_abs'] == 0 for group in fields.values() for f in group['fields'].values())
binary = repo/'build-restart-fix/aspect-fix'
qualified = binary.with_name('aspect-r3a-corrected-qualified')
if not qualified.exists():
    shutil.copy2(binary, qualified)
assert sha(binary) == sha(qualified)
assert sha(repo/'build-refactor-r3a/aspect-r3a-qualified') == '14553b4957c367027c297aa810fc7157b97f59179ffb60a6162e1d5f2641200b'
artifacts = [qualified, repo/'build-refactor-r3a/aspect-r3a-qualified',
             repo/'CMakeLists.txt',
             repo/'benchmarks/reconstructed_fault/refactoring_r3a/plugin-build/libphase_field_fault_stage_j_restart_create.release.so',
             repo/'benchmarks/reconstructed_fault/refactoring_boundary/plugin-build/libbp3_restore_150x50.release.so',
             repo/'benchmarks/reconstructed_fault/refactoring_r2b_cache/plugin-build/libcache_audit.release.so']
record = dict(changed_source=changed, unchanged_source_files=len(entry)-2,
              production_change_is_only_transient_initialization=True,
              expected_exit_codes=expected, exact_bp3_field_groups=len(fields),
              cohesive_checks=15, bp3_cache_work_checks=30,
              artifacts={str(p.relative_to(repo)): sha(p) for p in artifacts})
(evidence/'qualification.json').write_text(json.dumps(record, indent=2)+'\n')
(evidence/'corrected-r3a-source-hashes.json').write_text(json.dumps(current, indent=2)+'\n')
print(json.dumps(record, indent=2))
