#!/usr/bin/env python3
"""Check recorded results and preserve the R3b executable and source identity."""
from pathlib import Path
import hashlib
import json
import shutil
root = Path(__file__).resolve().parent
repo = root.parents[2]
e = root/'evidence'
expected = {'configure':0, 'build':0, 'separate-history':0,
            'cohesive-create-one':1, 'cohesive-resume-one':1}
for variant in ('reference','candidate'):
    for case in ('unit','stage-j','temperature','frozen-stress','rollback-original','rollback-open-top'):
        for rank in (1,2):
            expected[f'{variant}-{case}-{rank}'] = 1 if case=='stage-j' else 0
for mode in ('legacy-one','legacy-two','automatic-one','automatic-two','automatic-split'):
    expected[f'candidate-bp3-{mode}'] = 0
if (e/'build-final.json').exists():
    expected['build-final'] = 0
for label, code in expected.items():
    assert json.loads((e/f'{label}.json').read_text())['exit_code']==code,label
for name in ('source-verification','instantiations','history-lifecycle-comparison','cohesive-comparison','lifecycle-comparison'):
    data=json.loads((e/f'{name}.json').read_text())
    assert data and all(data.values()),name
fields=json.loads((e/'state-comparison.json').read_text())
assert len(fields)==405
assert all(f['max_abs']==0 for group in fields.values() for f in group['fields'].values())
obj=repo/'build-refactor-r3b/CMakeFiles/aspect.exe.release.dir/source/material_model/phase_field_fault/history.cc.o'
for name in ('source/material_model/phase_field_fault/history.cc','include/aspect/material_model/phase_field_fault.h'):
    assert obj.stat().st_mtime >= (repo/name).stat().st_mtime, name
source=json.loads((e/'entry-source-hashes.json').read_text())
source={name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in source}
(e/'qualified-source-hashes.json').write_text(json.dumps(source,indent=2)+'\n')
binary=repo/'build-refactor-r3b/aspect-release'
qualified=binary.with_name('aspect-r3b-qualified')
if not qualified.exists():shutil.copy2(binary,qualified)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(binary)==sha(qualified)
record=dict(reference_commit='bef79b31a',reference_binary_sha256=sha(repo/'build-restart-fix/aspect-r3a-corrected-qualified'),
            candidate_binary_sha256=sha(qualified),expected_exit_codes=expected,
            source_files=len(source),exact_field_groups=len(fields))
(e/'qualification.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
