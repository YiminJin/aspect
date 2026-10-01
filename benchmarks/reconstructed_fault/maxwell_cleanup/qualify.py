#!/usr/bin/env python3
"""Validate the focused rename checks and preserve the tested executable."""
from pathlib import Path
import hashlib,json,shutil
root=Path(__file__).resolve().parent
repo=root.parents[2]
e=root/'evidence'
labels=['build','build-final','plugin-configure','plugin-build','plugin-build-final','unit-1','unit-2']
labels += [f'candidate-{name}-{rank}' for name in ('temperature','frozen-stress') for rank in (1,2)]
labels += [f'candidate-bp3-{mode}' for mode in ('legacy-one','legacy-two','automatic-one','automatic-two','automatic-split')]
for label in labels:
    assert json.loads((e/f'{label}.json').read_text())['exit_code']==0,label
for file in ('frozen-comparison','lifecycle-comparison'):
    checks=json.loads((e/f'{file}.json').read_text());assert checks and all(checks.values()),file
fields=json.loads((e/'state-comparison.json').read_text())
assert len(fields)==405
assert all(f['max_abs']==0 for v in fields.values() for f in v['fields'].values())
for name in ('constitutive','history'):
    source=repo/f'source/material_model/phase_field_fault/{name}.cc'
    obj=repo/f'build-refactor-r3b/CMakeFiles/aspect.exe.release.dir/source/material_model/phase_field_fault/{name}.cc.o'
    assert obj.stat().st_mtime>=source.stat().st_mtime,name
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
reference=repo/'build-refactor-r3b/aspect-r3b-qualified'
assert sha(reference)=='fbd3a74e0238c0fcf5026c9ec8b554f5296302fe14f3d0de4d986096d2c109db'
binary=repo/'build-refactor-r3b/aspect-release'
qualified=binary.with_name('aspect-maxwell-qualified')
if not qualified.exists():shutil.copy2(binary,qualified)
assert sha(qualified)==sha(binary)
record=dict(reference_sha256=sha(reference),candidate_sha256=sha(qualified),
            passing_invocations=len(labels),exact_field_groups=len(fields),
            frozen_temperature_checks=12,cache_work_checks=30)
(e/'qualification.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
