#!/usr/bin/env python3
"""Protect entry-state code and distinguish the move from the behavior extension."""
import hashlib, json, difflib
from pathlib import Path
root = Path(__file__).resolve().parent
repo = root.parents[2]
evidence = root / 'evidence'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
entry = json.loads((evidence / 'entry-source-hashes.json').read_text())
changed = [name for name, digest in entry.items() if sha(repo / name) != digest]
allowed = {
    'include/aspect/material_model/phase_field_fault.h',
    'include/aspect/reconstructed_fault/manager.h',
    'source/material_model/phase_field_fault/normalization.cc',
    'source/reconstructed_fault/manager.cc',
    'source/reconstructed_fault/surface_system.cc',
}
assert set(changed) <= allowed, changed
normalization = repo / 'source/material_model/phase_field_fault/normalization.cc'
text = normalization.read_text()
for addition in (
'''      if (this->get_reconstructed_fault_manager().uses_automatic_boundary_completion())
        prepare_automatic_boundary_completion();
''',
'''      if (this->get_reconstructed_fault_manager().uses_automatic_boundary_completion())
        {
          apply_automatic_boundary_completion(profiles, profile_integrals);
          return;
        }
''',
'''      boundary_continuations.clear();
      boundary_continuation_generation = numbers::invalid_unsigned_int;
'''):
    assert text.count(addition) == 1
    text = text.replace(addition, '')
assert text == (evidence/'normalization.cc.move').read_text()
assert sha(repo/'build-refactor-boundary/aspect-completion-move') == (evidence/'move-artifact-sha256.txt').read_text().split()[0]
for name, current in [('normalization.cc',normalization), ('phase_field_fault.h',repo/'include/aspect/material_model/phase_field_fault.h')]:
    before=(evidence/(name+'.move')).read_text().splitlines(keepends=True)
    after=current.read_text().splitlines(keepends=True)
    (evidence/(name+'.extension.diff')).write_text(''.join(difflib.unified_diff(before,after,fromfile=name+'.move',tofile=str(current.relative_to(repo)))))
reference_checked = []
for line in (root.with_name('refactoring_r2b')/'evidence/candidate-artifact-sha256.txt').read_text().splitlines():
    digest, name = line.split(None,1)
    if name.startswith(('source/', 'include/')):
        continue
    assert sha(repo/name) == digest, name
    reference_checked.append(name)
record = dict(entry_files=len(entry), changed_existing=changed, protected_unchanged=len(entry)-len(changed),
              normalization_change='Only automatic preparation, dispatch and transient invalidation; remaining move-stage source exact.',
              preserved_move_binary=True, preserved_reference_artifacts=reference_checked)
(evidence/'source-scope.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
