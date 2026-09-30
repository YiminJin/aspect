#!/usr/bin/env python3
"""Verify exact moved spans and preserve every pre-R2a non-material source edit."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parent
repo = root.parents[2]
original = (root / 'evidence/phase_field_fault.before.cc').read_text()
main = (repo / 'source/material_model/phase_field_fault.cc').read_text()
normalization = (repo / 'source/material_model/phase_field_fault/normalization.cc').read_text()
spans = json.loads((root / 'evidence/moved-spans.json').read_text())
expected = original
for label, record in spans.items():
    text = record['text']
    assert hashlib.sha256(text.encode()).hexdigest() == record['sha256']
    assert expected.count(text) == normalization.count(text) == 1, label
    expected = expected.replace(text, '')

# Undo only the documented shared-helper linkage scaffolding for comparison.
restored = main.replace('Shared implementation helper and file-local helper types',
                        'File-local helper types')
restored = restored.replace(
    '  // Used by both history preparation and normalization. Keep one definition;\n'
    '  // normalization.cc declares it privately without adding a public header API.\n'
    '  namespace internal\n', '  namespace\n', 1)
restored = restored.replace(
    '  }\n\n  namespace\n  {\n    template <int dim>\n    bool\n'
    '    cohesive_history_is_initialized(',
    '    template <int dim>\n    bool\n    cohesive_history_is_initialized(', 1)
restored = restored.replace('    using aspect::internal::throw_if_history_error;\n', '', 1)
assert restored == expected, 'Changes beyond the recorded relocation/linkage scaffolding'

for line in (root / 'evidence/entry-source-sha256.txt').read_text().splitlines():
    digest, name = line.split('  ', 1)
    if name != 'source/material_model/phase_field_fault.cc':
        assert hashlib.sha256((repo / name).read_bytes()).hexdigest() == digest, name
assert main.count('ASPECT_REGISTER_MATERIAL_MODEL(PhaseFieldFault,') == 1
assert 'ASPECT_REGISTER_MATERIAL_MODEL(' not in normalization
print('All moved text is byte-identical; remaining method bodies, header and prior edits are unchanged.')
print(f'Lines: main {len(original.splitlines())} -> {len(main.splitlines())}; new file {len(normalization.splitlines())}.')
