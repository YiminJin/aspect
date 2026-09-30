#!/usr/bin/env python3
"""Check the bounded extraction against the recorded entry source, not HEAD."""
from pathlib import Path
import difflib
import hashlib
import json

root = Path(__file__).resolve().parent
repo = root.parents[2]
evidence = root / 'evidence'
source = 'source/material_model/phase_field_fault/normalization.cc'
header = 'include/aspect/material_model/phase_field_fault.h'
old = (evidence / 'normalization.cc.before').read_text()
new = (repo / source).read_text()

def identical(label, left, right):
    assert left == right, label + '\n' + ''.join(difflib.unified_diff(
        left.splitlines(True), right.splitlines(True)))

geometry_start = '      // Admit only mappings'
left = old[old.index(geometry_start):old.index('      const double geometry_seconds=')]
right = new[new.index(geometry_start):new.index('      return geometry;')]
right = right.replace('geometry.ends.resize(profiles.size());',
                      'std::vector<std::array<double,2>> ends(profiles.size());')
for name in ('ends', 'candidates', 'rebuilt', 'reused', 'intervals'):
    right = right.replace('geometry.' + name, name)
identical('geometry statements', left, right)

marker = '      // Local four/eight-point refinement'
left, right = old[old.index(marker):], new[new.index(marker):]
for name in ('candidates', 'rebuilt', 'reused', 'intervals'):
    right = right.replace('<< geometry.' + name, '<< ' + name)
instantiation = ('    template PhaseFieldFault<dim>::NormalizationCellGeometry \\\n'
                 '      PhaseFieldFault<dim>::prepare_cell_normalization_geometry('
                 'const std::vector<NormalizationProfile> &); \\\n')
assert right.count(instantiation) == 1
right = right.replace(instantiation, '')
identical('integration tail and later methods', left, right)
identical('earlier methods (including value reuse and completion)',
          old[:old.index('    template <int dim>\n    std::vector<double>\n'
                         '    PhaseFieldFault<dim>::integrate_cell')],
          new[:new.index('    template <int dim>\n'
                         '    typename PhaseFieldFault<dim>::NormalizationCellGeometry')])

old_header = (evidence / 'phase_field_fault.h.before').read_text()
new_header = (repo / header).read_text()
start = new_header.index('        /** Profile endpoints and local work counts')
end = new_header.index('        /** Integrate current phase values', start)
identical('header outside private declarations', old_header,
          new_header[:start] + new_header[end:])

hashes = json.loads((evidence / 'entry-source-hashes.json').read_text())
changed = [name for name, sha in hashes.items()
           if hashlib.sha256((repo / name).read_bytes()).hexdigest() != sha]
assert set(changed) == {source, header}, changed
for name in (source, header):
    before = evidence / (Path(name).name + '.before')
    (evidence / (Path(name).name + '.diff')).write_text(''.join(difflib.unified_diff(
        before.read_text().splitlines(True), (repo / name).read_text().splitlines(True),
        fromfile='R2a/' + name, tofile='R2b/' + name)))
result = dict(geometry_statements='exact after output qualification',
              integration_tail='exact after diagnostic qualification and instantiation',
              earlier_methods='byte-identical', existing_header='byte-identical',
              other_source_files_unchanged=len(hashes)-2, changed_source_files=changed)
(evidence / 'source-verification.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
