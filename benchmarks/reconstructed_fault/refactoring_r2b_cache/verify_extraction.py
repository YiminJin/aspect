#!/usr/bin/env python3
"""Prove the moved decision and unchanged caller against proposal 2."""
from pathlib import Path
import hashlib,json,difflib
root=Path(__file__).resolve().parent
repo=root.parents[2]; evidence=root/'evidence'
path=repo/'source/material_model/phase_field_fault/normalization.cc'
before=(evidence/'normalization.cc.before').read_text(); after=path.read_text()
body=(evidence/'reuse-body.before').read_text()
assert after.count(body)==1
start=after.index('    /** Inputs and decision for this preparation only;')
end=after.index('    template <int dim>\n    void\n    PhaseFieldFault<dim>::compute_normalization_integrals()',start)
restored=after[:start]+after[end:]
call='''      auto [phase_block, owned_indices, phase_values, fault_versions, fault_vertices,
            surface_compositions, composition_independent, global_hit, key_end, key_mpi_end] =
        prepare_normalization_reuse(previous_cache_valid);
'''
assert restored.count(call)==1
restored=restored.replace(call,body)
inst='    template PhaseFieldFault<dim>::NormalizationReuseDecision \\\n      PhaseFieldFault<dim>::prepare_normalization_reuse(const bool) const; \\\n'
assert restored.count(inst)==1
restored=restored.replace(inst,'')
assert restored==before, 'Caller/invalidation/publication or unrelated implementation changed'
entry=json.loads((evidence/'entry-source-hashes.json').read_text())
changed=[name for name,h in entry.items() if hashlib.sha256((repo/name).read_bytes()).hexdigest()!=h]
assert set(changed)=={'source/material_model/phase_field_fault/normalization.cc','include/aspect/material_model/phase_field_fault.h'},changed
for name in ('normalization.cc','phase_field_fault.h'):
 p=path if name=='normalization.cc' else repo/'include/aspect/material_model/phase_field_fault.h'
 (evidence/(name+'.diff')).write_text(''.join(difflib.unified_diff((evidence/(name+'.before')).read_text().splitlines(True),p.read_text().splitlines(True),fromfile=name+'.proposal2',tofile=str(p.relative_to(repo)))))
record=dict(moved_body_exact=True,caller_restored_exact=True,changed_source_files=changed,other_entry_files_unchanged=len(entry)-len(changed))
(evidence/'source-verification.json').write_text(json.dumps(record,indent=2)+'\n')
print(record)
