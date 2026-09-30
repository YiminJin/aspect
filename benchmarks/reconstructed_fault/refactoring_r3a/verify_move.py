#!/usr/bin/env python3
"""Check exact method/helper text, remaining entry code and protected source hashes."""
from pathlib import Path
import hashlib,json,re
root=Path(__file__).resolve().parent; repo=root.parents[2]; e=root/'evidence'
before=(e/'phase_field_fault.cc.before').read_text()
after=(repo/'source/material_model/phase_field_fault.cc').read_text()
spans=json.loads((e/'moved-methods.json').read_text())
files={name:(repo/f'source/material_model/phase_field_fault/{name}.cc').read_text() for name in ('constitutive','history')}
for s in spans:
 body=before[s['start']:s['end']]
 assert files[s['destination']].count(body)==1,s['name']
 assert not re.search(r'PhaseFieldFault<dim>::\s*'+s['name']+r'\(',after),s['name']
# Every file-local helper retains its complete definition exactly once.
for name,dest in [('cohesive_history_is_initialized','history'),('fault_scalar_property_is_initialized','history'),('validate_initial_fault_state_mapping','history'),('interpolate_surface_chemical_compositions','history'),('interpolate_fault_scalar','constitutive')]:
 m=re.search(r'\n    '+name+r'\(',before)
 start=before.rfind('    template <int dim>\n',0,m.start())
 end=before.index('\n    }',m.end())+len('\n    }')
 assert files[dest].count(before[start:end])==1,name
start=before.index('    void\n    throw_if_history_error(')
end=before.index('\n    }',start)+len('\n    }')
assert files['history'].count(before[start:end])==1
# Reconstruct the intended entry file without accepting changes to retained methods.
helper_start=before.index('  // -----------------------------------------------------------------------------')
helper_end=before.index('  namespace MaterialModel\n')
expected=before
for start,end in sorted([(s['start'],s['end']) for s in spans]+[(helper_start,helper_end)],reverse=True):expected=expected[:start]+expected[end:]
expected=expected.replace('    using aspect::internal::throw_if_history_error;\n','')
for title in ('Maxwell constitutive law','Cohesive constitutive law','Initial cohesive-state initialization'):
 expected=expected.replace('    // -----------------------------------------------------------------------------\n    // '+title+'\n    // -----------------------------------------------------------------------------\n','')
expected=re.sub(r'\n{5,}','\n\n\n\n',expected)
assert after==expected
entry=json.loads((e/'entry-source-hashes.json').read_text())
changed=[name for name,h in entry.items() if hashlib.sha256((repo/name).read_bytes()).hexdigest()!=h]
assert changed==['source/material_model/phase_field_fault.cc'],changed
record=dict(exact_moved_methods=len(spans),exact_helpers=6,remaining_entry_exact=True,protected_files_unchanged=len(entry)-1,protected_R2_files=['normalization.cc','boundary_completion.cc'],header_declarations_unchanged=True)
(e/'source-verification.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
