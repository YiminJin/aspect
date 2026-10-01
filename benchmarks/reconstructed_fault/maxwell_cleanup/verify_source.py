#!/usr/bin/env python3
"""Prove the bounded rename preserves all other entry C++ source."""
from pathlib import Path
import hashlib,json,re
root=Path(__file__).resolve().parent
repo=root.parents[2]
e=root/'evidence'
entry=json.loads((e/'entry-source-hashes.json').read_text())
selected=json.loads((e/'renamed-files.json').read_text())
pattern=re.compile(r'"(?:\\.|[^"\\])*"|\bkappa\b')
for name in selected:
    old=(e/'before'/name).read_text()
    expected=pattern.sub(lambda m:'eta_ve' if m[0]=='kappa' else m[0],old)
    if name.endswith('phase_field_fault/constitutive.cc'):
        expected=expected.replace('positive dt and kappa.','positive dt and eta_ve.').replace('effective viscosity kappa must','viscoelastic viscosity eta_ve must')
    if name.endswith('material_model/phase_field_fault.h'):
        expected=expected.replace('          double eta_ve;', '          /** Viscoelastic viscosity eta*(1-beta), evaluated with expm1. */\n          double eta_ve;',1)
    if name.endswith('phase_field_fault_stage_j_temperature.cc'):
        expected=expected.replace('kappa_Gamma','eta_ve_Gamma')
    assert (repo/name).read_text()==expected,name
for name,h in entry.items():
    if name not in selected:
        assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==h,name
reference=repo/'build-refactor-r3b/aspect-r3b-qualified'
assert hashlib.sha256(reference.read_bytes()).hexdigest()=='fbd3a74e0238c0fcf5026c9ec8b554f5296302fe14f3d0de4d986096d2c109db'
record=dict(renamed_files=len(selected),protected_Cpp_files=len(entry)-len(selected),
            original_frozen_stress_implementation_restored=True,
            diagnostic_column_names_unchanged=True,qualified_R3b_unchanged=True)
(e/'source-verification.json').write_text(json.dumps(record,indent=2)+'\n')
print(record)
