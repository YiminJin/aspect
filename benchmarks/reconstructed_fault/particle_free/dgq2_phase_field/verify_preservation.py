#!/usr/bin/env python3
"""Read-only closeout verification against task entry and frozen build identity."""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
root = Path(__file__).resolve().parents[4]
out = root/'benchmarks/reconstructed_fault/particle_free/evidence/dgq2_phase_field'
entry = json.loads((out/'entry.json').read_text())
candidate = root/'build-particle-free/dgq2-phase/candidate'
identity = json.loads((candidate/'source-identity.json').read_text())

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20), b''):h.update(block)
    return h.hexdigest()

def verify(mapping):
    changed={}
    for name,expected in mapping.items():
        actual=digest(root/name) if (root/name).is_file() else 'missing'
        if actual!=expected:changed[name]=actual
    return changed

protected=verify(entry['protected'])
source=verify(identity['source_sha256'])
plugins=verify(identity['matching_plugins_and_benchmark_sources'])
unchanged = {name:h for name,h in entry['source_sha256'].items()
             if name.startswith('source/material_model/history/')
             or name in ['source/material_model/phase_field_fault/stress_reaction.cc',
                         'source/material_model/phase_field_fault/stress_evolution.cc',
                         'source/simulator/solver/reconstructed_fault_stokes.cc',
                         'source/material_model/rheology/fault_friction.cc',
                         'include/aspect/reconstructed_fault/objective_stress.h']}
critical=verify(unchanged)
assert digest(candidate/'aspect-release') == identity['core_sha256']
current={str(p.relative_to(root)):digest(p) for top in ('include','source','unit_tests')
         for p in (root/top).rglob('*') if p.is_file() and p.suffix in ('.h','.cc')}
changed={n:h for n,h in current.items() if entry['source_sha256'].get(n)!=h}
result=dict(utc=datetime.now(timezone.utc).isoformat(),
            branch=subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip(),
            head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
            protected_count=len(entry['protected']),protected_changes=protected,
            candidate_source_changes=source,candidate_support_changes=plugins,
            critical_stress_friction_sources=len(unchanged),critical_changes=critical,
            selected_source_changes=changed,core_sha256=identity['core_sha256'],
            cleanup_losses=entry['cleanup_losses'])
(out/'preservation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='selected_source_changes'},indent=2))
assert not protected and not source and not plugins and not critical
