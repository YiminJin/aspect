#!/usr/bin/env python3
"""Check pinned assessment evidence; exit zero does not mean P3 qualification."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];E=HERE/'accuracy_energy_evidence'
r=json.loads((E/'manifest.json').read_text())
for name,digest in r['hashes'].items():
 assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest, name
n=json.loads((E/'normalization.json').read_text());s=json.loads((E/'stress.json').read_text());p=json.loads((E/'profile-energy.json').read_text())
strong=[a for a in n if a['state']=='strong' and a['spacing']==.025]
assert len(strong)==3 and all(a['reference_projection']['max_source_error']>.005 for a in strong)
assert all(a['min_I']>0 and a['constant_error']<2e-14 and a['conservation_error']<2e-15 for a in n)
assert s['native_updates']==336 and all(a['max_particle_error_Pa']<2e-10 and a['max_FE_drift_Pa']<2e-10 for a in s['patterns'].values())
assert all(a['max_bulk_identity_defect']<1e-8 and a['max_fault_identity_defect']<1e-8 for a in p['retained_trajectories'].values())
q=json.loads((E/'qualification.json').read_text())
assert q['direct_irreversibility_fixed_mesh']=='verified' and q['P3_ready'] is False and q['fresh_finer_coupled_runs']==0
print(f"{len(r['hashes'])} artifact hashes checked. Direct irreversibility verified; surface accuracy fails; profile energy unresolved; P3 not ready.")
