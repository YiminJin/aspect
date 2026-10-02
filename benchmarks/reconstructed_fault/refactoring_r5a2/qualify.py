#!/usr/bin/env python3
"""Validate focused outcomes and freeze the review candidate and manifests."""
from pathlib import Path
import hashlib,json,shutil,subprocess
r=Path(__file__).resolve().parent;repo=r.parents[2];e=r/'evidence'
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for script in ('verify_move','verify_symbols','compare'):
 subprocess.run(['python3',str(r/(script+'.py'))],check=True)
labels=['configure','build-independent-projection','build-final-check','independent',
 'reference-plugin-configure','reference-plugin-build','plugin-configure','plugin-build']
for v in ('reference','candidate'):
 for rank in (1,2):
  for name in ('unit','ih','no-composition','surface','cache-remote','cache-cell'):
   labels.append(f'{v}-{name}-{rank}')
 labels.append(f'{v}-cold-warm-2')
labels += [f'candidate-{name}-{rank}' for rank in (1,2) for name in ('pressure','rollback-original')]
labels.append('candidate-restart')
for label in labels:
 expected=1 if label=='candidate-restart' else 0
 assert json.loads((e/(label+'.json')).read_text())['exit_code']==expected,label
for name in ('reference-protected-hashes','reference-checkpoint-source-hashes','local-hashes'):
 entries=json.loads((e/(name+'.json')).read_text())
 assert all(h(repo/p)==expected for p,expected in entries.items()),name
artifacts=json.loads((e/'reference-executed-artifacts.json').read_text())
# The only changed entry is the explicitly documented per-source build setup.
assert all(h(repo/p)==expected for p,expected in artifacts.items() if p!='CMakeLists.txt')
source={str(p.relative_to(repo)):h(p) for d in ('source','include','unit_tests') for p in (repo/d).rglob('*') if p.is_file()}
(e/'candidate-source-hashes.json').write_text(json.dumps(source,indent=2)+'\n')
binary=repo/'build-refactor-r5a2/aspect-release';qualified=binary.with_name('aspect-r5a2-qualified')
if not qualified.exists():shutil.copy2(binary,qualified)
assert h(binary)==h(qualified)
paths=[qualified,repo/'build-refactor-r5a1/aspect-r5a1-qualified',repo/'CMakeLists.txt']
paths += list((r/'inputs').glob('*.prm'))+list((r/'plugin-build').glob('*.so'))+list((r/'reference-plugin-build').glob('*.so'))
paths += [p for p in (repo/'tests').glob('reconstructed_fault_particle_projection_cache*') if p.is_file()]
paths += [repo/'tests/reconstructed_fault_particle_projection_cache/screen-output']
paths += [repo/'tests'/n for n in ('phase_field_fault_ih.cc','phase_field_fault_ih_cache.cc','phase_field_fault_surface_system.cc','phase_field_fault_pressure_gauge.cc','phase_field_fault_stage_i_rollback.cc','phase_field_fault_stage_j_restart_create.cc','phase_field_fault_stage_j_restart.cc','phase_field_fault_stage_j.cc')]
(e/'executed-artifacts.json').write_text(json.dumps({str(p.relative_to(repo)):h(p) for p in paths},indent=2)+'\n')
record=dict(reference_commit='c3ce532be86765c7c9edcacb2f44377ff62820e1',candidate='uncommitted R5a2 over accepted R5a1',candidate_binary=str(qualified.relative_to(repo)),candidate_sha256=h(qualified),
 moved_methods=10,moved_exclusive_helpers=2,independent_symbols=20,
 source_entries=len(source),artifact_entries=len(paths),source_checks=len(json.loads((e/'source-checks.json').read_text())),matched_checks=len(json.loads((e/'comparison.json').read_text())),
 completed_build_runtime_outcomes=len(labels),unit_assertions_per_rank=834,unit_cases=16,ranks=[1,2],
 cold_warm='two ranks: one cold rebuild, zero warm rebuilds; exact values/support',
 restart='preserved one-rank checkpoint restored; unchanged step-two nonconvergence',
 limitations='No exhaustive migration/empty-owner/equal-volume-domain lifecycle suite, 3D runtime or performance conclusion; no demonstrated MPI correctness defect.')
(e/'qualification.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
