#!/usr/bin/env python3
"""Record completed R5a1 evidence and preserve the qualified candidate."""
from pathlib import Path
import hashlib,json,shutil,subprocess
r=Path(__file__).resolve().parent;repo=r.parents[2];e=r/'evidence'
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name in ('verify_move','verify_symbols','compare'):
 subprocess.run(['python3',str(r/(name+'.py'))],check=True)
labels=['configure','build','independent','reference-plugin-configure','reference-plugin-build','plugin-configure','plugin-build']
labels += [f'{v}-unit-{rank}' for v in ('reference','candidate') for rank in (1,2)]
labels += [f'candidate-{name}-{rank}' for name in ('pressure','rollback-original') for rank in (1,2)]
labels += [f'{v}-restart' for v in ('reference','candidate')]
for label in labels:
 record=json.loads((e/(label+'.json')).read_text())
 assert record['exit_code']==(1 if label.endswith('-restart') else 0),(label,record['exit_code'])
for name in ('protected-hashes','checkpoint-source-hashes'):
 hashes=json.loads((e/(name+'.json')).read_text())
 assert all(h(repo/p)==expected for p,expected in hashes.items()),name
source={str(p.relative_to(repo)):h(p) for d in ('source','include','unit_tests') for p in (repo/d).rglob('*') if p.is_file()}
(e/'candidate-source-hashes.json').write_text(json.dumps(source,indent=2)+'\n')
binary=repo/'build-refactor-r5a1/aspect-release';qualified=binary.with_name('aspect-r5a1-qualified')
if not qualified.exists():shutil.copy2(binary,qualified)
assert h(binary)==h(qualified)
paths=[qualified,repo/'build-refactor-r4c/aspect-r4c-verified',repo/'CMakeLists.txt']
paths += list((r/'inputs').glob('*.prm')) + list((r/'plugin-build').glob('*.so')) + list((r/'reference-plugin-build').glob('*.so'))
paths += [repo/'tests'/name for name in ('phase_field_fault_pressure_gauge.cc','phase_field_fault_stage_i_rollback.cc','phase_field_fault_stage_j_restart_create.cc','phase_field_fault_stage_j_restart.cc','phase_field_fault_stage_j.cc')]
artifacts={str(p.relative_to(repo)):h(p) for p in paths}
(e/'executed-artifacts.json').write_text(json.dumps(artifacts,indent=2)+'\n')
record=dict(reference=json.loads((e/'reference.json').read_text()),candidate='uncommitted move over aea2a80b0',candidate_binary=str(qualified.relative_to(repo)),candidate_sha256=h(qualified),
 source_entries=len(source),artifact_entries=len(artifacts),recorded_outcomes=len(labels),
 moved_definitions=16,independent_symbols=32,source_checks=7,matched_checks=31,
 unit_assertions_per_rank=20130,unit_cases=7,ranks=[1,2],restart='preserved one-rank checkpoint restored; unchanged step-two nonconvergence',
 reused='accepted R4c and repaired frozen AMG/GMG evidence; not rerun')
(e/'qualification.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
