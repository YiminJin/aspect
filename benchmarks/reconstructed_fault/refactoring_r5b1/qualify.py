#!/usr/bin/env python3
"""Record source, artifact and matched-result qualification for R5b1 review."""
from pathlib import Path
import hashlib,json,shutil,subprocess
r=Path(__file__).resolve().parent;repo=r.parents[2];e=r/'evidence'
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for script in ('verify_move','verify_symbols','compare'):subprocess.run(['python3',str(r/(script+'.py'))],check=True)
labels=['configure','build-direct-type-include','independent','reference-plugin-configure','reference-plugin-build','plugin-configure','plugin-build','plugin-final-build','reference-supplement-configure','reference-supplement-build-configured','supplement-configure','supplement-build']
for v in ('reference','candidate'):
 for rank in (1,2):
  for name in ('unit','dynamic','adiabatic','rate','explicit','filter','singular','singular-current','pressure','rollback','bp3'):
   if v=='reference' and name in ('pressure','rollback'):continue
   labels.append(f'{v}-{name}-{rank}')
for label in labels:assert json.loads((e/(label+'.json')).read_text())['exit_code']==(1 if '-singular-' in label else 0),label
for name in ('r5a2-executed-artifacts','r5a2-reference-protected-hashes','r5a2-reference-checkpoint-source-hashes','r5a2-local-hashes'):
 entries=json.loads((e/(name+'.json')).read_text());assert all(h(repo/p)==expected for p,expected in entries.items() if p!='CMakeLists.txt'),name
assert subprocess.check_output(['git','diff','HEAD','--','tests'],cwd=repo)==b''
source={str(p.relative_to(repo)):h(p) for d in ('source','include','unit_tests') for p in (repo/d).rglob('*') if p.is_file()}
(e/'candidate-source-hashes.json').write_text(json.dumps(source,indent=2)+'\n')
binary=repo/'build-refactor-r5b1/aspect-release';qualified=binary.with_name('aspect-r5b1-qualified')
if not qualified.exists():shutil.copy2(binary,qualified)
assert h(binary)==h(qualified)
paths=[qualified,repo/'build-refactor-r5a2/aspect-r5a2-qualified',repo/'CMakeLists.txt']
paths+=list((r/'inputs').glob('*.prm'))+list((r/'plugin-build').rglob('*.so'))+list((r/'reference-plugin-build').rglob('*.so'))
paths += [r/'plugin/CMakeLists.txt',r/'plugin/singular_current_diagnostic.py',r/'plugin-build/singular_current_diagnostic.cc',r/'reference-plugin-build/singular_current_diagnostic.cc']
paths+=[repo/'tests'/n for n in ('phase_field_fault_surface_dynamic_pressure.cc','phase_field_fault_surface_system.cc','phase_field_fault_surface_singular_system.cc','phase_field_fault_surface_singular.cc','phase_field_fault_ih.cc','phase_field_fault_ih_cache.cc','phase_field_fault_pressure_gauge.cc','phase_field_fault_stage_i_rollback.cc')]
(e/'executed-artifacts.json').write_text(json.dumps({str(p.relative_to(repo)):h(p) for p in paths},indent=2)+'\n')
record=dict(reference_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),candidate='uncommitted R5b1 over accepted R5a',candidate_binary=str(qualified.relative_to(repo)),candidate_sha256=h(qualified),
 source_entries=len(source),artifact_entries=len(paths),source_checks=8,backend_symbols=4,dispatcher_symbols=2,
 matched_checks=len(json.loads((e/'comparison.json').read_text())),recorded_build_runtime_outcomes=len(labels),
 unit_assertions_per_rank=636,unit_cases=3,ranks=[1,2],expected_failures='four unchanged original stale-diagnostic fixture failures; four intentional supplementary singular-failure passes verifying invalidation',
 limitations='No R5b2 extraction, 3D runtime, long BP3/BP5 trajectory or dedicated detailed-native-diagnostic observer run; historical restart/scientific limitations unchanged.')
(e/'qualification.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
