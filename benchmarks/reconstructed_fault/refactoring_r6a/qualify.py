#!/usr/bin/env python3
"""Record the tested source, input, plugin and immutable binary qualification."""
from pathlib import Path
import hashlib,json,shutil,subprocess
r=Path(__file__).resolve().parent;repo=r.parents[2];e=r/'evidence'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for n in ('verify_source','verify_symbols','compare'):subprocess.run(['python3',str(r/(n+'.py'))],cwd=repo,check=True)
cases=[f'bp3-{mode}-{rank}' for mode,rank in [('off',1),('on',1),('off',2),('on',2),('missing',1),('blocked',2)]]+[f'rollback-{mode}-{rank}' for mode in ('off','on') for rank in (1,2)]
labels=['configure','build','reference-plugin-configure','reference-plugin-build','plugin-configure','plugin-build','independent']+[v+'-'+c for v in ('reference','candidate') for c in cases]
for label in labels:assert json.loads((e/(label+'.json')).read_text())['exit_code']==0,label
selectors=json.loads((e/'switch-readers.json').read_text());inventory=(r/'switch_inventory.md').read_text();assert all(n in inventory for n in selectors)
binary=repo/'build-refactor-r6a/aspect-release';qualified=binary.with_name('aspect-r6a-qualified')
if not qualified.exists():shutil.copy2(binary,qualified)
assert digest(binary)==digest(qualified)
source={str(p.relative_to(repo)):digest(p) for d in ('source','include','unit_tests') for p in (repo/d).rglob('*') if p.is_file()}
(e/'candidate-source-hashes.json').write_text(json.dumps(source,indent=2)+'\n')
paths=[qualified,repo/'build-refactor-r5b2/aspect-r5b2-qualified',repo/'CMakeLists.txt']
paths+=list((r/'inputs').glob('*.prm'))+list((r/'plugin-build').rglob('*.so'))+list((r/'reference-plugin-build').rglob('*.so'))
paths+=list(r.glob('*.py'))+list((r/'plugin').glob('*'))+[r/'switch_inventory.md']
paths+=[repo/'tests/phase_field_fault_stage_i_rollback.cc',r.with_name('refactoring_boundary')/'inputs/auto-bp3-base.prm']
(e/'executed-artifacts.json').write_text(json.dumps({str(p.relative_to(repo)):digest(p) for p in paths},indent=2)+'\n')
record=dict(reference_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=repo).strip(),candidate='uncommitted R6a over accepted post-R5',candidate_binary=str(qualified.relative_to(repo)),candidate_sha256=digest(qualified),source_entries=len(source),artifact_entries=len(paths),source_checks=len(json.loads((e/'source-checks.json').read_text())),helper_symbols=12,comparisons=len(json.loads((e/'comparison.json').read_text())),runtime_cases=20,build_runtime_outcomes=len(labels),ranks=[1,2],literal_selectors=len(selectors),production_selectors=sum(any(a['file'].startswith(('source/','include/')) for a in v) for v in selectors.values()),rows_per_accepted_update=dict(stress_cycle=16875,continued_source=146),accepted_updates=6,limitations='No R6b/R6c or R7; no Debug/3D runtime, new restart/long trajectory, induced mid-write I/O failure or late history-validation failure. Existing rollback and open-failure paths covered. Repaired frozen evidence reused; original singular fixture and historical scientific/cache limitations unchanged.')
(e/'qualification.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
