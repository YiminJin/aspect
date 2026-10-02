#!/usr/bin/env python3
"""Freeze the tested candidate and record exact source/input/plugin provenance."""
from pathlib import Path
import hashlib,json,shutil,subprocess
r=Path(__file__).resolve().parent;repo=r.parents[2];e=r/'evidence'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name in ('verify_source','verify_symbols','compare'):
 subprocess.run(['python3',str(r/(name+'.py'))],cwd=repo,check=True)
cases=[f'bp3-{m}-{n}' for m,n in [('off',1),('on',1),('off',2),('on',2),('blocked',2)]]+[f'rollback-{m}-{n}' for m in ('off','on') for n in (1,2)]
labels=['configure','build','build-final','plugin-configure','plugin-build','independent','independent-final']+[v+'-'+c for v in ('reference','candidate') for c in cases]
for label in labels:assert json.loads((e/(label+'.json')).read_text())['exit_code']==0,label
# Only the current build list and rolling inventory deliberately change R6a artifacts.
prior=json.loads((r.with_name('refactoring_r6a')/'evidence/executed-artifacts.json').read_text())
exceptions={'CMakeLists.txt','benchmarks/reconstructed_fault/refactoring_r6a/switch_inventory.md'}
assert all(digest(repo/p)==h for p,h in prior.items() if p not in exceptions)
binary=repo/'build-refactor-r6b/aspect-release';qualified=binary.with_name('aspect-r6b-qualified')
if not qualified.exists():shutil.copy2(binary,qualified)
assert digest(binary)==digest(qualified)
source={str(p.relative_to(repo)):digest(p) for d in ('source','include','unit_tests') for p in (repo/d).rglob('*') if p.is_file()}
(e/'candidate-source-hashes.json').write_text(json.dumps(source,indent=2)+'\n')
paths=[qualified,repo/'build-refactor-r6a/aspect-r6a-qualified',repo/'CMakeLists.txt']
paths+=list((r/'inputs').glob('*.prm'))+list((r/'plugin-build').rglob('*.so'))+list((r.with_name('refactoring_r6a')/'plugin-build').rglob('*.so'))+list(r.glob('*.py'))
paths+=[repo/'tests/phase_field_fault_stage_i_rollback.cc',r.with_name('refactoring_boundary')/'inputs/auto-bp3-base.prm',r.with_name('refactoring_r6a')/'switch_inventory.md']
(e/'executed-artifacts.json').write_text(json.dumps({str(p.relative_to(repo)):digest(p) for p in paths},indent=2)+'\n')
record=dict(reference_commit=subprocess.check_output(['git','rev-parse','1a3b57eda'],cwd=repo,text=True).strip(),candidate='uncommitted R6b over accepted R6a',candidate_binary=str(qualified.relative_to(repo)),candidate_sha256=digest(qualified),source_entries=len(source),artifact_entries=len(paths),source_checks=len(json.loads((e/'source-checks.json').read_text())),symbols=9,runtime_cases=18,comparisons=len(json.loads((e/'comparison.json').read_text())),build_runtime_outcomes=len(labels),ranks=[1,2],limitations='No Debug/3D runtime, long/restart campaign, mid-write failure injection, R6c or R7. Repaired frozen evidence reused; unrelated scientific/cache limitations unchanged.')
(e/'qualification.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
