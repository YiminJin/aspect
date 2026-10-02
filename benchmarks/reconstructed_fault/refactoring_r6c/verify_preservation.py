#!/usr/bin/env python3
"""Verify this assessment leaves the accepted implementation/artifacts intact."""
from pathlib import Path
import hashlib,json,re,subprocess
r=Path(__file__).resolve().parent;repo=r.parents[2];e=r/'evidence'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
checks={};counts={}
exceptions={'benchmarks/reconstructed_fault/refactoring_r6a/switch_inventory.md'}
for label,path in [('assessment-baseline',e/'reference-hashes.json'),('qualified-source',r.with_name('refactoring_r6b')/'evidence/candidate-source-hashes.json'),('qualified-artifacts',r.with_name('refactoring_r6b')/'evidence/executed-artifacts.json')]:
 data=json.loads(path.read_text());counts[label]=len(data)
 failures=[p for p,h in data.items() if p not in exceptions and (not (repo/p).is_file() or digest(repo/p)!=h)]
 checks[label]=not failures
 if failures:print(label,failures)
allowed={'doc/reconstructed_fault/refactoring.md','doc/reconstructed_fault/refactoring/refactoring_plan.md','doc/reconstructed_fault/CURRENT_STATUS.md','doc/reconstructed_fault/refactor_review.md'}|exceptions
changed=subprocess.check_output(['git','diff','--name-only','00ad5ce1c'],cwd=repo,text=True).splitlines()
checks['tracked-diff-documentation-only']=set(changed)<=allowed
baseline=json.loads((e/'baseline.json').read_text())
checks['qualified-binary']=digest(repo/baseline['qualified_binary'])==baseline['sha256']
checks['environment-guard']=json.loads((e/'environment-guard.json').read_text())['exit_code']==0
missing=[]
for p in [r/'README.md']+[repo/p for p in changed]:
 for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
  if '://' in target or target.startswith('#'):continue
  target=target.split('#')[0]
  if target and not (p.parent/target).exists():missing.append((str(p),target))
# Historical entries may retain references to archived runtime evidence. Only
# assess links introduced in this pass; do not rewrite those historical records.
newtext=subprocess.check_output(['git','diff','--unified=0','00ad5ce1c'],cwd=repo,text=True)
newtargets={t for line in newtext.splitlines() if line.startswith('+') for t in re.findall(r'\]\(([^)]+)\)',line)}
missing=[(p,t) for p,t in missing if p==str(r/'README.md') or t in newtargets]
checks['new-document-links']=not missing
if missing:print('missing links',missing)
subprocess.run(['git','diff','--check'],cwd=repo,check=True)
checks['whitespace']=True
(e/'verification.json').write_text(json.dumps({'checks':checks,'counts':counts,'numerical_runs':'reused R6b; no new R6c numerical/MPI runs'},indent=2)+'\n')
print(json.dumps({'checks':checks,'counts':counts},indent=2));assert all(checks.values())
