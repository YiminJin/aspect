#!/usr/bin/env python3
"""Record the exact local artifacts and distinguish incoming edits from R7b evidence."""
from pathlib import Path
import hashlib, json, shutil, subprocess
r=Path(__file__).resolve().parent;repo=r.parents[2]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
baseline=json.loads((r/'evidence/baseline.json').read_text())
assert sha(r/baseline['baseline_binary'])==baseline['baseline_sha256']
binary=repo/'build-refactor-post-r6-gcc12-unity/aspect'
frozen=r/'build/aspect-r7b-qualified'
if not frozen.exists():shutil.copy2(binary,frozen)
assert sha(binary)==sha(frozen)
updated_docs={'doc/reconstructed_fault/'+p for p in ['CURRENT_STATUS.md','refactor_review.md','refactoring.md','refactoring/refactoring_plan.md']}
untouched={p:sha(repo/p)==value for p,value in baseline['incoming'].items() if p not in updated_docs}
assert all(untouched.values()),[p for p,ok in untouched.items() if not ok]
assert not subprocess.check_output(['git','diff','HEAD','--','source','include','CMakeLists.txt'],cwd=repo)
assert not subprocess.check_output(['git','diff','0f66d9869','HEAD','--','source','include'],cwd=repo)
paths=[r/'build/reference-aspect',frozen,r/'build/debug/debug_fault',
       repo/'build-refactor-post-r6-gcc12-unity/CMakeCache.txt',
       repo/'benchmarks/reconstructed_fault/post_r6_cleanup/build-gcc12/maintained/libbp3_restore_150x50.release.so',
       repo/'benchmarks/reconstructed_fault/post_r6_cleanup/build-gcc12/maintained/CMakeCache.txt']
paths+=list((r/'build/plugin').glob('*.so'))
paths+=list((r/'inputs').glob('*.prm'))+list((r/'plugin').glob('*'))+list((r/'debug').glob('*'))
paths+=list(r.glob('*.py'))
manifest={'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
 'reference_revision':'0f66d9869',
 'stack':'GCC 12.4.0; OpenMPI 5.0.6; deal.II 9.6.2; Trilinos 14.2 Epetra; Release unity/PCH ON',
 'source_include_unchanged_from_reference':True,'tracked_core_unchanged_from_HEAD':True,
 'preserved_incoming_files':untouched,'updated_documents':sorted(updated_docs),
 'artifacts':{str(p.relative_to(repo)):sha(p) for p in paths if p.is_file()}}
(r/'evidence/qualified-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Qualified executable:',sha(frozen));print('Unrelated incoming files preserved:',len(untouched))
