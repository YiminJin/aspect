#!/usr/bin/env python3
"""Audit protected inputs, source baseline, and fine continuation restoration."""
from pathlib import Path
import hashlib,json,subprocess,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];OUT=HERE.parent/'evidence/stage4_rsf'
sys.path.insert(0,str(HERE.parent/'overnight_free_turn'))
from observations import native,read,gather,np,RUNS
entry=json.loads((OUT/'entry.json').read_text())
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==entry['head']
checks={}
for name,expected in entry['protected'].items():
 with (ROOT/name).open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
 assert actual==expected,name
 checks[name]=actual
# The source-side incoming patch must be identical. Documentation changes are
# selected by this task; none of the incoming source/benchmark edits are ours.
def chunks(s):
 return {'diff --git '+x.splitlines()[0]: 'diff --git '+x for x in s.split('diff --git ')[1:]}
before=chunks((OUT/'entry.patch').read_text());after=chunks(subprocess.check_output(['git','diff','--binary'],cwd=ROOT,text=True))
kept=[]
for key,value in before.items():
 if ' b/doc/' in key:continue
 assert after.get(key)==value,key
 kept.append(key)
for key in after:
 if ' b/doc/' not in key:assert key in before,key
p=RUNS/'overnight-free-turn-c-half/output';q=RUNS/'stage4-free-turn-fine-endpoint/output'
old=native(p,16);restored=gather(q,'restore-phase-r',['cell','i'])
assert np.array_equal(old.cell,restored.cell) and np.array_equal(old.i,restored.i)
phase=float(max(abs(old.B-restored.B)));solution=float(max(abs(restored.B-restored.solution_B)))
a,b=read(p/'tip-sections-0016.csv'),read(q/'restore-sections.csv')
section={key:float(max(abs(a[key]-b[key]))) for key in b.dtype.names}
assert phase==0 and solution==0 and max(section.values())==0
parents={}
for src in (p/'restart/02').iterdir():
 if src.is_file():
  target=q/'restart/02'/src.name
  with src.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
  with target.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h
  parents[src.name]=h
source_parents={}
for label in ['stage4-free-turn-fine-endpoint','stage4-fixed-phase-stiffness-homogeneous']:
 manifest=json.loads((RUNS/label/'command.json').read_text())
 source=Path(manifest['arguments']['resume_from'])
 for rel,expected in manifest['restart_source_sha256'].items():
  with (source/rel).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==expected
 source_parents[label]=len(manifest['restart_source_sha256'])
manifest=json.loads((RUNS/'stage4-fixed-phase-stiffness-homogeneous/command.json').read_text())
with Path(manifest['arguments']['checker']).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==manifest['checker_sha256']
record=dict(original_restart_files_rechecked=source_parents,final_checker_sha256=manifest['checker_sha256'],protected_sha256=checks,unchanged_incoming_non_document_diff=kept,copied_parent_sha256=parents,
 fine_restore=dict(phase_error=phase,solution_phase_error=solution,section_errors=section),
 point_limit='The sparse parent has no full point frame at accepted16. Serialized checkpoint copies are byte-checked; pointwise tensor equality is not claimed.',
 cleanup='No old retained data deleted. Existing cleanup losses and broken historical aliases are unchanged. Only new closed CSVs were losslessly compressed, with verified roundtrip hashes.')
(OUT/'preservation.json').write_text(json.dumps(record,indent=2)+'\n')
print('Protected source/binaries/checkpoints and fine restore: PASS')
