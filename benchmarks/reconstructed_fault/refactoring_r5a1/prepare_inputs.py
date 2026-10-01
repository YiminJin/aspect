#!/usr/bin/env python3
"""Prepare matched small fixtures and private copies of a preserved checkpoint."""
from pathlib import Path
import hashlib,json,shutil
r=Path(__file__).resolve().parent;repo=r.parents[2];inputs=r/'inputs';inputs.mkdir(exist_ok=True)
for name in ('pressure','rollback-original'):
 for rank in (1,2):
  old=r.with_name('refactoring_r4c')/f'inputs/candidate-{name}-{rank}.prm'
  (inputs/old.name).write_text(old.read_text().replace('refactoring_r4c','refactoring_r5a1'))
src=r.with_name('restart_investigation')/'output-create-one'
hashes={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in src.rglob('*') if p.is_file()}
(r/'evidence/checkpoint-source-hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
for v in ('reference','candidate'):
 out=r/f'output-{v}-restart'
 # Refuse to replace an existing run or checkpoint branch.
 shutil.copytree(src,out)
 plugin='reference-plugin-build' if v=='reference' else 'plugin-build'
 s=(r.with_name('restart_fix')/'inputs/cohesive-resume-one.prm').read_text()
 s=s.replace('benchmarks/reconstructed_fault/refactoring_r3a/plugin-build',str((r/plugin).relative_to(repo)))
 s=s.replace('benchmarks/reconstructed_fault/restart_fix/output-cohesive-resume-one',str(out.relative_to(repo)))
 (inputs/f'{v}-restart.prm').write_text(s)
