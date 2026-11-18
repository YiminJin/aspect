#!/usr/bin/env python3
"""Stage 4: the selected 90-minute fine endpoint, with immutable parent/binaries."""
import datetime,gzip,hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/stage4_rsf'
label='stage4-free-turn-fine-endpoint'
run=ROOT/'build-particle-free/bulk-tip/runs'/label
assert not run.exists(), 'Never overwrite retained evidence'
entry=json.loads((OUT/'entry.json').read_text())
for name,expected in entry['protected'].items():
 with (ROOT/name).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==expected,name
parent=ROOT/'build-particle-free/bulk-tip/runs/overnight-free-turn-c-half/output'
env=os.environ.copy()
for k in list(env):
 if k.startswith('ASPECT_TIP_'):env.pop(k)
env.update(ASPECT_TIP_POSITIVE_METRIC_FIRST='1',ASPECT_TIP_POINT_EXPORT_STRIDE='9999',ASPECT_TIP_RESTORE_AUDIT='1')
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','DEAL_II_NUM_THREADS']:env[k]='1'
cmd=['python3',str(ROOT/'benchmarks/reconstructed_fault/particle_free/bulk_tip/run.py'),label,
 '--fixture',str(ROOT/'benchmarks/reconstructed_fault/particle_free/overnight_free_turn/fine-normal.prm'),
 '--executable',str(ROOT/'build-particle-free/first-free-turn/tested/aspect-release'),
 '--checker',str(ROOT/'build-particle-free/first-free-turn/tested/libbulk_tip_checks.release.so'),
 '--ranks','4','--bind-cores','--dt','.0025','--end','.08','--time-update','midpoint',
 '--phase-balance','--checkpoint-steps','1','--integrity-stop','--seconds','5400','--wall-hours','1.475',
 '--resume-from',str(parent),'--snapshot-id','2','--copy-selected-snapshot-only']
(OUT/'endpoint-launch.json').write_text(json.dumps(dict(command=cmd,start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),environment={k:v for k,v in env.items() if k.startswith(('ASPECT_TIP_','OMP_','OPENBLAS_','MKL_','DEAL_II_'))}),indent=2)+'\n')
code=subprocess.run(cmd,cwd=ROOT,env=env).returncode
archived=[]
for pattern in ['tip-points-*.csv','tip-balance-*.csv','restore-points-*.csv','restore-phase-*.csv']:
 for src in sorted((run/'output').glob(pattern)):
  dest=src.with_suffix('.csv.gz');assert not dest.exists();h=hashlib.sha256()
  with src.open('rb') as f,gzip.open(dest,'wb',compresslevel=6) as g:
   for block in iter(lambda:f.read(1048576),b''):h.update(block);g.write(block)
  with gzip.open(dest,'rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h.hexdigest()
  archived.append(dict(file=str(dest.relative_to(ROOT)),raw_bytes=src.stat().st_size,gzip_bytes=dest.stat().st_size,raw_sha256=h.hexdigest()));src.unlink()
(OUT/'endpoint-compression.json').write_text(json.dumps(archived,indent=2)+'\n')
raise SystemExit(code)
