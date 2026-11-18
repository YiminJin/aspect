#!/usr/bin/env python3
"""One native fixed-phase interval and six read-only restoring-response actions."""
from pathlib import Path
import os,subprocess,json,datetime,gzip,hashlib
ROOT=Path(__file__).resolve().parents[4];OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/stage4_rsf'
label='stage4-fixed-phase-stiffness-homogeneous';run=ROOT/'build-particle-free/bulk-tip/runs'/label
assert not run.exists()
env=os.environ.copy()
for k in list(env):
 if k.startswith('ASPECT_TIP_'):env.pop(k)
env.update(ASPECT_TIP_POSITIVE_METRIC_FIRST='1',ASPECT_TIP_POINT_EXPORT_STRIDE='9999',ASPECT_TIP_RESTORING_RESPONSE='1')
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','DEAL_II_NUM_THREADS']:env[k]='1'
cmd=['python3',str(ROOT/'benchmarks/reconstructed_fault/particle_free/bulk_tip/run.py'),label,
 '--fixture',str(ROOT/'benchmarks/reconstructed_fault/particle_free/first_free_turn/positive.prm'),
 '--executable',str(ROOT/'build-particle-free/first-free-turn/tested/aspect-release'),
 '--checker',str(ROOT/'build-particle-free/stage4-rsf/libbulk_tip_checks.release.so'),
 '--ranks','4','--bind-cores','--dt','.0025','--end','.08','--end-step','16','--time-update','midpoint',
 '--fixed-phase','--phase-balance','--checkpoint-steps','1','--seconds','2400','--wall-hours','.625',
 '--resume-from',str(ROOT/'build-particle-free/bulk-tip/runs/first-turn-positive-fourcore-end/output'),
 '--snapshot-id','1','--copy-selected-snapshot-only','--restart-dt','.0025']
(OUT/'stiffness-launch.json').write_text(json.dumps(dict(command=cmd,start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),environment={k:v for k,v in env.items() if k.startswith(('ASPECT_TIP_','OMP_','OPENBLAS_','MKL_','DEAL_II_'))}),indent=2)+'\n')
code=subprocess.run(cmd,cwd=ROOT,env=env).returncode
records=[]
for src in sorted((run/'output').glob('tip-balance-*.csv')):
 dest=src.with_suffix('.csv.gz');assert not dest.exists();h=hashlib.sha256()
 with src.open('rb') as f,gzip.open(dest,'wb',compresslevel=6) as g:
  for block in iter(lambda:f.read(1048576),b''):h.update(block);g.write(block)
 with gzip.open(dest,'rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h.hexdigest()
 records.append(dict(file=str(dest.relative_to(ROOT)),raw_sha256=h.hexdigest()));src.unlink()
(OUT/'stiffness-compression.json').write_text(json.dumps(records,indent=2)+'\n')
raise SystemExit(code)
