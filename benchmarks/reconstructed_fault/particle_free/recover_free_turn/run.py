#!/usr/bin/env python3
"""Run one selected recovery window; preserve inputs and compress only after MPI."""
import argparse,os,json,subprocess,gzip,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[4]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('label');p.add_argument('--executable',type=Path,required=True)
p.add_argument('--resume-from',type=Path);p.add_argument('--snapshot-id',type=int)
p.add_argument('--dt',type=float,default=.005);p.add_argument('--end-step',type=int);p.add_argument('--point-stride',type=int)
p.add_argument('--reflected',action='store_true');p.add_argument('--trace',action='store_true')
p.add_argument('--stop-iteration',type=int,default=40);p.add_argument('--seconds',type=int,default=7200)
a=p.parse_args();evidence=root/'benchmarks/reconstructed_fault/particle_free/evidence/recover_free_turn_step16'
evidence.mkdir(exist_ok=True);run=root/'build-particle-free/bulk-tip/runs'/a.label
assert not run.exists()
env=os.environ.copy()
for key in list(env):
 if key.startswith('ASPECT_TIP_'):env.pop(key)
env.update(ASPECT_TIP_POSITIVE_METRIC_FIRST='1',ASPECT_TIP_POINT_EXPORT_STRIDE=str(a.point_stride or (1 if a.resume_from else 4)))
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','DEAL_II_NUM_THREADS']:env[key]='1'
if a.trace:
 env.update(ASPECT_TIP_TRACE_SEARCH='1',ASPECT_TIP_TRACE_ROWS='1',ASPECT_TIP_TRACE_STOP_ITERATION=str(a.stop_iteration))
if a.resume_from:env['ASPECT_TIP_RESTORE_AUDIT']='1'
fixture=root/'benchmarks/reconstructed_fault/particle_free/first_free_turn'/('reflected.prm' if a.reflected else 'positive.prm')
cmd=['python3',str(root/'benchmarks/reconstructed_fault/particle_free/bulk_tip/run.py'),a.label,'--fixture',str(fixture),
 '--executable',str(a.executable.resolve()),'--checker',str(root/'build-particle-free/first-free-turn/tested/libbulk_tip_checks.release.so'),
 '--ranks','4','--bind-cores','--dt',str(a.dt),'--end','.08','--time-update','midpoint',
 '--phase-balance','--checkpoint-steps','1','--integrity-stop','--seconds',str(a.seconds)]
if a.resume_from:
 cmd+=['--resume-from',str(a.resume_from.resolve()),'--copy-selected-snapshot-only']
 if a.snapshot_id:cmd+=['--snapshot-id',str(a.snapshot_id)]
 if a.dt!=.005:cmd+=['--restart-dt',str(a.dt)]
if a.end_step:cmd+=['--end-step',str(a.end_step)]
(evidence/(a.label+'-launch.json')).write_text(json.dumps({'command':cmd,'environment':{k:v for k,v in env.items() if k.startswith(('ASPECT_TIP_','OMP_','OPENBLAS_','DEAL_II_'))}},indent=2)+'\n')
code=subprocess.run(cmd,env=env).returncode
records=[]
for pattern in ['tip-points-*.csv','tip-balance-*.csv','tip-seed-fracture-*.csv','restore-points-*.csv','restore-phase-*.csv','tip-search-points-*.csv']:
 for src in sorted((run/'output').glob(pattern)):
  dest=src.with_suffix('.csv.gz');assert not dest.exists()
  digest=hashlib.sha256()
  with src.open('rb') as f,gzip.open(dest,'wb',compresslevel=6) as g:
   for block in iter(lambda:f.read(1048576),b''):digest.update(block);g.write(block)
  with gzip.open(dest,'rb') as g:assert hashlib.file_digest(g,'sha256').hexdigest()==digest.hexdigest()
  records.append({'file':str(dest.relative_to(root)),'raw_sha256':digest.hexdigest(),'raw_bytes':src.stat().st_size,'gzip_bytes':dest.stat().st_size})
  src.unlink()
(evidence/(a.label+'-compression.json')).write_text(json.dumps(records,indent=2)+'\n')
raise SystemExit(code)
