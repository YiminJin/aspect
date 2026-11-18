#!/usr/bin/env python3
"""One selected physical window, bounded by the single overnight deadline."""
import argparse,csv,datetime,gzip,hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/overnight_free_turn'
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('case',choices=['a','a-tail','b-prefix','b-half','c-pilot','c-prefix','c-half'])
p.add_argument('--seconds',type=int,required=True)
p.add_argument('--resume-from',type=Path);p.add_argument('--snapshot-id',type=int)
p.add_argument('--end-step',type=int);p.add_argument('--point-stride',type=int)
a=p.parse_args()
budget=json.loads((OUT/'budget.json').read_text())
cutoff=datetime.datetime.fromisoformat(budget['simulation_cutoff_utc']).timestamp()
seconds=min(a.seconds,int(cutoff-time.time()-90));assert seconds>120,'Closeout reserve reached'
label='overnight-free-turn-'+a.case
run=ROOT/'build-particle-free/bulk-tip/runs'/label
assert not run.exists()
fixture=ROOT/'benchmarks/reconstructed_fault/particle_free/first_free_turn/positive.prm'
if a.case.startswith('b'):fixture=Path(__file__).with_name('fixed-offaxis.prm')
if a.case.startswith('c'):fixture=Path(__file__).with_name('fine-normal.prm')
dt=.00125 if a.case.startswith('a') else .0025 if a.case.endswith('half') else .005
parent=a.resume_from
if a.case=='a':parent=ROOT/'build-particle-free/bulk-tip/runs/first-turn-positive-fourcore-end/output'
if a.case=='a-tail':assert parent is not None,'Continuation requires its actual accepted parent'
if a.case.endswith('half') and parent is None:
 parent=ROOT/'build-particle-free/bulk-tip/runs'/('overnight-free-turn-'+a.case[0]+'-prefix')/'output'
if a.case.endswith('half'):
 with (parent/'bulk-accepted.csv').open() as f:last=list(csv.DictReader(f))[-1]
 assert int(last['step'])==15 and abs(float(last['time'])-.075)<1e-13,'Half-step continuation requires its own accepted .075 parent'
env=os.environ.copy()
for k in list(env):
 if k.startswith('ASPECT_TIP_'):env.pop(k)
env.update(ASPECT_TIP_POSITIVE_METRIC_FIRST='1',ASPECT_TIP_POINT_EXPORT_STRIDE=str(a.point_stride or (1 if parent else 15)))
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','DEAL_II_NUM_THREADS']:env[k]='1'
if parent:env['ASPECT_TIP_RESTORE_AUDIT']='1'
cmd=['python3',str(ROOT/'benchmarks/reconstructed_fault/particle_free/bulk_tip/run.py'),label,'--fixture',str(fixture),
 '--executable',str(ROOT/'build-particle-free/first-free-turn/tested/aspect-release'),
 '--checker',str(ROOT/'build-particle-free/first-free-turn/tested/libbulk_tip_checks.release.so'),
 '--ranks','4','--bind-cores','--dt',str(dt),'--end','.08','--time-update','midpoint',
 '--phase-balance','--checkpoint-steps','1','--integrity-stop','--seconds',str(seconds),
 '--wall-hours',str((seconds-90)/3600)]
end_step=a.end_step or (15 if a.case in ['b-prefix','c-prefix'] else 4 if a.case=='c-pilot' else None)
if end_step:cmd+=['--end-step',str(end_step)]
if parent:
 snapshot=a.snapshot_id or int((parent/'restart/last_good_checkpoint.txt').read_text().strip())
 cmd+=['--resume-from',str(parent.resolve()),'--snapshot-id',str(snapshot),'--copy-selected-snapshot-only']
 parent_dt=float(json.loads((parent.parent/'command.json').read_text())['arguments']['dt'])
 assert dt<=parent_dt,'Only unchanged or reduced pending timesteps are selected'
 if dt<parent_dt:cmd+=['--restart-dt',str(dt)]
record=dict(case=a.case,label=label,start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=cmd,common_budget=budget,
 environment={k:v for k,v in env.items() if k.startswith(('ASPECT_TIP_','OMP_','OPENBLAS_','MKL_','DEAL_II_'))})
(OUT/(label+'-launch.json')).write_text(json.dumps(record,indent=2)+'\n')
code=subprocess.run(cmd,cwd=ROOT,env=env).returncode
compressed=[]
for pattern in ['tip-points-*.csv','tip-balance-*.csv','tip-seed-fracture-*.csv','restore-points-*.csv','restore-phase-*.csv']:
 for src in sorted((run/'output').glob(pattern)):
  dest=src.with_suffix('.csv.gz');assert not dest.exists();h=hashlib.sha256()
  with src.open('rb') as f,gzip.open(dest,'wb',compresslevel=6) as g:
   for block in iter(lambda:f.read(1048576),b''):h.update(block);g.write(block)
  with gzip.open(dest,'rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h.hexdigest()
  compressed.append(dict(file=str(dest.relative_to(ROOT)),raw_bytes=src.stat().st_size,gzip_bytes=dest.stat().st_size,raw_sha256=h.hexdigest()));src.unlink()
(OUT/(label+'-compression.json')).write_text(json.dumps(compressed,indent=2)+'\n')
raise SystemExit(code)
