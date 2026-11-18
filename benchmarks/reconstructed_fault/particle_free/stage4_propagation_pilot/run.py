#!/usr/bin/env python3
"""One fresh, explicitly selected spatial-loading screen or free pilot."""
from pathlib import Path
import argparse,os,json,subprocess,gzip,hashlib,time
R=Path(__file__).resolve().parents[4];B=R/'build-particle-free/stage4-propagation-pilot'
p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('--amplitude',type=float,required=True);p.add_argument('--screen',action='store_true');p.add_argument('--settle',action='store_true');p.add_argument('--steps',type=int,default=0);p.add_argument('--dt',type=float,default=.00008);p.add_argument('--seconds',type=int,default=900);p.add_argument('--input',default='input');p.add_argument('--resume-from',type=Path);p.add_argument('--snapshot-id',type=int,choices=[1,2,3],help='Explicit native snapshot selector; source marker is unchanged');p.add_argument('--end-time',type=float);p.add_argument('--dense-stride',type=int,default=1);p.add_argument('--checkpoint-steps',type=int,default=8);p.add_argument('--search-order',choices=['positive-first','adaptive-exact-first'],default='positive-first');a=p.parse_args()
assert a.steps==0 if a.screen or a.settle else a.steps>0
assert a.snapshot_id is None or a.resume_from is not None
assert subprocess.run(['pgrep','-x','aspect-release'],capture_output=True).returncode==1
assert Path(a.label).name==a.label
assert 1<=a.dense_stride<=128 and a.seconds>=900
fixture=B/(a.label+'.prm');assert not fixture.exists()
t=(B/'base.prm').read_text()
# Constant boundary velocity, obtained from the same divergence-free displacement
# shape, loading along the chosen preparation ray on a .0365 s scale.
rate=a.amplitude/.0365
old='0.01*(y-2)-0.0041954981558863999*(x-4);0.0041954981558863999*(y-2)'
new=f'({.01+rate:.17g})*(y-2)-0.0041954981558863999*(x-4);0.0041954981558863999*(y-2)'
assert old in t;fixture.write_text(t.replace(old,new))
env=os.environ.copy()
for k in list(env):
 if k.startswith(('ASPECT_TIP_','ASPECT_PILOT_','ASPECT_PRELOAD_','ASPECT_PREP_')):env.pop(k)
env.update(ASPECT_WAVELENGTH_INPUT=str(B/a.input),ASPECT_PILOT_COMPACT='1',ASPECT_PILOT_AGE_PERTURBATION='0' if a.screen or a.settle else '.01',ASPECT_TIP_POINT_EXPORT_STRIDE='999999',ASPECT_TIP_POSITIVE_METRIC_FIRST='1',ASPECT_PRELOAD_GAMMA='.000365',ASPECT_PILOT_LOAD_AMPLITUDE=str(a.amplitude))
env['ASPECT_PILOT_DENSE_STRIDE']=str(a.dense_stride)
if a.search_order=='adaptive-exact-first':
 env.pop('ASPECT_TIP_POSITIVE_METRIC_FIRST',None)
 env['ASPECT_TIP_TRACE_SEARCH']='1'
if a.screen:env['ASPECT_PILOT_EQUILIBRIUM_SCREEN']='1'
if a.settle:env['ASPECT_PREP_SETTLE']='1'
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','DEAL_II_NUM_THREADS']:env[k]='1'
cmd=['python3',str(R/'benchmarks/reconstructed_fault/particle_free/bulk_tip/run.py'),'stage4-propagation-'+a.label,'--fixture',str(fixture),'--executable',str(R/'build-particle-free/first-free-turn/tested/aspect-release'),'--checker',str(B/'libbulk_tip_checks.release.so'),'--ranks','4','--bind-cores','--dt',str(a.dt),'--end',str(a.end_time if a.end_time is not None else max(1,a.steps)*a.dt),'--end-step',str(a.steps),'--seconds',str(a.seconds),'--wall-hours',str(max(300,a.seconds-1200)/3600),'--checkpoint-steps',str(a.checkpoint_steps),'--fixed-phase' if a.screen or a.settle else '--phase-balance']
if a.resume_from:
 cmd+=['--resume-from',str(a.resume_from.resolve()),'--copy-selected-snapshot-only','--restart-dt',str(a.dt)]
 if a.snapshot_id is not None:cmd+=['--snapshot-id',str(a.snapshot_id)]
 env['ASPECT_TIP_RESTORE_AUDIT']='1'
record=dict(argv=cmd,environment={k:v for k,v in env.items() if k.startswith(('ASPECT_','OMP_','OPENBLAS_','MKL_','DEAL_II_'))},started_unix=time.time())
(B/(a.label+'-launch.json')).write_text(json.dumps(record,indent=2)+'\n')
code=subprocess.call(cmd,cwd=R,env=env)
out=R/'build-particle-free/bulk-tip/runs'/('stage4-propagation-'+a.label)/'output';compression={}
for f in sorted(out.glob('*')):
 if f.is_file() and (f.name.startswith(('release-phase-','preload-scan-','prepared-history-','tip-balance-','tip-seed-fracture-','restore-'))) and f.suffix in ['.csv','.bin']:
  raw=f.read_bytes();target=f.with_suffix(f.suffix+'.gz');assert not target.exists()
  with gzip.open(target,'wb',compresslevel=4) as stream:stream.write(raw)
  digest=hashlib.sha256(raw).hexdigest();assert hashlib.sha256(gzip.decompress(target.read_bytes())).hexdigest()==digest
  compression[f.name]=dict(bytes=len(raw),sha256=digest,gzip_bytes=target.stat().st_size);f.unlink()
(out/'lossless-compression.json').write_text(json.dumps(compression,indent=2)+'\n')
record.update(exit_code=code,seconds=time.time()-record['started_unix']);(B/(a.label+'-finished.json')).write_text(json.dumps(record,indent=2)+'\n')
raise SystemExit(code)
