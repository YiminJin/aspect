#!/usr/bin/env python3
"""Fresh native cases from a common prepared phase and resistance distribution."""
from pathlib import Path
import argparse,os,subprocess,time,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];BUILD=ROOT/'build-particle-free/stage4-preparation'
p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('--law',choices=['weakening','neutral'],default='weakening');p.add_argument('--steps',type=int,default=0);p.add_argument('--seconds',type=int,default=7200);p.add_argument('--dt',type=float,default=.00008);p.add_argument('--settle',action='store_true');p.add_argument('--release',action='store_true');p.add_argument('--input',default='input-prepared');p.add_argument('--perturbation',type=float,default=.01);p.add_argument('--resume-from',type=Path);p.add_argument('--restart-dt',type=float);p.add_argument('--end-time',type=float);p.add_argument('--checkpoint-steps',type=int,default=16);p.add_argument('--snapshot-id',type=int,choices=[1,2,3]);p.add_argument('--extend-terminal-dt',type=float);p.add_argument('--restore-audit',action='store_true');a=p.parse_args()
assert 0<=a.steps<=128 and a.dt>0
entry=json.loads((BUILD/'entry.json').read_text());remaining=entry['deadline_unix']-time.time()
assert remaining>600,'Reserve ten minutes for retention and comparison.'
env=os.environ.copy()
for key in list(env):
    if key.startswith(('ASPECT_TIP_','ASPECT_PILOT_','ASPECT_PRELOAD_','ASPECT_PREP_')):env.pop(key)
env.update(ASPECT_WAVELENGTH_INPUT=str(BUILD/a.input),ASPECT_PILOT_COMPACT='1',ASPECT_PILOT_AGE_PERTURBATION=str(a.perturbation),ASPECT_TIP_POINT_EXPORT_STRIDE='999999',ASPECT_TIP_POSITIVE_METRIC_FIRST='1',ASPECT_PRELOAD_GAMMA='.000365')
if a.settle:env['ASPECT_PREP_SETTLE']='1'
if a.extend_terminal_dt:env['ASPECT_TIP_EXTEND_TERMINAL_RESTART_DT']=str(a.extend_terminal_dt)
if a.restore_audit:env['ASPECT_TIP_RESTORE_AUDIT']='1'
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','DEAL_II_NUM_THREADS'):env[key]='1'
cmd=['python3',str(HERE.parent/'bulk_tip/run.py'),'stage4-preparation-'+a.label,'--fixture',str(BUILD/f'256-{a.law}.prm'),'--executable',str(ROOT/'build-particle-free/first-free-turn/tested/aspect-release'),'--checker',str(BUILD/'libbulk_tip_checks.release.so'),'--ranks','4','--bind-cores','--dt',str(a.dt),'--end',str(a.end_time if a.end_time is not None else max(1,a.steps)*a.dt),'--end-step',str(a.steps),'--seconds',str(min(a.seconds,int(remaining-600))),'--checkpoint-steps',str(a.checkpoint_steps)]
allowed=min(a.seconds,int(remaining-600))
cmd+=['--wall-hours',str(max(300,allowed-300)/3600)]
if a.resume_from:cmd+=['--resume-from',str(a.resume_from),'--copy-selected-snapshot-only']
if a.snapshot_id:cmd+=['--snapshot-id',str(a.snapshot_id)]
if a.restart_dt:cmd+=['--restart-dt',str(a.restart_dt)]
if a.release:cmd+=['--phase-balance']
else:cmd+=['--fixed-phase']
requests=BUILD/'requests';requests.mkdir(exist_ok=True)
assert Path(a.label).name==a.label
request=dict(requested_unix=time.time(),arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},command=cmd,launcher_environment={k:v for k,v in env.items() if k.startswith(('ASPECT_TIP_','ASPECT_PILOT_','ASPECT_PRELOAD_','ASPECT_PREP_')) or k in ['ASPECT_WAVELENGTH_INPUT','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','DEAL_II_NUM_THREADS']},note='Actual environment supplied to the native benchmark launcher; that launcher also applies the recorded command-line diagnostic/restart flags.')
with (requests/(a.label+'.json')).open('x') as f:json.dump(request,f,indent=2);f.write('\n')
raise SystemExit(subprocess.call(cmd,cwd=ROOT,env=env))
