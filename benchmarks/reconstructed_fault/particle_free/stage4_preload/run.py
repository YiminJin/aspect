#!/usr/bin/env python3
"""Fresh native preload; optional bounded fixed-phase or release window."""
from pathlib import Path
import argparse,os,subprocess
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];BUILD=ROOT/'build-particle-free/stage4-preload'
p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('--law',choices=['weakening','neutral'],default='weakening');p.add_argument('--gamma',type=float,required=True);p.add_argument('--steps',type=int,default=0);p.add_argument('--seconds',type=int,default=2700);p.add_argument('--release',action='store_true');a=p.parse_args()
assert a.gamma>0 and 0<=a.steps<=8
env=os.environ.copy()
for key in list(env):
 if key.startswith(('ASPECT_TIP_','ASPECT_PILOT_','ASPECT_PRELOAD_')):env.pop(key)
env.update(ASPECT_WAVELENGTH_INPUT=str(ROOT/'build-particle-free/stage4-pilot/input-256'),ASPECT_PILOT_COMPACT='1',ASPECT_PILOT_AGE_PERTURBATION='.01',ASPECT_TIP_POINT_EXPORT_STRIDE='999999',ASPECT_TIP_POSITIVE_METRIC_FIRST='1',ASPECT_PRELOAD_GAMMA=str(a.gamma))
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','DEAL_II_NUM_THREADS'):env[key]='1'
cmd=['python3',str(HERE.parent/'bulk_tip/run.py'),'stage4-preload-'+a.label,
 '--fixture',str(BUILD/f'256-{a.law}.prm'),'--executable',str(ROOT/'build-particle-free/first-free-turn/tested/aspect-release'),
 '--checker',str(BUILD/'libbulk_tip_checks.release.so'),'--ranks','4','--bind-cores',
 '--dt','.00008','--end',str(max(1,a.steps)*8e-5),'--end-step',str(a.steps),'--seconds',str(a.seconds),'--checkpoint-steps','8']
if not a.release:cmd+=['--fixed-phase']
raise SystemExit(subprocess.call(cmd,cwd=ROOT,env=env))
