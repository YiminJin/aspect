#!/usr/bin/env python3
"""One fresh preloaded native fixed-phase history; at most four MPI ranks."""
from pathlib import Path
import argparse,os,subprocess
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];BUILD=ROOT/'build-particle-free/stage4-pilot'
p=argparse.ArgumentParser();p.add_argument('nx',type=int,choices=[256,512]);p.add_argument('law',choices=['weakening','neutral']);p.add_argument('case',choices=['background','perturbed']);p.add_argument('--steps',type=int,default=8);p.add_argument('--seconds',type=int,default=3600);p.add_argument('--ranks',type=int,choices=[1,2,4],default=4);p.add_argument('--suffix',default='');a=p.parse_args()
env=os.environ.copy()
for key in list(env):
 if key.startswith(('ASPECT_TIP_','ASPECT_PILOT_')):env.pop(key)
env.update(ASPECT_WAVELENGTH_INPUT=str(BUILD/f'input-{a.nx}'),ASPECT_PILOT_COMPACT='1',ASPECT_PILOT_AGE_PERTURBATION='.01' if a.case=='perturbed' else '0',ASPECT_TIP_POINT_EXPORT_STRIDE='999999',ASPECT_TIP_POSITIVE_METRIC_FIRST='1')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','DEAL_II_NUM_THREADS'):env[key]='1'
dt=8e-5
command=['python3',str(HERE.parent/'bulk_tip/run.py'),f'stage4-native-{a.nx}-{a.law}-{a.case}{a.suffix}',
 '--fixture',str(BUILD/f'{a.nx}-{a.law}.prm'),'--executable',str(ROOT/'build-particle-free/first-free-turn/tested/aspect-release'),
 '--checker',str(BUILD/'libbulk_tip_checks.release.so'),'--ranks',str(a.ranks),'--bind-cores','--dt',str(dt),'--end',str(a.steps*dt),
 '--end-step',str(a.steps),'--seconds',str(a.seconds),'--fixed-phase','--checkpoint-steps',str(max(1,a.steps))]
raise SystemExit(subprocess.call(command,cwd=ROOT,env=env))
