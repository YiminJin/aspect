#!/usr/bin/env python3
"""One selected fresh phase-release history, eight intervals, four ranks."""
from pathlib import Path
import argparse,os,subprocess
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];BUILD=ROOT/'build-particle-free/stage4-release'
p=argparse.ArgumentParser();p.add_argument('law',choices=['weakening','neutral']);p.add_argument('--steps',type=int,default=8);p.add_argument('--seconds',type=int,default=2700);p.add_argument('--suffix',default='');a=p.parse_args()
env=os.environ.copy()
for key in list(env):
 if key.startswith(('ASPECT_TIP_','ASPECT_PILOT_')):env.pop(key)
env.update(ASPECT_WAVELENGTH_INPUT=str(ROOT/'build-particle-free/stage4-pilot/input-256'),ASPECT_PILOT_COMPACT='1',ASPECT_PILOT_AGE_PERTURBATION='.01',ASPECT_TIP_POINT_EXPORT_STRIDE='999999',ASPECT_TIP_POSITIVE_METRIC_FIRST='1')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','DEAL_II_NUM_THREADS'):env[key]='1'
cmd=['python3',str(HERE.parent/'bulk_tip/run.py'),f'stage4-release-256-{a.law}{a.suffix}',
 '--fixture',str(BUILD/f'256-{a.law}.prm'),'--executable',str(ROOT/'build-particle-free/first-free-turn/tested/aspect-release'),
 '--checker',str(BUILD/'libbulk_tip_checks.release.so'),'--ranks','4','--bind-cores',
 '--dt','.00008','--end',str(a.steps*8e-5),'--end-step',str(a.steps),'--seconds',str(a.seconds),'--checkpoint-steps','2']
raise SystemExit(subprocess.call(cmd,cwd=ROOT,env=env))
