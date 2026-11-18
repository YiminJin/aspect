#!/usr/bin/env python3
"""Fresh selected Stage 4 native conditioning/control; no automatic continuation."""
from pathlib import Path
import argparse,json,os,subprocess
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('case',choices=['weakening-background','weakening-perturbed','neutral-background','neutral-perturbed'])
p.add_argument('--label',required=True);p.add_argument('--seconds',type=int,required=True)
p.add_argument('--free-phase',action='store_true',help='Fresh t0 free-phase counterpart; only after selected native qualification')
a=p.parse_args()
record=json.loads((HERE.parent/'evidence/stage4_rsf/native-candidate.json').read_text())['configurations'][a.case]
env=os.environ.copy()
for key in list(env):
 if key.startswith('ASPECT_TIP_'):env.pop(key)
env.update(ASPECT_TIP_POSITIVE_METRIC_FIRST='1',ASPECT_TIP_POINT_EXPORT_STRIDE='9999')
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','DEAL_II_NUM_THREADS']:env[key]='1'
cmd=['python3',str(HERE.parent/'bulk_tip/run.py'),a.label,'--fixture',str(ROOT/record['input']),
 '--executable',str(ROOT/'build-particle-free/first-free-turn/tested/aspect-release'),
 '--checker',str(ROOT/'build-particle-free/first-free-turn/tested/libbulk_tip_checks.release.so'),
 '--ranks','4','--bind-cores','--dt',str(record['dt']),'--end',str(record['end']),'--time-update','midpoint',
 '--checkpoint-steps','20','--seconds',str(a.seconds),'--wall-hours',str(max(1,a.seconds-90)/3600)]
if not a.free_phase:cmd+=['--fixed-phase']
else:cmd+=['--phase-balance','--integrity-stop']
raise SystemExit(subprocess.run(cmd,cwd=ROOT,env=env).returncode)
