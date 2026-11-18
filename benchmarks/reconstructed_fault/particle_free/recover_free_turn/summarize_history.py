#!/usr/bin/env python3
"""Join accepted physical intervals; never include failed nonlinear trials."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'first_free_turn'))
from analyze import RUNS,OUT,read
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--reflected',action='store_true')
a=p.parse_args();out=OUT.parent/'recover_free_turn_step16'
histories={'positive':['first-turn-positive-prefix','first-turn-positive-end','first-turn-positive-fourcore-end','recover-free-turn-positive-half']}
if a.reflected:histories['reflected']=['recover-free-turn-reflected-prefix','recover-free-turn-reflected-half']
results={}
for sign,labels in histories.items():
 rows=[];previous=0.;solver=0.;process=0.
 for label in labels:
  path=RUNS/label
  accepted=read(path/'output/bulk-accepted.csv')
  for r in accepted:
   assert abs(r.time-previous-r.dt)<1e-12,(label,r.time,previous,r.dt)
   assert int(r.publications)==int(r.step)
   work=float(r.Wext-r.energy_change-r.Db-r.Dc-r.Df)
   rows.append(dict(run=label,step=int(r.step),time=float(r.time),dt=float(r.dt),
      phase_KKT=float(r.phase_KKT),mechanical_residual=float(r.mechanical_residual),
      fracture=float(r.fracture),physical_work_error=work,ledger_error=float(r.ledger_error)))
   previous=float(r.time)
  profile=read(path/'output/tip-profile.csv');solver+=float(sum(profile.step_seconds))
  manifest=json.loads((path/'command.json').read_text());process+=manifest.get('seconds',0.)
 errors=np.array([r['physical_work_error'] for r in rows])
 window=[r for r in rows if r['time']>.075+1e-12]
 results[sign]=dict(steps=rows,signed_work_error=float(sum(errors)),absolute_work_error=float(sum(abs(errors))),
    recovery_window_signed=sum(r['physical_work_error'] for r in window),recovery_window_absolute=sum(abs(r['physical_work_error']) for r in window),
    accepted_solver_seconds=solver,process_seconds_including_original_failures=process)
 with (out/f'{sign}-accepted-history.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if a.reflected:
 assert [r['step'] for r in results['positive']['steps']]==[r['step'] for r in results['reflected']['steps']]
 positive=np.array([(r['time'],r['dt']) for r in results['positive']['steps']])
 reflected=np.array([(r['time'],r['dt']) for r in results['reflected']['steps']])
 assert np.max(abs(positive-reflected))<1e-14
(out/'accepted-history.json').write_text(json.dumps(results,indent=2)+'\n')
for sign,r in results.items():print(sign,{k:v for k,v in r.items() if k!='steps'})
