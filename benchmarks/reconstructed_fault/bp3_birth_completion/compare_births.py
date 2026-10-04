"""Compare actual birth events to an independent known-flow ancestry test."""
from pathlib import Path
import numpy as np,json
r=Path(__file__).resolve().parent
checks=[];summary=[];reused=[]
def check(name,ok,**data):checks.append(dict(check=name,passed=bool(ok),**data))
def read(p):return np.atleast_1d(np.genfromtxt(p,delimiter=',',names=True))
def population(folder,step):
 a=np.concatenate([read(p) for p in sorted(folder.glob(f'birth_identity_{step}_rank*.csv'))])
 assert len(np.unique(a['id']))==len(a)
 return a[np.argsort(a['id'])]
for case,ranks,oldcase in [('transport-A-final',2,'transport-A-final'),('transport-A-serial',1,'transport-A-serial-final')]:
 folder=r/('output-'+case);old=r/f'output-transport-control-{ranks}'
 previous=None
 for step in range(5):
  now=population(folder,step)
  check(f'{case}/{step}/baseline-H',np.array_equal(now['H'],now['baseline']))
  if previous is not None:
   ids,i,j=np.intersect1d(now['id'],previous['id'],return_indices=True)
   residual=np.hypot(now['x'][i]-previous['x'][j]-.4,now['y'][i]-previous['y'][j]-.24)
   bound=256*np.finfo(float).eps*np.maximum.reduce([np.ones(len(i)),abs(now['x'][i]),abs(now['y'][i])])
   retained=residual<bound
   expected=np.ones(len(now),dtype=bool);expected[i[retained]]=False
   check(f'{case}/{step}/actual-births-vs-trajectory',np.array_equal(now['born'].astype(bool),expected))
   check(f'{case}/{step}/survivor-H',np.array_equal(now['H'][i[retained]],previous['H'][j[retained]]))
   for ni,oj in zip(i[~retained],j[~retained]):
    reused.append(dict(case=case,step=step,id=int(now['id'][ni]),old_x=float(previous['x'][oj]),old_y=float(previous['y'][oj]),new_x=float(now['x'][ni]),new_y=float(now['y'][ni])))
   old_rule=now['id']>=previous['id'].max()+1
   summary.append(dict(case=case,step=step,particles=len(now),births=int(now['born'].sum()),reused_ids=int((~retained).sum()),old_threshold_births=int(old_rule.sum()),lost=len(previous)+int(now['born'].sum())-len(now)))
  else:check(f'{case}/startup-not-births',not now['born'].any())
  previous=now
  for rank in range(ranks):
   p=f'transport_{step}_rank{rank}.csv'
   new=read(folder/p);ref=read(old/p)
   check(f'{case}/{step}/{rank}/shape',new.shape==ref.shape)
   # Both sides use the shared production startup initializer. The only
   # difference is whether the maintained birth audit is attached.
   for col in new.dtype.names:
    error=float(abs(new[col]-ref[col]).max());scale=float(max(abs(new[col]).max(),abs(ref[col]).max()))
    check(f'{case}/{step}/{rank}/{col}',error==0.,max_abs=error)
   check(f'{case}/{step}/{rank}/newborn-H',new['newborn_H_max_error'].max()==0)
 check(f'{case}/reuse-nonvacuous',any(x['case']==case for x in reused))
 resume=r/('output-'+('transport-A' if ranks==2 else 'transport-A-serial')+'-resume')
 for step in (3,4):
  a,b=population(folder,step),population(resume,step)
  check(f'{case}/restart/{step}/exact-particles-and-birth-flags',np.array_equal(a,b))
(r/'results/birth_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
(r/'results/birth_counts.json').write_text(json.dumps(summary,indent=2)+'\n')
(r/'results/reused_ids.json').write_text(json.dumps(reused,indent=2)+'\n')
failed=[x for x in checks if not x['passed']]
print(f'{len(checks)-len(failed)}/{len(checks)} birth checks passed')
for x in failed:print(x)
raise SystemExit(bool(failed))
