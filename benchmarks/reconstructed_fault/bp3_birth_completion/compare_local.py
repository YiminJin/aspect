from pathlib import Path
import numpy as np,json,csv
r=Path(__file__).resolve().parent;checks=[]
for case in ['A','B']:
 new=r/f'output-local-{case}-final';old=r.parent/'bp3_local_tests'/f'output-{case}'
 for step in range(3):
  files=[*new.glob(f'velocity_{step}_rank*.csv'),*new.glob(f'friction_samples_{step}_rank*.csv'),new/f'fault_{step}.csv',new/'profiles'/f'fault_{step}.csv']
  for p in files:
   rel=p.relative_to(new);a=np.genfromtxt(p,delimiter=',',names=True);b=np.genfromtxt(old/rel,delimiter=',',names=True)
   checks.append(dict(check=f'{case}/{rel}',passed=np.array_equal(a,b)))
 for name in ['mesh_rank0.csv','mesh_rank1.csv','particle_summary.csv']:
  if name.startswith('mesh'):
   checks.append(dict(check=f'{case}/{name}',passed=(new/name).read_bytes()==(old/name).read_bytes()))
  else:
   a=np.genfromtxt(new/name,delimiter=',',names=True);b=np.genfromtxt(old/name,delimiter=',',names=True)[:3]
   checks.append(dict(check=f'{case}/{name}',passed=np.array_equal(a,b)))
 a,b=(list(csv.DictReader((folder/'accepted_steps.csv').open()))[:3] for folder in (new,old))
 keys=['time','dt','free','lower_active','newton_updates','krylov_iterations','min_alpha','fresh_linear_checks_passed']
 checks.append(dict(check=f'{case}/solver-decisions-and-fresh-residuals',passed=all(x['fresh_linear_checks_passed']=='1' and all(x[k]==y[k] for k in keys) for x,y in zip(a,b))))
(r/'results/local_comparison.json').write_text(json.dumps(checks,indent=2)+'\n')
print(sum(x['passed'] for x in checks),'/',len(checks),'exact local comparisons')
print([x for x in checks if not x['passed']])
assert all(x['passed'] for x in checks)
