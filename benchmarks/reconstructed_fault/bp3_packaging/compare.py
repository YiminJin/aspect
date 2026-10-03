#!/usr/bin/env python3
from pathlib import Path
import csv,json,hashlib,subprocess,shutil
r=Path(__file__).resolve().parent;repo=r.parents[2];base=r.with_name('bp3_runtime')
checks=[]
def check(name,ok,**details):checks.append(dict(check=name,passed=bool(ok),**details))
def records(p):return list(csv.DictReader(p.open()))
def out(case):return r/('output-'+case)
# Source change is observational: require exact files/fields, not relaxed tolerances.
for case,n in [('model-only',1),('create',2),('staggered',2),('direct',1),('retry',1),('resume',2)]:
 old=base/('output-'+case);new=out(case)
 for pattern in ['accepted_steps.csv','restored_growth.csv','profiles/fault_*.csv','audit_particles_*.csv','audit_bulk_*.csv','mature_history_*.csv','work_qp_*.csv']:
  for p in new.glob(pattern):
   rel=p.relative_to(new)
   check(case+'/'+str(rel),(old/rel).read_bytes()==p.read_bytes())
 if case!='model-only':
  for rank in range(n):
   check(case+f'/{rank}/lifecycle',records(old/f'lifecycle_rank{rank}.csv')==records(new/f'lifecycle_rank{rank}.csv'))
   births=records(new/f'birth_H_rank{rank}.csv')
   check(case+f'/{rank}/birth-H',all(x['H']==x['expected'] and x['audit_already_present']=='0' for x in births))
 # Independent global accounting from particle IDs, H and stress values.
 summaries=records(new/'particle_summary.csv');previous=None
 for summary in summaries:
  step=int(summary['step']);cloud=[]
  for rank in range(n):cloud+=list(csv.reader((new/f'audit_particles_{step}_rank{rank}.csv').open()))[1:]
  if not cloud:continue
  ids={row[0] for row in cloud}
  check(case+f'/{step}/count',int(summary['particles'])==len(ids))
  H=[float(row[3]) for row in cloud]
  stress=[float(row[key]) for row in cloud for key in [6,7,8]]
  check(case+f'/{step}/extrema',float(summary['H_min'])==min(H) and float(summary['H_max'])==max(H) and float(summary['stress_component_min'])==min(stress) and float(summary['stress_component_max'])==max(stress))
  if previous is not None:
   check(case+f'/{step}/births-losses',int(summary['births_since_backup'])==len(ids-previous) and int(summary['removed_or_exited_since_backup'])==len(previous-ids))
  previous=ids
# Quiet visualization/output cadence must not change physical acceptance or checkpoint recovery.
for case in ['quiet','resume','reschedule','branch']:
 reference=out('staggered');candidate=out(case)
 old=records(reference/'accepted_steps.csv');new=records(candidate/'accepted_steps.csv')
 check(case+'/accepted-states',old==new)
 check(case+'/particle-summary',records(reference/'particle_summary.csv')==records(candidate/'particle_summary.csv'))
 for rank in range(2):
  a={x['step']:x for x in records(reference/f'lifecycle_rank{rank}.csv') if x['stage']=='accepted'}
  b={x['step']:x for x in records(candidate/f'lifecycle_rank{rank}.csv') if x['stage']=='accepted'}
  check(case+f'/{rank}/accepted-history',all(a[k]==v for k,v in b.items()))
 check(case+'/checkpoint',bool(list((candidate/'restart').glob('*/resume.z'))))
check('quiet/no-native-bulk-particle',not list(out('quiet').glob('solution/*')) and not list(out('quiet').glob('particles/*')))
check('quiet/no-detailed-dumps',not list(out('quiet').glob('audit_*.csv')) and not list(out('quiet').glob('restored_raw_*.csv')))
check('quiet/no-cumulative-table',not (out('quiet')/'cumulative_slip.csv').exists())
check('quiet/profiles-retained',(out('quiet')/'profiles/fault_0.csv').exists() and (out('quiet')/'profiles/fault_2.csv').exists())
for case in ['material','mesh','limiter','filter','interpolator']:
 record=json.loads((r/f'evidence/reject-{case}-np2.json').read_text())
 log=(r/f'evidence/reject-{case}-np2.log').read_text(errors='replace')
 check('identity/'+case,record['exit_code']!=0 and 'BP3 restart requires the same geometry, material/profile' in log)
# Preserve unrelated local edits, all earlier evidence and core implementations.
previous=json.loads((base/'results/preservation.json').read_text())
expected=json.loads((r.with_name('bp3_geometry')/'filter_derivative/results/preservation.json').read_text())['preserved_local']
for path,digest in expected.items():check('preserve/'+path,hashlib.sha256((repo/path).read_bytes()).hexdigest()==digest)
for path in ['source','include','unit_tests','benchmarks/reconstructed_fault/bp3_runtime','benchmarks/reconstructed_fault/bp3/fixtures']:
 check('preserve/'+path,not subprocess.check_output(['git','diff','e2ff248fb','--',path],cwd=repo))
(r/'results/checks.json').write_text(json.dumps(checks,indent=2)+'\n')
failed=[x for x in checks if not x['passed']]
print(f'{len(checks)-len(failed)}/{len(checks)} checks pass')
for x in failed:print(x)
raise SystemExit(bool(failed))
