#!/usr/bin/env python3
"""Exact replay checks and explicit accounting of the selected H-only change."""
from pathlib import Path
import csv, json, shutil
r=Path(__file__).resolve().parent
old=r.parent/'particle_lifecycle'
results=r/'results';results.mkdir(exist_ok=True)
checks=[];differences=[]
def check(name,passed,**detail):
 checks.append(dict(check=name,passed=bool(passed),**detail))
 assert passed,(name,detail)
def records(path):return list(csv.DictReader(path.open()))
def particles(root,case,rank,step):
 path=root/f'output-{case}/audit_particles_{step}_rank{rank}.csv'
 return {int(x[0]):x[1:] for x in list(csv.reader(path.open()))[1:]}
def equal_file(a,b,name):check(name,a.read_bytes()==b.read_bytes())
for a,b,n,step in [('direct','retry',1,1),('staggered','resume',2,2),('direct2','retry2',2,2)]:
 for rank in range(n):
  for stem in ['audit_particles','audit_bulk','mature_history','work_qp']:
   file=f'{stem}_{step}_rank{rank}.csv'
   equal_file(r/f'output-{a}'/file,r/f'output-{b}'/file,f'{a}/{b}/{file}')
  aa,bb=(next(x for x in records(r/f'output-{c}/lifecycle_rank{rank}.csv') if x['stage']=='accepted' and int(x['step'])==step) for c in [a,b])
  check(f'{a}/{b}/rank{rank}/all-accepted-lifecycle-fields',aa==bb)
 for file in [f'profiles/fault_{step}.csv',f'common_fe_weak_{step}.csv',f'work_weak_{step}.csv']:
  equal_file(r/f'output-{a}'/file,r/f'output-{b}'/file,f'{a}/{b}/{file}')
 for file in ['accepted_steps.csv','restored_growth.csv']:
  aa,bb=(next(x for x in records(r/f'output-{c}'/file) if int(x['step'])==step) for c in [a,b])
  check(f'{a}/{b}/{file}',aa==bb)
for case,n,time in [('retry',1,'50'),('retry2',2,'150')]:
 for rank in range(n):
  rows=records(r/f'output-{case}/lifecycle_rank{rank}.csv')
  before=next(x for x in rows if x['stage']=='backup' and x['step']==('1' if case=='retry' else '2'))
  after=next(x for x in rows if x['stage']=='restored_before_advection' and x['time']==time)
  fields=['n','next_id','position_hash','property_hash','integrator_hash','rng_hash','membership_hash','audit_hash','audit_size']
  check(f'{case}/rank{rank}/exact-rollback',all(before[f]==after[f] for f in fields))
for case,ref,n in [('serial-fixed','final-serial',1),('mpi','final-mpi',2),('staggered','final-staggered',2)]:
 for rank in range(n):
  for step in range(3):
   a,b=particles(old,ref,rank,step),particles(r,case,rank,step)
   check(f'{case}/rank{rank}/step{step}/same-membership',a.keys()==b.keys())
   # Stored row is x,y,H,theta_initial,strengthening,Maxwell...,integrator...
   check(f'{case}/rank{rank}/step{step}/all-non-H-particle-data',all(a[i][:2]+a[i][3:]==b[i][:2]+b[i][3:] for i in a))
   changed=[i for i in a if a[i][2]!=b[i][2]]
   check(f'{case}/rank{rank}/step{step}/only-born-H-changes',all(i>=30000 for i in changed))
   if step==0:check(f'{case}/rank{rank}/startup-unchanged',not changed)
   differences.append(dict(case=case,rank=rank,step=step,H_changed=len(changed),
                           max_abs_H_difference=max(abs(float(a[i][2])-float(b[i][2])) for i in a)))
   file=f'audit_bulk_{step}_rank{rank}.csv'
   equal_file(old/f'output-{ref}'/file,r/f'output-{case}'/file,f'{case}/rank{rank}/step{step}/bulk-unchanged')
 for file in ['accepted_steps.csv','restored_growth.csv']:
  equal_file(old/f'output-{ref}'/file,r/f'output-{case}'/file,f'{case}/{file}/unchanged')
 for step in range(3):
  for file in [f'profiles/fault_{step}.csv',f'common_fe_weak_{step}.csv',f'work_weak_{step}.csv']:
   equal_file(old/f'output-{ref}'/file,r/f'output-{case}'/file,f'{case}/{file}/unchanged')
# Retained H stays committed, even when it originated under the old policy.
for case,n in [('serial-fixed',1),('mpi',2),('staggered',2),('old-resume',2)]:
 for rank in range(n):
  steps=[1,2] if case=='old-resume' else [0,1,2]
  for before,after in zip(steps,steps[1:]):
   a,b=(particles(r,case,rank,k) for k in [before,after])
   common=a.keys()&b.keys()
   check(f'{case}/rank{rank}/{before}->{after}/committed-H-unchanged',all(a[i][2]==b[i][2] for i in common),survivors=len(common))
# Same-rank checkpoint includes all streams; both ranks consume them at both events.
for rank in range(2):
 equal_file(r/f'output-create/rng-checkpoint-2-rank{rank}.txt',r/f'output-resume/rng-resume-2-rank{rank}.txt',f'rank{rank}/RNG-checkpoint')
 rows=[x for x in records(r/f'output-staggered/lifecycle_rank{rank}.csv') if x['stage']=='accepted']
 check(f'rank{rank}/RNG-on-both-events',len({x['rng_hash'] for x in rows})==3)
 check(f'rank{rank}/nonzero-incoming-Maxwell',float(rows[1]['stress_max'])>0)
births=[]
for case,n in [('serial-fixed',1),('mpi',2),('retry',1),('direct',1),('staggered',2),('create',2),('resume',2),('retry2',2),('direct2',2),('old-resume',2)]:
 for rank in range(n):
  path=r/f'output-{case}/birth_H_rank{rank}.csv';rows=records(path)
  check(f'{case}/rank{rank}/every-newborn-H',all(x['H']==x['expected'] and x['audit_already_present']=='0' for x in rows))
  check(f'{case}/rank{rank}/exterior-Hc',all(x['H']==x['Hc'] for x in rows if x['active']=='0'))
  births.append(dict(case=case,rank=rank,observations=len(rows),active=sum(x['active']=='1' for x in rows),outside=sum(x['active']=='0' for x in rows)))
  shutil.copyfile(path,results/f'{case}-birth_H_rank{rank}.csv')
  shutil.copyfile(r/f'output-{case}/lifecycle_rank{rank}.csv',results/f'{case}-lifecycle_rank{rank}.csv')
check('non-vacuous-active-and-exterior-births',sum(x['active'] for x in births)>0 and sum(x['outside'] for x in births)>0)
log=(r/'evidence/evolving-fixed-np2.log').read_text(errors='replace')
check('evolving-real-history-transfer',log.count('EVOLVING H TRANSFER PASS: intentional stop before mechanics.')>=2)
for name,data in [('checks.json',checks),('H_differences.json',differences),('birth_counts.json',births)]:
 (results/name).write_text(json.dumps(data,indent=2)+'\n')
print(f'PASS: {len(checks)} exact replay, retained-history, initializer and evolving-transfer checks')
print('Largest selected H change:',max(x['max_abs_H_difference'] for x in differences))
