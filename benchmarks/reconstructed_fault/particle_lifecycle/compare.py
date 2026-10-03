#!/usr/bin/env python3
"""Assert qualified comparisons and retain compact, full-precision evidence."""
from pathlib import Path
import csv, hashlib, json, shutil
r=Path(__file__).resolve().parent
out=r/'results';out.mkdir(exist_ok=True)
def rows(case, file):return list(csv.DictReader((r/f'output-{case}'/file).open()))
def accepted(case,rank,step):return next(x for x in rows(case,f'lifecycle_rank{rank}.csv') if x['stage']=='accepted' and int(x['step'])==step)
fields=['n','next_id','position_hash','membership_hash','property_hash','integrator_hash','rng_hash','partition_hash','stress_max','stress_mean','H_min','H_negative','pressure_min','pressure_max','mapped_stress_max','audit_size','audit_hash']
checks=[]
def check(label, condition, **detail):
 checks.append(dict(check=label,passed=bool(condition),**detail))
 assert condition,(label,detail)
for a,b,rank_count,step in [('final-direct','final-retry',1,1),('final-staggered','final-resume-fixed',2,2),('final-direct2-fixed','final-retry2-fixed',2,2),('random-direct','random-retry',2,1)]:
 for rank in range(rank_count):
  x,y=accepted(a,rank,step),accepted(b,rank,step)
  for f in fields:check(f'{a}/{b}/rank{rank}/{f}',x[f]==y[f],reference=x[f],candidate=y[f])
  for pattern in [f'audit_bulk_{step}_rank{rank}.csv',f'audit_particles_{step}_rank{rank}.csv',f'mature_history_{step}_rank{rank}.csv',f'work_qp_{step}_rank{rank}.csv']:
   pa,pb=(r/f'output-{c}'/pattern for c in [a,b])
   check(f'{a}/{b}/{pattern}',pa.read_bytes()==pb.read_bytes())
 for pattern in [f'profiles/fault_{step}.csv',f'work_weak_{step}.csv',f'common_fe_weak_{step}.csv']:
  pa,pb=(r/f'output-{c}'/pattern for c in [a,b])
  check(f'{a}/{b}/{pattern}',pa.read_bytes()==pb.read_bytes())
 for filename in ['accepted_steps.csv','restored_growth.csv']:
  x,y=(next(x for x in rows(c,filename) if int(x['step'])==step) for c in [a,b])
  check(f'{a}/{b}/{filename}',x==y)
# A rejected trial must restore every retained particle/ID/RNG/audit value.
for case,ranks,step,time in [('final-retry',1,1,'50'),('final-retry2-fixed',2,2,'150'),('random-retry',2,1,'50')]:
 for rank in range(ranks):
  rr=rows(case,f'lifecycle_rank{rank}.csv')
  backup=next(x for x in rr if x['stage']=='backup' and int(x['step'])==step)
  restored=next(x for x in rr if x['stage']=='restored_before_advection' and x['time']==time)
  for f in ['n','next_id','position_hash','membership_hash','property_hash','integrator_hash','rng_hash','audit_size','audit_hash']:
   check(f'{case}/rollback/rank{rank}/{f}',backup[f]==restored[f])
# Both ranks consume saved manager streams before and after the checkpoint,
# with nonzero incoming Maxwell history at the second event.
for rank in range(2):
 a,b,c=(accepted('final-staggered',rank,k) for k in range(3))
 check(f'rank{rank}/two-native-RNG-events',len({a['rng_hash'],b['rng_hash'],c['rng_hash']})==3)
 check(f'rank{rank}/nonzero-incoming-Maxwell',float(b['stress_max'])>0)
 checkpoint=(r/f'output-final-create-fixed/rng-checkpoint-2-rank{rank}.txt').read_bytes()
 resumed=(r/f'output-final-resume-fixed/rng-resume-2-rank{rank}.txt').read_bytes()
 check(f'rank{rank}/complete-streams-at-restart',checkpoint==resumed)
 # The negative-control snapshot changes RNG words only: placement must differ.
 negative=accepted('final-rng-negative',rank,2)
 check(f'rank{rank}/RNG-sensitive-placement',negative['position_hash']!=c['position_hash'])
 migrated=[x for x in rows('migration-native',f'lifecycle_rank{rank}.csv') if x['stage']=='migrated']
 check(f'rank{rank}/native-migration',any(int(x['received_ids'])>0 for x in migrated))
# All active managers carry separate streams and exact native particle state.
for manager in range(2):
 for rank in range(2):
  suffix=f'manager{manager}-rank{rank}.txt'
  a=r/f'output-streams/streams-accepted-2-{suffix}'
  b=r/f'output-streams-resume/streams-accepted-2-{suffix}'
  check(f'multiple-managers/{suffix}/accepted',a.read_bytes()==b.read_bytes())
  a=r/f'output-streams-create/streams-checkpoint-2-{suffix}'
  b=r/f'output-streams-resume/streams-resume-2-{suffix}'
  # This early resume observer runs before native mesh-particle unpack.
  # Compare the two restored streams and next-ID header here; complete
  # particle/integrator data are compared at accepted state above.
  check(f'multiple-managers/{suffix}/loaded-streams-and-next-ID',a.read_bytes().splitlines()[:3]==b.read_bytes().splitlines()[:3])
for case in ['corrupt-value','corrupt-missing']:
 log=(r/f'evidence/{case}-np2.log').read_text(errors='replace')
 check(case,'BP3 particle audit: changed survivor H' in log)
for case in ['legacy-active','changed-active']:
 log=(r/f'evidence/{case}-np1.log').read_text(errors='replace')
 check(case,'Particle population management requires a checkpoint with per-rank RNG' in log)
for case in ['legacy-disabled','changed-disabled']:
 log=(r/f'evidence/{case}-np1.log').read_text(errors='replace')
 check(case,'Particle RNG replay unavailable' in log and 'Aborting!' not in log)
log=(r/'evidence/reordered-np1.log').read_text()
prm=(r/'output-reordered/parameters.prm').read_text()
check('runtime-name-limiter-reordering',
      log.count('crack_driving_force components 3..3; boundary extrapolation disabled.')==1
      and log.count('maxwell stress components 0..2; boundary extrapolation disabled.')==1
      and 'true,true,true,true,false,false' in prm)
# Unrejected original fixture retains its qualified pre-correction physical results.
for case,old,ranks in [('final-serial','crossing-serial',1),('final-mpi','crossing-mpi',2)]:
 base=r.parent/'particle_replenishment'/f'output-{old}'
 for rank in range(ranks):
  original={x['step']:x for x in csv.DictReader((base/f'lifecycle_rank{rank}.csv').open()) if x['stage']=='accepted'}
  for step in range(3):
   actual=accepted(case,rank,step)
   for field in ['n','next_id','position_hash','property_hash','integrator_hash','stress_max','pressure_min','pressure_max','mapped_stress_max','H_min']:
    check(f'pre-correction/{case}/step{step}/rank{rank}/{field}',original[str(step)][field]==actual[field])
for name in ['final-serial','final-mpi','final-retry','final-direct','final-staggered','final-create-fixed','final-resume-fixed','final-retry2-fixed','final-direct2-fixed','final-rng-negative','migration-native','random-direct','random-retry']:
 for path in (r/f'output-{name}').glob('lifecycle_rank*.csv'):
  selected=[x for x in csv.DictReader(path.open()) if x['stage'] in ['accepted','backup','rejected','restored_before_advection','managed','migrated','checkpoint','resume']]
  with (out/f'{name}-{path.name}').open('w') as f:
   w=csv.DictWriter(f,fieldnames=selected[0],lineterminator="\n");w.writeheader();w.writerows(selected)
 for file in ['accepted_steps.csv','restored_growth.csv']:
  shutil.copyfile(r/f'output-{name}'/file,out/f'{name}-{file}')
(out/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print(f'PASS: {len(checks)} exact lifecycle/field/checkpoint and expected-failure checks')
