#!/usr/bin/env python3
"""Exact lifecycle replay plus scale-aware geometry comparisons; no solver gate changes."""
from pathlib import Path
import csv, json, math, shutil, sys
r=Path(__file__).resolve().parent
filtered='--filter-derivative' in sys.argv
endpoint='--endpoint' in sys.argv or filtered
prefix='filter' if filtered else 'endpoint'
results=r/('filter_derivative/results' if filtered else 'endpoint/results' if endpoint else 'results');results.mkdir(exist_ok=True)
checks=[];metrics=[]
def check(name,ok,**detail):
 checks.append(dict(check=name,passed=bool(ok),**detail))
def out(case):return r/f"output-{prefix if endpoint else 'qualified'}-{case}"
def records(p):return list(csv.DictReader(p.open()))
def rows(p):return list(csv.reader(p.open()))
def exact(a,b,name):check(name,a.read_bytes()==b.read_bytes())
def close_columns(a,b,name,skip=(),floor=1e-22):
 check(name+'/shape',len(a)==len(b) and all(len(x)==len(y) for x,y in zip(a,b)))
 for c in range(len(a[0])):
  if c in skip:continue
  x=[float(v[c]) for v in a];y=[float(v[c]) for v in b]
  error=max(abs(i-j) for i,j in zip(x,y));scale=max(abs(v) for v in x+y)
  # Column/component infinity norm: avoids meaningless pointwise relative
  # errors at zeros. 5e-10 is below the unchanged 1e-8 nonlinear gate.
  check(name+f'/column{c}',math.isfinite(error) and error<=5e-10*scale+floor,
        max_abs=error,scale=scale)
  metrics.append(dict(group=name,column=c,max_abs=error,scale=scale))
for a,b,n,step in [('direct','retry',1,1),('staggered','resume',2,2),('direct2','retry2',2,2)]:
 for rank in range(n):
  for stem in ['audit_particles','audit_bulk','mature_history','work_qp']:
   file=f'{stem}_{step}_rank{rank}.csv';exact(out(a)/file,out(b)/file,f'{a}/{b}/{file}')
  aa,bb=(next(x for x in records(out(c)/f'lifecycle_rank{rank}.csv') if x['stage']=='accepted' and int(x['step'])==step) for c in [a,b])
  check(f'{a}/{b}/rank{rank}/accepted-lifecycle',aa==bb)
 for file in [f'profiles/fault_{step}.csv',f'common_fe_weak_{step}.csv',f'work_weak_{step}.csv']:
  exact(out(a)/file,out(b)/file,f'{a}/{b}/{file}')
 for file in ['accepted_steps.csv','restored_growth.csv']:
  aa,bb=(next(x for x in records(out(c)/file) if int(x['step'])==step) for c in [a,b])
  check(f'{a}/{b}/{file}',aa==bb)
for case,n,time in [('retry',1,'50'),('retry2',2,'150')]:
 for rank in range(n):
  data=records(out(case)/f'lifecycle_rank{rank}.csv')
  before=next(x for x in data if x['stage']=='backup' and x['step']==('1' if case=='retry' else '2'))
  after=next(x for x in data if x['stage']=='restored_before_advection' and x['time']==time)
  keys=['n','next_id','position_hash','property_hash','integrator_hash','rng_hash','membership_hash','audit_hash','audit_size']
  check(f'{case}/{rank}/rollback',all(before[k]==after[k] for k in keys))
for case,n in [('serial-fixed',1),('mpi',2),('staggered',2)]:
 old=r.parent/'frozen_particle_H'/f'output-{case}'
 for rank in range(n):
  previous=None
  for step in range(3):
   file=f'audit_particles_{step}_rank{rank}.csv';a=rows(old/file)[1:];b=rows(out(case)/file)[1:]
   check(f'{case}/{rank}/{step}/membership',[x[0] for x in a]==[x[0] for x in b])
   close_columns(a,b,f'{case}/{rank}/{step}/particle',skip=(0,))
   current={x[0]:x[3] for x in b}
   if previous is not None:
    check(f'{case}/{rank}/{step}/committed-H',all(current[i]==previous[i] for i in current.keys()&previous.keys()))
   previous=current
   file=f'audit_bulk_{step}_rank{rank}.csv';a=rows(old/file)[1:];b=rows(out(case)/file)[1:]
   check(f'{case}/{rank}/{step}/dofs',[x[:2] for x in a]==[x[:2] for x in b])
   for component in sorted({x[1] for x in a}):
    close_columns([[x[2]] for x in a if x[1]==component],[[x[2]] for x in b if x[1]==component],f'{case}/{rank}/{step}/bulk{component}')
 for step in range(3):
  file=f'profiles/fault_{step}.csv';close_columns(rows(old/file)[1:],rows(out(case)/file)[1:],f'{case}/{step}/profile')
 a=records(old/'accepted_steps.csv');b=records(out(case)/'accepted_steps.csv')
 keys=['time','dt','free','lower_active','newton_updates','krylov_iterations','min_alpha','fresh_linear_checks_passed']
 check(f'{case}/solver-decisions',all(all(x[k]==y[k] for k in keys) for x,y in zip(a,b)))
for rank in range(2):
 exact(out('create')/f'rng-checkpoint-2-rank{rank}.txt',out('resume')/f'rng-resume-2-rank{rank}.txt',f'{rank}/checkpoint-RNG')
 data=[x for x in records(out('staggered')/f'lifecycle_rank{rank}.csv') if x['stage']=='accepted']
 check(f'{rank}/both-events-use-RNG',len({x['rng_hash'] for x in data})==3)
 check(f'{rank}/nonzero-stress',float(data[1]['stress_max'])>0.)
births=[]
for case,n in [('serial-fixed',1),('mpi',2),('retry',1),('direct',1),('staggered',2),('create',2),('resume',2),('retry2',2),('direct2',2)]:
 for rank in range(n):
  file=f'birth_H_rank{rank}.csv';data=records(out(case)/file)
  check(f'{case}/{rank}/newborn-H-before-audit',all(x['H']==x['expected'] and x['audit_already_present']=='0' for x in data))
  check(f'{case}/{rank}/exterior-Hc',all(x['H']==x['Hc'] for x in data if x['active']=='0'))
  births.append(dict(case=case,rank=rank,observations=len(data),active=sum(x['active']=='1' for x in data),exterior=sum(x['active']=='0' for x in data)))
  shutil.copyfile(out(case)/file,results/f'{case}-{file}')
  shutil.copyfile(out(case)/f'lifecycle_rank{rank}.csv',results/f'{case}-lifecycle_rank{rank}.csv')
check('nonvacuous-birth-regions',sum(x['active'] for x in births)>0 and sum(x['exterior'] for x in births)>0)
for geom,n in [('60',1),('60-reverse',2),('45',2),('45-reverse',2),('left',1),('small',2)]:
 case='guard-'+geom+'-qualified';log=(r/'evidence'/f'{case}-np{n}.log').read_text(errors='replace')
 check(case+'/pass-marker','BP3 GEOMETRY PASS:' in log and all(f'BP3 GEOMETRY CHECK COMPLETE rank {i}' in log for i in range(n)))
 shutil.copyfile(r/f'output-{case}/geometry.csv',results/f'{case}-geometry.csv')
for geom in ['60','45']:
 exact(results/f'guard-{geom}-qualified-geometry.csv',results/f'guard-{geom}-reverse-qualified-geometry.csv',f'{geom}/order-independent-frame')
 for step in range(3):
  file=f'profiles/fault_{step}.csv'
  # Resampling retains each anchor and input order; compare physical points.
  a=sorted(rows(out(f'dip-{geom}')/file)[1:],key=lambda x:(float(x[5]),float(x[6])))
  b=sorted(rows(out(f'dip-{geom}-reverse')/file)[1:],key=lambda x:(float(x[5]),float(x[6])))
  close_columns(a,b,f'{geom}/{step}/reversed-physical-profile',skip=(1,))
  for case in [f'dip-{geom}',f'dip-{geom}-reverse']:
   data=records(out(case)/'accepted_steps.csv')
   check(f'{case}/{step}/fresh-residuals',data[step]['fresh_linear_checks_passed']=='1')
for case in ['qualified-old-resume','qualified-changed-resume']:
 log=(r/'evidence'/f'{case}-np2.log').read_text(errors='replace')
 check(case+'/explicit-identity-rejection','BP3 restart requires the same geometry identity' in log)
log=(r/'evidence/qualified-evolving-np2.log').read_text(errors='replace')
check('evolving-fallback','EVOLVING H TRANSFER PASS:' in log)
log=(r/'evidence/unit-profile-bounds-np2.log').read_text(errors='replace')
check('profile-bounds-and-reference-unit-tests',log.count('All tests passed')==2)
if endpoint:
 for suffix in ['', '-reverse']:
  for n in [1,2]:
   case=f'{prefix}-points{suffix}-{n}'
   log=(r/'evidence'/f'{case}-np{n}.log').read_text()
   check(case+'/final-point-regression',all(log.count(f'ENDPOINT HANDOFF PASS rank {i}')==1 for i in range(n)))
 association=json.loads((results/'full_association_comparison.json').read_text())
 check('all-16875-source-admissions-match',association['corrected_order_admission_mismatches']==0)
 for order in ['forward','reverse']:
  check(order+'/preserve-all-previously-active-associations',association[order]['previously_active_preserved_exactly'])
 for suffix in ['', '-reverse']:
  serial=f'dip-45{suffix}-serial';mpi=f'dip-45{suffix}'
  for step in range(3):
   file=f'profiles/fault_{step}.csv'
   close_columns(rows(out(serial)/file)[1:],rows(out(mpi)/file)[1:],f'{mpi}/{step}/one-vs-two-ranks')
  for case,n in [(serial,1),(mpi,2)]:
   log=(r/'evidence'/f'{prefix}-{case}-np{n}.log').read_text()
   check(case+'/endpoint-regression-markers',all(log.count(f'ENDPOINT HANDOFF PASS rank {i}')==3 for i in range(n)))
 for step in range(3):
  file=f'profiles/fault_{step}.csv'
  a=sorted(rows(out('dip-45-serial')/file)[1:],key=lambda x:(float(x[5]),float(x[6])))
  b=sorted(rows(out('dip-45-reverse-serial')/file)[1:],key=lambda x:(float(x[5]),float(x[6])))
  close_columns(a,b,f'45/{step}/serial-reversed-physical-profile',skip=(1,))
 log=(r/f'evidence/{prefix}-unit-np2.log').read_text()
 check('endpoint-unit-tests',log.count('All tests passed')==2)
for name,data in [('checks.json',checks),('comparison_metrics.json',metrics),('birth_counts.json',births)]:
 (results/name).write_text(json.dumps(data,indent=2)+'\n')
failed=[x for x in checks if not x['passed']]
print(f'{len(checks)-len(failed)} passed; {len(failed)} failed geometry, field, lifecycle and MPI checks')
for x in failed:print(x)
raise SystemExit(bool(failed))
