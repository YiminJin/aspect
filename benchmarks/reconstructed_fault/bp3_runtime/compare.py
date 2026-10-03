#!/usr/bin/env python3
"""Unchanged physical tolerance and exact replay requirements from geometry qualification."""
from pathlib import Path
import csv,json,math
r=Path(__file__).resolve().parent;old=r.with_name('bp3_geometry')
checks=[]
def check(name,ok,**detail):checks.append(dict(check=name,passed=bool(ok),**detail))
def rows(path):return list(csv.reader(path.open()))[1:]
def records(path):return list(csv.DictReader(path.open()))
def close(a,b,name,skip=()):
 check(name+'/shape',len(a)==len(b) and all(len(x)==len(y) for x,y in zip(a,b)))
 for c in range(len(a[0])):
  if c in skip:continue
  x=[float(v[c]) for v in a];y=[float(v[c]) for v in b]
  error=max(abs(i-j) for i,j in zip(x,y));scale=max(abs(v) for v in x+y)
  check(name+f'/{c}',math.isfinite(error) and error<=5e-10*scale+1e-22,max_abs=error,scale=scale)
def output(case):return r/('output-'+case)
for case,base,n in [('dip-60-matched','dip-60',2),('dip-45-serial','dip-45-serial',1),('dip-45-reverse-serial','dip-45-reverse-serial',1),('staggered','staggered',2),('create','create',2),('direct','direct',1),('retry','retry',1),('direct2','direct2',2),('retry2','retry2',2),('resume','resume',2)]:
 a=old/('output-filter-'+base);b=output(case)
 for p in sorted(b.glob('profiles/fault_*.csv')):
  close(rows(a/'profiles'/p.name),rows(p),case+'/'+p.name)
 for rank in range(n):
  for p in sorted(b.glob(f'audit_particles_*_rank{rank}.csv')):
   aa,bb=rows(a/p.name),rows(p)
   check(case+'/'+p.name+'/IDs',[x[0] for x in aa]==[x[0] for x in bb]);close(aa,bb,case+'/'+p.name,skip=(0,))
  for p in sorted(b.glob(f'audit_bulk_*_rank{rank}.csv')):
   aa,bb=rows(a/p.name),rows(p)
   check(case+'/'+p.name+'/DoFs',[x[:2] for x in aa]==[x[:2] for x in bb])
   for c in sorted({x[1] for x in aa}):
    close([[x[2]] for x in aa if x[1]==c],[[x[2]] for x in bb if x[1]==c],case+'/'+p.name+'/component'+c)
  births=records(b/f'birth_H_rank{rank}.csv')
  check(case+f'/{rank}/births',all(x['H']==x['expected'] and x['audit_already_present']=='0' and (x['active']=='1' or x['H']==x['Hc']) for x in births))
 aa,bb=records(a/'accepted_steps.csv'),records(b/'accepted_steps.csv')
 keys=['time','dt','free','lower_active','newton_updates','krylov_iterations','min_alpha','fresh_linear_checks_passed']
 check(case+'/decisions',len(aa)==len(bb) and all(all(x[k]==y[k] for k in keys) for x,y in zip(aa,bb)))
 check(case+'/fresh-residuals',all(x['fresh_linear_checks_passed']=='1' for x in bb))
# Completion values and source associations must not change with loading setup.
import xml.etree.ElementTree as ET
for step in range(3):
 file=f'reconstructed_faults/reconstructed_faults-{step:05d}.vtu'
 if step in (0,2):
  arrays=[ET.parse(folder/file).find('.//DataArray[@Name="previous_I_h"]').text.split()
          for folder in [old/'output-filter-dip-60',output('dip-60-matched')]]
  check(f'60/{step}/completed-Ih-exact',arrays[0]==arrays[1])
 for rank in range(2):
  file=f'work_qp_{step}_rank{rank}.csv'
  aa,bb=(records(folder/file) for folder in [old/'output-filter-dip-60',output('dip-60-matched')])
  keys=['cell','qp','x','y','source_active','segment','xi','phi','Ih','chi']
  check(f'60/{step}/{rank}/source-and-normalization-exact',len(aa)==len(bb) and all(all(x[k]==y[k] for k in keys) for x,y in zip(aa,bb)))
# Replay must remain bitwise exact inside the new implementation.
for left,right,n,step in [('direct','retry',1,1),('staggered','resume',2,2),('direct2','retry2',2,2)]:
 for rank in range(n):
  for stem in ['audit_particles','audit_bulk','mature_history','work_qp']:
   file=f'{stem}_{step}_rank{rank}.csv'
   check(left+'/'+right+'/'+file,(output(left)/file).read_bytes()==(output(right)/file).read_bytes())
  a,b=(next(x for x in records(output(c)/f'lifecycle_rank{rank}.csv') if x['stage']=='accepted' and int(x['step'])==step) for c in [left,right])
  check(left+'/'+right+f'/{rank}/lifecycle',a==b)
 for file in [f'profiles/fault_{step}.csv',f'common_fe_weak_{step}.csv',f'work_weak_{step}.csv']:
  check(left+'/'+right+'/'+file,(output(left)/file).read_bytes()==(output(right)/file).read_bytes())
for case,n,time,step in [('retry',1,'50','1'),('retry2',2,'150','2')]:
 for rank in range(n):
  data=records(output(case)/f'lifecycle_rank{rank}.csv')
  before=next(x for x in data if x['stage']=='backup' and x['step']==step)
  after=next(x for x in data if x['stage']=='restored_before_advection' and x['time']==time)
  keys=['n','next_id','position_hash','property_hash','integrator_hash','rng_hash','membership_hash','audit_hash','audit_size']
  check(case+f'/{rank}/rollback',all(before[k]==after[k] for k in keys))
for rank in range(2):
 check(f'{rank}/checkpoint-RNG',(output('create')/f'rng-checkpoint-2-rank{rank}.txt').read_bytes()==(output('resume')/f'rng-resume-2-rank{rank}.txt').read_bytes())
 data=[x for x in records(output('staggered')/f'lifecycle_rank{rank}.csv') if x['stage']=='accepted']
 check(f'{rank}/nontrivial-RNG-stress',len({x['rng_hash'] for x in data})==3 and float(data[1]['stress_max'])>0.)
(r/'results/comparison.json').write_text(json.dumps(checks,indent=2)+'\n')
failed=[x for x in checks if not x['passed']]
print(f'{len(checks)-len(failed)}/{len(checks)} passed')
for x in failed:print(x)
raise SystemExit(bool(failed))
