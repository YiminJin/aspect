#!/usr/bin/env python3
"""Exact matched-rank results; timings, paths and executable metadata excluded."""
from pathlib import Path
import json,re
r=Path(__file__).resolve().parent;e=r/'evidence';old=r.with_name('refactoring_r5a1');checks={}
ansi=re.compile(r'\x1b\[[0-9;]*m')
def log(p):return ansi.sub('',p.read_text(errors='replace')).replace('\x00','')
def decisions(s):
 markers=('Iteration ', 'Fault linear solve:', 'Relative nonlinear residuals', 'line search accepted after',
 'Solving temperature system', 'Phase-field residual target:', 'Timestep ', 'All tests passed',
 'Distributed I_h:', 'cache:', 'Surface system:', 'surface system:', 'Coupled pressure gauge:',
 'Stage-I rollback after an accepted Newton update:', 'Reconstructed-fault Stage-I solve:',
 'Changed loading:', 'Stage-J history resolution:', 'Stage-J feedback step')
 return [line.strip() for line in s.splitlines() if any(m in line for m in markers)]
def work(s):
 rows=[]
 for line in s.splitlines():
  if line.startswith(('Fault quadrature work:', 'Fault I_h lookup work:', 'Cell I_h:')):
   line=re.sub(r', (?:preparation|geometry) seconds=[^,\n]*','',line)
   rows.append(line)
 return rows
def build_counts(s):
 return re.findall(r'\| (Fault: Cache build(?: total)?)\s*\|\s*(\d+)\s*\|',s)
def files(d,pattern):return {p.name:p.read_bytes() for p in d.glob(pattern)}
for rank in (1,2):
 for name in ('unit','ih','no-composition','surface','cache-remote','cache-cell','pressure','rollback-original'):
  label=f'{name}-{rank}'; reused=name in ('pressure','rollback-original')
  ar,av=(old,'candidate') if reused else (r,'reference')
  ap=ar/f'evidence/{av}-{label}';bp=e/f'candidate-{label}'
  a,b=log(ap.with_suffix('.log')),log(bp.with_suffix('.log'))
  checks[label+'/environment']=json.loads(ap.with_suffix('.json').read_text())['environment']==json.loads(bp.with_suffix('.json').read_text())['environment']
  checks[label+'/passed']=all(json.loads(p.with_suffix('.json').read_text())['exit_code']==0 for p in (ap,bp))
  checks[label+'/decisions']=bool(decisions(a)) and decisions(a)==decisions(b)
  if name=='unit':
   checks[label+'/all-ranks']=b.count('All tests passed (834 assertions in 16 test cases)')==rank
  else:
   da=ar/f'output-{av}-{label}';db=r/f'output-candidate-{label}'
   checks[label+'/statistics']=(da/'statistics').read_bytes()==(db/'statistics').read_bytes()
   # Always compare the existing coarse cache-build count. Detailed timers are
   # enabled only for the newly matched projection cases.
   ca,cb=build_counts(a),build_counts(b)
   if reused:ca=[x for x in ca if x[0]=='Fault: Cache build'];cb=[x for x in cb if x[0]=='Fault: Cache build']
   checks[label+'/build-counts']=bool(ca) and ca==cb
   if not reused:checks[label+'/work']=bool(work(a)) and work(a)==work(b)
   if name.startswith('cache-'):
    aa,bb=files(da,'cache_counts_*.csv'),files(db,'cache_counts_*.csv')
    checks[label+'/cache-counters']=len(aa)==rank and aa==bb
   if name=='surface':checks[label+'/surface-marker']='verified' in b and 'Surface' in b
   if name=='pressure':
    aa,bb=files(da,'gauge-state-*.txt'),files(db,'gauge-state-*.txt')
    checks[label+'/histories']=bool(aa) and aa==bb
   if name=='rollback-original':checks[label+'/accepted-update-rollback']=all('line search accepted after' in s and 'Stage-I rollback after an accepted Newton update: verified' in s for s in (a,b))
a,b=[log(e/f'{v}-cold-warm-2.log') for v in ('reference','candidate')]
checks['cold-warm/environment']=json.loads((e/'reference-cold-warm-2.json').read_text())['environment']==json.loads((e/'candidate-cold-warm-2.json').read_text())['environment']
for v,s in zip(('reference','candidate'),(a,b)):
 checks[v+'/cold-warm-pass']=json.loads((e/f'{v}-cold-warm-2.json').read_text())['exit_code']==0 and 'Particle projection cold/warm cache: verified' in s
 data=files(r/f'output-{v}-cold-warm-2','particle-cache-rank-*.txt')
 checks[v+'/cold-warm-counts']=len(data)==2 and all(x.startswith(b'cold rebuilds 1 warm rebuilds 0\n') for x in data.values())
checks['cold-warm/exact-values-support']=files(r/'output-reference-cold-warm-2','particle-cache-rank-*.txt')==files(r/'output-candidate-cold-warm-2','particle-cache-rank-*.txt')
checks['cold-warm/work']=work(a)==work(b) and len([x for x in work(a) if x.startswith('Fault quadrature work:')])==2
checks['cold-warm/build-counts']=bool(build_counts(a)) and build_counts(a)==build_counts(b)
a=log(old/'evidence/candidate-restart.log');b=log(e/'candidate-restart.log')
checks['restart/environment']=json.loads((old/'evidence/candidate-restart.json').read_text())['environment']==json.loads((e/'candidate-restart.json').read_text())['environment']
checks['restart/known-failure']=json.loads((e/'candidate-restart.json').read_text())['exit_code']==1 and all('Newton line search exhausted all admissible candidates' in s and 'Nonlinear solver failed to converge' in s for s in (a,b))
checks['restart/restored']=all('Stage-J checkpoint histories, V, geometry, and bulk: verified' in s for s in (a,b))
checks['restart/trace']=bool(decisions(a)) and decisions(a)==decisions(b)
for name in ('mesh','mesh.info','mesh_fixed.data','mesh_variable.data','resume.z'):
 original=r.with_name('restart_investigation')/'output-create-one/restart/01'/name
 checks['restart/preserved/'+name]=(r/'output-candidate-restart/restart/01'/name).read_bytes()==original.read_bytes()
(e/'comparison.json').write_text(json.dumps(checks,indent=2)+'\n')
print(len(checks),'checks; failures:',[k for k,v in checks.items() if not v]);assert all(checks.values())
