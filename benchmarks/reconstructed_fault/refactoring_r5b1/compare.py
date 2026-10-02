#!/usr/bin/env python3
"""Exact matched-rank surface/coupled data; omit timing and path metadata."""
from pathlib import Path
import json,re,xml.etree.ElementTree as ET
r=Path(__file__).resolve().parent;e=r/'evidence';old=r.with_name('refactoring_r5a2');checks={};counts={}
ansi=re.compile(r'\x1b\[[0-9;]*m')
def log(p):return ansi.sub('',p.read_text(errors='replace')).replace('\x00','')
def trace(s):
 markers=('Iteration ','Fault linear solve:','Relative nonlinear residuals','line search accepted after','Timestep ',
 'All tests passed','verified','derivative error=','relative error=','Normal filter range','raw/filtered row difference=',
 'Reconstructed-fault surface and bulk coupling:','Stage-I rollback after an accepted Newton update:')
 return [x.strip() for x in s.splitlines() if any(m in x for m in markers)]
def work(s):
 rows=[]
 for line in s.splitlines():
  if line.startswith(('Fault quadrature work:','Fault I_h lookup work:','Cell I_h:')) or 'Fault sparse G:' in line:
   rows.append(re.sub(r', (?:preparation|geometry) seconds=[^,\n]*','',line.strip()))
 return rows
def calls(s):return sorted(re.findall(r'\|\s*(Fault:[^|]+?)\s*\|\s*(\d+)\s*\|',s))
def files(d,pattern):return {str(p.relative_to(d)):p.read_bytes() for p in d.glob(pattern)}
def vtu(d):
 data={}
 for p in d.rglob('*.vtu'):
  data[str(p.relative_to(d))]=[(n.attrib, (n.text or '').split()) for n in ET.parse(p).findall('.//DataArray')]
 return data
for rank in (1,2):
 for name in ('unit','dynamic','adiabatic','rate','explicit','filter','singular','singular-current','pressure','rollback','bp3'):
  label=f'{name}-{rank}';reused=name in ('pressure','rollback')
  ar=old if reused else r;av='candidate' if reused else 'reference'
  alabel=f'rollback-original-{rank}' if name=='rollback' else label
  ap=ar/f'evidence/{av}-{alabel}';bp=e/f'candidate-{label}'
  aa,bb=[json.loads(p.with_suffix('.json').read_text()) for p in (ap,bp)]
  a,b=[log(p.with_suffix('.log')) for p in (ap,bp)]
  checks[label+'/outcomes']=aa['exit_code']==bb['exit_code']==(1 if name.startswith('singular') else 0)
  checks[label+'/environment']=aa['environment']==bb['environment']
  checks[label+'/trace']=bool(trace(a)) and trace(a)==trace(b)
  if name=='unit':
   checks[label+'/all-ranks']=b.count('All tests passed (636 assertions in 3 test cases)')==rank
   continue
  if name.startswith('singular'):
   marker='A singular K_V did not produce the expected factorization diagnostic.' if name=='singular' else 'Verified Stage-F singular K_V factorization'
   checks[label+'/expected-marker']=all(marker in x for x in (a,b))
   if name=='singular-current':checks[label+'/observed-lapack-diagnostic']=all('GTTRF failed, info=1, singular pivot vertex=0' in x for x in (a,b))
   continue
  da=ar/f'output-{av}-{alabel}';db=r/f'output-candidate-{label}'
  checks[label+'/statistics']=(da/'statistics').read_text().replace(str(da),'OUTPUT')==(db/'statistics').read_text().replace(str(db),'OUTPUT')
  checks[label+'/work']=work(a)==work(b)
  checks[label+'/call-counts']=bool(calls(a)) and calls(a)==calls(b)
  patterns=['domain_fault_geometry.csv']
  if name=='filter':patterns+=['filter_*.csv']
  if name=='pressure':patterns+=['gauge-state-*.txt']
  if name=='bp3':patterns=['*.csv','profiles/*.csv']
  for pattern in patterns:
   af,bf=files(da,pattern),files(db,pattern)
   if name in ('pressure','rollback') and pattern=='domain_fault_geometry.csv':continue
   checks[label+'/'+pattern]=bool(af) and af==bf;counts[label+'/'+pattern]=len(af)
  if name=='explicit':checks[label+'/explicit-reference-marker']='Sparse B/G basis/random reference actions: verified' in b
  if name=='filter':checks[label+'/filter-marker']='Free-equation normal filter coupling:' in b
  if name=='rollback':checks[label+'/accepted-update-rollback']='Stage-I rollback after an accepted Newton update: verified' in b and 'line search accepted after' in b
  if name=='bp3':
   af,bf=vtu(da),vtu(db);checks[label+'/all-vtu-data']=bool(af) and af==bf;counts[label+'/vtu-files']=len(af)
(e/'comparison.json').write_text(json.dumps(checks,indent=2)+'\n');(e/'compared-files.json').write_text(json.dumps(counts,indent=2)+'\n')
print(len(checks),'checks; failures:',[k for k,v in checks.items() if not v]);assert all(checks.values())
