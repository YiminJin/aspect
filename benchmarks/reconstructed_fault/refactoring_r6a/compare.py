#!/usr/bin/env python3
"""Exact versions and off/on comparisons; keep every scientific CSV column."""
from pathlib import Path
import json,re,csv,xml.etree.ElementTree as ET
r=Path(__file__).resolve().parent;e=r/'evidence';checks={};row_counts={}
ansi=re.compile(r'\x1b\[[0-9;]*m')
def log(label):return ansi.sub('',(e/(label+'.log')).read_text(errors='replace')).replace('\x00','')
def trace(s):
 marks=('Fault linear solve:','Relative nonlinear residuals','line search accepted after','*** Timestep','Stage-I rollback after an accepted Newton update:')
 return [x.strip() for x in s.splitlines() if any(m in x for m in marks)]
def counts(s):
 lines=[re.sub(r', (?:preparation|geometry) seconds=[^,\n]*','',x.strip()) for x in s.splitlines() if x.startswith(('Fault quadrature work:','Fault I_h lookup work:','Cell I_h:')) or 'Fault sparse G:' in x]
 return lines,sorted(re.findall(r'\|\s*(Fault:[^|]+?)\s*\|\s*(\d+)\s*\|',s))
def diagnostic(p):return p.name.startswith(('stress_update_','continued_source_history_','stress_transfer_','stress_trace_cells_'))
def fields(d):
 data={str(p.relative_to(d)):p.read_bytes() for pattern in ('*.csv','profiles/*.csv') for p in d.glob(pattern) if p.is_file() and not diagnostic(p)}
 for p in d.rglob('*.vtu'):
  data[str(p.relative_to(d))]=[(n.attrib,(n.text or '').split()) for n in ET.parse(p).findall('.//DataArray')]
 return data
def compare(a,b,key,rows=False):
 aa,bb=[json.loads((e/(v+'.json')).read_text()) for v in (a,b)]
 la,lb=log(a),log(b);da,db=[r/('output-'+v) for v in (a,b)]
 checks[key+'/outcome']=aa['exit_code']==bb['exit_code']==0
 checks[key+'/trace']=bool(trace(la)) and trace(la)==trace(lb)
 checks[key+'/work']=counts(la)==counts(lb)
 checks[key+'/statistics']=[line.split() for line in (da/'statistics').read_text().replace(str(da),'OUTPUT').splitlines()]==[line.split() for line in (db/'statistics').read_text().replace(str(db),'OUTPUT').splitlines()]
 if '-bp3-' in a:
  fa,fb=fields(da),fields(db);checks[key+'/fields']=bool(fa) and fa==fb
  if rows:
   fa={p.name:p.read_bytes() for p in da.glob('*') if p.is_file() and diagnostic(p)}
   fb={p.name:p.read_bytes() for p in db.glob('*') if p.is_file() and diagnostic(p)}
   checks[key+'/diagnostic-payloads']=bool(fa) and fa==fb
 else:checks[key+'/rollback-marker']=all('Stage-I rollback after an accepted Newton update: verified' in s for s in (la,lb))
 if key.startswith('matched/'):
  checks[key+'/environment']=aa['environment']==bb['environment']

cases=[f'bp3-{mode}-{rank}' for mode,rank in [('off',1),('on',1),('off',2),('on',2),('missing',1),('blocked',2)]]+ [f'rollback-{mode}-{rank}' for mode in ('off','on') for rank in (1,2)]
for case in cases:compare('reference-'+case,'candidate-'+case,'matched/'+case,rows=case.startswith('bp3-') and '-off-' not in case)
for v in ('reference','candidate'):
 for rank in (1,2):
  for family in ('bp3','rollback'):compare(f'{v}-{family}-off-{rank}',f'{v}-{family}-on-{rank}',f'neutral/{v}-{family}-{rank}')
 for mode,rank in [('missing',1),('blocked',2)]:compare(f'{v}-bp3-off-{rank}',f'{v}-bp3-{mode}-{rank}',f'neutral/{v}-{mode}-{rank}')
 for mode,rank in [('off',1),('on',1),('off',2),('on',2),('missing',1),('blocked',2)]:
  d=r/f'output-{v}-bp3-{mode}-{rank}';key=f'rows/{v}-{mode}-{rank}'
  if mode=='off':checks[key+'/disabled-no-streams']=not any(d.glob('stress_update_*')) and not any(d.glob('continued_source_history_*'));continue
  for prefix,columns in [('stress_update',30),('continued_source_history',20)]:
   files=sorted(d.glob(prefix+'*'))
   checks[key+'/'+prefix+'/file-count']=len(files)==6*rank
   if mode=='blocked':checks[key+'/'+prefix+'/silent-open-failure']=all(p.is_dir() for p in files);continue
   counts_=[]
   for p in files:
    with p.open() as f:rows=list(csv.reader(f))
    checks[key+'/'+p.name+'/schema']=all(len(row)==columns for row in rows) and len(rows[0])==columns
    counts_.append(len(rows)-1)
   row_counts[key+'/'+prefix]=counts_
   checks[key+'/'+prefix+'/selection']=all(n==0 if mode=='missing' and prefix=='stress_update' else n>0 for n in counts_)
  if mode=='missing':checks[key+'/missing-input']=not any(d.glob('stress_trace_cells_*'))
(e/'comparison.json').write_text(json.dumps(checks,indent=2)+'\n');(e/'row-counts.json').write_text(json.dumps(row_counts,indent=2)+'\n')
print(len(checks),'checks; failures:',[k for k,v in checks.items() if not v]);assert all(checks.values())
