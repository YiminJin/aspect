#!/usr/bin/env python3
"""Exact matched runs; distinguish off/on presentation and added audit work."""
from pathlib import Path
import csv,json,re,xml.etree.ElementTree as ET
r=Path(__file__).resolve().parent;e=r/'evidence';checks={};rows={};work={}
ansi=re.compile(r'\x1b\[[0-9;]*m')
def log(label):return ansi.sub('',(e/(label+'.log')).read_text(errors='replace')).replace('\x00','')
def trace(s,neutral=False):
 marks=('Fault linear solve:','Relative nonlinear residuals','line search accepted after','*** Timestep','Stage-I rollback after an accepted Newton update:')
 result=[]
 for line in s.splitlines():
  if any(m in line for m in marks):
   line=line.strip()
   if neutral and 'Fault linear solve:' in line:
    # The switch intentionally changes precision and adds detailed fields.
    values=dict(re.findall(r'(iterations|fresh|target)=([^, ]+)',line))
    line=(values['iterations'],format(float(values['fresh']),'.6e'),format(float(values['target']),'.6e'))
   result.append(line)
 return result
def counts(s):
 lines=[re.sub(r', (?:preparation|geometry) seconds=[^,\n]*','',x.strip()) for x in s.splitlines() if x.startswith(('Fault quadrature work:','Fault I_h lookup work:','Cell I_h:')) or 'Fault sparse G:' in x]
 return lines,sorted(re.findall(r'\|\s*(Fault:[^|]+?)\s*\|\s*(\d+)\s*\|',s))
def fields(d):
 result={str(p.relative_to(d)):p.read_bytes() for pattern in ('*.csv','profiles/*.csv') for p in d.glob(pattern) if p.is_file() and not p.name.startswith('nonlinear_bounds_')}
 for p in d.rglob('*.vtu'):
  result[str(p.relative_to(d))]=[(n.attrib,(n.text or '').split()) for n in ET.parse(p).findall('.//DataArray')]
 return result
def diagnostics(s):
 marks=('Fault bound audit:','Fault nonlinear residual:','Fault trial merit:')
 return [x.strip() for x in s.splitlines() if any(m in x for m in marks)]
def compare(a,b,key,neutral=False):
 aa,bb=[json.loads((e/(v+'.json')).read_text()) for v in (a,b)]
 la,lb=log(a),log(b);da,db=[r/('output-'+v) for v in (a,b)]
 checks[key+'/outcome']=aa['exit_code']==bb['exit_code']==0
 checks[key+'/solver-trace']=bool(trace(la)) and trace(la,neutral)==trace(lb,neutral)
 checks[key+'/statistics']=[line.split() for line in (da/'statistics').read_text().replace(str(da),'OUTPUT').splitlines()]==[line.split() for line in (db/'statistics').read_text().replace(str(db),'OUTPUT').splitlines()]
 ca,cb=counts(la),counts(lb)
 if not neutral:
  checks[key+'/work']=ca==cb
  checks[key+'/diagnostic-log']=diagnostics(la)==diagnostics(lb)
  checks[key+'/environment']=aa['environment']==bb['environment']
  fa,fb=[{p.name:p.read_bytes() for p in d.glob('nonlinear_bounds_*.csv') if p.is_file()} for d in (da,db)]
  checks[key+'/bound-payloads']=fa==fb
 else:work[key]={'equal':ca==cb,'off':ca,'on':cb}
 if '-bp3-' in a:
  fa,fb=fields(da),fields(db);checks[key+'/physical-history-fields']=bool(fa) and fa==fb
 else:checks[key+'/rollback-marker']=all('Stage-I rollback after an accepted Newton update: verified' in s for s in (la,lb))
cases=[f'bp3-{m}-{n}' for m,n in [('off',1),('on',1),('off',2),('on',2),('blocked',2)]]+[f'rollback-{m}-{n}' for m in ('off','on') for n in (1,2)]
for c in cases:compare('reference-'+c,'candidate-'+c,'matched/'+c)
for v in ('reference','candidate'):
 for n in (1,2):
  for family in ('bp3','rollback'):compare(f'{v}-{family}-off-{n}',f'{v}-{family}-on-{n}',f'neutral/{v}-{family}-{n}',True)
 # Both selector values are present: same on-mode behavior despite I/O failure.
 compare(f'{v}-bp3-on-2',f'{v}-bp3-blocked-2',f'blocked/{v}',True)
 checks[f'blocked/{v}/work']=counts(log(f'{v}-bp3-on-2'))==counts(log(f'{v}-bp3-blocked-2'))
 checks[f'blocked/{v}/diagnostic-log']=diagnostics(log(f'{v}-bp3-on-2'))==diagnostics(log(f'{v}-bp3-blocked-2'))
 for c in cases:
  d=r/f'output-{v}-{c}';files=sorted(d.glob('nonlinear_bounds_*.csv'));key=f'rows/{v}-{c}'
  if '-off-' in c:checks[key+'/absent']=not files;continue
  if '-blocked-' in c:checks[key+'/silent-open-failure']=len(files)==7 and all(p.is_dir() for p in files);continue
  checks[key+'/nonempty']=bool(files)
  if c.startswith('bp3'):checks[key+'/seven-timesteps']=len(files)==7
  rows[key]={}
  for p in files:
   with p.open() as f:data=list(csv.reader(f))
   checks[key+'/'+p.name]=len(data)>1 and data[0]==['iteration','fault','vertex','V','dV','prescribed','lower_active','Fmin_weak_density','alpha_max','bulk','surface'] and all(len(row)==11 for row in data)
   rows[key][p.name]=len(data)-1
(e/'comparison.json').write_text(json.dumps(checks,indent=2)+'\n')
(e/'row-counts.json').write_text(json.dumps(rows,indent=2)+'\n')
(e/'off-on-work.json').write_text(json.dumps(work,indent=2)+'\n')
print(len(checks),'checks; failures:',[k for k,v in checks.items() if not v]);assert all(checks.values())
