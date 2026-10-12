#!/usr/bin/env python3
"""Numerical gates independent of expected console output."""
import csv,json,math,sys,re
from pathlib import Path
here=Path(__file__).resolve().parent;build=here.parents[2]/'build-propagation-integration'
def rows(path):
 with path.open() as f:result=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
 assert result and all(math.isfinite(x) for r in result for x in r.values()),str(path)
 return result
def nodes(output,step):
 result={}
 for p in output.glob(f'phase-{step}-*.csv'):
  for r in rows(p):
   key=(r['x'],r['y']);assert key not in result
   result[key]=r
 assert result
 return result
def compare(a,b,fields=('phi',),tol=1.298e-11):
 assert a.keys()==b.keys()
 error=max(abs(a[k][f]-b[k][f]) for k in a for f in fields)
 assert error<=tol,(error,tol)
 return error
def assess(label,kind):
 output=build/('output-'+label);summary={}
 allowance={'straight':4.922e-12,'padded':1.298e-11}[kind]
 for step in range(5):
  r=rows(output/f'metrics-{step}.csv')[0]
  assert 0<=r['min']<=r['max']<=1
  assert -3e-14<=r['cpdi_min']<=r['cpdi_max']<=1+3e-14
  assert max(r[k] for k in ('free','dual','complementarity'))<=1e-11
  assert r['healing']<=allowance
  assert r['Ih_min']>0 and r['Ih_cold_difference']==0
  if step in (1,3):assert r['change']<=allowance
  if step in (2,4):assert r['change']>100*allowance
  summary[str(step)]=r
 ref=rows(here/'bounds_evidence'/f'bounds_{kind}_final.bound-nodes.csv')
 for step,field in [(0,'relaxed'),(4,'bounded_final')]:
  actual=nodes(output,step)
  error=max(abs(actual[(r['x'],r['y'])]['phi']-r[field]) for r in ref)
  assert error<=allowance,(label,step,error,allowance)
  summary[f'reference_error_{step}']=error
 for step in (3,4):
  if list(output.glob(f'phase-resume-{step}-*.csv')):
   summary[f'restart_error_{step}']=compare(nodes(output,step),nodes(output,f'resume-{step}'),('phi','old','older'),allowance)
 return summary
if __name__=='__main__':
 straight,padded,mpi,support_case=sys.argv[1:];result={}
 result['straight']=assess(straight,'straight');result['padded']=assess(padded,'padded');result['mpi']=assess(mpi,'padded')
 result['mpi_errors']={str(k):compare(nodes(build/('output-'+padded),k),nodes(build/('output-'+mpi),k),('phi','old','older')) for k in range(5)}
 support_log=(here/'integration_evidence'/(support_case+'.log')).read_text()
 support=[{k:float(v) for k,v in re.findall(r'(\w+)=([\deE+.-]+)',line)} for line in support_log.splitlines() if line.startswith('P1 support:')]
 assert len(support)>=3 and all(math.isfinite(v) for r in support for v in r.values())
 assert all('restart_error_'+str(k) in result[x] for x in ('padded','mpi') for k in (3,4))
 result['support']=support
 result['algebraic_phase_checks_pass']=True
 result['support_gate']=all(r['missed_strip_qp_phi']<=1e-6 and r['missed_strip_particle_phi']<=1e-6 for r in support)
 result['P1_production_gate']=result['support_gate'];result['mechanical_H_qualified']=False
 print(json.dumps(result,indent=2))
 sys.exit(0 if result["P1_production_gate"] else 2)
