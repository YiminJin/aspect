#!/usr/bin/env python3
"""Reduce local raw diagnostics to compact, common-point comparisons."""
from pathlib import Path
import csv,json,math,shutil
import numpy as np
r=Path(__file__).resolve().parent
out=r/'results';out.mkdir(exist_ok=True)
def rows(p):return list(csv.DictReader(p.open()))
def write(name,data):
 if data:
  with (out/name).open('w') as f:
   w=csv.DictWriter(f,lineterminator="\n",fieldnames=list(data[0]));w.writeheader();w.writerows(data)
summary=[];stages=[];bins=[];inserts=[]
for base in ['regular','random-5432','random-5433','random-5434']:
 for policy in ['','-limited']:
  case=base+policy;d=r/('output-'+case)
  if not (d/'cells_rank0.csv').exists():continue
  a=rows(d/'cells_rank0.csv');events=rows(d/'events_rank0.csv')
  if events[-1]['step']!='20':continue
  geometry=[x for x in a if x['family']=='0' and x['site']=='LLS_support']
  s=dict(case=case,initial=int(events[0]['total']),after_initial=int(events[1]['total']),startup_births=int(events[1]['born_or_received']),startup_removals=int(events[1]['lost_or_sent']),final=int(events[-1]['total']),transport_births=sum(int(x['born_or_received']) for x in events[2:]),transport_losses=sum(int(x['lost_or_sent']) for x in events[2:]),min_sigma_ratio=min(float(x['sigma_ratio']) for x in geometry),underdetermined_cells=sum(int(x['n'])<3 for x in geometry),ratio_below_1e_8=sum(float(x['sigma_ratio'])<1e-8 for x in geometry))
  for fam in range(3):s[['constant','affine','curved'][fam]+'_max_error_over_1e8']=max(float(x['max_error'])/1e8 for x in a if int(x['family'])==fam)
  summary.append(s)
  for stage,step in [('generated','0'),('initial_management','0')]+[('transport',str(k)) for k in range(21)]:
   for site in ['LLS_support','LLS_gauss3','Q2_support','Q2_gauss3']:
    for family in range(3):
     b=[x for x in a if x['stage']==stage and x['step']==step and x['site']==site and int(x['family'])==family]
     if not b:continue
     weights=np.array([float(x['volume']) for x in b]);weights/=weights.sum()
     stages.append(dict(case=case,stage=stage,step=step,site=site,family=family,count_min=min(int(x['n']) for x in b),count_max=max(int(x['n']) for x in b),min_ratio=min(float(x['sigma_ratio']) for x in b),rms_error_over_1e8=math.sqrt(sum(w*float(x['rms_error'])**2 for w,x in zip(weights,b)))/1e8,max_error_over_1e8=max(float(x['max_error']) for x in b)/1e8,mean_over_1e8=sum(w*float(x['mean']) for w,x in zip(weights,b))/1e8,reference_mean_over_1e8=sum(w*float(x['reference_mean']) for w,x in zip(weights,b))/1e8,min_over_1e8=min(float(x['min']) for x in b)/1e8,max_over_1e8=max(float(x['max']) for x in b)/1e8))
  for n in sorted({int(x['n']) for x in a}):
   b=[x for x in a if int(x['n'])==n and x['family']=='2' and x['site']=='LLS_support']
   bins.append(dict(case=case,n=n,cell_snapshots=len(b),min_ratio=min(float(x['sigma_ratio']) for x in b),rms_error_over_1e8=math.sqrt(sum(float(x['rms_error'])**2 for x in b)/len(b))/1e8,max_error_over_1e8=max(float(x['max_error']) for x in b)/1e8))
  unique={tuple(x[k] for k in ['step','cell','x','y','property']):x for x in rows(d/'insertion_rank0.csv')}
  for prop in sorted({x['property'] for x in unique.values()},key=int):
   b=[x for x in unique.values() if x['property']==prop]
   inserts.append(dict(case=case,property=prop,proposals=len(b),min_input=min(float(x['input_min']) for x in b),max_input=max(float(x['input_max']) for x in b),min_proposal=min(float(x['proposal']) for x in b),max_proposal=max(float(x['proposal']) for x in b),max_error_over_1e8=max(abs(float(x['proposal'])-float(x['reference'])) for x in b)/1e8))
  for pattern in ['worst_generated*.csv','worst_initial_management*.csv','worst_transport_step20_*.csv']:
   for p in d.glob(pattern):shutil.copyfile(p,out/(case+'-'+p.name))
write('comparison.csv',summary);write('stage_fields.csv',stages);write('count_conditioned.csv',bins);write('insertion_summary.csv',inserts)
runs=[]
for p in sorted((r/'evidence').glob('*.json')):
 x=json.loads(p.read_text());runs.append(dict(label=p.stem,exit_code=x['exit_code'],seconds=x['seconds'],child_max_rss_kib=x.get('child_max_rss_kib',''),command=' '.join(x['command'])))
write('runs.csv',runs)
print(json.dumps(summary,indent=2))
