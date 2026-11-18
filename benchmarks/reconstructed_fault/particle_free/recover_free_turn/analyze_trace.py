#!/usr/bin/env python3
"""Reduce sparse recovery diagnostics; keep current and trial limiting rows distinct."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,re,csv
import numpy as np
root=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'bulk_tip'))
from analyze_resolution_matched import gather
label=sys.argv[1] if len(sys.argv)>1 else 'recover-free-turn-trace'
path=root/'build-particle-free/bulk-tip/runs'/label/'output'
out=root/'benchmarks/reconstructed_fault/particle_free/evidence/recover_free_turn_step16'
rows=gather(path,'tip-search-rows-r',['iteration','shift','alpha','row'])
records=[]
for iteration in np.unique(rows.iteration):
 for shift in np.unique(rows.shift[rows.iteration==iteration]):
  for alpha in np.unique(rows.alpha[(rows.iteration==iteration)&(rows.shift==shift)]):
   selected=rows[(rows.iteration==iteration)&(rows.shift==shift)&(rows.alpha==alpha)]
   old=selected[np.argmax(abs(selected.current_projected))]
   new=selected[np.argmax(abs(selected.trial_projected))]
   actual=selected.trial_gradient_over_mass-selected.current_gradient_over_mass
   predicted=selected.tangent_over_mass
   err=abs(actual-predicted)/np.maximum(np.maximum(abs(actual),abs(predicted)),1e-30)
   records.append(dict(iteration=int(iteration),shift=float(shift),alpha=float(alpha),
     current_row=int(old.row),current_cell=str(old.cell),current_i=int(old.i),current_x=float(old.x),current_y=float(old.y),
     current_max=float(abs(old.current_projected)),current_trial_residual=float(old.trial_projected),
     trial_row=int(new.row),trial_cell=str(new.cell),trial_i=int(new.i),trial_x=float(new.x),trial_y=float(new.y),
     trial_current=float(new.current),trial_gap=float(new.gap),trial_displacement=float(new.direction),
     trial_max=float(abs(new.trial_projected)),trial_predicted=float(new.predicted_projected),max_linear_predicted=float(max(abs(selected.predicted_projected))),
     trial_current_gradient=float(new.current_gradient_over_mass),trial_gradient=float(new.trial_gradient_over_mass),
     trial_current_status=int(new.current_status),trial_status=int(new.trial_status),
     max_row_derivative_relative_error=float(max(err))))
with (out/(label+'-row-summary.csv')).open('w') as stream:
 writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
small=rows[(rows.iteration==36)&(rows.shift==0)&(rows.alpha<=1.00001e-5)]
fd=[]
for r in small:
 actual=r.trial_gradient_over_mass-r.current_gradient_over_mass
 predicted=r.tangent_over_mass
 fd.append(dict(row=int(r.row),cell=str(r.cell),i=int(r.i),alpha=float(r.alpha),
  current_value=float(r.current),gap=float(r.gap),current_status=int(r.current_status),trial_status=int(r.trial_status),
  actual_change=float(actual),linear_change=float(predicted),relative_error=float(abs(actual-predicted)/max(abs(actual),abs(predicted),1e-30)),
  subtraction_roundoff=float(4*np.finfo(float).eps*(abs(r.trial_gradient_over_mass)+abs(r.current_gradient_over_mass)))))
(out/(label+'-one-sided.json')).write_text(json.dumps(fd,indent=2)+'\n')
log=(path/'log.txt').read_text();iteration=None;trials=[]
for line in log.splitlines():
 m=re.search(r'Bulk tip coupled Newton (\d+):',line)
 if m:iteration=int(m[1])
 if 'Tip search trial:' in line:
  fields={k.strip():float(v) for k,v in re.findall(r'([^:,]+)=([^,\s]+)',line)}
  fields['iteration']=iteration
  fields['functional_pass']=fields['functional change']<=fields['Armijo']+fields['roundoff']
  trials.append(fields)
(out/(label+'-search-trials.json')).write_text(json.dumps(trials,indent=2)+'\n')
print('Sparse rows',len(rows),'trials',len(trials))
for i in (30,35,36,37,38,39):
 for r in records:
  if r['iteration']==i and r['alpha'] in (1.,.5,.00390625,.0009765625,.00048828125):print(json.dumps(r))
print('Functional failures',sum(not r['functional_pass'] for r in trials))

independent=[];alternatives=[]
for line in log.splitlines():
 if 'Tip independent functional:' in line or 'Tip alternative comparison:' in line:
  fields={k.strip():float(v) for k,v in re.findall(r'([^:,]+)=([^,\s]+)',line)}
  (independent if 'independent functional' in line else alternatives).append(fields)
(out/(label+'-functional.json')).write_text(json.dumps(independent,indent=2)+'\n')
(out/(label+'-alternative.json')).write_text(json.dumps(alternatives,indent=2)+'\n')
print('Independent functional records',len(independent),'alternative trials',len(alternatives))
for r in independent:
 if r['iteration']==36 and r['shift']==0 and r['alpha']<=1e-5:print('One-sided functional',r)
print('Alternative comparisons',alternatives)
