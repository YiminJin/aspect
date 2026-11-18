#!/usr/bin/env python3
"""Early native acceleration: stage rates/tractions, endpoint slip, own backgrounds."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-shortwave-mpl')
from pathlib import Path
import json,csv,hashlib,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];OUT=HERE.parent/'evidence/stage4_shortwave/native-pilot';OUT.mkdir(exist_ok=True)
def read(p):return np.atleast_1d(np.genfromtxt(p,delimiter=',',names=True))
def crossing(t,y,level):
    ids=np.flatnonzero(y>=level)
    if not len(ids):return None
    i=ids[0]
    return float(t[i] if i==0 else t[i-1]+(t[i]-t[i-1])*(level-y[i-1])/(y[i]-y[i-1]))
data={};records=[]
for nx in (256,512):
 for law in ('weakening','neutral'):
  for case in ('background','perturbed'):
   label=f'stage4-native-{nx}-{law}-{case}';run=ROOT/'build-particle-free/bulk-tip/runs'/label
   if not (run/'command.json').exists():continue
   command=json.loads((run/'command.json').read_text());out=run/'output'
   if not (out/'pilot-summary.csv').exists():continue
   summary=read(out/'pilot-summary.csv');data[nx,law,case]=(summary,out)
   preload=read(out/'preload-checks.csv')[0];accepted=read(out/'bulk-accepted.csv');initial=read(out/'tip-initial.csv')[0]
   power=read(out/'tip-power.csv')
   time=summary['stage_time'];end=int(summary['step'][-1]);sections=read(out/f'tip-sections-{end:04d}.csv')
   positive=sum(sections['ds']*sections['positive_w']);negative=sum(sections['ds']*sections['negative_w'])
   observed=json.loads((run/'completion-observed.json').read_text()) if (run/'completion-observed.json').exists() else {}
   row=dict(nx=nx,law=law,case=case,exit_code=command.get('exit_code'),seconds=command.get('seconds',observed.get('solver_wall_seconds')),
    completion_observed=observed.get('normal_termination_and_snapshot',False),
    accepted_steps=end,endpoint_time=float(summary['time'][-1]),last_rate_stage_time=float(time[-1]),
    initial_peak_rate=float(summary['peak_rate'][0]),last_peak_stage_rate=float(summary['peak_rate'][-1]),
    time_to_peak_rate_4e_4=crossing(time,summary['peak_rate'],4e-4),
    integrated_slip=float(summary['integrated_slip'][-1]),traction_change=float(summary['mean_patch_resistance'][-1]-summary['mean_patch_resistance'][0]),
    mixed_equilibrium=float(preload['relative_equilibrium_residual']),preload_work_error=float(preload['relative_work_error']),
    initial_bulk_and_cohesive_storage=float(preload['storage']),initial_fracture=float(initial['fracture']),
    maximum_fracture_inventory_change=float(max(abs(accepted['fracture']-initial['fracture']))),
    maximum_accepted_mechanical_residual=float(max(accepted['mechanical_residual'])),
    maximum_power_adjoint_error=float(max(abs(power['power_adjoint_error']))),
    maximum_section_residual_work=float(max(abs(power['section_residual_work']))),
    maximum_publication_count_error=float(max(abs(accepted['publications']-accepted['step']))),
    friction_inverse_error=float(preload['friction_inverse_error']),
    accumulated_absolute_physical_work_error=float(sum(abs(accepted['mechanical_work']))),
    signed_physical_work_error=float(sum(accepted['mechanical_work'])),
    absolute_external_work=float(sum(abs(accepted['Wext']))),
    last_integrated_positive_rate=float(positive),last_integrated_negative_rate=float(negative),
    last_integrated_cancellation_fraction=float(1-abs(positive+negative)/(positive-negative)))
   records.append(row)
   target=OUT/label;target.mkdir(exist_ok=True)
   for p in [run/'command.json',run/'input.prm',run/'run.log',*run.glob('completion-observed.json'),out/'preload-checks.csv',out/'pilot-summary.csv',out/'tip-initial.csv',out/'bulk-accepted.csv',out/'tip-power.csv',out/'tip-checks.csv',out/'tip-profile.csv',out/'tip-timers.csv',*out.glob('pilot-sections-*.csv'),out/f'tip-sections-{end:04d}.csv']:
    shutil.copy2(p,target/p.name)
comparisons=[];perturbations=[]
for nx in (256,512):
 for law in ('weakening','neutral'):
  if (nx,law,'background') not in data or (nx,law,'perturbed') not in data:continue
  bg,_=data[nx,law,'background'];s,_=data[nx,law,'perturbed'];n=min(len(bg),len(s))
  delta=s['integrated_rate'][:n]-bg['integrated_rate'][:n]
  perturbations.append(dict(nx=nx,law=law,common_endpoint_time=float(s['time'][n-1]),
   initial_integrated_rate_perturbation=float(delta[0]),last_integrated_rate_perturbation=float(delta[-1]),
   integrated_rate_perturbation_amplification=float(delta[-1]/delta[0]),
   relative_integrated_rate_perturbation=float(delta[-1]/bg['integrated_rate'][n-1]),
   relative_perturbation_amplification=float((delta[-1]/bg['integrated_rate'][n-1])/(delta[0]/bg['integrated_rate'][0])),
   added_integrated_slip=float(s['integrated_slip'][n-1]-bg['integrated_slip'][n-1]),
   mean_section_resistance_difference=float(s['mean_patch_resistance'][n-1]-bg['mean_patch_resistance'][n-1])))
for law in ('weakening','neutral'):
 for case in ('background','perturbed'):
  if (256,law,case) not in data or (512,law,case) not in data:continue
  a,pa=data[256,law,case];b,pb=data[512,law,case];n=min(len(a),len(b));a=a[:n];b=b[:n]
  assert np.max(abs(a['time']-b['time']))<1e-14
  row=dict(law=law,case=case,common_endpoint_time=float(a['time'][-1]))
  for nx,path in ((256,pa),(512,pb)):
   history=read(path/'bulk-accepted.csv')[:n-1]
   row[f'mesh_{nx}_common_signed_work_error']=float(sum(history['mechanical_work']))
   row[f'mesh_{nx}_common_absolute_work_error']=float(sum(abs(history['mechanical_work'])))
   row[f'mesh_{nx}_common_absolute_external_work']=float(sum(abs(history['Wext'])))
  for field in ('peak_rate','integrated_rate','integrated_slip'):
   row[field+'_relative_max_difference']=float(max(abs(a[field]-b[field]))/max(abs(b[field])))
  row['traction_change_absolute_max_difference']=float(max(abs((a['mean_patch_resistance']-a['mean_patch_resistance'][0])-(b['mean_patch_resistance']-b['mean_patch_resistance'][0]))))
  row['traction_change_relative_max_difference']=row['traction_change_absolute_max_difference']/float(max(abs(b['mean_patch_resistance']-b['mean_patch_resistance'][0])))
  ta=crossing(a['stage_time'],a['peak_rate'],4e-4);tb=crossing(b['stage_time'],b['peak_rate'],4e-4)
  row['rate_doubling_marker_time_difference']=None if ta is None or tb is None else ta-tb
  end=int(a['step'][-1]);sa=read(pa/f'pilot-sections-{end:04d}.csv');sb=read(pb/f'pilot-sections-{end:04d}.csv')
  for field in ('rate_stage','age_endpoint','slip_endpoint','resistance_stage'):
   ref=np.interp(sb['x'],sa['x'],sa[field]);row[field+'_relative_L2_difference']=float(np.sqrt(sum(sb['ds']*(ref-sb[field])**2)/sum(sb['ds']*sb[field]**2)))
  comparisons.append(row)
fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
colors={'weakening':'tab:red','neutral':'tab:blue'}
for (nx,law,case),(s,out) in data.items():
 if case!='perturbed':continue
 label=f'{nx} '+('weakening' if law=='weakening' else 'b=a control')
 style='--' if nx==256 else '-'
 kw=dict(color=colors[law],ls=style,label=label,marker='o' if nx==512 else None,ms=3)
 axes[0,0].plot(s['stage_time']*1e3,s['peak_rate']*1e3,**kw)
 axes[0,1].plot(s['time']*1e3,s['integrated_slip'],**kw)
 axes[1,0].plot(s['stage_time']*1e3,s['mean_patch_resistance']-s['mean_patch_resistance'][0],**kw)
 if (nx,law,'background') in data:
  bg,_=data[nx,law,'background'];n=min(len(bg),len(s))
  delta=s['integrated_rate'][:n]-bg['integrated_rate'][:n]
  relative=delta/bg['integrated_rate'][:n]
  axes[1,1].plot(s['stage_time'][:n]*1e3,relative/relative[0],**kw)
labels=[('Peak rate (mm/s)','Early loaded response'),('Integrated slip (m²)','Endpoint cumulative slip'),('Mean traction change (Pa)','Midpoint section resistance (τ−Q):S'),('Relative-rate perturbation / initial','Perturbation relative to own background')]
for ax,(ylabel,title) in zip(axes.flat,labels):ax.set_xlabel('Time (ms)');ax.set_ylabel(ylabel);ax.set_title(title);ax.grid(alpha=.2);ax.legend(fontsize=8)
for ext in ('png','pdf'):fig.savefig(OUT/f'early-native-comparison.{ext}',dpi=170)
plt.close(fig)
record=dict(runs=records,mesh_comparisons=comparisons,perturbations=perturbations,note='Short forced response; no complete autonomous event, time refinement, free growth, or selected spacing is established.')
(OUT/'comparison.json').write_text(json.dumps(record,indent=2)+'\n')
for name,rows in [('runs',records),('mesh-comparisons',comparisons),('perturbations',perturbations)]:
 if rows:
  with (OUT/(name+'.csv')).open('w') as f:
   writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)));writer.writeheader();writer.writerows(rows)
print(json.dumps(record,indent=2))
