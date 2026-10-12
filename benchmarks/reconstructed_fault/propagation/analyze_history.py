#!/usr/bin/env python3
"""Check native path experiments and unchanged reference reruns; never update H."""
import csv, hashlib, json, math, os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-history-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
here=Path(__file__).resolve().parent
build=here.parents[2]/'build-propagation-integration'
e=here/'history_evidence';e.mkdir(exist_ok=True)
def rows(p):
 with p.open() as f:
  return [{k:(v if k=='path' else float(v)) for k,v in r.items()} for r in csv.DictReader(f)]
native=rows(build/'output-history-native-v2/history-native.csv')
assert len(native)==560
summary={}
for eta in [1e8,1e22]:
 for path in ['loading','relaxation','cycle','hold']:
  results=[]
  for dt in [.125,.0625,.03125]:
   r=[v for v in native if v['eta']==eta and v['path']==path and v['dt']==dt]
   last=r[-1];T=last['time'];g=last['g'];H0=last['H0'];G=1e6
   if path=='loading': integral=600**2*T+600*200*T*T+200**2*T**3/3
   elif path=='hold': integral=600**2*T
   elif path=='cycle': integral=255000*T # exactly two complete periods
   else: integral=600**2*eta/(2*G)*(-math.expm1(-2*G*T/eta))
   work=last['state_energy']+integral/(eta*g*g)
   potential_error=max(abs(v['potential_derivative']-v['candidate'])/max(1,abs(v['candidate'])) for v in r)
   assert potential_error<2e-7
   assert all(v['H_max']==H0 for v in r)
   if path=='relaxation': assert max(abs(v['candidate'])/H0 for v in r)<1e-9
   if eta==1e22:
    assert abs(last['signed_sum']-(last['state_energy']-H0))/H0<1e-12
    assert all(abs(v['intact_candidate']-v['q']**2/(2*G))<1e-12 for v in r)
   results.append(dict(dt=dt,H0=H0,H_final=last['H_max'],max_candidate=max(v['candidate'] for v in r),
      min_candidate=min(v['candidate'] for v in r),state_energy_final=last['state_energy'],
      candidate_sum=last['signed_sum'],positive_candidate_sum=last['positive_sum'],
      signed_work_expected=work,relative_sum_work_error=abs(H0+last['signed_sum']-work)/H0,
      potential_derivative_error=potential_error,intact_final=last['intact_candidate'],near_intact_final=last['near_intact_candidate']))
  if eta==1e8 and path!='relaxation':
   assert results[-1]['relative_sum_work_error']<results[0]['relative_sum_work_error']
  summary[f'{eta:g}/{path}']=results
repeats={};series=[]
for old,new in [('material_reference_a016','history_reference_repeat'),('material_reference_a016_half','history_reference_repeat_half')]:
 d=build/('output-'+new);old_dir=build/('output-'+old)
 record=json.loads((here/'material_evidence'/(new+'.json')).read_text());assert record['code']==0
 # Compare every saved diagnostic CSV, including native publication checks,
 # phase entries, accepted fields and direct-normalization samples.
 files=list(old_dir.glob('*.csv'));assert files
 for p in files: assert p.read_bytes()==(d/p.name).read_bytes(),p.name
 f=rows(d/'feedback.csv');local=rows(d/'material-local.csv');particles=rows(d/'material-particles.csv')
 for v in f:
  assert 0<=v['phi_min']<=v['phi_max']<1 and v['phase_kkt']<=1e-11
  assert v['phase_healing']<=4.922e-12 and v['mechanical_residual']<=1e-8 and v['Ih_min']>0
  assert v['H_increment_integral']>=0 and v['offband_H_increases']==0
 norms={}
 for step in sorted({0,max(range(len(f)),key=lambda j:f[j]['phi_max'])}):
  rs=rows(d/f'material-normalization-accepted-{step}.csv')
  norms[str(step)]=max(abs(v['normalized_source_integral']-1) for v in rs)
 repeats[new]=dict(dt=record['dt'],accepted_steps=len(f)-1,byte_identical_csv_count=len(files),
     phase_gain=f[-1]['phi_max']-f[0]['phi_max'],H_increase=f[-1]['H_cumulative_integral'],
     KKT=max(v['phase_kkt'] for v in f),healing=max(v['phase_healing'] for v in f),
     mechanical_residual=max(v['mechanical_residual'] for v in f),source_errors=norms)
 series.append((f,local,particles))
common_times=[]
for c in series[0][0]:
 f=next(v for v in series[1][0] if v['time']==c['time'])
 common_times.append(dict(time=c['time'],phi_coarse=c['phi_max'],phi_fine=f['phi_max'],
  H_growth_coarse=c['H_cumulative_integral'],H_growth_fine=f['H_cumulative_integral']))
common_slip={}
for x in [0.,.5,1.]:
 local=[[v for v in s[1] if abs(v['x']-x)<1e-12] for s in series]
 ps=[[v for v in s[2] if v['target']==x] for s in series]
 slip=min(v[-1]['slip'] for v in local)
 common_slip[str(x)]=dict(slip=slip,values=[])
 for r,p in zip(local,ps):
  values={key:float(np.interp(slip,[v['slip'] for v in r],[v[key] for v in r])) for key in ['phi','q','time']}
  # A fixed near-surface parent is paired with the corresponding fault-node slip.
  values['nearby_parent_H']=float(np.interp(slip,[v['slip'] for v in r],[v['H_new'] for v in p]))
  values['nearby_parent_H0']=p[0]['H0']
  common_slip[str(x)]['values'].append(values)
result=dict(native_steps=len(native),native=summary,repeats=repeats,common_times=common_times,common_slip=common_slip,
 physical_definition_unresolved=True,production_law_changed=False,stress_scaling_run=False,feedback_refinement_pass=False)
(e/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
fig,axes=plt.subplots(1,3,figsize=(12,3.5),constrained_layout=True)
for dt in [.125,.0625,.03125]:
 r=[v for v in native if v['eta']==1e8 and v['dt']==dt and v['path']=='loading']
 axes[0].plot([v['time'] for v in r],[v['candidate']/v['H0'] for v in r],'.-',label=f'dt={dt}')
for (f,loc,ps),name in zip(series,['dt=0.125','dt=0.0625']):
 axes[1].plot([v['time'] for v in f],[v['phi_max']-f[0]['phi_max'] for v in f],'.-',label=name)
 r=[v for v in loc if v['x']==0]
 axes[2].plot([v['slip'] for v in r],[v['phi']-r[0]['phi'] for v in r],'.-',label=name)
axes[0].set(xlabel='Time (s)',ylabel='Native increment candidate / initial H',title='Fixed phase, prescribed smooth q')
axes[1].set(xlabel='Time (s)',ylabel='Phase maximum increase',title='Unchanged coupled reference')
axes[2].set(xlabel='Left-endpoint local slip (m)',ylabel='Local phase increase',title='Accepted states only')
for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.25)
fig.savefig(e/'history-diagnosis.png',dpi=180);fig.savefig(e/'history-diagnosis.pdf');plt.close(fig)
print(json.dumps(dict(native=summary,repeats=repeats,common_slip=common_slip),indent=2))
