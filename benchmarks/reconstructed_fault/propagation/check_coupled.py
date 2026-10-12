#!/usr/bin/env python3
"""Check actual partial coupled evidence; a diagnosed failure is not qualification."""
import csv,json,math,sys
from pathlib import Path
here=Path(__file__).resolve().parent;build=here.parents[2]/'build-propagation-integration'
def rows(p):
 with p.open() as f:r=[{k:float(v) for k,v in x.items()} for x in csv.DictReader(f)]
 assert r and all(math.isfinite(v) for x in r for v in x.values()),p
 return r
result={}
for label,failed_step in zip(sys.argv[1:],(2,3)):
 output=build/('output-'+label);trajectory=rows(output/'feedback.csv')
 for r in trajectory:
  assert r['phase_kkt']<=1e-11 and r['mechanical_residual']<1e-8
  assert 0<=r['phi_min']<=r['phi_max']<=1
  assert r['H_increment_integral']>=0 and r['H_cumulative_integral']>=0
  assert r['trial_observations']>0
  assert r['Ih_min']>0
 assert trajectory[-1]['time']==.25 and trajectory[-1]['H_cumulative_integral']>1000
 entry=rows(output/f'phase-entry-{failed_step}.csv')[0]
 assert entry['phase_kkt']<=1e-11 and 0<=entry['phi_min']<=entry['phi_max']<=1
 assert entry['change']>0.08 and entry['Ih_min']<-7000
 raw=rows(output/f'normalization-positive-profiles-{failed_step}.csv')
 projected=rows(output/f'normalization-projection-diagnosis-{failed_step}.csv')
 assert all(r['Ih_Q16']>0 and r['Ih_Q32']>0 for r in raw)
 assert min(r['Q16_projected'] for r in projected)<-7000 and min(r['Q32_projected'] for r in projected)<-7000
 discrepancy=max(abs(r['native_Ih']-r['Q32_projected']) for r in projected)
 assert discrepancy < abs(entry['Ih_min']) # The sign conclusion is resolved by the independent calculation.
 uncertainty=rows(output/f'failed-phase-uncertainty-{failed_step}.csv')[0]
 spread=rows(output/f'failed-phase-spread-{failed_step}.csv')[0]
 result[label]=dict(trajectory=trajectory,failed_phase_entry=entry,uncertainty=uncertainty,spread=spread,
   positive_profile_min=min(r['Ih_Q32'] for r in raw),Q16_Q32_integral_difference=max(abs(r['Ih_Q16']-r['Ih_Q32']) for r in raw),
   native_Q32_projection_difference=discrepancy,box_equilibrium=True,normalization_admissible=False,
   healing_within_retained_straight_allowance=entry['healing']<=4.922e-12)
a,b=list(result.values())
for field in ('phi_max','Ih_min','Ih_max','q_min','q_max'):
 assert a['trajectory'][0][field]==b['trajectory'][0][field]
result['same_initial_phase_and_cohesive_state']=True
result['common_accepted_time']=.25
result['half_dt_to_dt_cumulative_H_ratio']=b['trajectory'][-1]['H_cumulative_integral']/a['trajectory'][-1]['H_cumulative_integral']
result['full_requested_interval_completed']=False
result['feedback_refinement_conclusion']='inconclusive: normalization failure truncates both paths; one-step lag prevents comparing current phase as equal-H equilibrium'
result['P1_fixed_support_gate']=True;result['P2_qualified']=False
print(json.dumps(result,indent=2));sys.exit(2)
