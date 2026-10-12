#!/usr/bin/env python3
"""Verify preservation, direct irreversibility and the stated limited qualification."""
from pathlib import Path
import hashlib,json,csv
import numpy as np
root=Path(__file__).resolve().parents[3];b=root/'build-propagation-integration';e=Path(__file__).parent/'irreversibility_evidence';baseline=b/'irreversibility-baseline'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(p.open())]
def field(p,key):
 v=np.zeros((33,33))
 for r in rows(p):v[round(32*r['x']),round(32*r['y'])]=r[key]
 return v
old=json.loads((baseline/'manifest.json').read_text());m=json.loads((e/'manifest.json').read_text())
for name,h in old['hashes'].items():
 assert sha(baseline/name)==h,name
 if name.startswith(('benchmarks/reconstructed_fault/propagation/coupling_evidence/','build-propagation-integration/output-coupling_evolving','build-propagation-integration/output-coupling_diagnosis')):assert sha(root/name)==h,name
for name,h in m['hashes'].items():assert sha(root/name)==h,name
assert m['preserved_previous_artifacts']==3154 and not old['evolving_qualification_pass']
assert m['direct_irreversibility_checks_pass'] and m['common_end_time_comparison_complete']
assert not m['full_damage_feedback_baseline_qualified'] and not m['P3_run'] and not m['stress_scaling_run']
assert sha(b/'aspect-release')==sha(b/'irreversibility-final-aspect')==m['core_sha256']
assert sha(b/'plugin/libirreversibility_feedback.release.so')==sha(b/'irreversibility-final-plugin.so')
for tag,dt,k in [('125',.125,8),('0625',.0625,16),('03125',.03125,32)]:
 label='irreversibility_dt'+tag;p=b/('output-'+label);run=json.loads((e/(label+'.json')).read_text());f=rows(p/'feedback.csv')
 assert run['code']==0 and f[-1]['time']==1 and len(f)==k+1
 assert run['hashes']['build-propagation-integration/aspect-release']==m['core_sha256']
 previous=field(p/'material-phase-accepted-0.csv','phi')
 assert (p/'material-phase-accepted-0.csv').read_bytes()==(b/('output-coupling_evolving_dt'+tag)/'material-phase-accepted-0.csv').read_bytes()
 for step,r in enumerate(f):
  v=field(p/f'material-phase-accepted-{step}.csv','phi');assert np.all(v>=previous) and np.max(v)<1 and np.min(v)>=0;previous=v
  assert r['phase_kkt']<=1e-11 and r['mechanical_residual']<=1e-8 and r['phase_healing']==0 and r['Ih_min']>0 and r['H_increment_integral']>=0
 assert f[-1]['phi_max']-f[0]['phi_max']>.12 and f[-1]['H_cumulative_integral']>3000
 label='irreversibility_replay_guard_dt'+tag;p=b/('output-'+label);run=json.loads((e/(label+'.json')).read_text())
 assert run['code']==1 and 'frozen-H irreversibility replay completed; no history advanced' in (e/(label+'.log')).read_text()
 assert run['hashes']['build-propagation-integration/aspect-release']==m['core_sha256']
 v=field(p/'irreversibility-replay.csv','irreversible');lower=field(p/'irreversibility-replay.csv','lower');r=field(p/'irreversibility-replay.csv','residual');assert np.all(v>=lower) and np.max(v)<1
 active=v==lower;free=v>lower;assert np.max(r[active])<=1e-11 and np.max(abs(r[free]))<=1e-11
 z=rows(p/'irreversibility-replay-metrics.csv')[0];assert z['repeat_error']==0 and z['complementarity']<1e-14 and z['KKT']<=1e-11
 re=json.loads((e/'replays.json').read_text())[tag];assert re['identical_box_and_lower'] and re['core_difference_inf']<=1.6e-10
 for name in ['box','irreversible']:assert min(r['projected'] for r in rows(p/('replay-normalization-'+name+'.csv')))>0
zero=rows(b/'output-irreversibility_zero/feedback.csv');assert all(r['phase_change']==r['phase_healing']==r['H_cumulative_integral']==0 for r in zero)
assert '622 assertions in 7 test cases' in (b/'irreversibility-unit-final.log').read_text()
for r in json.loads((e/'configuration-guards.json').read_text()):assert r['exit_code']==1 and r['expected_message'] in (e/(r['label']+'.log')).read_text()
for r in json.loads((e/'mature-compatibility.json').read_text()):assert r['code']==0 and r['identical_csvs']==2
ref=json.loads((e/'refinement.json').read_text());z=ref['common_times'][-1];assert z['time']==1
assert z['pair_differences'][1]['phase_inf']<z['pair_differences'][0]['phase_inf']
assert z['pair_differences'][1]['q_inf']<z['pair_differences'][0]['q_inf']
c=json.loads((e/'comparison.json').read_text())
for z in c['cases'].values():assert z['normalization'][str(z['accepted_steps'])]['max_source_error']>.06
assert all(z['cumulative_profile_work']<-90 for z in ref['energetics'].values())
print('Verified',len(m['hashes']),'current artifacts and',len(old['hashes']),'preserved incoming artifacts.')
print('Three trajectories complete at t=1 with direct irreversibility; refinement improves but substantial temporal, normalization and energetic questions remain. Full baseline qualification: false.')
