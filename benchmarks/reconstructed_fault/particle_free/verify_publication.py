#!/usr/bin/env python3
"""Independent selected-state quadrature/work checks and native publication gates."""
import argparse,json,re,time
from pathlib import Path
import numpy as np
from assess_pf6a import state
p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('assessment',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
r=json.loads((a.run/'record.json').read_text());assessment=json.loads(a.assessment.read_text())
assert assessment['native_completed']
strict_reference=all(assessment['implementation_target_passes'].values())
checks=assessment['checks']
for key in ['source_recurrence_scaled','same_field_energy_scaled','same_field_shear_weak_load_scaled','same_field_cohesive_weak_load_scaled']:
 assert checks[key]<1e-9,(key,checks[key])
assert checks['healing']==0 and checks['KKT']<1e-9 and checks['fault_KKT_Pa']<1e-6
metrics=np.genfromtxt(a.run/'run/metrics.csv',delimiter=',',names=True)
assert len(metrics)==73 and np.array_equal(metrics['publications'],np.arange(73))
gates=np.genfromtxt(a.run/'run/gates.csv',delimiter=',',names=True,dtype=None,encoding='utf8');assert np.all(gates['passed']==1)
quad={}
for step in [1,68,72]:
 start=time.monotonic();root=a.run/'run'
 quad[str(step)]=state(root/f'state-{step}',root/f'state-{step-1}',1/128,step==72)
 quad[str(step)]['assessment_seconds']=time.monotonic()-start
 print('independent q12/24/32 and ray check',step,flush=True)
pub=re.findall(r'DG Maxwell publication: error=([^ ]+) evaluation allowance=([^ ]+) net-stress allowance=([^ ]+) maximum error/allowance=([^ ]+) old allowance failures=(\d+)',(a.run/'run.log').read_text())
assert len(pub)==72
summary=dict(samples=72*9216,accepted_publications=72,maximum_error_Pa=max(float(v[0])for v in pub),
             maximum_allowance_Pa=max(float(v[1])for v in pub),maximum_error_over_bound=max(float(v[3])for v in pub),
             old_allowance_full_tensor_failures=sum(int(v[4])for v in pub))
eval=np.genfromtxt(a.run/'run/stress-evaluation.csv',delimiter=',',names=True);compared=eval[eval['compared']==1]
assert compared['step'].tolist()==[0.,1.,68.,72.] and np.max(compared['scaled_error'])<1e-9
result=dict(operator_and_publication_checks_passed=True,strict_reference_match_passed=strict_reference,
            strict_reference_target=assessment['implementation_target'],strict_reference_targets=assessment['implementation_target_passes'],gate_count=len(gates),core_sha256=r['executable_sha256'],observer_sha256=r['plugin_sha256'],
            native_seconds=r['seconds'],maximum_RSS_KiB=r['max_rss_KiB'],publication=summary,
            native_reference_maximum=assessment['maximum'],independent_checks=checks,
            quadrature=quad,batch_point_comparison=dict(steps=compared['step'].tolist(),maximum_error_Pa=float(max(compared['error_Pa'])),
              maximum_scaled_error=float(max(compared['scaled_error'])),batch_seconds=float(sum(eval['batch_seconds'])),
              point_verification_seconds=float(sum(eval['point_seconds'])),points_per_snapshot=147456),
            scope='serial fixed 2x512, fixed reference surface, no advection; original equations/defaults/limiter settings')
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='quadrature'},indent=2))
